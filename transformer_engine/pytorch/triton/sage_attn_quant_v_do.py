import pytest
import torch
import triton.tools.experimental_descriptor

import triton
import triton.language as tl

import torch.nn.functional as F

RCP_LN2: tl.constexpr = 1.4426950408889634 # exp(x) = exp2(x * log2(e)) = exp2(x / ln(2)) = exp2(x * RCP_LN2)
LN2: tl.constexpr = 0.6931471824645996  # = ln(2)
MIN_SCALE: tl.constexpr = 1e-4

@triton.jit
def quant_per_block(Input, Output, Scale, 
                    stride_ib, stride_ih, stride_is,
                    stride_ob, stride_oh, stride_os,
                    stride_sb, stride_sh, stride_sk,
                    sm_scale: tl.constexpr,
                    HEAD_DIM: tl.constexpr,
                    SEQ_LEN: tl.constexpr,
                    BLOCK: tl.constexpr,
                    QUANT_TYPE: tl.constexpr): 
    start_m = tl.program_id(0)
    off_h = tl.program_id(1)
    off_b = tl.program_id(2)

    offs_n = start_m * BLOCK + tl.arange(0, BLOCK)
    offs_k = tl.arange(0, HEAD_DIM)

    input_ptrs = Input + off_b * stride_ib + off_h * stride_ih + offs_n[:, None] * stride_is + offs_k[None, :]
    output_ptrs = Output + off_b * stride_ob + off_h * stride_oh + offs_n[:, None] * stride_os + offs_k[None, :]
    scale_ptrs = Scale + off_b * stride_sb + off_h * stride_sh + start_m * stride_sk

    x = tl.load(input_ptrs, mask=offs_n[:, None] < SEQ_LEN)
    x = x.to(tl.float32)
    x *= sm_scale

    if QUANT_TYPE == 0 or QUANT_TYPE == 3:  # int8
        # scale = tl.max(tl.abs(x)) / 127. + 1e-8
        scale = tl.maximum(tl.max(tl.abs(x)) / 127. + 1e-8, MIN_SCALE)
        x_quant = ((x / scale) + 0.5 * tl.where(x >= 0, 1, -1)).to(tl.int8)
    elif QUANT_TYPE == 1:  # e4m3
        # scale = tl.max(tl.abs(x)) / 448. + 1e-8
        scale = tl.maximum(tl.max(tl.abs(x)) / 448. + 1e-8, MIN_SCALE)
        x_quant = (x / scale).to(tl.float8e4nv)
    elif QUANT_TYPE == 2:  # e5m2
        # scale = tl.max(tl.abs(x)) / 57344. + 1e-8
        scale = tl.maximum(tl.max(tl.abs(x)) / 57344. + 1e-8, MIN_SCALE)
        x_quant = (x / scale).to(tl.float8e5)
    else:
        tl.static_assert(False, "Unsupported quant type")
    
    tl.store(output_ptrs, x_quant, mask=offs_n[:, None] < SEQ_LEN)
    tl.store(scale_ptrs, scale)

'''
因果模式 (STAGE=3):
[矩阵上三角+对角线]
┌───────────────┐
│ ░░░░░░░░░░░░░░│  阶段1 (STAGE=1)：上三角部分，Q * K^T 无掩码限制
│ ░░░░░░░░░░░░░░│  
│ ░░░░░░░░░░░░░░│  
├───────────────┤
│ █████████████░│  阶段2 (STAGE=2)：当前块，需应用三角掩码
│ ███████████░░░│  
│ █████████░░░░░│  
└───────────────┘
'''
@triton.jit
def _attn_fwd_inner(acc, l_i, m_i, q,
                    K_ptrs, V_ptrs, 
                    stride_k, stride_v, start_m, 
                    q_scale, K_scale_ptr, V_scale_ptr, 
                    HEAD_DIM: tl.constexpr, SEQ_LEN: tl.constexpr, GROUPS: tl.constexpr,
                    BLOCK_Q: tl.constexpr, BLOCK_KV: tl.constexpr, 
                    STAGE: tl.constexpr, offs_m: tl.constexpr, offs_n: tl.constexpr):
    # range of values handled by this stage
    if STAGE == 1:
        lo, hi = 0, start_m * BLOCK_Q
    elif STAGE == 2:
        lo, hi = start_m * BLOCK_Q, (start_m + 1) * BLOCK_Q
        lo = tl.multiple_of(lo, BLOCK_Q)
        K_scale_ptr += lo // BLOCK_KV
        V_scale_ptr += lo // BLOCK_KV
        K_ptrs += stride_k * lo
        V_ptrs += stride_v * lo
    # causal = False
    else:
        lo, hi = 0, SEQ_LEN

    for start_n in range(lo, hi, BLOCK_KV):
        start_n = tl.multiple_of(start_n, BLOCK_KV)

        k = tl.load(K_ptrs, mask=offs_n[None, :] < (SEQ_LEN - start_n))
        k_scale = tl.load(K_scale_ptr)

        qk = tl.dot(q, k).to(tl.float32) * q_scale * k_scale 
        if STAGE == 2:
            mask = offs_m[:, None] >= (start_n + offs_n[None, :])
            qk = qk + tl.where(mask, 0, -1.0e6)
            m_ij = tl.maximum(m_i, tl.max(qk, 1))
            qk = qk - m_ij[:, None]
        else:
            m_ij = tl.maximum(m_i, tl.max(qk, 1))
            qk = qk - m_ij[:, None]

        p = tl.math.exp2(qk)
        l_ij = tl.sum(p, 1)
        
        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + l_ij

        acc = acc * alpha[:, None]
        
        v = tl.load(V_ptrs, mask=offs_n[:, None] < (SEQ_LEN - start_n))
        p = p.to(v.dtype) 
        acc = tl.dot(p, v, acc)

        m_i = m_ij
        K_ptrs += BLOCK_KV * stride_k
        V_ptrs += BLOCK_KV * stride_v
        K_scale_ptr += 1
        V_scale_ptr += 1
    return acc, l_i, m_i

@triton.jit
def _attn_fwd_inner_quant(acc, l_i, m_i, q,
                          K_ptrs, V_ptrs, stride_k, stride_v, 
                          start_m, 
                          q_scale, K_scale_ptr, V_scale_ptr, 
                          HEAD_DIM: tl.constexpr, SEQ_LEN: tl.constexpr, GROUPS: tl.constexpr,
                          BLOCK_Q: tl.constexpr, BLOCK_KV: tl.constexpr, 
                          STAGE: tl.constexpr, offs_m: tl.constexpr, offs_n: tl.constexpr,
                          QUANT_TYPE: tl.constexpr):
    # range of values handled by this stage
    if STAGE == 1:
        lo, hi = 0, start_m * BLOCK_Q
    elif STAGE == 2:
        lo, hi = start_m * BLOCK_Q, (start_m + 1) * BLOCK_Q
        lo = tl.multiple_of(lo, BLOCK_Q)
        K_scale_ptr += lo // BLOCK_KV
        V_scale_ptr += lo // BLOCK_KV
        K_ptrs += stride_k * lo
        V_ptrs += stride_v * lo
    # causal = False
    else:
        lo, hi = 0, SEQ_LEN

    for start_n in range(lo, hi, BLOCK_KV):
        start_n = tl.multiple_of(start_n, BLOCK_KV)
        k = tl.load(K_ptrs,  mask=offs_n[None, :] < (SEQ_LEN - start_n))
        k_scale = tl.load(K_scale_ptr)

        qk = tl.dot(q, k).to(tl.float32) * q_scale * k_scale 
        if STAGE == 2:
            mask = offs_m[:, None] >= (start_n + offs_n[None, :])
            qk = qk + tl.where(mask, 0, -1.0e6)
            m_ij = tl.maximum(m_i, tl.max(qk, 1))
            qk = qk - m_ij[:, None]
        else:
            m_ij = tl.maximum(m_i, tl.max(qk, 1))
            qk = qk - m_ij[:, None]

        p = tl.math.exp2(qk)
        l_ij = tl.sum(p, 1)

        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + l_ij

        acc = acc * alpha[:, None]
        
        if QUANT_TYPE == 0:  # int8
            p_scale = tl.maximum(tl.max(tl.abs(p)) / 127. + 1e-8, MIN_SCALE)
            p_quant = ((p / p_scale) + 0.5 * tl.where(p >= 0, 1, -1)).to(tl.int8)
        elif QUANT_TYPE == 1:  # e4m3
            p_scale = tl.maximum(tl.max(tl.abs(p)) / 448. + 1e-8,  MIN_SCALE)
            p_quant = (p / p_scale).to(tl.float8e4nv)
        elif QUANT_TYPE == 2:  # e5m2
            p_scale = tl.maximum(tl.max(tl.abs(p)) / 57344. + 1e-8, MIN_SCALE)
            p_quant = (p / p_scale).to(tl.float8e5)
        else:
            tl.static_assert(False, "Unsupported quant type")

        v = tl.load(V_ptrs, mask=offs_n[:, None] < (SEQ_LEN - start_n))
        v_scale = tl.load(V_scale_ptr)

        if QUANT_TYPE == 0:
            accumulator = tl.zeros([BLOCK_Q, HEAD_DIM], dtype=tl.int32)
            middle = tl.dot(p_quant, v, accumulator, out_dtype=tl.float32) * v_scale * p_scale
        else:
            accumulator = tl.zeros([BLOCK_Q, HEAD_DIM], dtype=tl.float32)
            middle = tl.dot(p_quant, v, accumulator).to(tl.float32) * v_scale * p_scale
        acc = acc + middle

        m_i = m_ij 
        K_ptrs += BLOCK_KV * stride_k
        V_ptrs += BLOCK_KV * stride_v
        K_scale_ptr += 1
        V_scale_ptr += 1
    return acc, l_i, m_i

@triton.jit
def _attn_fwd(Q, K, V, 
              Q_scale, K_scale, V_scale,
              LSE, Out, 
              stride_qb, stride_qh, stride_qs, stride_qd, 
              stride_kb, stride_kh, stride_ks, stride_kd, 
              stride_vb, stride_vh, stride_vs, stride_vd, 
              stride_ob, stride_oh, stride_os, stride_od, 
              HEAD_NUM: tl.constexpr, SEQ_Q: tl.constexpr, # Q
              GROUPS: tl.constexpr, HEAD_DIM: tl.constexpr, SEQ_KV: tl.constexpr, # K
              BLOCK_Q: tl.constexpr, BLOCK_KV: tl.constexpr, 
              STAGE: tl.constexpr,
              QUANT_TYPE: tl.constexpr,
              ):
    tl.static_assert(BLOCK_KV <= HEAD_DIM)
    
    start_m = tl.program_id(0)
    off_h = tl.program_id(1).to(tl.int64)
    off_b = tl.program_id(2).to(tl.int64)

    offs_m = start_m * BLOCK_Q + tl.arange(0, BLOCK_Q)
    offs_n = tl.arange(0, BLOCK_KV)
    offs_k = tl.arange(0, HEAD_DIM)

    q_scale_offset = (off_b * HEAD_NUM + off_h) * tl.cdiv(SEQ_Q, BLOCK_Q)
    k_scale_offset = (off_b * (HEAD_NUM // GROUPS) + off_h // GROUPS) * tl.cdiv(SEQ_KV, BLOCK_KV)  
    v_scale_offset = (off_b * (HEAD_NUM // GROUPS) + off_h // GROUPS) * tl.cdiv(SEQ_KV, BLOCK_KV)

    Q_ptrs = Q + (off_b * stride_qb + off_h * stride_qh) + offs_m[:, None] * stride_qs + offs_k[None, :]
    Q_scale_ptr = Q_scale + q_scale_offset + start_m
    q_scale = tl.load(Q_scale_ptr)
    K_ptrs = K + (off_b * stride_kb + (off_h // GROUPS) * stride_kh) + offs_n[None, :] * stride_ks + offs_k[:, None] 
    K_scale_ptr = K_scale + k_scale_offset
    V_ptrs = V + (off_b * stride_vb + (off_h // GROUPS) * stride_vh) + offs_n[:, None] * stride_vs + offs_k[None, :]
    V_scale_ptr = V_scale + v_scale_offset
    O_block_ptr = Out + (off_b * stride_ob + off_h * stride_oh) + offs_m[:, None] * stride_os + offs_k[None, :]
    
    m_i = tl.zeros([BLOCK_Q], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_Q], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_Q, HEAD_DIM], dtype=tl.float32)

    q = tl.load(Q_ptrs, mask=offs_m[:, None] < SEQ_Q) # load q: it will stay in SRAM throughout
    
    if QUANT_TYPE == 3:
        if STAGE & 1: # 1 or 3
            acc, l_i, m_i = _attn_fwd_inner(acc, l_i, m_i, q, K_ptrs, V_ptrs, 
                                            stride_ks, stride_vs, start_m, 
                                            q_scale, K_scale_ptr, V_scale_ptr, 
                                            HEAD_DIM, SEQ_KV, GROUPS,
                                            BLOCK_Q, BLOCK_KV,
                                            4 - STAGE, offs_m, offs_n)
        if STAGE & 2: # 2 or 3
            acc, l_i, m_i = _attn_fwd_inner(acc, l_i, m_i, q, K_ptrs, V_ptrs, 
                                            stride_ks, stride_vs, start_m, 
                                            q_scale, K_scale_ptr, V_scale_ptr, 
                                            HEAD_DIM, SEQ_KV, GROUPS,
                                            BLOCK_Q, BLOCK_KV, 
                                            2, offs_m, offs_n)
    else:
        if STAGE & 1: # 1 or 3
            acc, l_i, m_i = _attn_fwd_inner_quant(acc, l_i, m_i, q, K_ptrs, V_ptrs,
                                                  stride_ks, stride_vs, start_m, 
                                                  q_scale, K_scale_ptr, V_scale_ptr, 
                                                  HEAD_DIM, SEQ_KV, GROUPS,
                                                  BLOCK_Q, BLOCK_KV, 
                                                  4 - STAGE, offs_m, offs_n, QUANT_TYPE)
        if STAGE & 2: # 2 or 3
            acc, l_i, m_i = _attn_fwd_inner_quant(acc, l_i, m_i, q, K_ptrs, V_ptrs, 
                                                  stride_ks, stride_vs, start_m, 
                                                  q_scale, K_scale_ptr, V_scale_ptr, 
                                                  HEAD_DIM, SEQ_KV, GROUPS,
                                                  BLOCK_Q,  BLOCK_KV,
                                                  2, offs_m, offs_n,  QUANT_TYPE)
    acc = acc / l_i[:, None]
    tl.store(O_block_ptr, acc.to(Out.type.element_ty), mask=(offs_m[:, None] < SEQ_Q))
    
    lse_ptrs = LSE + (off_b * SEQ_Q * HEAD_NUM + off_h * SEQ_Q) + offs_m
    m_i += tl.math.log2(l_i)
    tl.store(lse_ptrs, m_i, mask=(offs_m < SEQ_Q))

# 计算 Delta = rowsum(O ⊙ dO)
@triton.jit
def _attn_bwd_preprocess(O, DO, Delta, 
                         SEQ_LEN, BLOCK_Q: tl.constexpr, HEAD_DIM: tl.constexpr):
    off_m = tl.program_id(0) * BLOCK_Q + tl.arange(0, BLOCK_Q)
    off_h = tl.program_id(1).to(tl.int64)
    off_b = tl.program_id(2).to(tl.int64)
    off_n = tl.arange(0, HEAD_DIM)
    # load
    o = tl.load(O + off_h * off_b * HEAD_DIM * SEQ_LEN + off_m[:, None] * HEAD_DIM + off_n[None, :])
    do = tl.load(DO + off_h * off_b * HEAD_DIM * SEQ_LEN + off_m[:, None] * HEAD_DIM + off_n[None, :])
    
    delta = tl.sum(o * do, axis=1)
    # write-back
    tl.store(Delta + off_h * off_b * SEQ_LEN + off_m, delta)

# The main inner-loop logic for computing dK and dV.
# 固定处理 K 和 V 的一个块（形状为 BLOCK_KV × HEAD_DIM），然后迭代处理 Q 和 dO 的多个块
@triton.jit
def _attn_bwd_dkdv(dk, dv, 
                   Q_ptrs, DO_ptrs, k, v, 
                   LSE, D, 
                   # shared by Q/K/V/DO.
                   stride_qs, stride_qd, 
                   Q_scale_ptr, DO_scale_ptr, k_scale, v_scale, 
                   HEAD_NUM, SEQ_LEN,
                   BLOCK_Q: tl.constexpr, 
                   BLOCK_KV: tl.constexpr, 
                   HEAD_DIM: tl.constexpr, 
                   # Filled in by the wrapper.
                   start_n, start_m, num_steps, 
                   MASK: tl.constexpr, QUANT_TYPE: tl.constexpr):
    offs_m = start_m + tl.arange(0, BLOCK_Q)
    offs_n = start_n + tl.arange(0, BLOCK_KV)
    offs_k = tl.arange(0, HEAD_DIM)

    qT_ptrs = Q_ptrs + offs_m[None, :] * stride_qs + offs_k[:, None] * stride_qd
    do_ptrs = DO_ptrs + offs_m[:, None] * stride_qs + offs_k[None, :] * stride_qd
    # BLOCK_KV must be a multiple of BLOCK_Q, otherwise the code wouldn't work.
    tl.static_assert(BLOCK_KV % BLOCK_Q == 0)
    curr_m = start_m
    step_m = BLOCK_Q
    for blk_idx in range(num_steps):
        qT = tl.load(qT_ptrs) 
        do = tl.load(do_ptrs)
        lse = tl.load(LSE + offs_m) # Load lse before computing qk to reduce pipeline stall.
        q_scale = tl.load(Q_scale_ptr)
        do_scale = tl.load(DO_scale_ptr)
        
        offs_m = curr_m + tl.arange(0, BLOCK_Q)
        #! qkT: 0.1 -> -0.000022 ~ 0.000024 | 1 -> -0.001836 ~ 0.001974 | 10 -> -0.183081 ~ 0.197608
        #! lse: 0.1 -> -0.000022 ~ 0.000024 | 1 -> -0.001836 ~ 0.001974 | 10 -> 10.000704 ~ 10.002090
        qkT = tl.dot(k, qT)
        pT = tl.math.exp2(qkT - lse[None, :])
        # max_qkT = tl.max(qkT, axis=1)
        # pT = tl.math.exp2(qkT - max_qkT[:, None] - lse[None, :])
        if MASK:
            mask = (offs_m[None, :] >= offs_n[:, None])
            pT = tl.where(mask, pT, 0.0)
        
        # 2 Quant pT & Compute dV
        if QUANT_TYPE == 0:  # int8
            # pT_scale = tl.max(tl.abs(pT)) / 127. + 1e-8
            pT_scale = tl.maximum(tl.max(tl.abs(pT)) / 127. + 1e-8, MIN_SCALE)
            pT_quant = ((pT / pT_scale) + 0.5 * tl.where(pT >= 0, 1, -1)).to(tl.int8)
        elif QUANT_TYPE == 1:  # e4m3
            # pT_scale = tl.max(tl.abs(pT)) / 448.  + 1e-8
            pT_scale = tl.maximum(tl.max(tl.abs(pT)) / 448. + 1e-8,  MIN_SCALE)
            pT_quant = (pT / pT_scale).to(tl.float8e4nv)
        elif QUANT_TYPE == 2:  # e5m2
            # pT_scale = tl.max(tl.abs(pT)) / 57344. + 1e-8
            pT_scale = tl.maximum(tl.max(tl.abs(pT)) / 57344. + 1e-8, MIN_SCALE)
            pT_quant = (pT / pT_scale).to(tl.float8e5)
        else:
            tl.static_assert(False, "Unsupported quant type")
        dv += tl.dot(pT_quant, do).to(tl.float32) * do_scale * pT_scale

        # 2 dP
        dpT = tl.dot(v, tl.trans(do)).to(tl.float32) * v_scale * do_scale
        Di = tl.load(D + offs_m) # D (= delta) is pre-divided by ds_scale.
        # 1 dK
        # dsT = (pT * (dpT - Di[None, :]))
        # qT = tl.trans(qT).to(tl.float32)
        # acc = tl.zeros([BLOCK_KV, HEAD_DIM], dtype=tl.float32)
        # dk += tl.dot(dsT, qT, acc) * k_scale
        dsT = (pT * (dpT - Di[None, :])).to(qT.dtype)
        dk += tl.dot(dsT, tl.trans(qT)) 
        # Increment pointers.
        curr_m += step_m
        qT_ptrs += step_m * stride_qs
        do_ptrs += step_m * stride_qs
        Q_scale_ptr += 1
        DO_scale_ptr += 1
    return dk, dv

# the main inner-loop logic for computing dQ
# 固定处理 Q 和 dO 的一个块（形状为 BLOCK_Q2 × HEAD_DIM），然后迭代处理 K 和 V 的多个块
@triton.jit
def _attn_bwd_dq(dq, q, K_ptrs, V_ptrs, do,
                lse, D,
                 # shared by Q/K/V/DO.
                 stride_qs, stride_qd, 
                 q_scale, do_scale, K_scale_ptr, V_scale_ptr, 
                 HEAD_NUM, SEQ_LEN, 
                 BLOCK_Q: tl.constexpr, 
                 BLOCK_KV: tl.constexpr, 
                 HEAD_DIM: tl.constexpr,
                 # Filled in by the wrapper.
                 start_m, start_n, num_steps, 
                 MASK: tl.constexpr, QUANT_TYPE: tl.constexpr):
    offs_m = start_m + tl.arange(0, BLOCK_Q)
    offs_n = start_n + tl.arange(0, BLOCK_KV)
    offs_k = tl.arange(0, HEAD_DIM)
    kT_ptrs = K_ptrs + offs_n[None, :] * stride_qs + offs_k[:, None] * stride_qd
    vT_ptrs = V_ptrs + offs_n[None, :] * stride_qs + offs_k[:, None] * stride_qd
    # D (= delta) is pre-divided by ds_scale.
    Di = tl.load(D + offs_m)
    # BLOCK_Q must be a multiple of BLOCK_KV, otherwise the code wouldn't work.
    tl.static_assert(BLOCK_Q % BLOCK_KV == 0)
    curr_n = start_n
    step_n = BLOCK_KV
    for blk_idx in range(num_steps):
        kT = tl.load(kT_ptrs)
        vT = tl.load(vT_ptrs)
        k_scale = tl.load(K_scale_ptr)
        v_scale = tl.load(V_scale_ptr)
        qk = tl.dot(q, kT).to(tl.float32) 
        p = tl.math.exp2(qk - lse)
        # Autoregressive masking.
        if MASK:
            offs_n = curr_n + tl.arange(0, BLOCK_KV)
            mask = (offs_m[:, None] >= offs_n[None, :])
            p = tl.where(mask, p, 0.0)
        # 2 dP
        dp = tl.dot(do, vT).to(tl.float32) * do_scale * v_scale 
        # 1 dQ
        ds = (p * (dp - Di[:, None])).to(kT.dtype)
        dq += tl.dot(ds, tl.trans(kT)) 

        # Increment pointers.
        curr_n += step_n
        kT_ptrs += step_n * stride_qs
        vT_ptrs += step_n * stride_qs
        K_scale_ptr += 1
        V_scale_ptr += 1
    return dq

@triton.jit
def _attn_bwd(Q, K, V, DO, 
              Q_scale, K_scale, V_scale, DO_scale,
              sm_scale, causal,
              DQ, DK, DV, 
              LSE, D,
              stride_qb, stride_qh, stride_qs, stride_qd, # shared by Q/K/V/DO.
              HEAD_NUM, SEQ_LEN, GROUPS,
              BLOCK_Q1: tl.constexpr, BLOCK_KV1: tl.constexpr, 
              BLOCK_Q2: tl.constexpr, BLOCK_KV2: tl.constexpr, 
              BLK_SLICE_FACTOR: tl.constexpr, 
              HEAD_DIM: tl.constexpr, QUANT_TYPE: tl.constexpr):
    pid = tl.program_id(0)
    off_h = tl.program_id(1).to(tl.int64)
    off_b = tl.program_id(2).to(tl.int64)
    adj = (stride_qh * off_b + stride_qb * off_h).to(tl.int64) 
    off_ch = (off_b * HEAD_NUM * SEQ_LEN).to(tl.int64) # 计算当前 batch 在 LSE 和 D 中的起始内存偏移量，挪动 HEAD_NUM * SEQ_LEN 个元素

    # offset pointers for batch/head
    Q  += adj;  K  += adj;  V  += adj
    DO += adj; DQ += adj; DK += adj; DV += adj
    LSE += off_ch;  D += off_ch

    # load scales
    offs_k = tl.arange(0, HEAD_DIM)

    start_n = pid * BLOCK_KV1
    offs_n = start_n + tl.arange(0, BLOCK_KV1)
    dv = tl.zeros([BLOCK_KV1, HEAD_DIM], dtype=tl.float32)
    dk = tl.zeros([BLOCK_KV1, HEAD_DIM], dtype=tl.float32)
    # load K and V: they stay in SRAM throughout the inner loop.
    k = tl.load(K + offs_n[:, None] * stride_qs + offs_k[None, :] * stride_qd)
    v = tl.load(V + offs_n[:, None] * stride_qs + offs_k[None, :] * stride_qd)

    q_scale_offset = (off_b * HEAD_NUM + off_h) * tl.cdiv(SEQ_LEN, BLOCK_Q2)
    do_scale_offset = (off_b * HEAD_NUM + off_h) * tl.cdiv(SEQ_LEN, BLOCK_Q2)
    k_scale_offset = (off_b * HEAD_NUM + off_h) * tl.cdiv(SEQ_LEN, BLOCK_KV2)  
    v_scale_offset = (off_b * HEAD_NUM + off_h) * tl.cdiv(SEQ_LEN, BLOCK_KV2)
    Q_scale_ptr = Q_scale + q_scale_offset
    DO_scale_ptr = DO_scale + do_scale_offset
    K_scale_ptr = K_scale + k_scale_offset
    V_scale_ptr = V_scale + v_scale_offset
    q_scale = tl.load(Q_scale_ptr + start_n)
    do_scale = tl.load(DO_scale_ptr + start_n)
    k_scale = tl.load(K_scale_ptr + start_n)
    v_scale = tl.load(V_scale_ptr + start_n)

    start_m = start_n
    if causal:
        MASK_BLOCK_Q1: tl.constexpr = BLOCK_Q1 // BLK_SLICE_FACTOR
        num_steps = BLOCK_KV1 // MASK_BLOCK_Q1
        dk, dv = _attn_bwd_dkdv(dk, dv, 
                                Q, DO, k, v,
                                LSE, D, 
                                stride_qs, stride_qd, 
                                Q_scale_ptr, DO_scale_ptr, k_scale, v_scale,
                                HEAD_NUM, SEQ_LEN, 
                                MASK_BLOCK_Q1, BLOCK_KV1, HEAD_DIM, 
                                start_n, start_m, num_steps, 
                                MASK=True, QUANT_TYPE=QUANT_TYPE)
                                
        start_m += num_steps * MASK_BLOCK_Q1
        num_steps = (SEQ_LEN - start_m) // BLOCK_Q1
        dk, dv = _attn_bwd_dkdv(dk, dv, 
                                Q, DO, k, v, 
                                LSE, D, 
                                stride_qs, stride_qd, 
                                Q_scale_ptr, DO_scale_ptr, k_scale, v_scale,
                                HEAD_NUM, SEQ_LEN, 
                                BLOCK_Q1, BLOCK_KV1, HEAD_DIM, 
                                start_n, start_m, num_steps, 
                                MASK=False, QUANT_TYPE=QUANT_TYPE)
    else:
        num_steps = SEQ_LEN // BLOCK_Q1
        dk, dv = _attn_bwd_dkdv(dk, dv, 
                                Q, DO, k, v,
                                LSE, D, 
                                stride_qs, stride_qd, 
                                Q_scale_ptr, DO_scale_ptr, k_scale, v_scale,
                                HEAD_NUM, SEQ_LEN, 
                                BLOCK_Q1, BLOCK_KV1, HEAD_DIM, 
                                start_n, 0, num_steps, 
                                MASK=False, QUANT_TYPE=QUANT_TYPE)
        
    dv_ptrs = DV + offs_n[:, None] * stride_qs + offs_k[None, :] * stride_qd
    tl.store(dv_ptrs, dv)
    dk_ptrs = DK + offs_n[:, None] * stride_qs + offs_k[None, :] * stride_qd
    dk = dk * sm_scale
    tl.store(dk_ptrs, dk)

    start_m = pid * BLOCK_Q2
    offs_m = start_m + tl.arange(0, BLOCK_Q2)
    q = tl.load(Q + offs_m[:, None] * stride_qs + offs_k[None, :] * stride_qd)
    dq = tl.zeros([BLOCK_Q2, HEAD_DIM], dtype=tl.float32)
    do = tl.load(DO + offs_m[:, None] * stride_qs + offs_k[None, :] * stride_qd)

    lse = tl.load(LSE + offs_m)
    lse = lse[:, None]
    if causal:
        end_n = start_m + BLOCK_Q2
        MASK_BLOCK_Q2: tl.constexpr = BLOCK_KV2 // BLK_SLICE_FACTOR
        num_steps = BLOCK_Q2 // MASK_BLOCK_Q2
        dq = _attn_bwd_dq(dq, q, K, V, do,
                          lse, D, 
                          stride_qs, stride_qd, 
                          q_scale, do_scale, K_scale_ptr, V_scale_ptr,
                          HEAD_NUM, SEQ_LEN, 
                          BLOCK_Q2, MASK_BLOCK_Q2, HEAD_DIM, 
                          start_m, end_n - num_steps * MASK_BLOCK_Q2, num_steps, 
                          MASK=True, QUANT_TYPE=QUANT_TYPE)
                        
        end_n -= num_steps * MASK_BLOCK_Q2
        num_steps = end_n // BLOCK_KV2
        dq = _attn_bwd_dq(dq, q, K, V, do,
                          lse, D, 
                          stride_qs, stride_qd, 
                          q_scale, do_scale, K_scale_ptr, V_scale_ptr,
                          HEAD_NUM, SEQ_LEN, 
                          BLOCK_Q2, BLOCK_KV2, HEAD_DIM, 
                          start_m, end_n - num_steps * BLOCK_KV2, num_steps, 
                          MASK=False, QUANT_TYPE=QUANT_TYPE)
    else:
        num_steps = SEQ_LEN // BLOCK_KV2
        dq = _attn_bwd_dq(dq, q, K, V, do,
                          lse, D, 
                          stride_qs, stride_qd, 
                          q_scale, do_scale, K_scale_ptr, V_scale_ptr,
                          HEAD_NUM, SEQ_LEN, 
                          BLOCK_Q2, BLOCK_KV2, HEAD_DIM, 
                          start_m, 0, num_steps,  
                          MASK=False, QUANT_TYPE=QUANT_TYPE)
                               
    dq_ptrs = DQ + offs_m[:, None] * stride_qs + offs_k[None, :] * stride_qd
    dq = dq * LN2
    tl.store(dq_ptrs, dq)

QUANT_CONFIG = {
    "int8": (0, torch.int8),
    "e4m3": (1, torch.float8_e4m3fn),
    "e5m2": (2, torch.float8_e5m2),
    "none": (3, torch.int8)
}
    
class _attention(torch.autograd.Function):
    @staticmethod
    def forward(ctx,
                q, k, v, 
                sm_scale,
                output_dtype=torch.float16, 
                causal=False,
                quant_type="int8"):
        BLOCK_Q, BLOCK_KV, = 128, 128

        try:
            quant_code, quant_dtype = QUANT_CONFIG[quant_type]
        except KeyError:
            raise ValueError(f"不支持的量化类型: {quant_type}")
        # q k v 的 shape 是 [B, HEAD_NUM, S, D]
        BATCH, HEAD_N_Q, SEQ_Q, HEAD_DIM_Q = q.shape
        _, HEAD_N_K, SEQ_KV, HEAD_DIM_K = k.shape
        HEAD_DIM_V = v.shape[-1] 
        GROUPS =  max(HEAD_N_Q // HEAD_N_K, 1)
        assert HEAD_DIM_K in {64, 128, 256}
            
        q_quant = torch.empty(q.shape, dtype=quant_dtype, device=q.device)
        k_quant = torch.empty(k.shape, dtype=quant_dtype, device=k.device)
        v_quant = torch.empty(v.shape, dtype=quant_dtype, device=v.device)

        q_scale = torch.empty((BATCH, HEAD_N_Q, (SEQ_Q + BLOCK_Q - 1) // BLOCK_Q), device=q.device, dtype=torch.float32)
        k_scale = torch.empty((BATCH, HEAD_N_K, (SEQ_KV + BLOCK_KV - 1) // BLOCK_KV), device=k.device, dtype=torch.float32)
        v_scale = torch.empty((BATCH, HEAD_N_K, (SEQ_KV + BLOCK_KV - 1) // BLOCK_KV), device=v.device, dtype=torch.float32)

        o = torch.empty(q.shape, dtype=output_dtype, device=q.device)
        lse = torch.empty((BATCH, HEAD_N_Q, SEQ_Q), dtype=torch.float32, device=q.device)

        if sm_scale is None:
            sm_scale = HEAD_DIM_Q**-0.5    

        k_mean = None
        lse_correction = None
        
        k_mean = k.mean(dim=2, keepdim=True)
        lse_correction = torch.matmul(q, k_mean.transpose(2, 3)).squeeze(-1).to(torch.float32)
        k = k - k_mean
        grid_q = (triton.cdiv(SEQ_Q, BLOCK_Q), HEAD_N_Q, BATCH)
        quant_per_block[grid_q](
            q, q_quant, q_scale,
            q.stride(0), q.stride(1), q.stride(2),
            q_quant.stride(0), q_quant.stride(1), q_quant.stride(2),
            q_scale.stride(0), q_scale.stride(1), q_scale.stride(2),
            sm_scale=(sm_scale * RCP_LN2),
            HEAD_DIM=HEAD_DIM_Q,
            SEQ_LEN=SEQ_Q,
            BLOCK=BLOCK_Q,
            QUANT_TYPE=quant_code
        )

        grid_kv = (triton.cdiv(SEQ_KV, BLOCK_KV), HEAD_N_K, BATCH)
        quant_per_block[grid_kv](
            k, k_quant, k_scale,
            k.stride(0), k.stride(1), k.stride(2),
            k_quant.stride(0), k_quant.stride(1), k_quant.stride(2),
            k_scale.stride(0), k_scale.stride(1), k_scale.stride(2),
            sm_scale=1.0,
            HEAD_DIM=HEAD_DIM_K,
            SEQ_LEN=SEQ_KV,
            BLOCK=BLOCK_KV,
            QUANT_TYPE=quant_code
        )
        quant_per_block[grid_kv](
            v, v_quant, v_scale,
            v.stride(0), v.stride(1), v.stride(2),
            v_quant.stride(0), v_quant.stride(1), v_quant.stride(2),
            v_scale.stride(0), v_scale.stride(1), v_scale.stride(2),
            sm_scale=1.0,
            HEAD_DIM=HEAD_DIM_V,
            SEQ_LEN=SEQ_KV,
            BLOCK=BLOCK_KV,
            QUANT_TYPE=quant_code
        )

        stage = 3 if causal else 1
        extra_kern_args = {}
        
        # grid = lambda args: (triton.cdiv(SEQ_L, args["BLOCK_Q"]), BATCH * HEAD_N_Q, 1) # 沿 seq 维度分块 BLOCK_Q；将 Batch 和 Head 维度合并；表示每个 warp 只处理 1 个 batch 和 head
        ctx.grid = grid_q
        _attn_fwd[grid_q](
            q_quant, k_quant, v if quant_type == "none" else v_quant,
            q_scale, k_scale, v_scale,
            lse, o, 
            q_quant.stride(0), q_quant.stride(1), q_quant.stride(2), q_quant.stride(3), 
            k_quant.stride(0), k_quant.stride(1), k_quant.stride(2), k_quant.stride(3),
            v_quant.stride(0), v_quant.stride(1), v_quant.stride(2), v_quant.stride(3),
            o.stride(0), o.stride(1), o.stride(2), o.stride(3),
            HEAD_NUM=HEAD_N_Q, SEQ_Q=SEQ_Q, GROUPS=GROUPS, 
            HEAD_DIM=HEAD_DIM_K, SEQ_KV=SEQ_KV,
            BLOCK_Q=BLOCK_Q, BLOCK_KV=BLOCK_KV,
            STAGE=stage,
            QUANT_TYPE=quant_code, 
            **extra_kern_args)
        ctx.save_for_backward(q, k, v, o, lse) # 对齐  exp2/log2 
        ctx.sm_scale = sm_scale
        ctx.causal = causal
        lse = lse * LN2
        lse = lse + lse_correction * sm_scale
        return o, lse

    @staticmethod
    def backward(ctx, do, dlse=None):
        q, k, v, o, lse = ctx.saved_tensors
        maybe_contiguous = lambda x: x.contiguous() if x.is_contiguous() == False else x
        do, q, k, v, o = [maybe_contiguous(x) for x in (do, q, k, v, o)]
        assert do.is_contiguous()
        # assert q.stride() == k.stride() == v.stride() == o.stride() == do.stride()
        #! dq 对 e4m3 支持不好，首先试试e5m2 q k v（qk 和 v的quant type可以不一样）
        # TODO bmm2 fp8 gemm , bmm1 fp16 gemm ->  bmm2 + bmm1 fp8 gemm
        # TODO softmax input grad dynamic quant
        dq = torch.empty_like(q)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)

        BATCH, HEAD_N_Q, SEQ_Q, HEAD_DIM_Q = q.shape
        _, HEAD_N_K, SEQ_KV, HEAD_DIM_K = k.shape
        PRE_BLOCK_Q = 128
        assert SEQ_Q % PRE_BLOCK_Q == 0
        GROUPS =  max(HEAD_N_Q // HEAD_N_K, 1)
        BLOCK_Q1, BLOCK_KV1, BLOCK_Q2, BLOCK_KV2 = 64, 128, 128, 64
        BLK_SLICE_FACTOR = 2
        # TODO 不量化 dv，量化 dq和dk
        quant_code, quant_dtype = QUANT_CONFIG["e4m3"]
        q_quant = torch.empty(q.shape, dtype=quant_dtype, device=q.device)
        do_quant = torch.empty(do.shape, dtype=quant_dtype, device=do.device)
        k_quant = torch.empty(k.shape, dtype=quant_dtype, device=k.device)
        v_quant = torch.empty(v.shape, dtype=quant_dtype, device=v.device)
        q_scale = torch.empty((BATCH, HEAD_N_Q, (SEQ_Q + BLOCK_Q2 - 1) // BLOCK_Q2), device=q.device, dtype=torch.float32)
        do_scale = torch.empty((BATCH, HEAD_N_Q, (SEQ_Q + BLOCK_Q2 - 1) // BLOCK_Q2), device=do.device, dtype=torch.float32)
        k_scale = torch.empty((BATCH, HEAD_N_K, (SEQ_KV + BLOCK_KV2 - 1) // BLOCK_KV2), device=k.device, dtype=torch.float32)
        v_scale = torch.empty((BATCH, HEAD_N_K, (SEQ_KV + BLOCK_KV2 - 1) // BLOCK_KV2), device=v.device, dtype=torch.float32)
        k = k * (ctx.sm_scale * RCP_LN2) 

        grid_q = (triton.cdiv(SEQ_Q, BLOCK_Q2), HEAD_N_Q, BATCH)
        quant_per_block[grid_q](
            q, q_quant, q_scale,
            q.stride(0), q.stride(1), q.stride(2),
            q_quant.stride(0), q_quant.stride(1), q_quant.stride(2),
            q_scale.stride(0), q_scale.stride(1), q_scale.stride(2),
            sm_scale=1.0,
            HEAD_DIM=HEAD_DIM_Q,
            SEQ_LEN=SEQ_Q,
            BLOCK=BLOCK_Q2,
            QUANT_TYPE=quant_code
        )
        quant_per_block[grid_q](
            do, do_quant, do_scale,
            do.stride(0), do.stride(1), do.stride(2),
            do_quant.stride(0), do_quant.stride(1), do_quant.stride(2),
            do_scale.stride(0), do_scale.stride(1), do_scale.stride(2),
            sm_scale=1.0,
            HEAD_DIM=HEAD_DIM_Q,
            SEQ_LEN=SEQ_Q,
            BLOCK=BLOCK_Q2,
            QUANT_TYPE=quant_code
        )
        
        grid_kv = (triton.cdiv(SEQ_KV, BLOCK_KV2), HEAD_N_K, BATCH)
        quant_per_block[grid_kv](
            k, k_quant, k_scale,
            k.stride(0), k.stride(1), k.stride(2),
            k_quant.stride(0), k_quant.stride(1), k_quant.stride(2),
            k_scale.stride(0), k_scale.stride(1), k_scale.stride(2),
            sm_scale=1.0,
            HEAD_DIM=HEAD_DIM_K,
            SEQ_LEN=SEQ_KV,
            BLOCK=BLOCK_KV2,
            QUANT_TYPE=quant_code
        )
        quant_per_block[grid_kv](
            v, v_quant, v_scale,
            v.stride(0), v.stride(1), v.stride(2),
            v_quant.stride(0), v_quant.stride(1), v_quant.stride(2),
            v_scale.stride(0), v_scale.stride(1), v_scale.stride(2),
            sm_scale=1.0,
            HEAD_DIM=HEAD_DIM_K,
            SEQ_LEN=SEQ_KV,
            BLOCK=BLOCK_KV2,
            QUANT_TYPE=quant_code
        )

        pre_grid = (SEQ_Q // PRE_BLOCK_Q, BATCH * HEAD_N_Q)
        delta = torch.empty_like(lse)
        _attn_bwd_preprocess[pre_grid](
            o, do, delta, 
            SEQ_Q, BLOCK_Q=PRE_BLOCK_Q, HEAD_DIM=HEAD_DIM_Q 
        )

        # NUM_WARPS, NUM_STAGES = 4, 5
        grid = (SEQ_Q // BLOCK_KV1, HEAD_N_Q, BATCH)
        _attn_bwd[grid](
            q, k, v_quant, do_quant, 
            q_scale, k_scale, v_scale, do_scale,
            ctx.sm_scale, 
            ctx.causal,
            dq, dk, dv, 
            lse, delta, 
            q.stride(0), q.stride(1), q.stride(2), q.stride(3), 
            HEAD_N_Q, SEQ_Q, GROUPS,
            BLOCK_Q1=BLOCK_Q1, BLOCK_KV1=BLOCK_KV1, 
            BLOCK_Q2=BLOCK_Q2, BLOCK_KV2=BLOCK_KV2, 
            BLK_SLICE_FACTOR=BLK_SLICE_FACTOR, 
            HEAD_DIM=HEAD_DIM_K, 
            QUANT_TYPE=quant_code, 
            num_warps=4 if HEAD_DIM_K == 64 else 8,
            num_stages=3 if HEAD_DIM_K == 64 else 2
        )
        dq = dq[..., : do.shape[-1]].contiguous()
        dk = dk[..., : do.shape[-1]].contiguous()
        dv = dv[..., : do.shape[-1]].contiguous()
        print(f"NAN in dK: {dk.isnan().any()}, INF in dK: {dk.isinf().any()}")
        print(f"NAN in dV: {dv.isnan().any()}, INF in dV: {dv.isinf().any()}")
        print(f"NAN in dQ: {dq.isnan().any()}, INF in dQ: {dq.isinf().any()}")
        return dq, dk, dv, None, None, None, None

def sage_attention(q, k, v, sm_scale, output_dtype=torch.float16, causal=False, quant_type="int8"):
    return _attention.apply(q, k, v, sm_scale, output_dtype, causal, quant_type)