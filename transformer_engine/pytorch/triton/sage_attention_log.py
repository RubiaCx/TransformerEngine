import pytest
import torch
import triton.tools.experimental_descriptor

import triton
import triton.language as tl

import torch.nn.functional as F

RCP_LN2: tl.constexpr = 1.4426950408889634 # exp(x) = exp2(x * log2(e)) = exp2(x / ln(2)) = exp2(x * RCP_LN2)
LN2: tl.constexpr = 0.6931471824645996  # = ln(2)

@triton.jit
def quant_per_block(Input, Output, Scale, 
                    stride_ib, stride_ih, stride_is,
                    stride_ob, stride_oh, stride_os,
                    stride_sb, stride_sh, 
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
    scale_ptrs = Scale + off_b * stride_sb + off_h * stride_sh + start_m

    x = tl.load(input_ptrs, mask=offs_n[:, None] < SEQ_LEN)
    x = x.to(tl.float32)
    x *= sm_scale

    if QUANT_TYPE == 0 or QUANT_TYPE == 3:  # int8
        scale = tl.max(tl.abs(x)) / 127. + 1e-8
        x_quant = x / (tl.max(tl.abs(x)) / 127.0 + 1e-8)
        x_quant += 0.5 * tl.where(x_quant >= 0, 1, -1)
    elif QUANT_TYPE == 1:  # e4m3
        scale = tl.max(tl.abs(x)) / 448. + 1e-8
        x_quant = (x / scale).to(tl.float8e4nv)
    elif QUANT_TYPE == 2:  # e5m2
        scale = tl.max(tl.abs(x)) / 57344. + 1e-8
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
def _attn_fwd_inner_quant(acc, l_i, m_i, 
                          q, K_ptrs, V_ptrs, stride_k, stride_v,
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
            p_scale = tl.max(tl.abs(p)) / 127. + 1e-8
            p_quant = (p / p_scale + 0.5 * tl.where(p >= 0, 1, -1)).to(tl.int8)
        elif QUANT_TYPE == 1:  # e4m3
            p_scale = tl.max(tl.abs(p)) / 448. + 1e-8
            p_quant = (p / p_scale).to(tl.float8e4nv)
        elif QUANT_TYPE == 2:  # e5m2
            p_scale = tl.max(tl.abs(p)) / 57344. + 1e-8
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
                                                  4 - STAGE, offs_m, offs_n,  QUANT_TYPE)
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
def _attn_bwd_preprocess(O, DO, 
                         Delta,  
                         BS, HEAD_NUM, SEQ_LEN, 
                         BLOCK_Q: tl.constexpr, HEAD_DIM: tl.constexpr 
                         ):
    off_m = tl.program_id(0) * BLOCK_Q + tl.arange(0, BLOCK_Q)
    off_h = tl.program_id(1).to(tl.int64)
    off_b = tl.program_id(2).to(tl.int64)
    off_n = tl.arange(0, HEAD_DIM)
    # load
    o = tl.load(O + off_h * off_b * HEAD_DIM * SEQ_LEN + off_m[:, None] * HEAD_DIM + off_n[None, :])
    do = tl.load(DO + off_h * off_b * HEAD_DIM * SEQ_LEN + off_m[:, None] * HEAD_DIM + off_n[None, :]).to(tl.float32)
    delta = tl.sum(o * do, axis=1)
    # write-back
    tl.store(Delta + off_h * off_b * SEQ_LEN + off_m, delta)

# The main inner-loop logic for computing dK and dV.
# 固定处理 K 和 V 的一个块（形状为 BLOCK_KV × HEAD_DIM），然后迭代处理 Q 和 dO 的多个块
@triton.jit
def _attn_bwd_dkdv(dk, dv, 
                   Q, k, v, 
                   DO, LSE, D, 
                   # shared by Q/K/V/DO.
                   stride_qs, stride_qd, 
                   HEAD_NUM, SEQ_LEN,
                   BLOCK_Q: tl.constexpr, 
                   BLOCK_KV: tl.constexpr, 
                   HEAD_DIM: tl.constexpr, 
                   # Filled in by the wrapper.
                   start_n, start_m, num_steps, 
                   debug_pT_ptr, debug_dsT_ptr,
                   MASK: tl.constexpr):
    offs_m = start_m + tl.arange(0, BLOCK_Q)
    offs_n = start_n + tl.arange(0, BLOCK_KV)
    offs_k = tl.arange(0, HEAD_DIM)

    qT_ptrs = Q + offs_m[None, :] * stride_qs + offs_k[:, None] * stride_qd
    do_ptrs = DO + offs_m[:, None] * stride_qs + offs_k[None, :] * stride_qd
    # BLOCK_KV must be a multiple of BLOCK_Q, otherwise the code wouldn't work.
    tl.static_assert(BLOCK_KV % BLOCK_Q == 0)
    curr_m = start_m
    step_m = BLOCK_Q
    for blk_idx in range(num_steps):
        qT = tl.load(qT_ptrs)
        # Load m before computing qk to reduce pipeline stall.
        offs_m = curr_m + tl.arange(0, BLOCK_Q)
        m = tl.load(LSE + offs_m)
        qkT = tl.dot(k, qT)
        pT = tl.math.exp2(qkT - m[None, :])
        if blk_idx == 0:
            tl.device_print("pT", pT)
            tl.store(debug_pT_ptr + offs_n[:, None] * BLOCK_Q + tl.arange(0, BLOCK_Q)[None, :], pT)
        # Autoregressive masking.
        if MASK:
            mask = (offs_m[None, :] >= offs_n[:, None])
            pT = tl.where(mask, pT, 0.0)
        do = tl.load(do_ptrs)
        # Compute dV.
        # ppT = pT.to(tl.float16)
        ppT = pT.to(do.dtype) 
        dv += tl.dot(ppT, do)
        # D (= delta) is pre-divided by ds_scale.
        Di = tl.load(D + offs_m)
        # Compute dP and dS.
        dpT = tl.dot(v, tl.trans(do)).to(tl.float32)
        dsT = pT * (dpT - Di[None, :])
        # dsT = dsT.to(tl.float16)
        if blk_idx == 0:
            # tl.device_print("dsT", dsT)
            tl.store(debug_dsT_ptr + offs_n[:, None] * BLOCK_Q + tl.arange(0, BLOCK_Q)[None, :], dsT)
        dsT = dsT.to(qT.dtype) 
        dk += tl.dot(dsT, tl.trans(qT))
        
        # Increment pointers.
        curr_m += step_m
        qT_ptrs += step_m * stride_qs
        do_ptrs += step_m * stride_qs
    return dk, dv

# the main inner-loop logic for computing dQ
# 固定处理 Q 和 dO 的一个块（形状为 BLOCK_Q2 × HEAD_DIM），然后迭代处理 K 和 V 的多个块
@triton.jit
def _attn_bwd_dq(dq, q, K, V, 
                 do, m, D,
                 # shared by Q/K/V/DO.
                 stride_qs, stride_qd, 
                 HEAD_NUM, SEQ_LEN, 
                 BLOCK_Q: tl.constexpr, 
                 BLOCK_KV: tl.constexpr, 
                 HEAD_DIM: tl.constexpr,
                 # Filled in by the wrapper.
                 start_m, start_n, num_steps, 
                 debug_p_ptr, debug_ds_ptr, 
                 MASK: tl.constexpr):
    offs_m = start_m + tl.arange(0, BLOCK_Q)
    offs_n = start_n + tl.arange(0, BLOCK_KV)
    offs_k = tl.arange(0, HEAD_DIM)
    kT_ptrs = K + offs_n[None, :] * stride_qs + offs_k[:, None] * stride_qd
    vT_ptrs = V + offs_n[None, :] * stride_qs + offs_k[:, None] * stride_qd
    # D (= delta) is pre-divided by ds_scale.
    Di = tl.load(D + offs_m)
    # BLOCK_Q must be a multiple of BLOCK_KV, otherwise the code wouldn't work.
    tl.static_assert(BLOCK_Q % BLOCK_KV == 0)
    curr_n = start_n
    step_n = BLOCK_KV
    for blk_idx in range(num_steps):
        kT = tl.load(kT_ptrs)
        vT = tl.load(vT_ptrs)
        qk = tl.dot(q, kT)
        p = tl.math.exp2(qk - m)
        if blk_idx == 0:
            tl.store(debug_p_ptr + offs_n[:, None] * BLOCK_Q + tl.arange(0, BLOCK_Q)[None, :],
                     tl.trans(p))
        # Autoregressive masking.
        if MASK:
            offs_n = curr_n + tl.arange(0, BLOCK_KV)
            mask = (offs_m[:, None] >= offs_n[None, :])
            p = tl.where(mask, p, 0.0)
        # Compute dP and dS.
        dp = tl.dot(do, vT).to(tl.float32)
        ds = p * (dp - Di[:, None])
        if blk_idx == 0:
            tl.store(debug_ds_ptr + offs_n[:, None] * BLOCK_Q + tl.arange(0, BLOCK_Q)[None, :],
                     tl.trans(ds))
        # ds = ds.to(tl.float16)
        ds = ds.to(kT.dtype) 
        # Compute dQ.
        # NOTE: We need to de-scale dq in the end, because kT was pre-scaled.
        dq += tl.dot(ds, tl.trans(kT))
        
        # Increment pointers.
        curr_n += step_n
        kT_ptrs += step_n * stride_qs
        vT_ptrs += step_n * stride_qs
    return dq

@triton.jit
def _attn_bwd(Q, K, V,
              sm_scale, causal,
              DO, DQ, DK, DV, 
              LSE, D,
              # shared by Q/K/V/DO.
              stride_qb, stride_qh, stride_qs, stride_qd,
              HEAD_NUM, SEQ_LEN, GROUPS,
              debug_pT_ptr, debug_dsT_ptr, debug_p_ptr, debug_ds_ptr,
              BLOCK_Q1: tl.constexpr, BLOCK_KV1: tl.constexpr, 
              BLOCK_Q2: tl.constexpr, BLOCK_KV2: tl.constexpr, 
              BLK_SLICE_FACTOR: tl.constexpr, 
              HEAD_DIM: tl.constexpr):
    pid = tl.program_id(0)
    off_h = tl.program_id(1).to(tl.int64)
    off_b = tl.program_id(2).to(tl.int64)
    adj = (stride_qh * off_b + stride_qb * off_h).to(tl.int64) 
    off_chz = (off_b * HEAD_NUM * SEQ_LEN).to(tl.int64) # 计算当前 batch 在 LSE 和 D 中的起始内存偏移量，挪动 HEAD_NUM * SEQ_LEN 个元素

    # offset pointers for batch/head
    Q += adj
    K += adj
    V += adj
    DO += adj
    DQ += adj
    DK += adj
    DV += adj
    LSE += off_chz
    D += off_chz

    # load scales
    offs_k = tl.arange(0, HEAD_DIM)

    start_n = pid * BLOCK_KV1
    start_m = start_n
    offs_n = start_n + tl.arange(0, BLOCK_KV1)
    dv = tl.zeros([BLOCK_KV1, HEAD_DIM], dtype=tl.float32)
    dk = tl.zeros([BLOCK_KV1, HEAD_DIM], dtype=tl.float32)
    # load K and V: they stay in SRAM throughout the inner loop.
    k = tl.load(K + offs_n[:, None] * stride_qs + offs_k[None, :] * stride_qd)
    v = tl.load(V + offs_n[:, None] * stride_qs + offs_k[None, :] * stride_qd)

    if causal:
        MASK_BLOCK_Q1: tl.constexpr = BLOCK_Q1 // BLK_SLICE_FACTOR
        num_steps = BLOCK_KV1 // MASK_BLOCK_Q1
        dk, dv = _attn_bwd_dkdv(dk, dv, 
                                Q, k, v, 
                                DO, LSE, D, 
                                stride_qs, stride_qd, 
                                HEAD_NUM, SEQ_LEN, 
                                MASK_BLOCK_Q1, BLOCK_KV1, HEAD_DIM, 
                                start_n, start_m, num_steps, 
                                debug_pT_ptr, debug_dsT_ptr,
                                MASK=True)
                                
        start_m += num_steps * MASK_BLOCK_Q1
        num_steps = (SEQ_LEN - start_m) // BLOCK_Q1
        dk, dv = _attn_bwd_dkdv(dk, dv, 
                                Q, k, v, 
                                DO, LSE, D, 
                                stride_qs, stride_qd, 
                                HEAD_NUM, SEQ_LEN, 
                                BLOCK_Q1, BLOCK_KV1, HEAD_DIM, 
                                start_n, start_m, num_steps, 
                                debug_pT_ptr, debug_dsT_ptr,
                                MASK=False)
    else:
        num_steps = SEQ_LEN // BLOCK_Q1
        dk, dv = _attn_bwd_dkdv(dk, dv, 
                                Q, k, v, 
                                DO, LSE, D, 
                                stride_qs, stride_qd, 
                                HEAD_NUM, SEQ_LEN, 
                                BLOCK_Q1, BLOCK_KV1, HEAD_DIM, 
                                start_n, 0, num_steps, 
                                debug_pT_ptr, debug_dsT_ptr,
                                MASK=False)
        
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

    m = tl.load(LSE + offs_m)
    m = m[:, None]
    if causal:
        end_n = start_m + BLOCK_Q2
        MASK_BLOCK_Q2: tl.constexpr = BLOCK_KV2 // BLK_SLICE_FACTOR
        num_steps = BLOCK_Q2 // MASK_BLOCK_Q2
        dq = _attn_bwd_dq(dq, q, K, V, 
                          do, m, D, 
                          stride_qs, stride_qd, 
                          HEAD_NUM, SEQ_LEN, 
                          BLOCK_Q2, MASK_BLOCK_Q2, HEAD_DIM, 
                          start_m, end_n - num_steps * MASK_BLOCK_Q2, num_steps, 
                          debug_p_ptr, debug_ds_ptr,
                          MASK=True)
                        
        end_n -= num_steps * MASK_BLOCK_Q2
        num_steps = end_n // BLOCK_KV2
        dq = _attn_bwd_dq(dq, q, K, V, 
                          do, m, D, 
                          stride_qs, stride_qd, 
                          HEAD_NUM, SEQ_LEN, 
                          BLOCK_Q2, BLOCK_KV2, HEAD_DIM, 
                          start_m, end_n - num_steps * BLOCK_KV2, num_steps, 
                          debug_p_ptr, debug_ds_ptr,
                          MASK=False)
    else:
        num_steps = SEQ_LEN // BLOCK_KV2
        dq = _attn_bwd_dq(dq, q, K, V, 
                          do, m, D, 
                          stride_qs, stride_qd, 
                          HEAD_NUM, SEQ_LEN, 
                          BLOCK_Q2, BLOCK_KV2, HEAD_DIM, 
                          start_m, 0, num_steps,  
                          debug_p_ptr, debug_ds_ptr,
                          MASK=False) 
                               
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
        BLOCK_Q = 128
        BLOCK_KV = 64

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
            q_scale.stride(0), q_scale.stride(1),
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
            k_scale.stride(0), k_scale.stride(1),
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
            v_scale.stride(0), v_scale.stride(1),
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
        lse = lse * LN2
        lse = lse + lse_correction * sm_scale

        ctx.sm_scale = sm_scale
        ctx.causal = causal

        return o, lse

    @staticmethod
    def backward(ctx, do, dlse=None):
        q, k, v, o, lse = ctx.saved_tensors
        maybe_contiguous = lambda x: x.contiguous() if x.is_contiguous() == False else x
        do, q, k, v, o = [maybe_contiguous(x) for x in (do, q, k, v, o)]
        assert do.is_contiguous()
        # assert q.stride() == k.stride() == v.stride() == o.stride() == do.stride()
        # print("SHAPE: ", q.shape, k.shape, v.shape, o.shape, do.shape)
        # FP 16 
        #! dq 对 e4m3 支持不好，首先试试e5m2 q k v（qk 和 v的quant type可以不一样）
        dq = torch.empty_like(q)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)

        BATCH, HEAD_N_Q, SEQ_Q, HEAD_DIM_Q = q.shape
        _, HEAD_N_K, SEQ_KV, HEAD_DIM_K = k.shape
        PRE_BLOCK_Q = 128
        GROUPS =  max(HEAD_N_Q // HEAD_N_K, 1)
        assert HEAD_DIM_K in {64, 128, 256}
        #! 前向反向的quant scale block size 对齐
        BLOCK_Q1, BLOCK_KV1, BLOCK_Q2, BLOCK_KV2 = 32, 128, 128, 32
        BLK_SLICE_FACTOR = 2

        num_blocks_kv = SEQ_KV // BLOCK_KV2
        total_kv = num_blocks_kv * BATCH * HEAD_N_K * BLOCK_KV2
        num_blocks_q  = SEQ_Q// BLOCK_Q2
        total_q  = num_blocks_q * BATCH * HEAD_N_Q * BLOCK_Q2
        debug_pT = torch.empty((BATCH * BLOCK_KV2, BLOCK_Q2), device=q.device, dtype=torch.float32)
        debug_dsT = torch.empty_like(debug_pT)
        debug_p  = torch.empty((BATCH * BLOCK_KV1, BLOCK_Q1), device=q.device, dtype=torch.float32)
        debug_ds = torch.empty_like(debug_p)

        k = k * (ctx.sm_scale * RCP_LN2) 
        assert SEQ_Q % PRE_BLOCK_Q == 0
        pre_grid = (SEQ_Q // PRE_BLOCK_Q, BATCH * HEAD_N_Q)
        delta = torch.empty_like(lse)
        _attn_bwd_preprocess[pre_grid](
            o, do, 
            delta, 
            BATCH, HEAD_N_Q, SEQ_Q, 
            BLOCK_Q=PRE_BLOCK_Q, HEAD_DIM=HEAD_DIM_Q 
        )
        # NUM_WARPS, NUM_STAGES = 4, 5
        grid = (triton.cdiv(SEQ_Q, BLOCK_Q2), HEAD_N_Q, BATCH)

        _attn_bwd[grid](
            q, k, v, 
            ctx.sm_scale, 
            ctx.causal,
            do, dq, dk, dv, 
            lse, delta, 
            q.stride(0), q.stride(1), q.stride(2), q.stride(3), 
            HEAD_N_Q, SEQ_Q, GROUPS,
            debug_pT, debug_dsT, debug_p, debug_ds, 
            BLOCK_Q1=BLOCK_Q1, BLOCK_KV1=BLOCK_KV1, 
            BLOCK_Q2=BLOCK_Q2, BLOCK_KV2=BLOCK_KV2, 
            BLK_SLICE_FACTOR=BLK_SLICE_FACTOR, 
            HEAD_DIM=HEAD_DIM_K, 
            num_warps=4 if HEAD_DIM_K == 64 else 8,
            num_stages=3 if HEAD_DIM_K == 64 else 2
        )
        torch.cuda.synchronize()
        dq = dq[..., : do.shape[-1]].contiguous()
        dk = dk[..., : do.shape[-1]].contiguous()
        dv = dv[..., : do.shape[-1]].contiguous()
        print(f"NAN in dK: {dk.isnan().any()}, INF in dK: {dk.isinf().any()}")
        print(f"NAN in dV: {dv.isnan().any()}, INF in dV: {dv.isinf().any()}")
        print(f"NAN in dQ: {dq.isnan().any()}, INF in dQ: {dq.isinf().any()}")
        pT_arr  = debug_pT.cpu().numpy().reshape(-1)
        dsT_arr = debug_dsT.cpu().numpy().reshape(-1)
        p_arr   = debug_p.cpu().numpy().reshape(-1)
        ds_arr  = debug_ds.cpu().numpy().reshape(-1)
        import matplotlib.pyplot as plt
        import numpy as np
        def plot_hist(data, title):
            data = data[~np.isnan(data)]  # 过滤NaN
            data = data[np.isfinite(data)]  # 过滤INF
            plt.figure()
            ax = plt.gca()
            plt.hist(data)
            plt.title(title)
            plt.title(title, fontsize=14, fontweight='bold', pad=20)
            plt.xlabel('Value', fontsize=12, labelpad=10)
            plt.ylabel('Frequency', fontsize=12, labelpad=10)
            plt.xticks(fontsize=10)
            plt.yticks(fontsize=10)
            
            # 添加网格线
            ax.yaxis.grid(True, linestyle='--', alpha=0.4)
            ax.xaxis.grid(False)
            
            # 优化边框
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_alpha(0.4)
            ax.spines['bottom'].set_alpha(0.4)
            
            # 自动调整科学计数法显示
            if np.max(np.abs(data)) > 1e4 or np.max(np.abs(data)) < 1e-4:
                ax.ticklabel_format(axis='both', style='sci', scilimits=(0,0))
            textstr = '\n'.join((
                f'Total samples: {len(data):,}',
                f'Mean: {np.mean(data):.2e}',
                f'Std: {np.std(data):.2e}',
                f'Max: {np.max(data):.2e}',
                f'Min: {np.min(data):.2e}'))
            ax.text(0.98, 0.95, textstr, transform=ax.transAxes,
                    fontsize=10, verticalalignment='top', horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            # 设置对数坐标
            ax.set_yscale('log')
            plt.tight_layout()
            plt.savefig(f"{title}.png")

        plot_hist(pT_arr,  "Distribution of pT = tl.math.exp2(qkT - m[None, :])")
        plot_hist(dsT_arr, "Distribution of dsT = pT * (tl.dot(v, tl.trans(do)) - Di[None, :])")
        plot_hist(p_arr,   "Distribution of p = tl.math.exp2(qk - m)")
        plot_hist(ds_arr,  "Distribution of ds = p * (tl.dot(do, vT) - Di[:, None])")
        return dq, dk, dv, None, None, None, None

def sage_attention(q, k, v, sm_scale, output_dtype=torch.float16, causal=False, quant_type="int8"):
    return _attention.apply(q, k, v, sm_scale, output_dtype, causal, quant_type)



def debug_backward(q, k, v, sm_scale, causal, do):
    """调试用反向传播"""
    class DebugFunction(torch.autograd.Function):
        @staticmethod
        def forward(ctx, q, k, v):
            output, lse = _attention.apply(q, k, v, sm_scale, torch.float16, False, "int8")
            ctx.save_for_backward(q, k, v, output, lse)
            ctx.sm_scale = sm_scale
            ctx.causal = False
            return output

        @staticmethod
        def backward(ctx, grad_output):
            # 取出 forward 时保存的
            q, k, v, output, lse = ctx.saved_tensors
            # 调用原始 Triton backward，返回七个（dq, dk, dv, None, None, None, None）
            grads = _attention.backward(ctx, grad_output)
            # 只取前 3 个：dq, dk, dv
            dq, dk, dv = grads[:3]
            return dq, dk, dv

    q = q.detach().requires_grad_(True)
    k = k.detach().requires_grad_(True)
    v = v.detach().requires_grad_(True)
    output = DebugFunction.apply(q, k, v)
    output.backward(do)
    
    return DebugFunction.backward.ctx.saved_tensors[-4:]

if __name__ == "__main__":
    # 测试配置
    B, H, S, D = 16, 16, 1024, 128
    dtype = torch.float16
    torch.manual_seed(42)
    def generate_tensor(shape, min_val, max_val, target_mean, target_std, device="cuda", dtype=torch.float16):
        tensor = torch.randn(shape, device=device, dtype=dtype)  # 正态分布
        tensor = (tensor - tensor.mean()) / (tensor.std() + 1e-8) # 标准化到均值为0，标准差为1
        tensor = tensor * target_std + target_mean
        if min_val is not None and max_val is not None:
            tensor = torch.clamp(tensor, min_val, max_val)
            tensor = tensor - tensor.mean() + target_mean
        
        return tensor
    shape = (B, H, S, D)
    q = generate_tensor(
        shape=shape,
        min_val=-4.65625,
        max_val=4.5625,
        target_mean=-4.676e-07,
        target_std=1,
        device="cuda",
        dtype=dtype
    )

    k = generate_tensor(
        shape=shape,
        min_val=-4.96875,
        max_val=4.71875,
        target_mean=-5.920e-07,
        target_std=1,
        device="cuda",
        dtype=dtype
    )

    v = generate_tensor(
        shape=shape,
        min_val=-3.5,
        max_val=3.53125,
        target_mean=-0.00517,
        target_std=0.6734,
        device="cuda",
        dtype=dtype
    )

    do = torch.randn_like(q)
    sm_scale = D ** -0.5

    # 执行反向传播并获取调试数据
    pT_arr, dsT_arr, p_arr, ds_arr = debug_backward(q, k, v, sm_scale, False, do)
