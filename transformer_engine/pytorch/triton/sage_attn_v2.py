import pytest
import torch
import triton.tools.experimental_descriptor

import triton
import triton.language as tl

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

    if QUANT_TYPE == 0:  # int8
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
                    BLOCK_Q: tl.constexpr, BLOCK_N: tl.constexpr, 
                    STAGE: tl.constexpr, offs_m: tl.constexpr, offs_n: tl.constexpr):
    # range of values handled by this stage
    if STAGE == 1:
        lo, hi = 0, start_m * BLOCK_Q
    elif STAGE == 2:
        lo, hi = start_m * BLOCK_Q, (start_m + 1) * BLOCK_Q
        lo = tl.multiple_of(lo, BLOCK_Q)
        K_scale_ptr += lo // BLOCK_N
        V_scale_ptr += lo // BLOCK_N
        K_ptrs += stride_k * lo
        V_ptrs += stride_v * lo
    # causal = False
    else:
        lo, hi = 0, SEQ_LEN

    for start_n in range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)

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
        K_ptrs += BLOCK_N * stride_k
        V_ptrs += BLOCK_N * stride_v
        K_scale_ptr += 1
        V_scale_ptr += 1
    return acc, l_i, m_i

@triton.jit
def _attn_fwd_inner_quant(acc, l_i, m_i, q,
                          K_ptrs, V_ptrs, stride_k, stride_v, 
                          start_m, 
                          q_scale, K_scale_ptr, V_scale_ptr, 
                          HEAD_DIM: tl.constexpr, SEQ_LEN: tl.constexpr, GROUPS: tl.constexpr,
                          BLOCK_Q: tl.constexpr, BLOCK_N: tl.constexpr, 
                          STAGE: tl.constexpr, offs_m: tl.constexpr, offs_n: tl.constexpr,
                          QUANT_TYPE: tl.constexpr):
    # range of values handled by this stage
    if STAGE == 1:
        lo, hi = 0, start_m * BLOCK_Q
    elif STAGE == 2:
        lo, hi = start_m * BLOCK_Q, (start_m + 1) * BLOCK_Q
        lo = tl.multiple_of(lo, BLOCK_Q)
        K_scale_ptr += lo // BLOCK_N
        V_scale_ptr += lo // BLOCK_N
        K_ptrs += stride_k * lo
        V_ptrs += stride_v * lo
    # causal = False
    else:
        lo, hi = 0, SEQ_LEN

    for start_n in range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
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
        K_ptrs += BLOCK_N * stride_k
        V_ptrs += BLOCK_N * stride_v
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
              BLOCK_Q: tl.constexpr, BLOCK_N: tl.constexpr, 
              STAGE: tl.constexpr,
              QUANT_TYPE: tl.constexpr,
              ):
    tl.static_assert(BLOCK_N <= HEAD_DIM)
    
    start_m = tl.program_id(0)
    off_h = tl.program_id(1).to(tl.int64)
    off_b = tl.program_id(2).to(tl.int64)

    offs_m = start_m * BLOCK_Q + tl.arange(0, BLOCK_Q)
    offs_n = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, HEAD_DIM)

    q_scale_offset = (off_b * HEAD_NUM + off_h) * tl.cdiv(SEQ_Q, BLOCK_Q)
    k_scale_offset = (off_b * (HEAD_NUM // GROUPS) + off_h // GROUPS) * tl.cdiv(SEQ_KV, BLOCK_N)  
    v_scale_offset = (off_b * (HEAD_NUM // GROUPS) + off_h // GROUPS) * tl.cdiv(SEQ_KV, BLOCK_N)

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
                                            BLOCK_Q, BLOCK_N,
                                            4 - STAGE, offs_m, offs_n)
        if STAGE & 2: # 2 or 3
            acc, l_i, m_i = _attn_fwd_inner(acc, l_i, m_i, q, K_ptrs, V_ptrs, 
                                            stride_ks, stride_vs, start_m, 
                                            q_scale, K_scale_ptr, V_scale_ptr, 
                                            HEAD_DIM, SEQ_KV, GROUPS,
                                            BLOCK_Q, BLOCK_N, 
                                            2, offs_m, offs_n)
    else:
        if STAGE & 1: # 1 or 3
            acc, l_i, m_i = _attn_fwd_inner_quant(acc, l_i, m_i, q, K_ptrs, V_ptrs,
                                                  stride_ks, stride_vs, start_m, 
                                                  q_scale, K_scale_ptr, V_scale_ptr, 
                                                  HEAD_DIM, SEQ_KV, GROUPS,
                                                  BLOCK_Q, BLOCK_N, 
                                                  4 - STAGE, offs_m, offs_n,  QUANT_TYPE)
        if STAGE & 2: # 2 or 3
            acc, l_i, m_i = _attn_fwd_inner_quant(acc, l_i, m_i, q, K_ptrs, V_ptrs, 
                                                  stride_ks, stride_vs, start_m, 
                                                  q_scale, K_scale_ptr, V_scale_ptr, 
                                                  HEAD_DIM, SEQ_KV, GROUPS,
                                                  BLOCK_Q,  BLOCK_N,
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
    off_hz = tl.program_id(1)
    off_n = tl.arange(0, HEAD_DIM)
    # load
    o = tl.load(O + off_hz * HEAD_DIM * SEQ_LEN + off_m[:, None] * HEAD_DIM + off_n[None, :])
    do = tl.load(DO + off_hz * HEAD_DIM * SEQ_LEN + off_m[:, None] * HEAD_DIM + off_n[None, :]).to(tl.float32)
    delta = tl.sum(o * do, axis=1)
    # write-back
    tl.store(Delta + off_hz * SEQ_LEN + off_m, delta)

# The main inner-loop logic for computing dK and dV.
# 固定处理 K 和 V 的一个块（形状为 BLOCK_N × HEAD_DIM），然后迭代处理 Q 和 dO 的多个块
@triton.jit
def _attn_bwd_dkdv(dk, dv, 
                   Q, k, v, 
                   q_scale, K_scale_ptr, V_scale_ptr,
                   DO, LSE, D, 
                   # shared by Q/K/V/DO.
                   stride_tok, stride_d, 
                   HEAD_NUM, SEQ_LEN,
                   BLOCK_Q: tl.constexpr, 
                   BLOCK_N: tl.constexpr, 
                   HEAD_DIM: tl.constexpr, 
                   # Filled in by the wrapper.
                   start_n, start_m, num_steps, 
                   MASK: tl.constexpr):
    offs_m = start_m + tl.arange(0, BLOCK_Q)
    offs_n = start_n + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, HEAD_DIM)

    qT_ptrs = Q + offs_m[None, :] * stride_tok + offs_k[:, None] * stride_d
    do_ptrs = DO + offs_m[:, None] * stride_tok + offs_k[None, :] * stride_d
    # BLOCK_N must be a multiple of BLOCK_Q, otherwise the code wouldn't work.
    tl.static_assert(BLOCK_N % BLOCK_Q == 0)
    curr_m = start_m
    step_m = BLOCK_Q
    for blk_idx in range(num_steps):
        k_scale = tl.load(K_scale_ptr)
        v_scale = tl.load(V_scale_ptr)
        qT = tl.load(qT_ptrs)
        # Load m before computing qk to reduce pipeline stall.
        offs_m = curr_m + tl.arange(0, BLOCK_Q)
        m = tl.load(LSE + offs_m)
        qkT = tl.dot(k, qT).to(tl.float32) * q_scale * k_scale 
        pT = tl.math.exp2(qkT - m[None, :])
        # Autoregressive masking.
        if MASK:
            mask = (offs_m[None, :] >= offs_n[:, None])
            pT = tl.where(mask, pT, 0.0)
        do = tl.load(do_ptrs)
        # Compute dV.
        dv += tl.dot(pT.to(v.dtype), do).to(tl.float32) * v_scale
        # D (= delta) is pre-divided by ds_scale.
        Di = tl.load(D + offs_m)
        # Compute dP and dS.
        dpT = tl.dot(v, tl.trans(do)).to(tl.float32)
        dsT = pT * (dpT - Di[None, :])
        dsT = dsT * (q_scale * k_scale)
        dsT = dsT.to(tl.float16)
        dk += tl.dot(dsT, tl.trans(qT))
        # Increment pointers.
        curr_m += step_m
        qT_ptrs += step_m * stride_tok
        do_ptrs += step_m * stride_tok
        K_scale_ptr += 1
        V_scale_ptr += 1
    return dk, dv

# the main inner-loop logic for computing dQ
# 固定处理 Q 和 dO 的一个块（形状为 BLOCK_Q_Q × HEAD_DIM），然后迭代处理 K 和 V 的多个块
@triton.jit
def _attn_bwd_dq(dq, q, K, V, 
                 q_scale, K_scale_ptr, V_scale_ptr,
                 do, m, D,
                 # shared by Q/K/V/DO.
                 stride_tok, stride_d, 
                 HEAD_NUM, SEQ_LEN, 
                 BLOCK_Q: tl.constexpr, 
                 BLOCK_N: tl.constexpr, 
                 HEAD_DIM: tl.constexpr,
                 # Filled in by the wrapper.
                 start_m, start_n, num_steps, 
                 MASK: tl.constexpr):
    offs_m = start_m + tl.arange(0, BLOCK_Q)
    offs_n = start_n + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, HEAD_DIM)
    kT_ptrs = K + offs_n[None, :] * stride_tok + offs_k[:, None] * stride_d
    vT_ptrs = V + offs_n[None, :] * stride_tok + offs_k[:, None] * stride_d
    # D (= delta) is pre-divided by ds_scale.
    Di = tl.load(D + offs_m)
    # BLOCK_Q must be a multiple of BLOCK_N, otherwise the code wouldn't work.
    tl.static_assert(BLOCK_Q % BLOCK_N == 0)
    curr_n = start_n
    step_n = BLOCK_N
    for blk_idx in range(num_steps):
        kT = tl.load(kT_ptrs)
        vT = tl.load(vT_ptrs)
        k_scale = tl.load(K_scale_ptr)
        qk = tl.dot(q, kT).to(tl.float32) * q_scale * k_scale 
        p = tl.math.exp2(qk - m)
        # Autoregressive masking.
        if MASK:
            offs_n = curr_n + tl.arange(0, BLOCK_N)
            mask = (offs_m[:, None] >= offs_n[None, :])
            p = tl.where(mask, p, 0.0)
        # Compute dP and dS.
        dp = tl.dot(do, vT).to(tl.float32)
        ds = p * (dp - Di[:, None])
        ds = ds.to(tl.float16) * q_scale * k_scale 
        # Compute dQ.
        # NOTE: We need to de-scale dq in the end, because kT was pre-scaled.
        dq += tl.dot(ds, tl.trans(kT))
        # Increment pointers.
        curr_n += step_n
        kT_ptrs += step_n * stride_tok
        vT_ptrs += step_n * stride_tok
    return dq


@triton.jit
def _attn_bwd(Q, K, V,
              Q_scale, K_scale, V_scale,
              DO, DQ, DK, DV, 
              LSE, D,
              # shared by Q/K/V/DO.
              stride_z, stride_h, stride_tok, stride_d,
              HEAD_NUM, SEQ_LEN, GROUPS,
              BLOCK_Q_KV: tl.constexpr, 
              BLOCK_N_KV: tl.constexpr, 
              BLOCK_Q_Q: tl.constexpr, 
              BLOCK_N_Q: tl.constexpr, 
              BLK_SLICE_FACTOR: tl.constexpr, 
              HEAD_DIM: tl.constexpr):
    LN2: tl.constexpr = 0.6931471824645996  # = ln(2)
    pid = tl.program_id(0)
    bhid = tl.program_id(2)
    off_b = (bhid // HEAD_NUM).to(tl.int64)
    off_h = (bhid % HEAD_NUM).to(tl.int64)
    off_chz = (bhid * SEQ_LEN).to(tl.int64)
    adj = (stride_h * off_h + stride_z * off_b).to(tl.int64)

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

    start_n = pid * BLOCK_N_KV
    start_m = start_n

    MASK_BLOCK_Q_KV: tl.constexpr = BLOCK_Q_KV // BLK_SLICE_FACTOR
    offs_n = start_n + tl.arange(0, BLOCK_N_KV)

    dv = tl.zeros([BLOCK_N_KV, HEAD_DIM], dtype=tl.float32)
    dk = tl.zeros([BLOCK_N_KV, HEAD_DIM], dtype=tl.float32)

    # load K and V: they stay in SRAM throughout the inner loop.
    k = tl.load(K + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d)
    v = tl.load(V + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d)

    q_scale_offset = off_b * HEAD_NUM * tl.cdiv(SEQ_LEN, BLOCK_Q_Q) + off_h * tl.cdiv(SEQ_LEN, BLOCK_Q_Q)
    Q_scale_ptr = Q_scale + q_scale_offset + start_m  
    q_scale = tl.load(Q_scale_ptr)

    kv_scale_offset = off_b * (HEAD_NUM // GROUPS) * tl.cdiv(SEQ_LEN, BLOCK_N_KV) + (off_h // GROUPS) * tl.cdiv(SEQ_LEN, BLOCK_N_KV)
    K_scale_ptr = K_scale + kv_scale_offset
    V_scale_ptr = V_scale + kv_scale_offset

    num_steps = BLOCK_N_KV // MASK_BLOCK_Q_KV

    dk, dv = _attn_bwd_dkdv(dk, dv, 
                            Q, k, v, 
                            q_scale, K_scale_ptr, V_scale_ptr,
                            DO, LSE, D, 
                            stride_tok, stride_d, 
                            HEAD_NUM, SEQ_LEN, 
                            MASK_BLOCK_Q_KV, BLOCK_N_KV, HEAD_DIM, 
                            start_n, start_m, num_steps, 
                            MASK=True 
                            )

    start_m += num_steps * MASK_BLOCK_Q_KV
    num_steps = (SEQ_LEN - start_m) // BLOCK_Q_KV

    # Compute dK and dV for non-masked blocks.
    dk, dv = _attn_bwd_dkdv( 
        dk, dv, 
        Q, k, v, 
        q_scale, K_scale_ptr, V_scale_ptr,
        DO, LSE, D, 
        stride_tok, stride_d, 
        HEAD_NUM, SEQ_LEN,
        BLOCK_Q_KV, BLOCK_N_KV, HEAD_DIM, 
        start_n, start_m, num_steps, 
        MASK=False
    )
    dv_ptrs = DV + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d
    tl.store(dv_ptrs, dv)

    dk_ptrs = DK + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d
    tl.store(dk_ptrs, dk)

    # THIS BLOCK DOES DQ:
    start_m = pid * BLOCK_Q_Q
    end_n = start_m + BLOCK_Q_Q

    MASK_BLOCK_N_Q: tl.constexpr = BLOCK_N_Q // BLK_SLICE_FACTOR
    offs_m = start_m + tl.arange(0, BLOCK_Q_Q)

    q = tl.load(Q + offs_m[:, None] * stride_tok + offs_k[None, :] * stride_d)
    dq = tl.zeros([BLOCK_Q_Q, HEAD_DIM], dtype=tl.float32)
    do = tl.load(DO + offs_m[:, None] * stride_tok + offs_k[None, :] * stride_d)

    m = tl.load(LSE + offs_m)
    m = m[:, None]

    # Compute dQ for masked (diagonal) blocks.
    # NOTE: This code scans each row of QK^T backward (from right to left,
    # but inside each call to _attn_bwd_dq, from left to right), but that's
    # not due to anything important.  I just wanted to reuse the loop
    # structure for dK & dV above as much as possible.
    num_steps = BLOCK_Q_Q // MASK_BLOCK_N_Q
    dq = _attn_bwd_dq(dq, q, K, V, 
                      do, m, D, 
                      stride_tok, stride_d, 
                      HEAD_NUM, SEQ_LEN, 
                      BLOCK_Q_Q, MASK_BLOCK_N_Q, HEAD_DIM, 
                      start_m, end_n - num_steps * MASK_BLOCK_N_Q, num_steps, 
                      MASK=True 
                      )
    end_n -= num_steps * MASK_BLOCK_N_Q
    # stage 2
    num_steps = end_n // BLOCK_N_Q
    dq = _attn_bwd_dq(dq, q, K, V, 
                      do, m, D, 
                      stride_tok, stride_d, 
                      HEAD_NUM, SEQ_LEN, 
                      BLOCK_Q_Q, BLOCK_N_Q, HEAD_DIM, 
                      start_m, end_n - num_steps * BLOCK_N_Q, num_steps, 
                      MASK=False 
                      )
    # Write back dQ.
    dq_ptrs = DQ + offs_m[:, None] * stride_tok + offs_k[None, :] * stride_d
    dq *= LN2
    tl.store(dq_ptrs, dq)

class _attention(torch.autograd.Function):
    @staticmethod
    def forward(ctx,
                q, k, v, 
                sm_scale,
                output_dtype=torch.float16, 
                causal=False,
                quant_type="int8"):
        BLOCK_Q = 128
        BLOCK_N = 64
        quant_type_map = {
            "int8": 0,
            "e4m3": 1,
            "e5m2": 2,
            "none": 3
        }
        assert quant_type in quant_type_map, f"Unsupported quant type: {quant_type}"

        # q k v 的 shape 是 [B, HEAD_NUM, S, D]
        BATCH, HEAD_N_Q, SEQ_Q, HEAD_DIM_Q = q.shape
        _, HEAD_N_K, SEQ_KV, HEAD_DIM_K = k.shape
        HEAD_DIM_V = v.shape[-1] 
        assert HEAD_DIM_Q == HEAD_DIM_K and HEAD_DIM_K == HEAD_DIM_V
        assert HEAD_DIM_K in {64, 128, 256}
        GROUPS =  max(HEAD_N_Q // HEAD_N_K, 1)

        q_quant = torch.empty(q.shape, dtype=quant_type_map[quant_type], device=q.device)
        k_quant = torch.empty(k.shape, dtype=quant_type_map[quant_type], device=k.device)
        v_quant = torch.empty(v.shape, dtype=quant_type_map[quant_type], device=v.device)
        o = torch.empty(q.shape, dtype=output_dtype, device=q.device)
        lse = torch.empty((BATCH, HEAD_N_Q, SEQ_Q), dtype=torch.float32, device=q.device)

        q_scale = torch.empty((batch_size, num_heads_q, (seq_len_q + BLKQ - 1) // BLKQ), 
                         device=q.device, dtype=torch.float32)
        k_scale = torch.empty((batch_size, num_heads_kv, (seq_len_kv + BLKK - 1) // BLKK), 
                            device=k.device, dtype=torch.float32)
        v_scale = torch.empty((batch_size, num_heads_kv, (seq_len_kv + BLKV - 1) // BLKV), 
                            device=v.device, dtype=torch.float32)
        
        # # q_scale: [BATCH, HEAD_N_Q, num_q_blocks]
        # num_q_blocks  = (SEQ_Q + BLOCK_Q - 1) // BLOCK_Q
        # num_kv_blocks = (SEQ_Q + BLOCK_N - 1) // BLOCK_N
        # assert q_scale.ndim == 3 and q_scale.shape == (BATCH, HEAD_N_Q, num_q_blocks), \
        #     "q_scale.shape: {}, expected: (BATCH, HEAD_N_Q, num_q_blocks)".format(q_scale.shape)
        # assert k_scale.ndim == 3 and k_scale.shape == (BATCH, HEAD_N_K, num_kv_blocks), \
        #     "k_scale.shape: {}, expected: (BATCH, HEAD_N_K, num_kv_blocks)".format(k_scale.shape)
        # assert v_scale.shape == k_scale.shape



        stage = 3 if causal else 1
        extra_kern_args = {}
        # 沿 seq 维度分块 BLOCK_Q；将 Batch 和 Head 维度合并；表示每个 warp 只处理 1 个 batch 和 head
        # grid = lambda args: (triton.cdiv(SEQ_L, args["BLOCK_Q"]), BATCH * HEAD_N_Q, 1)
        grid = (triton.cdiv(SEQ_Q, BLOCK_Q), HEAD_N_Q, BATCH)
        ctx.grid = grid
        _attn_fwd[grid](
            q, k, v, 
            q_scale, k_scale, v_scale,
            lse, o, 
            q.stride(0), q.stride(1), q.stride(2), q.stride(3), 
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            o.stride(0), o.stride(1), o.stride(2), o.stride(3),
            HEAD_NUM=HEAD_N_Q, SEQ_Q=SEQ_Q, GROUPS=GROUPS, 
            HEAD_DIM=HEAD_DIM_K, SEQ_KV=SEQ_KV,
            BLOCK_Q=BLOCK_Q, BLOCK_N=BLOCK_N,
            STAGE=stage,
            QUANT_TYPE=quant_type_map[quant_type], 
            **extra_kern_args)
        ctx.save_for_backward(q, k, v, o, lse, q_scale, k_scale, v_scale)
        ctx.HEAD_DIM = HEAD_DIM_K
        ctx.GROUPS = GROUPS
        ctx.causal = causal
        return o, lse

    @staticmethod
    def backward(ctx, do):
        q, k, v, o, lse, q_scale, k_scale, v_scale = ctx.saved_tensors

        assert do.is_contiguous()
        assert q.stride() == k.stride() == v.stride() == o.stride() == do.stride()

        dq = torch.empty_like(q)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)
        BATCH, N_HEAD, SEQ_L = q.shape[:3]
        PRE_BLOCK = 128
        NUM_WARPS, NUM_STAGES = 4, 5
        #! 前向反向的quant scale block size 对齐
        BLOCK_Q_KV, BLOCK_N_KV, BLOCK_Q_Q, BLOCK_N_Q = 32, 128, 128, 32
        BLK_SLICE_FACTOR = 2

        k = k * 1.4426950408889634  # = 1.0 / ln(2)
        PRE_BLOCK = 128
        assert SEQ_L % PRE_BLOCK == 0
        pre_grid = (SEQ_L // PRE_BLOCK, BATCH * N_HEAD)
        delta = torch.empty_like(lse)
        _attn_bwd_preprocess[pre_grid](
            o, do, 
            delta, 
            BATCH, N_HEAD, SEQ_L, 
            BLOCK_Q=PRE_BLOCK, HEAD_DIM=ctx.HEAD_DIM 
        )
        grid = (SEQ_L // BLOCK_N_KV, 1, BATCH * N_HEAD)
        _attn_bwd[grid](
            q, k, v, 
            q_scale, k_scale, v_scale,
            do, dq, dk, dv, 
            lse, delta, 
            q.stride(0), q.stride(1), q.stride(2), q.stride(3), 
            N_HEAD, SEQ_L, ctx.GROUPS,
            BLOCK_Q_KV=BLOCK_Q_KV, BLOCK_N_KV=BLOCK_N_KV, 
            BLOCK_Q_Q=BLOCK_Q_Q, BLOCK_N_Q=BLOCK_N_Q, 
            BLK_SLICE_FACTOR=BLK_SLICE_FACTOR, 
            HEAD_DIM=ctx.HEAD_DIM, 
            num_warps=NUM_WARPS, 
            num_stages=NUM_STAGES 
        )

        return dq, dk, dv, None, None, None, None, None, None

def sage_attention(
    q, k, v, 
    sm_scale,
    output_dtype=torch.float16, 
    causal=False,
    quant_type="int8"):

    return _attention.apply(q, k, v, sm_scale, output_dtype, causal, quant_type)
