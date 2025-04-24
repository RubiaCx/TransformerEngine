import pytest
import torch
import triton.tools.experimental_descriptor

import triton
import triton.language as tl


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
                    K_block_ptr, V_block_ptr,
                    start_m, 
                    q_scale, K_scale_ptr, V_scale_ptr, 
                    HEAD_DIM: tl.constexpr, SEQ_LEN: tl.constexpr, GROUPS: tl.constexpr,
                    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                    STAGE: tl.constexpr, offs_m: tl.constexpr, offs_n: tl.constexpr):
    # range of values handled by this stage
    if STAGE == 1:
        lo, hi = 0, start_m * BLOCK_M
    elif STAGE == 2:
        lo, hi = start_m * BLOCK_M, (start_m + 1) * BLOCK_M
        lo = tl.multiple_of(lo, BLOCK_M)
        K_scale_ptr += lo // BLOCK_N
        V_scale_ptr += lo // BLOCK_N
    # causal = False
    else:
        lo, hi = 0, SEQ_LEN
    K_block_ptr = tl.advance(K_block_ptr, (0, lo))
    V_block_ptr = tl.advance(V_block_ptr, (lo, 0))

    for start_n in range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)

        k = tl.load(K_block_ptr)
        k_scale = tl.load(K_scale_ptr)
        qk = tl.dot(q, k).to(tl.float32) * q_scale * k_scale 
        if STAGE == 2:
            mask = offs_m[:, None] >= (start_n + offs_n[None, :])
            qk = qk + tl.where(mask, 0, -1.0e6)
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        qk = qk - m_ij[:, None]

        p = tl.math.exp2(qk)
        l_ij = tl.sum(p, 1)

        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + l_ij

        acc = acc * alpha[:, None]
        
        v = tl.load(V_block_ptr)
        p = p.to(v.dtype) 
        acc = tl.dot(p, v, acc)

        m_i = m_ij
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
        K_scale_ptr += 1
        V_scale_ptr += 1
    return acc, l_i, m_i

@triton.jit
def _attn_fwd_inner_quant(acc, l_i, m_i, q,
                          K_block_ptr, V_block_ptr,
                          start_m, 
                          q_scale, K_scale_ptr, V_scale_ptr, 
                          HEAD_DIM: tl.constexpr, SEQ_LEN: tl.constexpr, GROUPS: tl.constexpr,
                          BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                          STAGE: tl.constexpr, offs_m: tl.constexpr, offs_n: tl.constexpr,
                          QUANT_TYPE: tl.constexpr):
    # range of values handled by this stage
    if STAGE == 1:
        lo, hi = 0, start_m * BLOCK_M
    elif STAGE == 2:
        lo, hi = start_m * BLOCK_M, (start_m + 1) * BLOCK_M
        lo = tl.multiple_of(lo, BLOCK_M)
        K_scale_ptr += lo // BLOCK_N
        V_scale_ptr += lo // BLOCK_N
    # causal = False
    else:
        lo, hi = 0, SEQ_LEN
    K_block_ptr = tl.advance(K_block_ptr, (0, lo))
    V_block_ptr = tl.advance(V_block_ptr, (lo, 0))

    for start_n in range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)

        k = tl.load(K_block_ptr)
        k_scale = tl.load(K_scale_ptr)
        qk = tl.dot(q, k).to(tl.float32) * q_scale * k_scale 
        if STAGE == 2:
            mask = offs_m[:, None] >= (start_n + offs_n[None, :])
            qk = qk + tl.where(mask, 0, -1.0e6)
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

        v = tl.load(V_block_ptr)
        v_scale = tl.load(V_scale_ptr)

        if QUANT_TYPE == 0:
            accumulator = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.int32)
            middle = tl.dot(p_quant, v, accumulator, out_dtype=tl.float32) * v_scale * p_scale
        else:
            accumulator = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
            middle = tl.dot(p_quant, v, accumulator).to(tl.float32) * v_scale * p_scale
        acc = acc + middle

        m_i = m_ij 
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
        K_scale_ptr += 1
        V_scale_ptr += 1
    return acc, l_i, m_i

@triton.jit
def _attn_fwd(Q, K, V, 
              Q_scale, K_scale, V_scale,
              LSE, Out, 
              stride_qb, stride_qh, stride_qm, stride_qk, 
              stride_kb, stride_kh, stride_kn, stride_kk, 
              stride_vb, stride_vh, stride_vk, stride_vn, 
              stride_ob, stride_oh, stride_om, stride_on, 
              HEAD_NUM: tl.constexpr, SEQ_LEN: tl.constexpr, # Q
              GROUPS: tl.constexpr, HEAD_DIM: tl.constexpr, # K
              BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, 
              STAGE: tl.constexpr,
              QUANT_TYPE: tl.constexpr,
              ):
    tl.static_assert(BLOCK_N <= HEAD_DIM)
    
    start_m = tl.program_id(0)
    off_b = tl.program_id(2).to(tl.int64)
    off_h = tl.program_id(1).to(tl.int64)
    off_kv_h = (off_h // GROUPS).to(tl.int64) 

    q_offset = off_b * stride_qb + off_h * stride_qh     
    k_offset = off_b * stride_kb + off_kv_h * stride_kh 
    v_offset = off_b * stride_vb + off_kv_h * stride_vh 

    q_scale_offset = off_b * HEAD_NUM * tl.cdiv(SEQ_LEN, BLOCK_M) + off_h * tl.cdiv(SEQ_LEN, BLOCK_M)
    Q_scale_ptr = Q_scale + q_scale_offset + start_m
    q_scale = tl.load(Q_scale_ptr)

    # kv_scale_offset = off_b * (HEAD_NUM // GROUPS) * tl.cdiv(SEQ_LEN, BLOCK_N) + (off_h // GROUPS) * tl.cdiv(SEQ_LEN, BLOCK_N)
    # kv_scale_offset = off_b * HEAD_NUM * tl.cdiv(SEQ_LEN, BLOCK_N) // GROUPS + off_kv_h * tl.cdiv(SEQ_LEN, BLOCK_N)
    kv_scale_offset = (off_b * (HEAD_NUM // GROUPS) + off_kv_h) * tl.cdiv(SEQ_LEN, BLOCK_N)
    K_scale_ptr = K_scale + kv_scale_offset
    V_scale_ptr = V_scale + kv_scale_offset

    Q_block_ptr = tl.make_block_ptr(
        base=Q + q_offset,
        shape=(SEQ_LEN, HEAD_DIM),
        strides=(stride_qm, stride_qk),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, HEAD_DIM),
        order=(1, 0),
    )
    V_block_ptr = tl.make_block_ptr(
        base=V + v_offset,
        shape=(SEQ_LEN, HEAD_DIM),
        strides=(stride_vk, stride_vn),
        offsets=(0, 0),
        block_shape=(BLOCK_N, HEAD_DIM), 
        order=(1, 0), # (1, 0) 行优先 (0, 1) 列优先
    )
    K_block_ptr = tl.make_block_ptr(
        base=K + k_offset,
        shape=(HEAD_DIM, SEQ_LEN),
        strides=(stride_kk, stride_kn),
        offsets=(0, 0),
        block_shape=(HEAD_DIM, BLOCK_N),
        order=(1, 0),
    )
    O_block_ptr = tl.make_block_ptr(
        base=Out + q_offset,
        shape=(SEQ_LEN, HEAD_DIM),
        strides=(stride_om, stride_on),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, HEAD_DIM),
        order=(1, 0),
    )
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    q = tl.load(Q_block_ptr) # load q: it will stay in SRAM throughout
    
    if QUANT_TYPE == 3:
        if STAGE & 1: # 1 or 3
            acc, l_i, m_i = _attn_fwd_inner(acc, l_i, m_i, q, K_block_ptr, V_block_ptr, 
                                            start_m, 
                                            q_scale, K_scale_ptr, V_scale_ptr, 
                                            HEAD_DIM, SEQ_LEN, GROUPS,
                                            BLOCK_M, BLOCK_N, 
                                            4 - STAGE, offs_m, offs_n)
        if STAGE & 2: # 2 or 3
            acc, l_i, m_i = _attn_fwd_inner(acc, l_i, m_i, q, K_block_ptr, V_block_ptr, 
                                            start_m, 
                                            q_scale, K_scale_ptr, V_scale_ptr, 
                                            HEAD_DIM, SEQ_LEN, GROUPS,
                                            BLOCK_M, BLOCK_N, 
                                            2, offs_m, offs_n)
    else:
        if STAGE & 1: # 1 or 3
            acc, l_i, m_i = _attn_fwd_inner_quant(acc, l_i, m_i, q, K_block_ptr, V_block_ptr, 
                                                  start_m, 
                                                  q_scale, K_scale_ptr, V_scale_ptr, 
                                                  HEAD_DIM, SEQ_LEN, GROUPS,
                                                  BLOCK_M, BLOCK_N, 
                                                  4 - STAGE, offs_m, offs_n,  QUANT_TYPE)
        if STAGE & 2: # 2 or 3
            acc, l_i, m_i = _attn_fwd_inner_quant(acc, l_i, m_i, q, K_block_ptr, V_block_ptr, 
                                                  start_m, 
                                                  q_scale, K_scale_ptr, V_scale_ptr, 
                                                  HEAD_DIM, SEQ_LEN, GROUPS,
                                                  BLOCK_M,  BLOCK_N, 
                                                  2, offs_m, offs_n,  QUANT_TYPE)
    m_i += tl.math.log2(l_i)
    acc = acc / l_i[:, None]
    lse_ptrs = LSE + (off_b * SEQ_LEN * HEAD_NUM + off_h * SEQ_LEN) + offs_m
    tl.store(lse_ptrs, m_i, mask=(offs_m < SEQ_LEN))
    tl.store(O_block_ptr, acc.to(Out.type.element_ty))

# 计算 Delta = rowsum(O ⊙ dO)
@triton.jit
def _attn_bwd_preprocess(O, DO, 
                         Delta,  
                         BS, HEAD_NUM, SEQ_LEN, 
                         BLOCK_M: tl.constexpr, HEAD_DIM: tl.constexpr 
                         ):
    off_m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
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
                   BLOCK_M: tl.constexpr, 
                   BLOCK_N: tl.constexpr, 
                   HEAD_DIM: tl.constexpr, 
                   # Filled in by the wrapper.
                   start_n, start_m, num_steps, 
                   MASK: tl.constexpr):
    offs_m = start_m + tl.arange(0, BLOCK_M)
    offs_n = start_n + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, HEAD_DIM)

    qT_ptrs = Q + offs_m[None, :] * stride_tok + offs_k[:, None] * stride_d
    do_ptrs = DO + offs_m[:, None] * stride_tok + offs_k[None, :] * stride_d
    # BLOCK_N must be a multiple of BLOCK_M, otherwise the code wouldn't work.
    tl.static_assert(BLOCK_N % BLOCK_M == 0)
    curr_m = start_m
    step_m = BLOCK_M
    for blk_idx in range(num_steps):
        k_scale = tl.load(K_scale_ptr)
        v_scale = tl.load(V_scale_ptr)
        qT = tl.load(qT_ptrs)
        # Load m before computing qk to reduce pipeline stall.
        offs_m = curr_m + tl.arange(0, BLOCK_M)
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
# 固定处理 Q 和 dO 的一个块（形状为 BLOCK_M_Q × HEAD_DIM），然后迭代处理 K 和 V 的多个块
@triton.jit
def _attn_bwd_dq(dq, q, K, V, 
                 q_scale, K_scale_ptr, V_scale_ptr,
                 do, m, D,
                 # shared by Q/K/V/DO.
                 stride_tok, stride_d, 
                 HEAD_NUM, SEQ_LEN, 
                 BLOCK_M: tl.constexpr, 
                 BLOCK_N: tl.constexpr, 
                 HEAD_DIM: tl.constexpr,
                 # Filled in by the wrapper.
                 start_m, start_n, num_steps, 
                 MASK: tl.constexpr):
    offs_m = start_m + tl.arange(0, BLOCK_M)
    offs_n = start_n + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, HEAD_DIM)
    kT_ptrs = K + offs_n[None, :] * stride_tok + offs_k[:, None] * stride_d
    vT_ptrs = V + offs_n[None, :] * stride_tok + offs_k[:, None] * stride_d
    # D (= delta) is pre-divided by ds_scale.
    Di = tl.load(D + offs_m)
    # BLOCK_M must be a multiple of BLOCK_N, otherwise the code wouldn't work.
    tl.static_assert(BLOCK_M % BLOCK_N == 0)
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
              BLOCK_M_KV: tl.constexpr, 
              BLOCK_N_KV: tl.constexpr, 
              BLOCK_M_Q: tl.constexpr, 
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

    MASK_BLOCK_M_KV: tl.constexpr = BLOCK_M_KV // BLK_SLICE_FACTOR
    offs_n = start_n + tl.arange(0, BLOCK_N_KV)

    dv = tl.zeros([BLOCK_N_KV, HEAD_DIM], dtype=tl.float32)
    dk = tl.zeros([BLOCK_N_KV, HEAD_DIM], dtype=tl.float32)

    # load K and V: they stay in SRAM throughout the inner loop.
    k = tl.load(K + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d)
    v = tl.load(V + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d)

    q_scale_offset = off_b * HEAD_NUM * tl.cdiv(SEQ_LEN, BLOCK_M_Q) + off_h * tl.cdiv(SEQ_LEN, BLOCK_M_Q)
    Q_scale_ptr = Q_scale + q_scale_offset + start_m  
    q_scale = tl.load(Q_scale_ptr)

    kv_scale_offset = off_b * (HEAD_NUM // GROUPS) * tl.cdiv(SEQ_LEN, BLOCK_N_KV) + (off_h // GROUPS) * tl.cdiv(SEQ_LEN, BLOCK_N_KV)
    K_scale_ptr = K_scale + kv_scale_offset
    V_scale_ptr = V_scale + kv_scale_offset

    num_steps = BLOCK_N_KV // MASK_BLOCK_M_KV

    dk, dv = _attn_bwd_dkdv(dk, dv, 
                            Q, k, v, 
                            q_scale, K_scale_ptr, V_scale_ptr,
                            DO, LSE, D, 
                            stride_tok, stride_d, 
                            HEAD_NUM, SEQ_LEN, 
                            MASK_BLOCK_M_KV, BLOCK_N_KV, HEAD_DIM, 
                            start_n, start_m, num_steps, 
                            MASK=True 
                            )

    start_m += num_steps * MASK_BLOCK_M_KV
    num_steps = (SEQ_LEN - start_m) // BLOCK_M_KV

    # Compute dK and dV for non-masked blocks.
    dk, dv = _attn_bwd_dkdv( 
        dk, dv, 
        Q, k, v, 
        q_scale, K_scale_ptr, V_scale_ptr,
        DO, LSE, D, 
        stride_tok, stride_d, 
        HEAD_NUM, SEQ_LEN,
        BLOCK_M_KV, BLOCK_N_KV, HEAD_DIM, 
        start_n, start_m, num_steps, 
        MASK=False
    )
    dv_ptrs = DV + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d
    tl.store(dv_ptrs, dv)

    dk_ptrs = DK + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d
    tl.store(dk_ptrs, dk)

    # THIS BLOCK DOES DQ:
    start_m = pid * BLOCK_M_Q
    end_n = start_m + BLOCK_M_Q

    MASK_BLOCK_N_Q: tl.constexpr = BLOCK_N_Q // BLK_SLICE_FACTOR
    offs_m = start_m + tl.arange(0, BLOCK_M_Q)

    q = tl.load(Q + offs_m[:, None] * stride_tok + offs_k[None, :] * stride_d)
    dq = tl.zeros([BLOCK_M_Q, HEAD_DIM], dtype=tl.float32)
    do = tl.load(DO + offs_m[:, None] * stride_tok + offs_k[None, :] * stride_d)

    m = tl.load(LSE + offs_m)
    m = m[:, None]

    # Compute dQ for masked (diagonal) blocks.
    # NOTE: This code scans each row of QK^T backward (from right to left,
    # but inside each call to _attn_bwd_dq, from left to right), but that's
    # not due to anything important.  I just wanted to reuse the loop
    # structure for dK & dV above as much as possible.
    num_steps = BLOCK_M_Q // MASK_BLOCK_N_Q
    dq = _attn_bwd_dq(dq, q, K, V, 
                      do, m, D, 
                      stride_tok, stride_d, 
                      HEAD_NUM, SEQ_LEN, 
                      BLOCK_M_Q, MASK_BLOCK_N_Q, HEAD_DIM, 
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
                      BLOCK_M_Q, BLOCK_N_Q, HEAD_DIM, 
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
                q_scale, k_scale, v_scale, 
                output_dtype=torch.float16, 
                causal=False,
                quant_type="int8"):
        BLOCK_M = 128
        BLOCK_N = 64
        # q k v 的 shape 是 [B, H, S, D]
        BATCH, HEAD_N_Q, SEQ_L, HEAD_DIM_Q = q.shape
        _, HEAD_N_K, _, HEAD_DIM_K = k.shape
        HEAD_DIM_V = v.shape[-1] 
        assert HEAD_DIM_Q == HEAD_DIM_K and HEAD_DIM_K == HEAD_DIM_V
        assert HEAD_DIM_K in {64, 128, 256}
        GROUPS =  max(HEAD_N_Q // HEAD_N_K, 1)

        # q_scale: [BATCH, HEAD_N_Q, num_q_blocks]
        num_q_blocks  = (SEQ_L + BLOCK_M - 1) // BLOCK_M
        num_kv_blocks = (SEQ_L + BLOCK_N - 1) // BLOCK_N
        assert q_scale.ndim == 3 and q_scale.shape == (BATCH, HEAD_N_Q, num_q_blocks), \
            "q_scale.shape: {}, expected: (BATCH, HEAD_N_Q, num_q_blocks)".format(q_scale.shape)
        assert k_scale.ndim == 3 and k_scale.shape == (BATCH, HEAD_N_K, num_kv_blocks), \
            "k_scale.shape: {}, expected: (BATCH, HEAD_N_K, num_kv_blocks)".format(k_scale.shape)
        assert v_scale.shape == k_scale.shape

        o = torch.empty(q.shape, dtype=output_dtype, device=q.device)
        lse = torch.empty((BATCH, HEAD_N_Q, SEQ_L), dtype=torch.float32, device=q.device)

        quant_type_map = {
            "int8": 0,
            "e4m3": 1,
            "e5m2": 2,
            "none": 3
        }
        assert quant_type in quant_type_map, f"Unsupported quant type: {quant_type}"
        stage = 3 if causal else 1
        extra_kern_args = {}
        # 沿 seq 维度分块 BLOCK_M；将 Batch 和 Head 维度合并；表示每个 warp 只处理 1 个 batch 和 head
        # grid = lambda args: (triton.cdiv(SEQ_L, args["BLOCK_M"]), BATCH * HEAD_N_Q, 1)
        grid = (triton.cdiv(SEQ_L, BLOCK_M), HEAD_N_Q, BATCH)
        ctx.grid = grid
        _attn_fwd[grid](
            q, k, v, 
            q_scale, k_scale, v_scale,
            lse, o, 
            q.stride(0), q.stride(1), q.stride(2), q.stride(3), 
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            o.stride(0), o.stride(1), o.stride(2), o.stride(3),
            HEAD_NUM=HEAD_N_Q, SEQ_LEN=SEQ_L, GROUPS=GROUPS, HEAD_DIM=HEAD_DIM_K,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
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
        BLOCK_M_KV, BLOCK_N_KV, BLOCK_M_Q, BLOCK_N_Q = 32, 128, 128, 32
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
            BLOCK_M=PRE_BLOCK, HEAD_DIM=ctx.HEAD_DIM 
        )
        grid = (SEQ_L // BLOCK_N_KV, 1, BATCH * N_HEAD)
        _attn_bwd[grid](
            q, k, v, 
            q_scale, k_scale, v_scale,
            do, dq, dk, dv, 
            lse, delta, 
            q.stride(0), q.stride(1), q.stride(2), q.stride(3), 
            N_HEAD, SEQ_L, ctx.GROUPS,
            BLOCK_M_KV=BLOCK_M_KV, BLOCK_N_KV=BLOCK_N_KV, 
            BLOCK_M_Q=BLOCK_M_Q, BLOCK_N_Q=BLOCK_N_Q, 
            BLK_SLICE_FACTOR=BLK_SLICE_FACTOR, 
            HEAD_DIM=ctx.HEAD_DIM, 
            num_warps=NUM_WARPS, 
            num_stages=NUM_STAGES 
        )

        return dq, dk, dv, None, None, None, None, None, None

def sage_attention(
    q, k, v, 
    q_scale, k_scale, v_scale, 
    output_dtype=torch.float16, 
    causal=False,
    quant_type="int8"):

    return _attention.apply(q, k, v, q_scale, k_scale, v_scale, output_dtype, causal, quant_type)
