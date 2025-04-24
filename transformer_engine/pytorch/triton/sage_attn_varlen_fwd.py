import pytest
import torch
import triton.tools.experimental_descriptor

import triton
import triton.language as tl

@triton.jit
def _attn_fwd_inner(acc, l_i, m_i, q,
                    K_block_ptr, V_block_ptr,
                    start_m, 
                    q_scale, K_scale_ptr, V_scale_ptr, 
                    HEAD_NUM: tl.constexpr, HEAD_DIM: tl.constexpr, SEQ_LEN: tl.constexpr,
                    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                    STAGE: tl.constexpr, offs_m: tl.constexpr, offs_n: tl.constexpr):
    # range of values handled by this stage
    if STAGE == 1:
        lo, hi = 0, start_m * BLOCK_M
    elif STAGE == 2:
        lo, hi = start_m * BLOCK_M, (start_m + 1) * BLOCK_M
        lo = tl.multiple_of(lo, BLOCK_M)
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
        K_scale_ptr += HEAD_NUM
        V_scale_ptr += 1
    return acc, l_i, m_i

@triton.jit
def _attn_fwd_inner_quant(acc, l_i, m_i, q,
                          K_block_ptr, V_block_ptr,
                          start_m, 
                          q_scale, K_scale_ptr, V_scale_ptr, 
                          HEAD_NUM: tl.constexpr, HEAD_DIM: tl.constexpr, SEQ_LEN: tl.constexpr,
                          BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                          STAGE: tl.constexpr, offs_m: tl.constexpr, offs_n: tl.constexpr,
                          QUANT_TYPE: tl.constexpr):
    # range of values handled by this stage
    if STAGE == 1:
        lo, hi = 0, start_m * BLOCK_M
    elif STAGE == 2:
        lo, hi = start_m * BLOCK_M, (start_m + 1) * BLOCK_M
        lo = tl.multiple_of(lo, BLOCK_M)
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
        K_scale_ptr += HEAD_NUM
        V_scale_ptr += 1
    return acc, l_i, m_i


@triton.jit
def _attn_fwd(Q, K, V, cu_seqlens_q, cu_seqlens_k, max_seqlen_q,
              Q_scale, K_scale, V_scale, cu_seqlens_q_scale, cu_seqlens_k_scale, cu_seqlens_v_scale,
              LSE, Out, 
              stride_qb, stride_qh, stride_qm, 
              stride_kb, stride_kh, stride_kn, 
              stride_vb, stride_vh, stride_vk, 
              stride_ob, stride_oh, stride_om, 
              BS, MAX_SEQ_LEN, GROUPS,
              HEAD_NUM: tl.constexpr,
              HEAD_DIM: tl.constexpr, 
              BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, 
              STAGE: tl.constexpr,
              QUANT_TYPE: tl.constexpr,
              ):
    tl.static_assert(BLOCK_N <= HEAD_DIM)
    
    start_m = tl.program_id(0)
    off_hb = tl.program_id(1)
    off_b = (off_hb // HEAD_NUM).to(tl.int64)
    off_h = (off_hb % HEAD_NUM).to(tl.int64)

    cu_seqlens_q_start = tl.load(cu_seqlens_q + off_b)
    cu_seqlens_q_end = tl.load(cu_seqlens_q + off_b + 1)
    qo_len = cu_seqlens_q_end - cu_seqlens_q_start

    if (start_m * BLOCK_M) >= qo_len:
        return
    
    qkv_offset = off_b * stride_qb + off_h * stride_qh

    q_scale_offset = off_b * HEAD_NUM * tl.cdiv(SEQ_LEN, BLOCK_M) + off_h * tl.cdiv(SEQ_LEN, BLOCK_M)
    Q_scale_ptr = Q_scale + q_scale_offset + start_m  
    q_scale = tl.load(Q_scale_ptr)

    kv_scale_offset = off_b * (HEAD_NUM // GROUPS) * tl.cdiv(SEQ_LEN, BLOCK_N) + (off_h // GROUPS) * tl.cdiv(SEQ_LEN, BLOCK_N)
    K_scale_ptr = K_scale + kv_scale_offset
    V_scale_ptr = V_scale + kv_scale_offset

    Q_block_ptr = tl.make_block_ptr(
        base=Q + qkv_offset,
        shape=(SEQ_LEN, HEAD_DIM),
        strides=(stride_qm, stride_qk),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, HEAD_DIM),
        order=(1, 0),
    )
    v_order: tl.constexpr = (0, 1) if V.dtype.element_ty == tl.float8e5 else (1, 0)
    V_block_ptr = tl.make_block_ptr(
        base=V + qkv_offset,
        shape=(SEQ_LEN, HEAD_DIM),
        strides=(stride_vk, stride_vn),
        offsets=(0, 0),
        block_shape=(BLOCK_N, HEAD_DIM),
        order=v_order,
    )
    K_block_ptr = tl.make_block_ptr(
        base=K + qkv_offset,
        shape=(HEAD_DIM, SEQ_LEN),
        strides=(stride_kk, stride_kn),
        offsets=(0, 0),
        block_shape=(HEAD_DIM, BLOCK_N),
        order=(0, 1),
    )
    O_block_ptr = tl.make_block_ptr(
        base=Out + qkv_offset,
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
    if QUANT_TYPE != 3:
        if STAGE & 1: # 1 or 3
            acc, l_i, m_i = _attn_fwd_inner_quant(acc, l_i, m_i, q, K_block_ptr, V_block_ptr, 
                                                  start_m, 
                                                  q_scale, K_scale_ptr, V_scale_ptr, 
                                                  BLOCK_M, HEAD_DIM, BLOCK_N, 
                                                  4 - STAGE, offs_m, offs_n, SEQ_LEN, QUANT_TYPE)
        if STAGE & 2: # 2 or 3
            acc, l_i, m_i = _attn_fwd_inner_quant(acc, l_i, m_i, q, K_block_ptr, V_block_ptr, 
                                                  start_m, 
                                                  q_scale, K_scale_ptr, V_scale_ptr, 
                                                  BLOCK_M, HEAD_DIM, BLOCK_N, 
                                                  2, offs_m, offs_n, SEQ_LEN, QUANT_TYPE)
    else:
        if STAGE & 1: # 1 or 3
            acc, l_i, m_i = _attn_fwd_inner(acc, l_i, m_i, q, K_block_ptr, V_block_ptr, 
                                            start_m, 
                                            q_scale, K_scale_ptr, V_scale_ptr, 
                                            BLOCK_M, HEAD_DIM, BLOCK_N, 
                                            4 - STAGE, offs_m, offs_n, SEQ_LEN)
        if STAGE & 2: # 2 or 3
            acc, l_i, m_i = _attn_fwd_inner(acc, l_i, m_i, q, K_block_ptr, V_block_ptr, 
                                            start_m, 
                                            q_scale, K_scale_ptr, V_scale_ptr,  
                                            BLOCK_M, HEAD_DIM, BLOCK_N, 
                                            2, offs_m, offs_n, SEQ_LEN)
    m_i += tl.math.log2(l_i)
    acc = acc / l_i[:, None]
    lse_ptrs = LSE + off_hb * SEQ_LEN + offs_m
    tl.store(lse_ptrs, m_i)
    tl.store(O_block_ptr, acc.to(Out.type.element_ty))

def forward(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q,
            q_scale, k_scale, v_scale, cu_seqlens_q_scale, cu_seqlens_k_scale, cu_seqlens_v_scale,
            output_dtype=torch.float16, 
            causal=False,
            quant_type="int8"):
        BLOCK_M = 128
        BLOCK_N = 64
        # q k v 的 shape 是 [T, H, D]
        _, HEAD_N_Q, HEAD_DIM_Q = q.shape
        _, HEAD_N_K, HEAD_DIM_K = k.shape
        HEAD_DIM_V = v.shape[-1] 
        BATCH = cu_seqlens_q.shape[0] - 1
        assert HEAD_DIM_Q == HEAD_DIM_K and HEAD_DIM_K == HEAD_DIM_V
        assert HEAD_DIM_K in {16, 32, 64, 128, 256}
        GROUPS =  max(HEAD_N_Q // HEAD_N_K, 1)
        o = torch.empty(q.shape, dtype=output_dtype, device=q.device)
        lse = torch.empty((BATCH, HEAD_N_Q, max_seqlen_q), dtype=torch.float32, device=q.device)

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
        grid = lambda args: (triton.cdiv(max_seqlen_q,  BLOCK_M), HEAD_N_Q * BATCH, 1)

        _attn_fwd[grid](
            q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q,
            q_scale, k_scale, v_scale, cu_seqlens_q_scale, cu_seqlens_k_scale, cu_seqlens_v_scale,
            lse, o, 
            q.stride(0), q.stride(1), q.stride(2), 
            k.stride(0), k.stride(1), k.stride(2), 
            v.stride(0), v.stride(1), v.stride(2), 
            o.stride(0), o.stride(1), o.stride(2), 
            BATCH, HEAD_N_Q, max_seqlen_q, GROUPS, HEAD_DIM=HEAD_DIM_K,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
            STAGE=stage,
            QUANT_TYPE=quant_type_map[quant_type], 
            **extra_kern_args)
        
        return o, lse