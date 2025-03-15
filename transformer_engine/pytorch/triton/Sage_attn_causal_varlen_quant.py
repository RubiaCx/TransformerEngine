"""
Copyright (c) 2024 by SageAttention team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import torch, math
import triton
import triton.language as tl

@triton.jit
def _attn_fwd_inner(acc, l_i, m_i, q, q_scale, kv_len,
                    K_ptrs, K_scale_ptr, V_ptrs, V_scale_ptr, stride_kn, stride_vn, 
                    start_m,  
                    H: tl.constexpr,
                    BLOCK_M: tl.constexpr, HEAD_DIM: tl.constexpr, BLOCK_N: tl.constexpr,  
                    STAGE: tl.constexpr, offs_m: tl.constexpr, offs_n: tl.constexpr,  
                    QUANT_TYPE: tl.constexpr):
    if STAGE == 1:
        lo, hi = 0, start_m * BLOCK_M
    elif STAGE == 2:
        lo, hi = start_m * BLOCK_M, (start_m + 1) * BLOCK_M
        lo = tl.multiple_of(lo, BLOCK_M)
        K_scale_ptr += (lo // BLOCK_N) * H
        K_ptrs += stride_kn * lo
        V_ptrs += stride_vn * lo
    for start_n in range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        k_mask = offs_n[None, :] < (kv_len - start_n)   
        k = tl.load(K_ptrs, mask=k_mask)
        k_scale = tl.load(K_scale_ptr)
        qk = tl.dot(q, k).to(tl.float32) * q_scale * k_scale 

        if STAGE == 2:
            mask = offs_m[:, None] >= (start_n + offs_n[None, :])
            qk = qk + tl.where(mask, 0, -1.0e6)
            m_ij = tl.maximum(m_i, tl.max(qk, 1))
            qk -= m_ij[:, None]
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

        v = tl.load(V_ptrs, mask=offs_n[:, None] < (kv_len - start_n))
        v_scale = tl.load(V_scale_ptr)

        if QUANT_TYPE == 0:
            accumulator = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.int32)
            middle = tl.dot(p_quant, v, accumulator, out_dtype=tl.float32) * v_scale * p_scale
        else:
            accumulator = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
            middle = tl.dot(p_quant, v, accumulator).to(tl.float32) * v_scale * p_scale
        
        acc = acc + middle
        m_i = m_ij
        K_ptrs += BLOCK_N * stride_kn
        K_scale_ptr += H
        V_ptrs += BLOCK_N * stride_vn
        V_scale_ptr += 1
    return acc, l_i, m_i

@triton.jit
def _attn_fwd(Q, K, V, 
              cu_seqlens_q, cu_seqlens_k, max_seqlen_q,
              Q_scale, K_scale, V_scale, cu_seqlens_q_scale, cu_seqlens_k_scale, cu_seqlens_v_scale,
              Out, Lse,
              stride_qh, stride_qn,
              stride_kh, stride_kn,  
              stride_vh, stride_vn,  
              stride_oh, stride_on,  
              H: tl.constexpr, num_kv_groups: tl.constexpr,
              HEAD_DIM: tl.constexpr,  
              BLOCK_M: tl.constexpr,  
              BLOCK_N: tl.constexpr, 
              STAGE: tl.constexpr,
              RETURN_LSE: tl.constexpr,
              QUANT_TYPE: tl.constexpr
              ):
    start_m = tl.program_id(0)
    off_z = tl.program_id(2).to(tl.int64)
    off_h = tl.program_id(1).to(tl.int64)

    cu_seqlens_q_start = tl.load(cu_seqlens_q + off_z)
    cu_seqlens_q_end = tl.load(cu_seqlens_q + off_z + 1)
    qo_len = cu_seqlens_q_end - cu_seqlens_q_start

    if (start_m * BLOCK_M) >= qo_len:
        return

    cu_seq_lens_q_scale_start = tl.load(cu_seqlens_q_scale + off_z)
    cu_seq_lens_k_scale_start = tl.load(cu_seqlens_k_scale + off_z)    
    cu_seq_lens_v_scale_start = tl.load(cu_seqlens_v_scale + off_z)
    q_scale_offset = cu_seq_lens_q_scale_start * H + off_h + start_m * H
    k_scale_offset = cu_seq_lens_k_scale_start * (H // num_kv_groups) + off_h // num_kv_groups
    v_scale_offset = cu_seq_lens_v_scale_start * (H // num_kv_groups) + off_h // num_kv_groups

    cu_seqlens_k_start = tl.load(cu_seqlens_k + off_z)
    cu_seqlens_k_end = tl.load(cu_seqlens_k + off_z + 1)
    kv_len = cu_seqlens_k_end - cu_seqlens_k_start

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, HEAD_DIM)
    Q_ptrs = Q + (cu_seqlens_q_start * stride_qn + off_h * stride_qh) + offs_m[:, None] * stride_qn + offs_k[None, :]
    Q_scale_ptr = Q_scale + q_scale_offset
    K_ptrs = K + (cu_seqlens_k_start * stride_kn + (off_h // num_kv_groups) * stride_kh) + offs_n[None, :] * stride_kn + offs_k[:, None] 
    K_scale_ptr = K_scale + k_scale_offset
    V_ptrs = V + (cu_seqlens_k_start * stride_vn + (off_h // num_kv_groups) * stride_vh) + offs_n[:, None] * stride_vn + offs_k[None, :]
    V_scale_ptr = V_scale + v_scale_offset
    O_block_ptr = Out + (cu_seqlens_q_start * stride_on + off_h * stride_oh) + offs_m[:, None] * stride_on + offs_k[None, :]
    
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    
    q = tl.load(Q_ptrs, mask=offs_m[:, None] < qo_len)
    q_scale = tl.load(Q_scale_ptr)
    acc, l_i, m_i = _attn_fwd_inner(acc, l_i, m_i, q, q_scale, kv_len, K_ptrs, K_scale_ptr, V_ptrs, V_scale_ptr, 
                                   stride_kn, stride_vn, start_m, H // num_kv_groups,
                                   BLOCK_M, HEAD_DIM, BLOCK_N, 4 - STAGE, offs_m, offs_n, QUANT_TYPE)
    acc, l_i, m_i = _attn_fwd_inner(acc, l_i, m_i, q, q_scale, kv_len, K_ptrs, K_scale_ptr, V_ptrs, V_scale_ptr, 
                                 stride_kn, stride_vn, start_m, H // num_kv_groups,
                                 BLOCK_M, HEAD_DIM, BLOCK_N, 2, offs_m, offs_n, QUANT_TYPE)
    acc = acc / l_i[:, None]
    tl.store(O_block_ptr, acc.to(Out.type.element_ty), mask=(offs_m[:, None] < qo_len))
    if RETURN_LSE:
        lse_ptrs = Lse + (off_z * H + off_h) * max_seqlen_q + offs_m
        l_i = tl.log2(l_i) + m_i 
        tl.store(lse_ptrs, l_i, mask=(offs_m < qo_len))
def forward(q, k, v, 
            cu_seqlens_q, cu_seqlens_k, max_seqlen_q, 
            q_scale, k_scale, v_scale, cu_seqlens_q_scale, cu_seqlens_k_scale, cu_seqlens_v_scale,
            return_lse=False, 
            output_dtype=torch.float16,
            quant_type="int8"):
    BLOCK_M = 128
    BLOCK_N = 64
    stage = 3

    quant_type_map = {
        "int8": 0,
        "e4m3": 1,
        "e5m2": 2
    }
    assert quant_type in quant_type_map, f"Unsupported quant type: {quant_type}"

    o = torch.empty(q.shape, dtype=output_dtype, device=q.device)
    batch_size = cu_seqlens_q.shape[0] - 1
    _, num_heads_q, head_dim = q.shape
    _, num_heads_kv, _ = k.shape

    HEAD_DIM_K = head_dim
    num_kv_groups = num_heads_q // num_heads_kv
    
    lse = torch.empty((batch_size, num_heads_q, max_seqlen_q), 
                      dtype=torch.float32, device=q.device) if return_lse else torch.empty([0], device=q.device)

    grid = (triton.cdiv(max_seqlen_q, BLOCK_M), num_heads_q, batch_size)
    _attn_fwd[grid](
        q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q,
        q_scale, k_scale, v_scale, cu_seqlens_q_scale, cu_seqlens_k_scale, cu_seqlens_v_scale,
        o, lse,
        q.stride(1), q.stride(0), 
        k.stride(1), k.stride(0),  
        v.stride(1), v.stride(0), 
        o.stride(1), o.stride(0),
        num_heads_q, num_kv_groups,
        HEAD_DIM=HEAD_DIM_K,  
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, 
        STAGE=stage,  
        RETURN_LSE=return_lse,
        QUANT_TYPE=quant_type_map[quant_type],
        num_warps=4 if head_dim == 64 else 8,
        num_stages=4
    )
    return o, lse