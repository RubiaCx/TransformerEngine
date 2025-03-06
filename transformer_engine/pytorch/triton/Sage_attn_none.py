# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import torch
import triton
import triton.language as tl

@triton.jit
def _attn_fwd_inner(acc, l_i, m_i, q, kv_len,
                    K_ptrs, V_ptrs, stride_kn, stride_vn, 
                    start_m,  
                    BLOCK_M: tl.constexpr, HEAD_DIM: tl.constexpr, BLOCK_N: tl.constexpr,  
                    STAGE: tl.constexpr, offs_m: tl.constexpr, offs_n: tl.constexpr):
    lo, hi = 0, kv_len
    for start_n in range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        k_mask = offs_n[None, :] < (kv_len - start_n)   
        k = tl.load(K_ptrs, mask = k_mask)
        
        qk = tl.dot(q, k).to(tl.float32)
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        qk = qk - m_ij[:, None]
        p = tl.math.exp2(qk)
        l_ij = tl.sum(p, 1)

        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        acc = acc * alpha[:, None]

        v = tl.load(V_ptrs, mask=offs_n[:, None] < (kv_len - start_n))
        acc += tl.dot(p.to(v.dtype), v)
        
        m_i = m_ij
        K_ptrs += BLOCK_N * stride_kn
        V_ptrs += BLOCK_N * stride_vn
    return acc, l_i, m_i

@triton.jit
def _attn_fwd(Q, K, V, Out, Lse, 
              stride_qz, stride_qh, stride_qn,
              stride_kz, stride_kh, stride_kn,  
              stride_vz, stride_vh, stride_vn,  
              stride_oz, stride_oh, stride_on,  
              qo_len, kv_len, H:tl.constexpr, num_kv_groups:tl.constexpr,
              HEAD_DIM: tl.constexpr,  
              BLOCK_M: tl.constexpr,  
              BLOCK_N: tl.constexpr,  
              STAGE: tl.constexpr,
              RETURN_LSE: tl.constexpr):
    
    start_m = tl.program_id(0)
    off_z = tl.program_id(2).to(tl.int64)
    off_h = tl.program_id(1).to(tl.int64)

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, HEAD_DIM)
    
    Q_ptrs = Q + (off_z * stride_qz + off_h * stride_qh) + offs_m[:, None] * stride_qn + offs_k[None, :]
    K_ptrs = K + (off_z * stride_kz + (off_h // num_kv_groups) * stride_kh) + offs_n[None, :] * stride_kn + offs_k[:, None]
    V_ptrs = V + (off_z * stride_vz + (off_h // num_kv_groups) * stride_vh) + offs_n[:, None] * stride_vn + offs_k[None, :]
    O_block_ptr = Out + (off_z * stride_oz + off_h * stride_oh) + offs_m[:, None] * stride_on + offs_k[None, :]
    
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    
    q = tl.load(Q_ptrs, mask=offs_m[:, None] < qo_len)
    acc, l_i, m_i = _attn_fwd_inner(acc, l_i, m_i, q, kv_len, 
                                   K_ptrs, V_ptrs, stride_kn, stride_vn,
                                   start_m, BLOCK_M, HEAD_DIM, BLOCK_N, 
                                   4 - STAGE, offs_m, offs_n)
    
    acc = acc / l_i[:, None]
    tl.store(O_block_ptr, acc.to(Out.type.element_ty), mask=(offs_m[:, None] < qo_len))

    if RETURN_LSE:
        lse_ptrs = Lse + (off_z * qo_len * H + off_h * qo_len) + offs_m
        l_i = tl.log2(l_i) + m_i
        tl.store(lse_ptrs, l_i, mask=(offs_m < qo_len))

def forward(q, k, v, 
            output_dtype=torch.float16, return_lse=False, tensor_layout=None):
    BLOCK_M, BLOCK_N = 128, 64
    stage = 1
    
    o = torch.empty(q.shape, dtype=output_dtype, device=q.device)
    if tensor_layout == "bhsd":
        batch_size, num_heads_q, seq_len_q,  head_dim = q.shape
        _, num_heads_kv, seq_len_kv, _ = k.shape
        stride_batch_q, stride_heads_q, stride_seq_q, stride_dim_q = q.stride()
        stride_batch_k, stride_heads_k, stride_seq_k, stride_dim_k = k.stride()
        stride_batch_v, stride_heads_v, stride_seq_v, stride_dim_v = v.stride()
        stride_batch_o, stride_heads_o, stride_seq_o, stride_dim_o = o.stride()
    elif tensor_layout == "bshd":
        batch_size, seq_len_q, num_heads_q, head_dim = q.shape
        _, seq_len_kv, num_heads_kv, _ = k.shape
        stride_batch_q, stride_seq_q, stride_heads_q, stride_dim_q = q.stride()
        stride_batch_k, stride_seq_k, stride_heads_k, stride_dim_k = k.stride()
        stride_batch_v, stride_seq_v, stride_heads_v, stride_dim_v = v.stride()
        stride_batch_o, stride_seq_o, stride_heads_o, stride_dim_o = o.stride()
    else:
        raise ValueError(f"tensor_layout {tensor_layout} not supported")

    if return_lse:
        lse = torch.empty([batch_size, num_heads_q, seq_len_q], dtype=torch.float32, device=q.device)
    else:
        lse = torch.empty([0], dtype=torch.float32, device='cpu')

    grid = (triton.cdiv(seq_len_q, BLOCK_M), num_heads_q, batch_size)
    _attn_fwd[grid](
        q, k, v, o, lse,
        stride_batch_q, stride_heads_q, stride_seq_q, 
        stride_batch_k, stride_heads_k, stride_seq_k, 
        stride_batch_v, stride_heads_v, stride_seq_v, 
        stride_batch_o, stride_heads_o, stride_seq_o, 
        seq_len_q, seq_len_kv,
        H=num_heads_q, 
        num_kv_groups=max(num_heads_q // num_heads_kv, 1),
        HEAD_DIM=head_dim,  
        BLOCK_M=BLOCK_M, 
        BLOCK_N=BLOCK_N,
        STAGE=stage, 
        RETURN_LSE=return_lse,
        num_warps=4 if head_dim == 64 else 8,
        num_stages=3 if head_dim == 64 else 4
    )
    return o, lse