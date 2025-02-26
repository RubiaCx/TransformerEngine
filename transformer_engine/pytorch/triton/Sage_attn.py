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

import torch
import triton
import triton.language as tl

@triton.jit
def _attn_fwd_inner(acc, l_i, m_i, q, q_scale, kv_len,
                    K_ptrs, K_scale_ptr, V_ptrs, stride_kn, stride_vn, 
                    start_m,  
                    BLOCK_M: tl.constexpr, HEAD_DIM: tl.constexpr, BLOCK_N: tl.constexpr,  
                    STAGE: tl.constexpr, offs_m: tl.constexpr, offs_n: tl.constexpr,  
                    ):
    lo, hi = 0, kv_len
    for start_n in range(lo, hi, BLOCK_N):
        # Attention(Q,K,V) = softmax(QK^T/sqrt(d))V
        start_n = tl.multiple_of(start_n, BLOCK_N)
        k_mask = offs_n[None, :] < (kv_len - start_n)   
        k = tl.load(K_ptrs, mask = k_mask)
        k_scale = tl.load(K_scale_ptr)
        # QK^T，即公式中的 S_i^j
        qk = tl.dot(q, k).to(tl.float32) * q_scale * k_scale 
        # rows_sum(exp(S_i^j - m_i^j)
        m_ij = tl.maximum(m_i, tl.max(qk, 1)) # 减去当前块的最大值，归一化
        qk = qk - m_ij[:, None]
        p = tl.math.exp2(qk) 
        l_ij = tl.sum(p, 1) 
        # exp(m_i^j-1 - m_i^j) 对于上一块累积的 l_i，按照当前块的尺度修正：
        alpha = tl.math.exp2(m_i - m_ij)
        #! l_i^j = exp(m_i^j-1 - m_i^j) + rows_sum(exp(S_i^j - m_i^j)
        l_i = l_i * alpha + l_ij

        acc = acc * alpha[:, None]
        v = tl.load(V_ptrs, mask = offs_n[:, None] < (kv_len - start_n))
        # p = p.to(tl.float16)
        # acc += tl.dot(p, v, out_dtype=tl.float16)  
        p = p.to(v.dtype)
        acc += tl.dot(p, v)
         
        m_i = m_ij
        K_ptrs += BLOCK_N * stride_kn
        K_scale_ptr += 1
        V_ptrs += BLOCK_N * stride_vn
    return acc, l_i, m_i

@triton.jit
def _attn_fwd(Q, K, V, Q_scale, K_scale, Out, Lse, 
              stride_qz, stride_qh, stride_qn,
              stride_kz, stride_kh, stride_kn,  
              stride_vz, stride_vh, stride_vn,  
              stride_oz, stride_oh, stride_on,  
              qo_len, kv_len, H:tl.constexpr, num_kv_groups:tl.constexpr,
              HEAD_DIM: tl.constexpr,  
              BLOCK_M: tl.constexpr,  
              BLOCK_N: tl.constexpr,  
              STAGE: tl.constexpr,
              RETURN_LSE: tl.constexpr,
              ):
    start_m = tl.program_id(0)

    off_z = tl.program_id(2).to(tl.int64)
    off_h = tl.program_id(1).to(tl.int64)

    q_scale_offset = (off_z * H + off_h) * tl.cdiv(qo_len, BLOCK_M)
    k_scale_offset = (off_z * (H // num_kv_groups) + off_h // num_kv_groups) * tl.cdiv(kv_len, BLOCK_N)  
    
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, HEAD_DIM)
    Q_ptrs = Q + (off_z * stride_qz + off_h * stride_qh) + offs_m[:, None] * stride_qn + offs_k[None, :]
    Q_scale_ptr = Q_scale + q_scale_offset + start_m
    K_ptrs = K + (off_z * stride_kz + (off_h // num_kv_groups) * stride_kh) + offs_n[None, :] * stride_kn + offs_k[:, None] 
    K_scale_ptr = K_scale + k_scale_offset
    V_ptrs = V + (off_z * stride_vz + (off_h // num_kv_groups) * stride_vh) + offs_n[:, None] * stride_vn + offs_k[None, :]
    O_block_ptr = Out + (off_z * stride_oz + off_h * stride_oh) + offs_m[:, None] * stride_on + offs_k[None, :]
    
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    
    q = tl.load(Q_ptrs, mask = offs_m[:, None] < qo_len)
    q_scale = tl.load(Q_scale_ptr)
    acc, l_i, m_i = _attn_fwd_inner(acc, l_i, m_i, q, q_scale, kv_len, K_ptrs, K_scale_ptr, V_ptrs, stride_kn, stride_vn,
                                    start_m,  
                                    BLOCK_M, HEAD_DIM, BLOCK_N,  
                                    4 - STAGE, offs_m, offs_n 
                                    )
    acc = acc / l_i[:, None]
    tl.store(O_block_ptr, acc.to(Out.type.element_ty), mask = (offs_m[:, None] < qo_len))

    if RETURN_LSE:
        lse_ptrs = Lse + (off_z * qo_len * H + off_h * qo_len) + offs_m  #!
        l_i = tl.log2(l_i) + m_i
        tl.store(lse_ptrs, l_i, mask = (offs_m < qo_len))

def forward(q, k, v, 
            q_scale, k_scale, v_scale, 
            output_dtype=torch.float16, return_lse=False, tensor_layout=None):
    BLOCK_M = 128
    BLOCK_N = 64
    stage = 1
    print("tensor_layout:", tensor_layout)
    #! bhsd == HND；bshd == NHD 
    o = torch.empty(q.shape, dtype=output_dtype, device=q.device)
    if tensor_layout == "bhsd":
        print("bhsd")
        batch_size, num_heads_q, seq_len_q,  head_dim = q.shape
        _, num_heads_kv, seq_len_kv, _ = k.shape
        stride_batch_q, stride_heads_q, stride_seq_q, stride_dim_q = q.stride()
        stride_batch_k, stride_heads_k, stride_seq_k, stride_dim_k = k.stride()
        stride_batch_v, stride_heads_v, stride_seq_v, stride_dim_v = v.stride()
        stride_batch_o, stride_heads_o, stride_seq_o, stride_dim_o = o.stride()
    elif tensor_layout == "bshd":
        print("bshd")
        batch_size, seq_len_q, num_heads_q, head_dim = q.shape
        _, seq_len_kv, num_heads_kv, _ = k.shape
        stride_batch_q, stride_seq_q, stride_heads_q, stride_dim_q = q.stride()
        stride_batch_k, stride_seq_k, stride_heads_k, stride_dim_k = k.stride()
        stride_batch_v, stride_seq_v, stride_heads_v, stride_dim_v = v.stride()
        stride_batch_o, stride_seq_o, stride_heads_o, stride_dim_o = o.stride()
    else:
        raise ValueError(f"tensor_layout {tensor_layout} not supported")

    num_kv_groups = max(num_heads_q // num_heads_kv, 1)
    
    assert num_heads_kv > 0, "num_heads_kv must be positive"
    assert num_kv_groups > 0, "num_kv_groups must be positive"

    if return_lse:
        lse = torch.empty([batch_size, num_heads_q, seq_len_q], dtype=torch.float32, device=q.device)
    else:
        lse = torch.empty([0], dtype=torch.float32, device='cpu')

    grid = (triton.cdiv(seq_len_q, BLOCK_M), num_heads_q, batch_size)
    _attn_fwd[grid](
        q, k, v, q_scale, k_scale, o, lse,
        stride_batch_q, stride_heads_q, stride_seq_q, 
        stride_batch_k, stride_heads_k, stride_seq_k, 
        stride_batch_v, stride_heads_v, stride_seq_v, 
        stride_batch_o, stride_heads_o, stride_seq_o, 
        seq_len_q, seq_len_kv,
        H=num_heads_q, num_kv_groups=num_kv_groups,
        HEAD_DIM=head_dim,  
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, 
        STAGE=stage, RETURN_LSE=return_lse,
        num_warps=4 if head_dim == 64 else 8,
        num_stages=3 if head_dim == 64 else 4
    )

    return o, lse