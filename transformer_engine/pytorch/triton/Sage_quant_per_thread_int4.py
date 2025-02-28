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
# 为什么不需要x *= sm_scale
#TODO(cx)
# dot2 fp8
# 不同精度lse

@triton.jit
def quant_query_per_thread_int4_kernel(Input, Output, Scale, 
                                       stride_iz, stride_ih, stride_in,
                                       stride_oz, stride_oh, stride_on,
                                       stride_sz, stride_sh,
                                       L: tl.constexpr,
                                       C: tl.constexpr, 
                                       BLK: tl.constexpr): 
    off_blk = tl.program_id(0) // 8
    off_tld = tl.program_id(0) % 8
    off_h = tl.program_id(1)
    off_b = tl.program_id(2)

    offs_n = off_blk * BLK + tl.arange(0, BLK // 8) * 8 + off_tld
    offs_k = tl.arange(0, C)

    input_ptrs = Input + off_b * stride_iz + off_h * stride_ih + offs_n[:, None] * stride_in + offs_k[None, :]
    output_ptrs = Output + off_b * stride_oz + off_h * stride_oh + offs_n[:, None] * stride_on + offs_k[None, :]
    scale_ptrs = Scale + off_b * stride_sz + off_h * stride_sh + off_blk * 8 + off_tld

    x = tl.load(input_ptrs, mask=offs_n[:, None] < L)
    x = x.to(tl.float32)
    scale = tl.max(tl.abs(x)) / 7. + 1e-8
    x_int8 = x / scale
    x_int8 += 0.5 * tl.where(x_int8 >= 0, 1, -1)
    x_int8 = x_int8.to(tl.int8)
    tl.store(output_ptrs, x_int8, mask=offs_n[:, None] < L)
    tl.store(scale_ptrs, scale)

@triton.jit
def quant_key_per_thread_int4_kernel(Input, Output, Scale, 
                                     stride_iz, stride_ih, stride_in,
                                     stride_oz, stride_oh, stride_on,
                                     stride_sz, stride_sh,
                                     L: tl.constexpr,
                                     C: tl.constexpr, 
                                     BLK: tl.constexpr):   
    off_blk = tl.program_id(0) // 4
    off_tld = tl.program_id(0) % 4
    off_h = tl.program_id(1)
    off_b = tl.program_id(2)

    offs_n = off_blk * BLK + tl.cat(tl.arange(0, BLK // 8) * 8, tl.arange(0, BLK // 8) * 8 + 1, True) + off_tld * 2
    offs_k = tl.arange(0, C)

    input_ptrs = Input + off_b * stride_iz + off_h * stride_ih + offs_n[:, None] * stride_in + offs_k[None, :]
    output_ptrs = Output + off_b * stride_oz + off_h * stride_oh + offs_n[:, None] * stride_on + offs_k[None, :]
    scale_ptrs = Scale + off_b * stride_sz + off_h * stride_sh + off_blk * 4 + off_tld

    x = tl.load(input_ptrs, mask=offs_n[:, None] < L)
    x = x.to(tl.float32)
    scale = tl.max(tl.abs(x)) / 7. + 1e-8
    x_int8 = x / scale
    x_int8 += 0.5 * tl.where(x_int8 >= 0, 1, -1)
    x_int8 = x_int8.to(tl.int8)
    tl.store(output_ptrs, x_int8, mask=offs_n[:, None] < L)
    tl.store(scale_ptrs, scale)

def per_thread_int4(q, k, km=None, BLKQ=128, WARPQ=32, BLKK=64, WARPK=64, sm_scale=None, tensor_layout="bhsd"):
    q_int8 = torch.empty(q.shape, dtype=torch.int8, device=q.device)
    k_int8 = torch.empty(k.shape, dtype=torch.int8, device=k.device)

    if tensor_layout == "bhsd":
        batch_size, num_heads_q, seq_len_q, head_dim = q.shape
        _, num_heads_kv, seq_len_kv, _ = k.shape
        stride_batch_q, stride_heads_q, stride_seq_q, stride_dim_q = q.stride()
        stride_batch_k, stride_heads_k, stride_seq_k, stride_dim_k = k.stride()
    else:
        raise ValueError(f"Unknown tensor layout: {tensor_layout}")

    q_scale = torch.empty((batch_size, num_heads_q, (seq_len_q + BLKQ - 1) // BLKQ * (BLKQ // WARPQ) * 8), device=q.device, dtype=torch.float32)
    k_scale = torch.empty((batch_size, num_heads_kv, (seq_len_kv + BLKK - 1) // BLKK * (BLKK // WARPK) * 4), device=q.device, dtype=torch.float32)

    if sm_scale is None:
        sm_scale = head_dim**-0.5

    grid = ((seq_len_q + BLKQ - 1) // BLKQ * (BLKQ // WARPQ) * 8, num_heads_q, batch_size)
    quant_query_per_thread_int4_kernel[grid](
        q, q_int8, q_scale, 
        stride_batch_q, stride_heads_q, stride_seq_q,
        stride_batch_q, stride_heads_q, stride_seq_q,
        q_scale.stride(0), q_scale.stride(1),
        # sm_scale=(sm_scale * 1.44269504),
        L=seq_len_q, C=head_dim, 
        BLK=WARPQ #!
    )

    grid = ((seq_len_kv + BLKK - 1) // BLKK * (BLKK // WARPK) * 4, num_heads_kv, batch_size)
    quant_key_per_thread_int4_kernel[grid](
        k, k_int8, k_scale, 
        stride_batch_k, stride_heads_k, stride_seq_k,
        stride_batch_k, stride_heads_k, stride_seq_k,
        k_scale.stride(0), k_scale.stride(1),
        # sm_scale=1.0,
        L=seq_len_kv, C=head_dim, 
        BLK=WARPK #!
    )

    return q_int8, q_scale, k_int8, k_scale

