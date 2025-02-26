# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import torch
import triton
import triton.language as tl

@triton.jit
def quant_per_block_e4m3_kernel(Input, Output, Scale, 
                                stride_iz, stride_ih, stride_in,
                                stride_oz, stride_oh, stride_on,
                                stride_sz, stride_sh,
                                sm_scale: tl.constexpr,
                                L: tl.constexpr,
                                C: tl.constexpr, 
                                BLK: tl.constexpr):
    off_blk = tl.program_id(0)
    off_h = tl.program_id(1)
    off_b = tl.program_id(2)

    offs_n = off_blk * BLK + tl.arange(0, BLK)
    offs_k = tl.arange(0, C)

    input_ptrs = Input + off_b * stride_iz + off_h * stride_ih + offs_n[:, None] * stride_in + offs_k[None, :]
    output_ptrs = Output + off_b * stride_oz + off_h * stride_oh + offs_n[:, None] * stride_on + offs_k[None, :]
    scale_ptrs = Scale + off_b * stride_sz + off_h * stride_sh + off_blk

    x = tl.load(input_ptrs, mask=offs_n[:, None] < L)
    x = x.to(tl.float32)
    x *= sm_scale
    scale = tl.max(tl.abs(x)) / 448. + 1e-8
    x_e4m3 = x / scale
    x_e4m3 = x_e4m3.to(tl.float8e4nv)
    tl.store(output_ptrs, x_e4m3, mask=offs_n[:, None] < L)
    tl.store(scale_ptrs, scale)

def per_block_e4m3(q, k, v, BLKQ=128, BLKK=64, BLKV=64, sm_scale=None, tensor_layout="bhsd"):
    q_e4m3 = torch.empty(q.shape, dtype=torch.float8_e4m3fn, device=q.device)
    k_e4m3 = torch.empty(k.shape, dtype=torch.float8_e4m3fn, device=k.device)
    v_e4m3 = torch.empty(v.shape, dtype=torch.float8_e4m3fn, device=v.device)

    if tensor_layout == "bhsd":
        batch_size, num_heads_q, seq_len_q, head_dim = q.shape
        _, num_heads_kv, seq_len_kv, _ = k.shape
        stride_batch_q, stride_heads_q, stride_seq_q, stride_dim_q = q.stride()
        stride_batch_k, stride_heads_k, stride_seq_k, stride_dim_k = k.stride()
        stride_batch_v, stride_heads_v, stride_seq_v, stride_dim_v = v.stride()
    elif tensor_layout == "bshd":
        batch_size, seq_len_q, num_heads_q, head_dim = q.shape
        _, seq_len_kv, num_heads_kv, _ = k.shape
        stride_batch_q, stride_seq_q, stride_heads_q, stride_dim_q = q.stride()
        stride_batch_k, stride_seq_k, stride_heads_k, stride_dim_k = k.stride()
        stride_batch_v, stride_seq_v, stride_heads_v, stride_dim_v = v.stride()
    else:
        raise ValueError(f"tensor_layout {tensor_layout} not supported")

    q_scale = torch.empty((batch_size, num_heads_q, (seq_len_q + BLKQ - 1) // BLKQ), 
                           device=q.device, dtype=torch.float32)
    k_scale = torch.empty((batch_size, num_heads_kv, (seq_len_kv + BLKK - 1) // BLKK), 
                           device=k.device, dtype=torch.float32)
    v_scale = torch.empty((batch_size, num_heads_kv, (seq_len_kv + BLKV - 1) // BLKV), 
                           device=v.device, dtype=torch.float32)

    if sm_scale is None:
        sm_scale = head_dim**-0.5

    grid = ((seq_len_q + BLKQ - 1) // BLKQ, num_heads_q, batch_size)
    quant_per_block_e4m3_kernel[grid]( # q_e4m3 == q
        q, q_e4m3, q_scale, 
        stride_batch_q, stride_heads_q, stride_seq_q,
        stride_batch_q, stride_heads_q, stride_seq_q,
        q_scale.stride(0), q_scale.stride(1),
        sm_scale=(sm_scale * 1.44269504),
        L=seq_len_q, C=head_dim, BLK=BLKQ
    )

    grid = ((seq_len_kv + BLKK - 1) // BLKK, num_heads_kv, batch_size)
    quant_per_block_e4m3_kernel[grid](
        k, k_e4m3, k_scale, 
        stride_batch_k, stride_heads_k, stride_seq_k,
        stride_batch_k, stride_heads_k, stride_seq_k,
        k_scale.stride(0), k_scale.stride(1),
        sm_scale=1.0,
        L=seq_len_kv, C=head_dim, BLK=BLKK
    )

    grid = ((seq_len_kv + BLKV - 1) // BLKV, num_heads_kv, batch_size)
    quant_per_block_e4m3_kernel[grid](
        v, v_e4m3, v_scale, 
        stride_batch_v, stride_heads_v, stride_seq_v,
        stride_batch_v, stride_heads_v, stride_seq_v,
        v_scale.stride(0), v_scale.stride(1),
        sm_scale=1.0,
        L=seq_len_kv, C=head_dim, BLK=BLKV
    )
    return q_e4m3, q_scale, k_e4m3, k_scale, v_e4m3, v_scale