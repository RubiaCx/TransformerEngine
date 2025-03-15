# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import torch
import triton
import triton.language as tl

@triton.jit
def quant_per_block_kernel(Input, Output, Scale, 
                          stride_iz, stride_ih, stride_in,
                          stride_oz, stride_oh, stride_on,
                          stride_sz, stride_sh,
                          sm_scale: tl.constexpr,
                          L: tl.constexpr,
                          C: tl.constexpr, 
                          BLK: tl.constexpr,
                          QUANT_TYPE: tl.constexpr): 
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
    
    tl.store(output_ptrs, x_quant, mask=offs_n[:, None] < L)
    tl.store(scale_ptrs, scale)

def per_block_quant(q, k, v, 
                   quant_type="int8",  
                   BLKQ=128, BLKK=64, BLKV=64, 
                   sm_scale=None, tensor_layout="bhsd"):
    dtype_map = {
        "int8": torch.int8,
        "e4m3": torch.float8_e4m3fn,
        "e5m2": torch.float8_e5m2
    }
    assert quant_type in dtype_map, f"Unsupported quant type: {quant_type}"
    
    q_quant = torch.empty(q.shape, dtype=dtype_map[quant_type], device=q.device)
    k_quant = torch.empty(k.shape, dtype=dtype_map[quant_type], device=k.device)
    v_quant = torch.empty(v.shape, dtype=dtype_map[quant_type], device=v.device)

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

    quant_type_code = {"int8":0, "e4m3":1, "e5m2":2}[quant_type]

    def launch_kernel(tensor_in, tensor_out, scale_out, blk_size, strides, is_query=False):
        grid = ((tensor_in.size(2) + blk_size - 1) // blk_size, 
                tensor_in.size(1), 
                tensor_in.size(0))
        quant_per_block_kernel[grid](
            tensor_in, tensor_out, scale_out,
            strides[0], strides[1], strides[2],
            strides[0], strides[1], strides[2],
            scale_out.stride(0), scale_out.stride(1),
            sm_scale=(sm_scale * 1.44269504) if is_query else 1.0,
            L=tensor_in.size(2),
            C=head_dim,
            BLK=blk_size,
            QUANT_TYPE=quant_type_code
        )

    launch_kernel(q, q_quant, q_scale, BLKQ, 
                 (stride_batch_q, stride_heads_q, stride_seq_q), is_query=True)
    launch_kernel(k, k_quant, k_scale, BLKK,
                 (stride_batch_k, stride_heads_k, stride_seq_k))
    launch_kernel(v, v_quant, v_scale, BLKV,
                 (stride_batch_v, stride_heads_v, stride_seq_v))

    return q_quant, q_scale, k_quant, k_scale, v_quant, v_scale 