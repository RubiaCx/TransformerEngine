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
def quant_per_block_varlen_kernel(Input, Output, Scale,
                                 cu_seqlens_input, cu_seqlens_scale,
                                 stride_ih, stride_in,
                                 stride_oh, stride_on,
                                 sm_scale: tl.constexpr,
                                 H: tl.constexpr,
                                 C: tl.constexpr, 
                                 BLK: tl.constexpr,
                                 QUANT_TYPE: tl.constexpr):  # 新增量化类型参数
    off_blk = tl.program_id(0)
    off_h = tl.program_id(1)
    off_b = tl.program_id(2)

    cu_seqlens_input_start = tl.load(cu_seqlens_input + off_b)
    cu_seqlens_input_end = tl.load(cu_seqlens_input + off_b + 1)
    L = cu_seqlens_input_end - cu_seqlens_input_start

    if (off_blk * BLK) >= L:
        return
    
    cu_seqlens_scale_start = tl.load(cu_seqlens_scale + off_b)

    offs_n = off_blk * BLK + tl.arange(0, BLK)
    offs_k = tl.arange(0, C)

    input_ptrs = Input + cu_seqlens_input_start * stride_in + off_h * stride_ih + offs_n[:, None] * stride_in + offs_k[None, :]
    output_ptrs = Output + cu_seqlens_input_start * stride_on + off_h * stride_oh + offs_n[:, None] * stride_on + offs_k[None, :]
    scale_ptrs = Scale + cu_seqlens_scale_start * H + off_h + off_blk * H

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

def per_block_varlen_quant(q, k, v, 
                          cu_seqlens_q, cu_seqlens_kv, 
                          max_seqlen_q, max_seqlen_kv, 
                          quant_type="int8",  
                          BLKQ=128, BLKK=64, BLKV=64, 
                          sm_scale=None):
    dtype_map = {
        "int8": torch.int8,
        "e4m3": torch.float8_e4m3fn,
        "e5m2": torch.float8_e5m2
    }
    assert quant_type in dtype_map, f"Unsupported quant type: {quant_type}"
    
    q_quant = torch.empty(q.shape, dtype=dtype_map[quant_type], device=q.device)
    k_quant = torch.empty(k.shape, dtype=dtype_map[quant_type], device=k.device)
    v_quant = torch.empty(v.shape, dtype=dtype_map[quant_type], device=v.device)
    #! THD
    batch_size = cu_seqlens_q.shape[0] - 1
    num_heads_q = q.shape[1]
    num_heads_kv = k.shape[1]
    head_dim = q.shape[-1]
    
    q_batch_len = cu_seqlens_q[1:] - cu_seqlens_q[:-1]
    kv_batch_len = cu_seqlens_kv[1:] - cu_seqlens_kv[:-1]

    q_scale_len = (q_batch_len + BLKQ - 1) // BLKQ
    k_scale_len = (kv_batch_len + BLKK - 1) // BLKK
    v_scale_len = (kv_batch_len + BLKV - 1) // BLKV

    cu_seqlens_q_scale = torch.nn.functional.pad(torch.cumsum(q_scale_len, dim=0), (1, 0), value=0)
    cu_seqlens_k_scale = torch.nn.functional.pad(torch.cumsum(k_scale_len, dim=0), (1, 0), value=0)
    cu_seqlens_v_scale = torch.nn.functional.pad(torch.cumsum(v_scale_len, dim=0), (1, 0), value=0)

    q_scale = torch.empty((cu_seqlens_q_scale[-1], num_heads_q), device=q.device, dtype=torch.float32)
    k_scale = torch.empty((cu_seqlens_k_scale[-1], num_heads_kv), device=k.device, dtype=torch.float32)
    v_scale = torch.empty((cu_seqlens_v_scale[-1], num_heads_kv), device=v.device, dtype=torch.float32)

    if sm_scale is None:
        sm_scale = head_dim**-0.5

    quant_type_code = {"int8":0, "e4m3":1, "e5m2":2}[quant_type]

    def launch_varlen_kernel(tensor_in, tensor_out, scale_out, seq_lens, cu_scale, max_len, blk_size, is_query=False):
        grid = ((max_len + blk_size - 1) // blk_size, 
                tensor_in.shape[1],  # num_heads
                batch_size)
        quant_per_block_varlen_kernel[grid](
            tensor_in, tensor_out, scale_out,
            seq_lens, cu_scale,
            tensor_in.stride(1), tensor_in.stride(0),
            tensor_out.stride(1), tensor_out.stride(0),
            sm_scale=(sm_scale * 1.44269504) if is_query else 1.0,
            H=tensor_in.shape[1],  # num_heads
            C=head_dim,
            BLK=blk_size,
            QUANT_TYPE=quant_type_code
        )

    launch_varlen_kernel(q, q_quant, q_scale, cu_seqlens_q, cu_seqlens_q_scale, max_seqlen_q, BLKQ, is_query=True)
    launch_varlen_kernel(k, k_quant, k_scale, cu_seqlens_kv, cu_seqlens_k_scale, max_seqlen_kv, BLKK)
    launch_varlen_kernel(v, v_quant, v_scale, cu_seqlens_kv, cu_seqlens_v_scale, max_seqlen_kv, BLKV)

    return q_quant, q_scale, k_quant, k_scale, v_quant, v_scale, cu_seqlens_q_scale, cu_seqlens_k_scale, cu_seqlens_v_scale 