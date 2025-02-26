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
def quant_per_block_e4m3_kernel(Input, Output, Scale, scale_stride,
                                L: tl.constexpr, C: tl.constexpr,
                                sm_scale: tl.constexpr,
                                BLK: tl.constexpr):
    off_blk = tl.program_id(0)
    off_b = tl.program_id(1)

    input_offset = off_b * L * C 
    offs_m = off_blk*BLK + tl.arange(0, BLK)
    offs_k = tl.arange(0, C)

    input_ptrs = Input + input_offset + offs_m[:, None] * C + offs_k[None, :]
    output_ptrs = Output + input_offset + offs_m[:, None] * C + offs_k[None, :]
    scale_ptrs = Scale + off_b * scale_stride + off_blk  

    x = tl.load(input_ptrs, mask=offs_m[:, None] < L)
    x *= sm_scale
    scale = tl.max(tl.abs(x)) / 448. + 1e-8
    x_e4m3 = x / scale
    x_e4m3 = x_e4m3.to(tl.float8e4nv)
    tl.store(output_ptrs, x_e4m3, mask=offs_m[:, None] < L)
    tl.store(scale_ptrs, scale)

def per_block_e4m3(q, k, v, BLKQ=64, BLKK=64, BLKV=256, sm_scale=None, tensor_layout="bhsd"):
    q_e4m3 = torch.empty(q.shape, dtype=torch.float8_e4m3fn, device=q.device)
    k_e4m3 = torch.empty(k.shape, dtype=torch.float8_e4m3fn, device=k.device)
    v_e4m3 = torch.empty(v.shape, dtype=torch.float8_e4m3fn, device=v.device)

    q_scale = torch.empty((q.shape[-4], q.shape[-3], (q.shape[-2] + BLKQ - 1) // BLKQ, 1), device=q.device, dtype=torch.float32)
    k_scale = torch.empty((k.shape[-4], k.shape[-3], (k.shape[-2] + BLKK - 1) // BLKK, 1), device=q.device, dtype=torch.float32)
    v_scale = torch.empty((v.shape[-4], v.shape[-3], (v.shape[-2] + BLKV - 1) // BLKV, 1), device=q.device, dtype=torch.float32)

    # 合并batch和heads维度，转换为3D张量 [batch, heads, seq, dim] -> [batch*heads, seq, dim]
    if tensor_layout == "bhsd":
        q = q.view(-1, q.size(-2), q.size(-1))
        k = k.view(-1, k.size(-2), k.size(-1))
        v = v.view(-1, v.size(-2), v.size(-1))
    elif tensor_layout == "bshd":
        q = q.permute(0, 2, 1, 3).contiguous().view(-1, q.size(1), q.size(3))
        k = k.permute(0, 2, 1, 3).contiguous().view(-1, k.size(1), k.size(3))
        v = v.permute(0, 2, 1, 3).contiguous().view(-1, v.size(1), v.size(3))
    else:
        raise ValueError(f"tensor_layout {tensor_layout} not supported")

    B, Lq, C = q.shape
    Lk = k.size(1)

    grid_q = ((Lq+BLKQ-1)//BLKQ, B, )
    quant_per_block_e4m3_kernel[grid_q](
        q, q_e4m3, q_scale, q_scale.stride(1),
        L=Lq, C=C,
        sm_scale=(C**-0.5 * 1.44269504),
        BLK=BLKQ
    )

    grid_k = ((Lk+BLKK-1)//BLKK, B, )
    quant_per_block_e4m3_kernel[grid_k](
        k, k_e4m3, k_scale, k_scale.stride(1),
        L=Lk, C=C,
        sm_scale=1.0,
        BLK=BLKK
    )
    grid_v = ((Lk+BLKV-1)//BLKV, B, )
    quant_per_block_e4m3_kernel[grid_v](
        v, v_e4m3, v_scale, v_scale.stride(1),
        L=Lk, C=C,
        sm_scale=1.0,
        BLK=BLKV
    )

    return q_e4m3, q_scale, k_e4m3, k_scale, v_e4m3, v_scale