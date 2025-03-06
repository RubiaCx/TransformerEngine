# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import functools
import logging
import math
import os
from importlib.metadata import version
from typing import Any, Dict, List, Tuple, Union, Optional
from contextlib import contextmanager
from einops import rearrange, repeat

import pytest
import torch

from transformer_engine.pytorch.attention import (
    SageAttention,
    FlashAttention,
    UnfusedDotProductAttention,
)

torch.backends.cuda.enable_flash_sdp(False)   
torch.backends.cuda.enable_math_sdp(True)  
from torch.nn.functional import scaled_dot_product_attention as sdpa

import flash_attn
from flash_attn.flash_attn_interface import _flash_attn_forward
from flash_attn.flash_attn_interface import flash_attn_varlen_func, flash_attn_func

torch.manual_seed(2025)
torch.set_printoptions(precision=4, threshold=None, edgeitems=None, linewidth=None, profile=None, sci_mode=False)

class ModelConfig:
    def __init__(
        self,
        batch_size: int,
        num_heads: int,
        num_gqa_groups: int,
        head_dim_qk: int,
        max_seqlen_q: int,
        max_seqlen_kv: int,
        dropout_p: float,
        attn_mask_type: str,
        attn_bias_type: str,
        head_dim_v: int = None,
        alibi_type: str = "none",
        num_layers: int = 1,
        bias_shape: str = "1hss",
        window_size: Tuple[int, int] = (-1, -1),
    ):
        self.batch_size = batch_size
        self.num_heads = num_heads
        self.num_gqa_groups = num_gqa_groups
        self.head_dim_qk = head_dim_qk
        self.head_dim_v = head_dim_qk if head_dim_v is None else head_dim_v
        self.hidden_size = num_heads * head_dim_qk
        self.hidden_size_kv = num_gqa_groups * self.head_dim_v
        self.max_seqlen_q = max_seqlen_q
        self.max_seqlen_kv = max_seqlen_kv
        self.dropout_p = dropout_p
        self.attn_mask_type = attn_mask_type
        self.attn_bias_type = attn_bias_type
        self.alibi_type = alibi_type
        self.attn_type = "self" if (max_seqlen_q == max_seqlen_kv) else "cross"
        self.num_layers = num_layers
        self.bias_shape = bias_shape
        self.window_size = window_size

model_configs_sage = {
    # batch_size, num_heads, num_gqa_groups, head_dim_qk, max_seqlen_q, max_seqlen_kv, dropout_p, attn_mask_type, attn_bias_type, head_dim_v
    # "sage_1": ModelConfig(2, 16, 16, 64, 512, 512, 0.0, "causal", "no_bias", head_dim_v=64),
    # "sage_2": ModelConfig(4, 24, 24, 128, 2048, 2048, 0.0, "no_mask", "post_scale_bias", head_dim_v=128),
    # "sage_3": ModelConfig(2, 32, 4, 128, 8192, 8192, 0.0, "padding_causal", "no_bias", head_dim_v=128),
    # "sage_4": ModelConfig(1, 8, 8, 32, 1024, 1024, 0.0, "causal", "no_bias", head_dim_v=32),
    # "sage_5": ModelConfig(8, 12, 12, 64, 4096, 4096, 0.0, "no_mask", "post_scale_bias", head_dim_v=64),
    "sage_6": ModelConfig(batch_size=1, num_heads=4, num_gqa_groups=4, head_dim_qk=16, head_dim_v=16, max_seqlen_q=32, max_seqlen_kv=32, dropout_p=0.0, attn_mask_type="no_mask", attn_bias_type="no_bias"), #TODO: 缩小参数
}

qkv_layouts = ["sbh3d"]
# qkv_layouts = ["sbh3d", "sbhd_sb2hd", "sbhd_sbh2d", "sbhd_sbhd_sbhd"]
#! BSHD
def ref_attention(q, k, v, scale):
    attn = torch.matmul(q, k.transpose(-2, -1)) * scale
    attn = torch.softmax(attn, dim=-1)
    output = torch.matmul(attn, v) 
    return output

#! BSHD -> BHSD -> BSHD 
def ref_attention2(q, k, v, scale, num_heads):
    # 合并batch和heads维度
    q = rearrange(q, 'b s h d -> (b h) s d')
    #  k = rearrange(k, 'b s h d -> (b h) d s')  
    k = rearrange(k, 'b s h d -> (b h) s d')
    v = rearrange(v, 'b s h d -> (b h) s d')
    
    # 计算注意力分数 (Q @ K^T)
    # attn = torch.bmm(q, k) * scale  # [batch*heads, seqlen, seqlen]
    attn = torch.bmm(q, k.transpose(-2, -1)) * scale  # [batch*heads, seqlen, seqlen]
    attn = torch.softmax(attn, dim=-1)
    output = torch.bmm(attn, v)  # [batch*heads, seqlen, head_dim]
    
    output = rearrange(output, '(b h) s d -> b s h d', h=num_heads)
    return output

def print_diff(name, out1, out2):
    diff = (out1 - out2).abs()
    print(f"\n{name} absolute differences:")
    print(f"max={diff.max().item()}, mean={diff.mean().item()}, std={diff.std().item()}")
    # 计算余弦相似度
    def cosine_similarity(x, y):
        x_flat = x.reshape(-1)
        y_flat = y.reshape(-1)
        cos_sim = torch.nn.functional.cosine_similarity(x_flat, y_flat, dim=0)
        return cos_sim.item()
    print(f"cosine_similarity: {cosine_similarity(out1, out2)}")

@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float8_e4m3fn, torch.float8_e5m2])
@pytest.mark.parametrize("model_configs", [model_configs_sage])
@pytest.mark.parametrize("model", model_configs_sage.keys())
@pytest.mark.parametrize("qkv_layout", qkv_layouts)
@pytest.mark.parametrize("value_range", [1, 10, 100])
def test_sage_attention_1(dtype, model_configs, model, qkv_layout, value_range):
    config = model_configs[model]
    tols = dict(atol=1e-3, rtol=1e-3)
    
    scale = 1.0 / (config.head_dim_qk ** 0.5)

    if qkv_layout == "sbh3d":
        # Q, K, V 在 head 维度交错存储
        qkv = torch.randn(
            config.max_seqlen_q, config.batch_size, config.num_heads, 3 * config.head_dim_qk,
            dtype=dtype, device="cuda"
        ) * value_range
        q, k, v = torch.split(qkv, config.head_dim_qk, dim=-1)
    elif qkv_layout == "sbhd_sb2hd":
        # Q 单独存储，K 和 V 在 seq 维度交错存储
        q = torch.randn(
            config.max_seqlen_q, config.batch_size, config.num_heads, config.head_dim_qk,
            dtype=dtype, device="cuda"
        ) * value_range
        kv = torch.randn(
            config.max_seqlen_kv, config.batch_size, config.num_heads, 2 * config.head_dim_qk,
            dtype=dtype, device="cuda"
        ) * value_range
        k, v = torch.split(kv, config.head_dim_qk, dim=-1)
    elif qkv_layout == "sbhd_sbh2d":
        # Q 单独存储，K 和 V 在 head 维度交错存储
        q = torch.randn(
            config.max_seqlen_q, config.batch_size, config.num_heads, config.head_dim_qk,
            dtype=dtype, device="cuda"
        ) * value_range
        kv = torch.randn(
            config.max_seqlen_kv, config.batch_size, config.num_heads, 2 * config.head_dim_qk,
            dtype=dtype, device="cuda"
        ) * value_range
        k, v = torch.split(kv, config.head_dim_qk, dim=-1)
    elif qkv_layout == "sbhd_sbhd_sbhd":
        # Q, K, V 完全分开存储
        q = torch.randn(
            config.max_seqlen_q, config.batch_size, config.num_heads, config.head_dim_qk,
            dtype=dtype, device="cuda"
        ) * value_range
        k = torch.randn(
            config.max_seqlen_kv, config.batch_size, config.num_heads, config.head_dim_qk,
            dtype=dtype, device="cuda"
        ) * value_range
        v = torch.randn(
            config.max_seqlen_kv, config.batch_size, config.num_heads, config.head_dim_qk,
            dtype=dtype, device="cuda"
        ) * value_range
    else:
        raise ValueError(f"Unsupported QKV layout: {qkv_layout}")

    ref_out = ref_attention(q, k, v, scale)

    sdpa_out = sdpa(q, k, v, is_causal=(config.attn_mask_type=="causal"))
    
    # unfused_attn = UnfusedDotProductAttention(
    #     softmax_scale=scale,
    #     attention_dropout=0.0,
    # ).cuda().to(dtype=q.dtype)

    # unfused_output = unfused_attn(q, k, v, qkv_layout=qkv_layout)
    flash_attn = FlashAttention(
        softmax_scale=scale,
        attention_dropout=0.0,
    ).cuda().to(dtype=q.dtype)
    
    flash_output = flash_attn(q, k, v, qkv_layout=qkv_layout).to(q.dtype)
    
    sage_attn_int8 = SageAttention(
        softmax_scale=scale,
        quantization_backend="triton",
        quantization_type="int8",
        smooth_k=True,
        return_lse=False,
        attention_dropout=0.0,
    ).cuda()

    sage_attn_e4m3 = SageAttention(
        softmax_scale=scale,
        quantization_backend="triton",
        quantization_type="e4m3",
        smooth_k=True,
        return_lse=False,
        attention_dropout=0.0,
    ).cuda()

    sage_attn_e5m2 = SageAttention(
        softmax_scale=scale,
        quantization_backend="triton",
        quantization_type="e5m2",
        smooth_k=True,
        return_lse=False,
        attention_dropout=0.0,
    ).cuda()

    # 将输入转换为 SBHD 布局（如果内核需要）
    q_sbhd = rearrange(q, 'b s h d -> s b h d').contiguous()  # 确保内存连续
    k_sbhd = rearrange(k, 'b s h d -> s b h d').contiguous()
    v_sbhd = rearrange(v, 'b s h d -> s b h d').contiguous()
    
    sage_int8_out = sage_attn_int8(q_sbhd, k_sbhd, v_sbhd, qkv_layout="sbhd_sbhd_sbhd").to(dtype)
    sage_e4m3_out = sage_attn_e4m3(q_sbhd, k_sbhd, v_sbhd, qkv_layout="sbhd_sbhd_sbhd").to(dtype)
    sage_e5m2_out = sage_attn_e5m2(q_sbhd, k_sbhd, v_sbhd, qkv_layout="sbhd_sbhd_sbhd").to(dtype)
    print("================================================")
    print("dtype:", dtype, "qkv_layout:", qkv_layout, "shape:", ref_out.shape, "\nbatch_size:", config.batch_size, "num_heads:", config.num_heads, "head_dim:", config.head_dim_qk)
    # print("Shapes: \n ref:", ref_out.shape, "sdpa:", sdpa_out.shape, "flash:", flash_output.shape,
        #   "sage_int8:", sage_int8_out.shape, "sage_e4m3:", sage_e4m3_out.shape, "sage_e5m2:", sage_e5m2_out.shape)
    print("value_range:", -value_range, "~", value_range)
    
    print("\nComparisons (vs ref):")
    print_diff("sdpa vs ref", sdpa_out, ref_out)
    print_diff("flash vs ref", flash_output, ref_out)
    print_diff("sage_int8 vs ref", sage_int8_out, ref_out)
    print_diff("sage_e4m3 vs ref", sage_e4m3_out, ref_out)
    print_diff("sage_e5m2 vs ref", sage_e5m2_out, ref_out)
    
    print("\nComparisons (sage vs sdpa):")
    print_diff("sage_int8 vs sdpa", sage_int8_out, sdpa_out)
    print_diff("sage_e4m3 vs sdpa", sage_e4m3_out, sdpa_out)
    print_diff("sage_e5m2 vs sdpa", sage_e5m2_out, sdpa_out)

    print("\nComparisons (sage vs flash):")
    print_diff("sage_int8 vs flash", sage_int8_out, flash_output)
    print_diff("sage_e4m3 vs flash", sage_e4m3_out, flash_output)
    print_diff("sage_e5m2 vs flash", sage_e5m2_out, flash_output)
    
    print("\nComparisons (sages):")
    print_diff("sage_int8 vs sage_e4m3", sage_int8_out, sage_e4m3_out)
    print_diff("sage_int8 vs sage_e5m2", sage_int8_out, sage_e5m2_out)
    print_diff("sage_e4m3 vs sage_e5m2", sage_e4m3_out, sage_e5m2_out)


    torch.testing.assert_close(ref_out, sdpa_out, **tols)
    torch.testing.assert_close(sage_int8_out, ref_out, **tols)
    torch.testing.assert_close(sage_e4m3_out, ref_out, **tols)
    torch.testing.assert_close(sage_e5m2_out, ref_out, **tols)

    # assert not torch.isnan(sage_output_int8).any(), "Sage output contains NaN values"
    # assert not torch.isinf(sage_output_int8).any(), "Sage output contains Inf values"

    # assert not torch.isnan(sage_output_e4m3).any(), "Sage output contains NaN values"
    # assert not torch.isinf(sage_output_e4m3).any(), "Sage output contains Inf values"
    
    # sage_output_e5m2 = sage_attn_e5m2(q, k, v, qkv_layout=qkv_layout).to(q.dtype) 
    # assert not torch.isnan(sage_output_e5m2).any(), "Sage output contains NaN values"
    # assert not torch.isinf(sage_output_e5m2).any(), "Sage output contains Inf values"



    # ref_output = ref_attention(q, k, v, scale)
    # assert not torch.isnan(ref_output).any(), "Ref output contains NaN values"
    # assert not torch.isinf(ref_output).any(), "Ref output contains Inf values"




    # sdpa_output = sdpa(q, k, v, is_causal=True)
    # assert not torch.isnan(sdpa_output).any(), "Sdpa output contains NaN values"
    # assert not torch.isinf(sdpa_output).any(), "Sdpa output contains Inf values"

    # print(f"\nModel: {model}, dtype: {dtype}, qkv_layout: {qkv_layout}")
    # print(f"Input shapes: q={q.shape}, k={k.shape}, v={v.shape}")

    # # print(f"Sage output shape: {sage_output.shape}")
    # # print(f"Ref output shape: {ref_output.shape}")
    # # print(f"Sdpa output shape: {sdpa_output.shape}")

    # sage_diff = (sage_output - ref_output).abs()
    # print(f"\nSage vs Ref:")
    # print(f"Max absolute difference: {sage_diff.max().item()}")
    # print(f"Mean absolute difference: {sage_diff.mean().item()}")
    # print(f"Std of absolute difference: {sage_diff.std().item()}")
    # print(f"Min absolute difference: {sage_diff.min().item()}")

    # sdpa_diff = (sdpa_output - ref_output).abs()
    # print(f"\nSdpa vs Ref:")
    # print(f"Max absolute difference: {sdpa_diff.max().item()}")
    # print(f"Mean absolute difference: {sdpa_diff.mean().item()}")
    # print(f"Std of absolute difference: {sdpa_diff.std().item()}")
    # print(f"Min absolute difference: {sdpa_diff.min().item()}")

    # diff = (sage_output - sdpa_output).abs()
    # print(f"\nSage vs Sdpa:")
    # print(f"Max absolute difference: {diff.max().item()}")
    # print(f"Mean absolute difference: {diff.mean().item()}")
    # print(f"Std of absolute difference: {diff.std().item()}")
    # print(f"Min absolute difference: {diff.min().item()}")

    # torch.testing.assert_close(sage_output, sdpa_output, **tols)
    # torch.testing.assert_close(sage_output, ref_output, **tols)
    # torch.testing.assert_close(sdpa_output, ref_output, **tols)    

    # #todo LSE 的 精度 diff
    # #todo 各种attn backend 的精度 diff（base on ref）
    # #todo 区分有无mask（focus on NOmask）

@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("model_configs", [model_configs_sage])
@pytest.mark.parametrize("model", model_configs_sage.keys())
@pytest.mark.parametrize("qkv_layout", qkv_layouts)
@pytest.mark.parametrize("value_range", [0.1, 1, 10, 100])
def test_flash(dtype, model_configs, model, qkv_layout, value_range):
    config = model_configs[model]
    tols = dict(atol=1e-3, rtol=1e-3)
    scale = 1.0 / (config.head_dim_qk ** 0.5)

    if qkv_layout == "sbh3d": # Q, K, V 在 head 维度交错存储
        qkv = torch.randn(
            config.max_seqlen_q, config.batch_size, config.num_heads, 3 * config.head_dim_qk,
            dtype=dtype, device="cuda"
        ) * value_range
        q, k, v = torch.split(qkv, config.head_dim_qk, dim=-1)
    else:
        raise ValueError(f"Unsupported QKV layout: {qkv_layout}")

    ref_out = ref_attention2(q, k, v, scale, config.num_heads).to(dtype)
    ref_out = ref_out.reshape(config.max_seqlen_q, config.batch_size, -1).contiguous()
    flash_attn = FlashAttention(
        softmax_scale=scale,
        attention_dropout=0.0,
    ).cuda()
    flash_output = flash_attn(q, k, v, qkv_layout=qkv_layout).to(dtype)
    flash_output_api = flash_attn_func(q, k, v).to(dtype)
    flash_output_api = flash_output_api.reshape(config.max_seqlen_q, config.batch_size, -1).contiguous()

    print("================================================")
    print("dtype:", dtype, "qkv_layout:", qkv_layout, "shape:", ref_out.shape, "\nbatch_size:", config.batch_size, "num_heads:", config.num_heads, "head_dim:", config.head_dim_qk)
    print("Shapes: \n ref:", ref_out.shape, "flash:", flash_output.shape, "flash_api:", flash_output_api.shape)
    print("value_range:", -value_range, "~", value_range)
    
    print("\nComparisons:")
    print_diff("flash_api vs ref", flash_output_api, ref_out)
    print_diff("flash vs ref", flash_output, ref_out)
    print_diff("flash vs flash_api", flash_output, flash_output_api)


    torch.testing.assert_close(flash_output, ref_out, **tols)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("model_configs", [model_configs_sage])
@pytest.mark.parametrize("model", model_configs_sage.keys())
@pytest.mark.parametrize("value_range", [0.1, 1, 10, 100])
def test_flash2(dtype, model_configs, model, value_range):
    config = model_configs[model]
    tols = dict(atol=1e-3, rtol=1e-3)
    scale = 1.0 / (config.head_dim_qk ** 0.5)

    q = torch.randn((config.batch_size, config.max_seqlen_q, config.num_heads, config.head_dim_qk), dtype=dtype).cuda() * value_range
    k = torch.randn((config.batch_size, config.max_seqlen_q, config.num_heads, config.head_dim_qk), dtype=dtype).cuda() * value_range
    v = torch.randn((config.batch_size, config.max_seqlen_vv, config.num_heads, config.head_dim_v), dtype=dtype).cuda() * value_range
    
    ref_out = ref_attention(q, k, v, scale).to(dtype)
    sdpa_out = sdpa(q, k, v, is_causal=(config.attn_mask_type == "causal")).to(dtype)

    flash_attn = FlashAttention(
        softmax_scale=scale,
        attention_dropout=0.0,
    ).cuda()
    flash_output = flash_attn(q, k, v, qkv_layout="sbhd_sbhd_sbhd").to(dtype)

    flash_output_api = flash_attn_func(q, k, v).to(dtype)

    print("================================================")
    print("dtype:", dtype, "shape:", ref_out.shape, "\nbatch_size:", config.batch_size, "num_heads:", config.num_heads, "head_dim:", config.head_dim_qk)
    print("Shapes: \n ref:", ref_out.shape, "sdpa:", sdpa_out.shape, "flash:", flash_output.shape, "flash_api:", flash_output_api.shape)
    print("value_range:", -value_range, "~", value_range)
    
    print("\nComparisons:")
    print_diff("sdpa vs ref", sdpa_out, ref_out)
    print_diff("flash vs ref", flash_output, ref_out)
    print_diff("flash_api vs ref", flash_output_api, ref_out)
    print_diff("flash vs flash_api", flash_output, flash_output_api)

    # flash_output = flash_output.reshape(-1)
    # flash_output_api = flash_output_api.reshape(-1)
    # cosi = torch.nn.CosineSimilarity(dim=0) 
    # print(f"cosine_similarity: {cosi(flash_output, flash_output_api)}")
    torch.testing.assert_close(ref_out, sdpa_out, **tols)
    torch.testing.assert_close(flash_output, ref_out, **tols)


@pytest.mark.parametrize("dtype", [torch.float16])
@pytest.mark.parametrize("model_configs", [model_configs_sage])
@pytest.mark.parametrize("model", model_configs_sage.keys())
@pytest.mark.parametrize("value_range", [1])
def test_flash3(dtype, model_configs, model, value_range):
    config = model_configs[model]
    tols = dict(atol=1e-3, rtol=1e-3)
    scale = 1.0 / (config.head_dim_qk ** 0.5)
    # B S H D
    q = torch.randn((config.batch_size, config.max_seqlen_q, config.num_heads, config.head_dim_qk), dtype=dtype).cuda() * value_range
    k = torch.randn((config.batch_size, config.max_seqlen_kv, config.num_heads, config.head_dim_qk), dtype=dtype).cuda() * value_range
    v = torch.randn((config.batch_size, config.max_seqlen_kv, config.num_heads, config.head_dim_v), dtype=dtype).cuda() * value_range
    # b s h d
    ref_out = ref_attention(q, k, v, scale).to(dtype)
    ref_out2 = ref_attention2(q, k, v, scale, config.num_heads).to(dtype)
    #! BHSD
    q_sdpa = q.permute(0, 2, 1, 3).contiguous()
    k_sdpa = k.permute(0, 2, 1, 3).contiguous()
    v_sdpa = v.permute(0, 2, 1, 3).contiguous()
    
    sdpa_out = sdpa(q_sdpa, k_sdpa, v_sdpa, is_causal=(config.attn_mask_type == "causal")).to(dtype)
    sdpa_out = sdpa_out.permute(0, 2, 1, 3).contiguous()
    flash_attn = FlashAttention(
        softmax_scale=scale,
        attention_dropout=0.0,
    ).cuda()
    flash_output = flash_attn(q, k, v, qkv_layout="bshd_bshd_bshd").to(dtype)

    flash_output_api = flash_attn_func(q, k, v).to(dtype)

    print("================================================")
    print("dtype:", dtype, "shape:", ref_out.shape, "\nbatch_size:", config.batch_size, "num_heads:", config.num_heads, "head_dim:", config.head_dim_qk)
    print("Shapes: \n ref:", ref_out.shape, "sdpa:", sdpa_out.shape, "flash:", flash_output.shape, "flash_api:", flash_output_api.shape)
    print("value_range:", -value_range, "~", value_range)
    
    print("\nComparisons:")
    print_diff("ref_bmm vs ref", ref_out2, ref_out)
    print_diff("sdpa vs ref", sdpa_out, ref_out)
    print_diff("flash vs ref", flash_output, ref_out)
    print_diff("flash_api vs ref", flash_output_api, ref_out)
    print_diff("sdpa vs ref_bmm", sdpa_out, ref_out2)
    print_diff("flash vs ref_bmm", flash_output, ref_out2)
    print_diff("flash_api vs ref_bmm", flash_output_api, ref_out2)
    print_diff("flash vs flash_api", flash_output, flash_output_api)
    print_diff("sdpa vs flash_api", sdpa_out, flash_output_api)

    # flash_output = flash_output.reshape(-1)
    # flash_output_api = flash_output_api.reshape(-1)
    # cosi = torch.nn.CosineSimilarity(dim=0) 
    # print(f"cosine_similarity: {cosi(flash_output, flash_output_api)}")
    torch.testing.assert_close(ref_out, sdpa_out, **tols)
    torch.testing.assert_close(flash_output, ref_out, **tols)

@pytest.mark.parametrize("dtype", [torch.float16])
@pytest.mark.parametrize("model_configs", [model_configs_sage])
@pytest.mark.parametrize("model", model_configs_sage.keys())
@pytest.mark.parametrize("value_range", [0.1, 1, 10])
def test_sage1(dtype, model_configs, model, value_range):
    config = model_configs[model]
    tols = dict(atol=1e-3, rtol=1e-3)
    scale = 1.0 / (config.head_dim_qk ** 0.5)
    # B S H D
    q = torch.randn((config.batch_size, config.max_seqlen_q, config.num_heads, config.head_dim_qk), dtype=dtype).cuda() * value_range
    k = torch.randn((config.batch_size, config.max_seqlen_kv, config.num_heads, config.head_dim_qk), dtype=dtype).cuda() * value_range
    v = torch.randn((config.batch_size, config.max_seqlen_kv, config.num_heads, config.head_dim_v), dtype=dtype).cuda() * value_range
    # b s h d
    ref_out = ref_attention(q, k, v, scale).to(dtype)
    ref_out2 = ref_attention2(q, k, v, scale, config.num_heads).to(dtype)
    
    sage_attn_int8 = SageAttention(
        softmax_scale=scale,
        quantization_backend="triton",
        quantization_type="int8",
        smooth_k=True,
        return_lse=False,
        attention_dropout=0.0,
    ).cuda()

    sage_attn_e4m3 = SageAttention(
        softmax_scale=scale,
        quantization_backend="triton",
        quantization_type="e4m3",
        smooth_k=True,
        return_lse=False,
        attention_dropout=0.0,
    ).cuda()

    sage_attn_e5m2 = SageAttention(
        softmax_scale=scale,
        quantization_backend="triton",
        quantization_type="e5m2",
        smooth_k=True,
        return_lse=False,
        attention_dropout=0.0,
    ).cuda()

    sage_int8_out = sage_attn_int8(q, k, v, qkv_layout="bshd_bshd_bshd").to(dtype)
    sage_e4m3_out = sage_attn_e4m3(q, k, v, qkv_layout="bshd_bshd_bshd").to(dtype)
    sage_e5m2_out = sage_attn_e5m2(q, k, v, qkv_layout="bshd_bshd_bshd").to(dtype)

    print("================================================")
    print("dtype:", dtype, "shape:", ref_out.shape, "\nbatch_size:", config.batch_size, "num_heads:", config.num_heads, "head_dim:", config.head_dim_qk)
    print("Shapes: \n ref:", ref_out.shape, "ref_bmm: ", ref_out2.shape, "sage_int8: ", sage_int8_out.shape, "sage_e4m3: ", sage_e4m3_out.shape, "sage_e5m2: ", sage_e5m2_out.shape)
    print("value_range:", -value_range, "~", value_range)
    
    print("\nComparisons:")
    print_diff("sage_int8 vs ref", sage_int8_out, ref_out)
    print_diff("sage_e4m3 vs ref", sage_e4m3_out, ref_out)
    print_diff("sage_e5m2 vs ref", sage_e5m2_out, ref_out)
    print_diff("sage_int8 vs ref_bmm", sage_int8_out, ref_out2)
    print_diff("sage_e4m3 vs ref_bmm", sage_e4m3_out, ref_out2)
    print_diff("sage_e5m2 vs ref_bmm", sage_e5m2_out, ref_out2)

    torch.testing.assert_close(ref_out, sage_int8_out, **tols)
    

@pytest.mark.parametrize("dtype", [torch.float16])
@pytest.mark.parametrize("model_configs", [model_configs_sage])
@pytest.mark.parametrize("model", model_configs_sage.keys())
@pytest.mark.parametrize("value_range", [0.1, 1])
def test_sage2(dtype, model_configs, model, value_range):
    config = model_configs[model]
    tols = dict(atol=1e-3, rtol=1e-3)
    scale = 1.0 / (config.head_dim_qk ** 0.5)
    # B H S D
    q = torch.randn((config.batch_size, config.num_heads, config.max_seqlen_q, config.head_dim_qk), dtype=dtype).cuda() * value_range
    k = torch.randn((config.batch_size, config.num_heads, config.max_seqlen_kv, config.head_dim_qk), dtype=dtype).cuda() * value_range
    v = torch.randn((config.batch_size, config.num_heads, config.max_seqlen_kv, config.head_dim_v), dtype=dtype).cuda() * value_range
    
    # 确保 num_heads_q 是 num_heads_kv 的整数倍
    assert config.num_heads % config.num_gqa_groups == 0, "num_heads must be divisible by num_gqa_groups"
    
    # for i in range(300):
    #     start = i * 32
    #     end = i*32 + 32
    #     k[:, :, start:end, :] = k[:, :, start:end, :] * k[0, 0, i, 0] * (i%6 + 1)
    # b s h d
    ref_out = ref_attention2(q.clone().transpose(1,2).contiguous(), k.clone().transpose(1,2).contiguous(), v.clone().transpose(1,2).contiguous(), scale, config.num_heads).to(torch.float32)
    ref_out = ref_out.transpose(1,2)
    flash_api_out = flash_attn_func(q.clone().transpose(1,2).contiguous(), k.clone().transpose(1,2).contiguous(), v.clone().transpose(1,2).contiguous()).to(torch.float32)
    flash_api_out = flash_api_out.transpose(1,2)
    
    sage_attn_int8 = SageAttention(
        softmax_scale=scale,
        quantization_backend="triton",
        quantization_type="int8",
        smooth_k=True,
        return_lse=False,
        attention_dropout=0.0,
    ).cuda()

    sage_attn_e4m3 = SageAttention(
        softmax_scale=scale,
        quantization_backend="triton",
        quantization_type="e4m3",
        smooth_k=True,
        return_lse=False,
        attention_dropout=0.0,
    ).cuda()

    sage_attn_e5m2 = SageAttention(
        softmax_scale=scale,
        quantization_backend="triton",
        quantization_type="e5m2",
        smooth_k=True,
        return_lse=False,
        attention_dropout=0.0,
    ).cuda()

    sage_int8_out = sage_attn_int8(q, k, v, qkv_layout="bhsd_bhsd_bhsd", attn_mask_type=config.attn_mask_type).to(torch.float32)
    sage_e4m3_out = sage_attn_e4m3(q, k, v, qkv_layout="bhsd_bhsd_bhsd", attn_mask_type=config.attn_mask_type).to(torch.float32)
    # sage_e5m2_out = sage_attn_e5m2(q, k, v, qkv_layout="bhsd_bhsd_bhsd", attn_mask_type=config.attn_mask_type).to(torch.float32)

    # print("================================================")
    # print("dtype:", dtype, "shape:", ref_out.shape, "\nbatch_size:", config.batch_size, "num_heads:", config.num_heads, "head_dim:", config.head_dim_qk)
    # print("Shapes: \n ref:", ref_out.shape, "flash_api: ", flash_api_out.shape, "sage_int8: ", sage_int8_out.shape) # "sage_e4m3: ", sage_e4m3_out.shape, "sage_e5m2: ", sage_e5m2_out.shape)
    # print("value_range:", -value_range, "~", value_range)
    
    # print("\nComparisons:")
    print_diff("flash_api vs ref", flash_api_out, ref_out)
    print_diff("sage_int8 vs ref", sage_int8_out, ref_out)
    print_diff("sage_e4m3 vs ref", sage_e4m3_out, ref_out)
    print_diff("sage_e5m2 vs ref", sage_e5m2_out, ref_out)

    torch.testing.assert_close(ref_out, sage_int8_out, **tols)
    # torch.testing.assert_close(ref_out, sage_e4m3_out, **tols)
    