# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
import torch
from transformer_engine.pytorch import TransformerLayer
from transformer_engine.pytorch.attention import (
    SageAttention,
    FlashAttention,
    DotProductAttention, 
    _attention_backends,
    SageAttentionFunc
)
import transformer_engine.pytorch.attention as te_attention
from importlib import reload

import torch.nn.functional as F
from datetime import datetime
import pandas as pd

from flash_attn.flash_attn_interface import flash_attn_func, _flash_attn_forward, _flash_attn_varlen_backward

import os
import math

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import torch

from flash_attn import flash_attn_func

import logging
from typing import Tuple
from transformer_engine.pytorch.attention import (
    DotProductAttention,
    MultiheadAttention,
    RotaryPositionEmbedding)
from transformer_engine.pytorch.utils import (
    get_device_compute_capability,
    init_method_normal,
    scaled_init_method_normal,
    is_bf16_compatible,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
        head_dim_kv: int = None,
        alibi_type: str = "none",
        num_layers: int = 1,
        bias_shape: str = "1hss",
        window_size: Tuple[int, int] = (-1, -1),
    ):
        self.batch_size = batch_size
        self.num_heads = num_heads
        self.num_gqa_groups = num_gqa_groups
        self.head_dim_qk = head_dim_qk
        self.head_dim_kv = head_dim_qk if head_dim_kv is None else head_dim_kv
        self.hidden_size = num_heads * head_dim_qk
        self.hidden_size_kv = num_gqa_groups * self.head_dim_kv
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

model_configs_sage_layer = {
    #   测试配置:         b,  h, hg,   d,   sq,  skv,   p,      mask,       bias
    # "sage_1_0": ModelConfig(2, 16, 4, 64, 128, 128, 0.0, "no_mask", "no_bias"),
    # "sage_1_1": ModelConfig(4, 16, 4, 64, 128, 128, 0.0, "causal", "no_bias"),
    # "sage_1_2": ModelConfig(2, 16, 4, 64, 128, 128, 0.0, "padding", "no_bias"),
    "sage_2_0": ModelConfig(1, 16, 4, 64, 512, 512, 0.0, "causal", "no_bias"),
}

seed = 1234
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
_cpu_rng_state = torch.get_rng_state()
_cuda_rng_state = torch.cuda.get_rng_state()

def reset_rng_states() -> None:
    """Revert back to initial RNG state"""
    torch.set_rng_state(_cpu_rng_state)
    torch.cuda.set_rng_state(_cuda_rng_state)


def _run_transformer_layer(
    dtype: torch.dtype,
    config: ModelConfig,
    backend: str,
    ckpt_attn: bool,
    qkv_format: str,
    workspace_opt: bool,
    fused_qkv_params: bool,
    RoPE: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """运行TransformerLayer模块进行前向传播和反向传播"""

    # 设置环境变量
    reset_rng_states()
    os.environ["NVTE_FLASH_ATTN"] = "0"
    os.environ["NVTE_FUSED_ATTN"] = "0"
    os.environ["NVTE_SAGE_ATTN"] = "0"
    os.environ["NVTE_UNFUSED_ATTN"] = "0"
    
    if backend == "FlashAttention":
        os.environ["NVTE_FLASH_ATTN"] = "1"
    elif backend == "SageAttention":
        os.environ["NVTE_SAGE_ATTN"] = "1"
    
    # 更新后端选择
    global _attention_backends
    _attention_backends["backend_selection_requires_update"] = True

    # 创建输入张量
    if qkv_format == "thd":
        # [s, b, h*d] 格式
        inp = torch.randn(
            config.max_seqlen_q,
            config.batch_size,
            config.hidden_size,
            dtype=dtype,
            device="cuda",
            requires_grad=True,
        )
    else:  # bshd格式
        # [b, s, h*d] 格式
        inp = torch.randn(
            config.batch_size,
            config.max_seqlen_q,
            config.hidden_size,
            dtype=dtype,
            device="cuda",
            requires_grad=True,
        )
    
    # 处理序列长度
    if "padding" in config.attn_mask_type:
        seqlens_q = torch.randint(
            max(1, config.max_seqlen_q//2), config.max_seqlen_q, 
            [config.batch_size], dtype=torch.int32, device="cuda"
        )
    else:
        seqlens_q = torch.full(
            [config.batch_size], config.max_seqlen_q, dtype=torch.int32, device="cuda"
        )
    
    # 创建注意力掩码
    attention_mask = None
    if "padding" in config.attn_mask_type:
        if qkv_format == "bshd":
            # 批处理优先格式的掩码
            attention_mask = torch.zeros(
                config.batch_size, 1, 1, config.max_seqlen_q, dtype=torch.bool, device="cuda"
            )
            for i in range(config.batch_size):
                attention_mask[i, 0, 0, seqlens_q[i]:] = True
        else:
            # thd格式的掩码
            attention_mask_q = torch.zeros(
                config.batch_size, 1, config.max_seqlen_q, dtype=torch.bool, device="cuda"
            )
            for i in range(config.batch_size):
                attention_mask_q[i, 0, seqlens_q[i]:] = True
            attention_mask = attention_mask_q
    
    # 创建累积序列长度向量(cu_seqlens) - 仅用于thd格式
    cu_seqlens_q = None
    if qkv_format == "thd":
        cu_seqlens_q = torch.zeros(config.batch_size + 1, dtype=torch.int32, device="cuda")
        cu_seqlens_q[1:] = torch.cumsum(seqlens_q, dim=0)
    
    # 初始化权重
    sigma = 0.02
    init_method = init_method_normal(sigma)
    output_layer_init_method = scaled_init_method_normal(sigma, 1)
    
    # 初始化RoPE
    rotary_pos_emb = None
    if RoPE:
        PE = RotaryPositionEmbedding(dim=config.head_dim_qk)
        rotary_pos_emb = PE(config.max_seqlen_q).to(device="cuda")
    
    # 创建Transformer层 - 注意GQA配置
    block = TransformerLayer(
        config.hidden_size,
        4 * config.hidden_size,
        config.num_heads,
        num_gqa_groups=config.num_gqa_groups,  # 这里设置GQA组数
        layernorm_epsilon=1e-5,
        hidden_dropout=0.0,
        attention_dropout=config.dropout_p,
        init_method=init_method,
        output_layer_init_method=output_layer_init_method,
        layer_number=1,
        kv_channels=config.head_dim_qk,
        self_attn_mask_type=config.attn_mask_type,
        tp_group=None,
        tp_size=1,
        params_dtype=dtype,
        fuse_qkv_params=fused_qkv_params,
        attn_input_format=qkv_format,
    ).to(dtype=dtype, device="cuda")
    
    # 执行前向传播
    out = block(
        inp,
        attention_mask=attention_mask,
        self_attn_mask_type=config.attn_mask_type,
        checkpoint_core_attention=ckpt_attn,
        rotary_pos_emb=rotary_pos_emb,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_kv=cu_seqlens_q,
        max_seqlen_q=config.max_seqlen_q if qkv_format == "thd" else None,
        max_seqlen_kv=config.max_seqlen_kv if qkv_format == "thd" else None,
    )
    
    # 执行反向传播
    loss = out.sum()
    loss.backward()
    
    return out, inp.grad

def test_transformer_layer_sage(
    dtype, 
    config, 
    ckpt_attn=False, 
    qkv_format="thd", 
    fused_qkv_params=False, 
    RoPE=False
):
    """测试SageAttention的TransformerLayer模块"""
    
    # 容差设置
    tols = dict(atol=5e-1, rtol=5e-2)
    workspace_opt = True
    
    # 记录测试信息
    logger.info(f"测试配置: dtype={dtype}, 模型配置={config.__dict__}")
    logger.info(f"Q头数={config.num_heads}, GQA组数={config.num_gqa_groups}, QKV格式={qkv_format}")
    
    try:
        # 运行SageAttention后端测试
        logger.info("运行SageAttention后端测试...")
        sage_attn_fwd, sage_attn_bwd = _run_transformer_layer(
            dtype,
            config,
            "SageAttention",
            ckpt_attn,
            qkv_format,
            workspace_opt,
            fused_qkv_params,
            RoPE,
        )
        
        # 运行FlashAttention作为基准对比
        logger.info("运行FlashAttention后端测试...")
        flash_attn_fwd, flash_attn_bwd = _run_transformer_layer(
            dtype,
            config,
            "FlashAttention",
            ckpt_attn,
            qkv_format,
            workspace_opt,
            fused_qkv_params,
            RoPE,
        )
        
        # 验证结果一致性
        fwd_max_diff = torch.max(torch.abs(sage_attn_fwd - flash_attn_fwd))
        fwd_mean_diff = torch.mean(torch.abs(sage_attn_fwd - flash_attn_fwd))
        fwd_cosine = torch.nn.functional.cosine_similarity(
            sage_attn_fwd.reshape(-1), flash_attn_fwd.reshape(-1), dim=0
        )
        
        bwd_max_diff = torch.max(torch.abs(sage_attn_bwd - flash_attn_bwd))
        bwd_mean_diff = torch.mean(torch.abs(sage_attn_bwd - flash_attn_bwd))
        bwd_cosine = torch.nn.functional.cosine_similarity(
            sage_attn_bwd.reshape(-1), flash_attn_bwd.reshape(-1), dim=0
        )
        
        # 输出结果
        logger.info(f"前向传播对比 - 最大差异: {fwd_max_diff.item():.6f}, 平均差异: {fwd_mean_diff.item():.6f}, 余弦相似度: {fwd_cosine.item():.6f}")
        logger.info(f"反向传播对比 - 最大差异: {bwd_max_diff.item():.6f}, 平均差异: {bwd_mean_diff.item():.6f}, 余弦相似度: {bwd_cosine.item():.6f}")
        
        # 验证是否在容差范围内
        is_fwd_close = fwd_max_diff <= tols["atol"] or fwd_max_diff <= tols["rtol"] * torch.max(torch.abs(flash_attn_fwd))
        is_bwd_close = bwd_max_diff <= tols["atol"] or bwd_max_diff <= tols["rtol"] * torch.max(torch.abs(flash_attn_bwd))
        
        if is_fwd_close and is_bwd_close:
            logger.info("测试通过: SageAttention在TransformerLayer中的结果与FlashAttention一致")
            return True
        else:
            logger.error("测试失败: SageAttention与FlashAttention结果差异超出容差范围")
            return False
            
    except Exception as e:
        logger.exception(f"测试异常: {e}")
        return False

if __name__ == "__main__":
    # 设置随机种子
    torch.manual_seed(2025)
    
    # 测试所有配置
    success_count = 0
    total_count = 0
    
    # 测试不同的数据类型和布局
    for dtype in [torch.float16, torch.bfloat16]:
        for qkv_format in ["thd"]: #, "bshd"]:
            # 测试不同的模型配置
            for model_name, config in model_configs_sage_layer.items():
                total_count += 1
                logger.info(f"\n===== 测试 {total_count}: 数据类型 {dtype}, 模型配置 {model_name}, QKV格式 {qkv_format} =====")
                
                if test_transformer_layer_sage(dtype, config, qkv_format=qkv_format):
                    success_count += 1
    
    # 结果汇总
    logger.info(f"\n===== 测试完成: {success_count}/{total_count} 成功 =====")
    
    if success_count == total_count:
        logger.info("🎉 所有测试都通过了!")
    else:
        logger.warning(f"⚠️ {total_count - success_count} 个测试失败")