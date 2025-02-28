# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import pytest
import torch
import logging
from transformer_engine.pytorch.attention import DotProductAttention

class ModelConfig:
    """变长序列测试配置"""
    def __init__(
        self,
        batch_size: int,
        num_heads: int,
        head_dim_qk: int,
        head_dim_v: int,
        max_seqlen: int,
        dropout_p: float = 0.0,
        attn_mask_type: str = "causal",
        qkv_format: str = "bshd"
    ):
        self.batch_size = batch_size
        self.num_heads = num_heads
        self.head_dim_qk = head_dim_qk
        self.head_dim_v = head_dim_v
        self.max_seqlen = max_seqlen
        self.dropout_p = dropout_p
        self.attn_mask_type = attn_mask_type
        self.qkv_format = qkv_format

def generate_varlen_inputs(config: ModelConfig, dtype: torch.dtype):
    """生成变长序列输入"""
    # 随机生成每个样本的实际长度
    seq_lens = torch.randint(
        low=int(0.5 * config.max_seqlen),
        high=config.max_seqlen,
        size=(config.batch_size,),
        device="cuda"
    )
    
    # 生成累积序列长度
    cu_seqlens = torch.zeros(config.batch_size + 1, dtype=torch.int32, device="cuda")
    cu_seqlens[1:] = torch.cumsum(seq_lens, dim=0)
    
    # 生成随机数据并填充
    total_tokens = int(cu_seqlens[-1])
    
    # 根据布局生成不同形状的张量
    if config.qkv_format == "bshd":
        q_shape = (total_tokens, config.num_heads, config.head_dim_qk)
        kv_shape = (total_tokens, config.num_heads, config.head_dim_v)
    elif config.qkv_format == "sbhd":
        q_shape = (total_tokens, config.num_heads, config.head_dim_qk)
        kv_shape = (total_tokens, config.num_heads, config.head_dim_v)
    else:
        raise ValueError(f"不支持的布局格式: {config.qkv_format}")
    
    q = torch.randn(q_shape, dtype=dtype, device="cuda", requires_grad=True)
    k = torch.randn(kv_shape, dtype=dtype, device="cuda", requires_grad=True)
    v = torch.randn(kv_shape, dtype=dtype, device="cuda", requires_grad=True)
    
    return q, k, v, cu_seqlens, seq_lens

def run_varlen_flash_attention(config: ModelConfig, dtype: torch.dtype):
    """运行变长FlashAttention测试"""
    # 生成输入
    q, k, v, cu_seqlens, seq_lens = generate_varlen_inputs(config, dtype)
    
    # 初始化注意力模块
    attn = DotProductAttention(
        num_attention_heads=config.num_heads,
        kv_channels=(config.head_dim_qk, config.head_dim_v),
        attn_mask_type=config.attn_mask_type,
        qkv_format=config.qkv_format,
        attention_dropout=config.dropout_p
    ).cuda()
    
    # 前向传播
    with torch.cuda.amp.autocast(dtype=dtype):
        out = attn(
            q, k, v,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_kv=cu_seqlens,
            max_seqlen_q=config.max_seqlen,
            max_seqlen_kv=config.max_seqlen
        )
    
    # 反向传播
    grad = torch.randn_like(out)
    out.backward(grad)
    
    return out, q.grad, k.grad, v.grad

# 测试配置参数化
varlen_configs = [
    # (batch_size, num_heads, head_dim_qk, head_dim_v, max_seqlen, qkv_format)
    ModelConfig(2, 8, 64, 64, 128, qkv_format="bshd"),
    ModelConfig(4, 16, 64, 128, 512, qkv_format="sbhd"),
    ModelConfig(2, 16, 128, 64, 1024, attn_mask_type="padding"),
    ModelConfig(1, 24, 64, 256, 2048, qkv_format="bshd")
]

@pytest.mark.parametrize("config", varlen_configs)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_varlen_flash_attention(config: ModelConfig, dtype: torch.dtype):

    
    outputs = run_varlen_flash_attention(config, dtype)
    
    expected_shape = (config.batch_size * config.max_seqlen, 
                     config.num_heads, config.head_dim_v)
    assert outputs[0].shape == expected_shape, f"输出形状错误: {outputs[0].shape} vs {expected_shape}"
    assert not torch.isnan(outputs[0]).any(), "输出包含NaN值"
    
    max_val = outputs[0].abs().max().item()
    assert max_val < 1e4, f"输出值过大: {max_val}"

if __name__ == "__main__":
    pytest.main([__file__])
