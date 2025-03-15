# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
import torch
from transformer_engine.pytorch.attention import (
    SageAttention,
    FlashAttention,
    DotProductAttention, 
    _attention_backends
)

import torch.nn.functional as F
from datetime import datetime
import pandas as pd

from flash_attn.flash_attn_interface import flash_attn_func
from flash_attn import flash_attn_varlen_func

import os

torch.manual_seed(2025)
torch.set_printoptions(precision=4, threshold=None, edgeitems=None, linewidth=None, profile=None, sci_mode=False)

class TestConfig:
    def __init__(self, batch_size, num_heads, seq_len, head_dim, total_seq, 
                 dtype=torch.float16, q_range = 1.0, k_range = 1.0, v_range = 1.0, layout='bhsd'):
        self.batch_size = batch_size
        self.num_heads = num_heads
        self.seq_len = seq_len
        self.head_dim = head_dim
        self.total_seq = total_seq
        self.dtype = dtype
        self.layout = layout  
        self.q_range = q_range
        self.k_range = k_range
        self.v_range = v_range

def create_tensors(config, q_range, k_range, v_range, requires_grad=False):
    if config.layout == 'bhsd':
        shape = (config.batch_size, config.num_heads, config.seq_len, config.head_dim)
    elif config.layout == 'bshd':
        shape = (config.batch_size, config.seq_len, config.num_heads, config.head_dim)
    elif config.layout == 'thd':
        shape = (config.total_seq, config.num_heads, config.head_dim)
    else:
        raise ValueError(f"Unsupported layout: {config.layout}")

    def generate_tensor(value_range):
        t = torch.randn(shape, device="cuda")
        t = t / t.std() * value_range
        t = t.to(config.dtype).contiguous()
        t.requires_grad = requires_grad
        return t

    return generate_tensor(q_range), generate_tensor(k_range), generate_tensor(v_range)

def calculate_similarity(output1, output2):
    output1 = output1.flatten().float()
    output2 = output2.flatten().float()
    
    metrics = {
        'max_diff': torch.max(torch.abs(output1 - output2)),
        'mean_diff': torch.mean(torch.abs(output1 - output2)),
        'cos_sim': torch.nn.functional.cosine_similarity(output1, output2, dim=0)
    }
    return metrics

def print_metrics(name, metrics):
    print(f"\n{name} Metrics:")
    print(f"Max Difference: {metrics['max_diff'].item():.6f}")
    print(f"Mean Difference: {metrics['mean_diff'].item():.6f}")
    print(f"Cosine Similarity: {metrics['cos_sim'].item():.6f}")

range_combinations = [
    # q_range, k_range, v_range
    (0.1, 0.1, 0.1),     
    # (1.0, 1.0, 1.0),    
    # (10.0, 10.0, 10.0), 
    # (0.1, 1.0, 10.0),  
    # (10.0, 1.0, 0.1),   
    # (0.1, 10.0, 0.1),  
    # (10.0, 0.1, 10.0),  
    # (0.1, 0.1, 10.0),
    # (10.0, 10.0, 0.1),
    # (0.1, 10.0, 10.0),
    # (10.0, 0.1, 0.1),
]
base_configs = [
    # (batch_size, num_heads, seq_len, head_dim, layout)
    (1, 4, 512, 64, 'bshd'), 
    # (4, 4, 512, 64, 'bshd'),
    # (8, 4, 512, 64, 'bshd'),
    # (16, 16, 2048, 128, 'bshd'),
    # (32, 8, 4096, 64, 'bshd'),
]

test_configs = []
for bs, h, s, d, layout in base_configs:
    for dtype in [torch.float16, torch.bfloat16]:
        for q_range, k_range, v_range in range_combinations:
            test_configs.append(
                TestConfig(
                    batch_size=bs,
                    num_heads=h,
                    seq_len=s,
                    head_dim=d,
                    total_seq=bs * s,
                    dtype=dtype,
                    q_range=q_range,
                    k_range=k_range,
                    v_range=v_range,
                    layout=layout
                )
            )

var_configs = [
    # total_seq, num_heads, head_dim, layout
    (2048, 4, 16, 'thd'),
    (2048, 4, 64, 'thd'),
    (2048, 4, 128, 'thd'),
    (2048, 4, 256, 'thd'),
]
test_var_configs = []
for t, h, d, layout in var_configs:
    for dtype in [torch.float16, torch.bfloat16]:
        for q_range, k_range, v_range in range_combinations:
            test_var_configs.append(
                TestConfig(
                    batch_size=0,
                    num_heads=h,
                    seq_len=0,
                    head_dim=d,
                    total_seq=t,
                    dtype=dtype,
                    q_range=q_range,
                    k_range=k_range,
                    v_range=v_range,
                    layout=layout
                )
            )

class ResultLogger:
    def __init__(self):
        self.columns = [
            'Quant Type', 
            'Batch Size', 'Num Heads', 'Seq Len', 'Head Dim', 'Total Seq',  
            'Data Type', 'Q K V Range', 'Layout', 
            'Max Diff', 'Mean Diff', 'Cos Sim',
            'Grad Q Cos Sim', 'Grad K Cos Sim', 'Grad V Cos Sim',
            'Test Type'
        ]
        self.results = []
    
    def add_result(self, config, qtype, metrics, grad_metrics=None, test_type="Forward"):
        grad_q_sim = grad_metrics['q']['cos_sim'].item() if grad_metrics else None
        grad_k_sim = grad_metrics['k']['cos_sim'].item() if grad_metrics else None
        grad_v_sim = grad_metrics['v']['cos_sim'].item() if grad_metrics else None
        
        self.results.append([
            qtype.upper(),
            config.batch_size,
            config.num_heads,
            config.seq_len,
            config.head_dim,
            config.total_seq,
            str(config.dtype).split('.')[-1],
            f"{config.q_range} - {config.k_range} - {config.v_range}",
            config.layout.upper(),  
            metrics['max_diff'].item(),
            metrics['mean_diff'].item(),
            metrics['cos_sim'].item(),
            grad_q_sim,
            grad_k_sim,
            grad_v_sim,
            test_type
        ])
    
    def save(self, filename="test_results"):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{filename}_{timestamp}"
        df = pd.DataFrame(self.results, columns=self.columns)
        df.to_excel(f"{filename}.xlsx", index=False)
        with open(f"{filename}.txt", "w") as f:
            f.write(df.to_string(index=False))
        print(f"Results saved to {filename}.[xlsx|txt]")

logger = ResultLogger()

def run_backward_test(config):
    """测试前向和反向传播的对比"""
    tols = {"atol": 1e-2, "rtol": 1e-2}
    scale = 1.0 / (config.head_dim ** 0.5)
    
    # 创建不同量化类型的SageAttention
    sage_e4m3 = SageAttention(
        softmax_scale=scale,
        quantization_backend="triton",
        quantization_type="e4m3",
        smooth_k=True,
        return_lse=False,
        attention_dropout=0.0,
    ).cuda()
    sage_e4m3.train()  # 设置为训练模式
    
    flash = FlashAttention(
        softmax_scale=scale,
        attention_dropout=0.0,
    ).cuda()
    flash.train()  # 设置为训练模式

    # 创建需要梯度的输入张量
    q, k, v = create_tensors(config, config.q_range, config.k_range, config.v_range, requires_grad=True)
    q_sage, k_sage, v_sage = create_tensors(config, config.q_range, config.k_range, config.v_range, requires_grad=True)
    
    # 前向传播
    sage_output = sage_e4m3(
        q_sage, k_sage, v_sage,
        qkv_layout=f"{config.layout}_{config.layout}_{config.layout}",
        attn_mask_type="no_mask"
    )
    
    # 准备FlashAttention输入
    if config.layout == 'bhsd': # [b,h,s,d] -> [b,s,h,d]
        q_flash = q.transpose(1, 2).contiguous()  
        k_flash = k.transpose(1, 2).contiguous()
        v_flash = v.transpose(1, 2).contiguous()
    else:  
        q_flash = q.contiguous()  
        k_flash = k.contiguous()
        v_flash = v.contiguous()
    
    # 运行FlashAttention
    flash_output = flash_attn_func(
        q_flash, k_flash, v_flash,
        dropout_p=0.0,
        softmax_scale=scale,
    )
    
    # 调整输出形状以匹配
    if config.layout == 'bshd':
        flash_output = flash_output.reshape(flash_output.size(0), flash_output.size(1), -1).contiguous()
    
    # 计算前向传播相似度
    forward_metrics = calculate_similarity(sage_output, flash_output)
    print_metrics(f"前向传播 - Sage e4m3 vs Flash (B={config.batch_size}, H={config.num_heads}, S={config.seq_len}, D={config.head_dim})", forward_metrics)
    
    # 创建随机梯度并执行反向传播
    grad_output_sage = torch.randn_like(sage_output)
    grad_output_flash = grad_output_sage.clone()
    
    sage_output.backward(grad_output_sage)
    flash_output.backward(grad_output_flash)
    
    # 比较梯度
    grad_metrics = {
        'q': calculate_similarity(q_sage.grad, q.grad),
        'k': calculate_similarity(k_sage.grad, k.grad),
        'v': calculate_similarity(v_sage.grad, v.grad)
    }
    
    print(f"\n===== 梯度对比 - SAGE vs FLASH =====")
    print_metrics("Q梯度", grad_metrics['q'])
    print_metrics("K梯度", grad_metrics['k'])
    print_metrics("V梯度", grad_metrics['v'])
    
    # 记录结果
    logger.add_result(config, 'e4m3', forward_metrics, grad_metrics, "Forward+Backward")

def run_var_backward_test(config):
    """可变长度序列的前向和反向传播测试"""
    tols = {"atol": 1e-2, "rtol": 1e-2}
    scale = 1.0 / (config.head_dim ** 0.5)
    # 创建cu_seqlens
    cu_seqlens_q = cu_seqlens_kv = torch.tensor([0, 300, 1100, 2048], device="cuda", dtype=torch.int32)
    
    # 创建模型
    sage_e4m3 = SageAttention(
        softmax_scale=scale,
        quantization_backend="triton",
        quantization_type="e4m3",
        smooth_k=True,
        return_lse=False,
        attention_dropout=0.0,
    ).cuda()
    sage_e4m3.train()
    
    # 创建需要梯度的输入张量
    q_sage, k_sage, v_sage = create_tensors(config, config.q_range, config.k_range, config.v_range, requires_grad=True)
    q_flash, k_flash, v_flash = create_tensors(config, config.q_range, config.k_range, config.v_range, requires_grad=True)
    
    # 前向传播
    sage_output = sage_e4m3(
        q_sage, k_sage, v_sage,
        qkv_layout="thd",
        attn_mask_type="no_mask",
        cu_seqlens_q=cu_seqlens_q, 
        cu_seqlens_kv=cu_seqlens_kv,
        max_seqlen_q=config.total_seq,
        max_seqlen_kv=config.total_seq,
    )
    
    # FlashAttention可变长度
    flash_output = flash_attn_varlen_func(
        q_flash, k_flash, v_flash,
        cu_seqlens_q=cu_seqlens_q, 
        cu_seqlens_k=cu_seqlens_kv,
        max_seqlen_q=config.total_seq,
        max_seqlen_k=config.total_seq,
        causal=False,
        dropout_p=0.0,
        softmax_scale=scale,
        return_attn_probs=False,
    )
    
    # 前向对比
    forward_metrics = calculate_similarity(sage_output, flash_output)
    print_metrics(f"前向传播 - Sage e4m3 vs Flash Varlen (T={config.total_seq}, H={config.num_heads}, D={config.head_dim})", forward_metrics)
    
    # 反向传播
    grad_output = torch.randn_like(sage_output)
    sage_output.backward(grad_output.clone())
    flash_output.backward(grad_output.clone())
    
    # 梯度对比
    grad_metrics = {
        'q': calculate_similarity(q_sage.grad, q_flash.grad),
        'k': calculate_similarity(k_sage.grad, k_flash.grad),
        'v': calculate_similarity(v_sage.grad, v_flash.grad)
    }
    
    print(f"\n===== 可变长度梯度对比 - SAGE vs FLASH =====")
    print_metrics("Q梯度", grad_metrics['q'])
    print_metrics("K梯度", grad_metrics['k'])
    print_metrics("V梯度", grad_metrics['v'])
    
    # 记录结果
    logger.add_result(config, 'e4m3', forward_metrics, grad_metrics, "Varlen Forward+Backward")

def run_test(config):
    tols = {"atol": 1e-2, "rtol": 1e-2}
    scale = 1.0 / (config.head_dim ** 0.5)
    sage_int8 = SageAttention(
        softmax_scale=scale,
        quantization_backend="triton",
        quantization_type="int8",
        smooth_k=True,
        return_lse=False,
        attention_dropout=0.0,
    ).cuda()
    sage_int8.eval()
    sage_e4m3 = SageAttention(
        softmax_scale=scale,
        quantization_backend="triton",
        quantization_type="e4m3",
        smooth_k=True,
        return_lse=False,
        attention_dropout=0.0,
    ).cuda()
    sage_e4m3.eval()
    sage_e5m2 = SageAttention(
        softmax_scale=scale,
        quantization_backend="triton",
        quantization_type="e5m2",
        smooth_k=True,
        return_lse=False,
        attention_dropout=0.0,
    ).cuda()
    sage_e5m2.eval()
    flash = FlashAttention(
        softmax_scale=scale,
        attention_dropout=0.0,
    ).cuda()

    q, k, v = create_tensors(config, config.q_range, config.k_range, config.v_range)
    
    sage_int8_output = sage_int8(
        q, k, v,
        qkv_layout=f"{config.layout}_{config.layout}_{config.layout}",
        attn_mask_type="no_mask"
    )
    
    sage_e4m3_output = sage_e4m3(
        q, k, v,
        qkv_layout=f"{config.layout}_{config.layout}_{config.layout}",
        attn_mask_type="no_mask"
    )

    sage_e5m2_output = sage_e5m2(
        q, k, v,
        qkv_layout=f"{config.layout}_{config.layout}_{config.layout}",
        attn_mask_type="no_mask"
    )

    flash_ans = flash(
        q, k, v,
        qkv_layout= "bshd_bshd_bshd",
        attn_mask_type="no_mask"
    )

    # -> bshd
    if config.layout == 'bhsd': # [b,h,s,d] -> [b,s,h,d]
        q_flash = q.transpose(1, 2).contiguous()  
        k_flash = k.transpose(1, 2).contiguous()
        v_flash = v.transpose(1, 2).contiguous()
    else:  
        q_flash = q.contiguous()  
        k_flash = k.contiguous()
        v_flash = v.contiguous()
    
    flash_output = flash_attn_func(
        q_flash, k_flash, v_flash
    )
    
    flash_output = flash_output.reshape(flash_output.size(0), flash_output.size(1), -1).contiguous()

    attention_kernel = DotProductAttention(config.num_heads, config.head_dim, softmax_scale=scale)
    attention_kernel.eval()
    os.environ["NVTE_FUSED_ATTN"] = "0"
    os.environ["NVTE_SAGE_ATTN"] = "1"
    os.environ["NVTE_UNFUSED_ATTN"] = "0"
    os.environ["NVTE_FLASH_ATTN"] = "0"
    dpa_output_1 = attention_kernel(q, k, v, qkv_format=config.layout, attn_mask_type="no_mask")


    metrics = calculate_similarity(sage_int8_output, flash_output)
    print_metrics(f"Sage int8 vs Flash (B={config.batch_size}, H={config.num_heads}, S={config.seq_len}, D={config.head_dim})", metrics)

    metrics_e4m3 = calculate_similarity(sage_e4m3_output, flash_output)
    print_metrics(f"Sage e4m3 vs Flash (B={config.batch_size}, H={config.num_heads}, S={config.seq_len}, D={config.head_dim})", metrics_e4m3)

    metrics_e5m2 = calculate_similarity(sage_e5m2_output, flash_output)
    print_metrics(f"Sage e5m2 vs Flash (B={config.batch_size}, H={config.num_heads}, S={config.seq_len}, D={config.head_dim})", metrics_e5m2)

    metrics_2 = calculate_similarity(flash_ans, flash_output)
    print_metrics(f"Flash vs Flash (B={config.batch_size}, H={config.num_heads}, S={config.seq_len}, D={config.head_dim})", metrics)

    metrics_dpa = calculate_similarity(dpa_output_1, flash_output)
    print_metrics(f"DPA vs Flash (B={config.batch_size}, H={config.num_heads} , H={config.num_heads}, D={config.head_dim})", metrics_dpa)


    logger.add_result(config, 'int8', metrics)
    logger.add_result(config, 'e4m3', metrics_e4m3)
    logger.add_result(config, 'e5m2', metrics_e5m2)
    logger.add_result(config, 'flash', metrics_2)
    logger.add_result(config, 'dpa Sage', metrics_dpa)

def run_var_test(config):
    tols = {"atol": 1e-2, "rtol": 1e-2}
    scale = 1.0 / (config.head_dim ** 0.5)
    cu_seqlens_q = cu_seqlens_kv = torch.tensor([0, 300, 1100, 2048], device="cuda", dtype=torch.int32)

    sage_int8 = SageAttention(
        softmax_scale=scale,
        quantization_backend="triton",
        quantization_type="int8",
        smooth_k=True,
        return_lse=False,
        attention_dropout=0.0,
    ).cuda()
    sage_int8.eval()
    sage_e4m3 = SageAttention(
        softmax_scale=scale,
        quantization_backend="triton",
        quantization_type="e4m3",
        smooth_k=True,
        return_lse=False,
        attention_dropout=0.0,
    ).cuda()
    sage_e4m3.eval()
    sage_e5m2 = SageAttention(
        softmax_scale=scale,
        quantization_backend="triton",
        quantization_type="e5m2",
        smooth_k=True,
        return_lse=False,
        attention_dropout=0.0,
    ).cuda()
    sage_e5m2.eval()
    flash = FlashAttention(
        softmax_scale=scale,
        attention_dropout=0.0,
    ).cuda()

    q, k, v = create_tensors(config, config.q_range, config.k_range, config.v_range)
    
    sage_int8_output = sage_int8(
        q, k, v,
        qkv_layout="thd",
        attn_mask_type="no_mask",
        cu_seqlens_q=cu_seqlens_q, 
        cu_seqlens_kv=cu_seqlens_kv,
        max_seqlen_q=config.total_seq,
        max_seqlen_kv=config.total_seq,
    )
    
    sage_e4m3_output = sage_e4m3(
        q, k, v,
        qkv_layout="thd",
        attn_mask_type="no_mask",
        cu_seqlens_q=cu_seqlens_q, 
        cu_seqlens_kv=cu_seqlens_kv,
        max_seqlen_q=config.total_seq,
        max_seqlen_kv=config.total_seq,
    )

    sage_e5m2_output = sage_e5m2(
        q, k, v,
        qkv_layout="thd",
        attn_mask_type="no_mask",
        cu_seqlens_q=cu_seqlens_q, 
        cu_seqlens_kv=cu_seqlens_kv,
        max_seqlen_q=config.total_seq,
        max_seqlen_kv=config.total_seq,
    )

    flash_ans = flash(
        q, k, v,
        qkv_layout="thd_thd_thd",
        attn_mask_type='padding', 
        cu_seqlens_q=cu_seqlens_q, 
        cu_seqlens_kv=cu_seqlens_kv,
        max_seqlen_q=config.total_seq,
        max_seqlen_kv=config.total_seq,
    )
    
    flash_output = flash_attn_varlen_func(
        q, k, v,
        cu_seqlens_q=cu_seqlens_q, 
        cu_seqlens_k=cu_seqlens_kv,
        max_seqlen_q=config.total_seq,
        max_seqlen_k=config.total_seq,
        causal=False,
        dropout_p=0.0,
        softmax_scale=scale,
        return_attn_probs=False,
    )

    attention_kernel = DotProductAttention(config.num_heads, config.head_dim)
    
    os.environ["NVTE_FUSED_ATTN"] = "0"
    os.environ["NVTE_SAGE_ATTN"] = "1"
    os.environ["NVTE_UNFUSED_ATTN"] = "0"
    os.environ["NVTE_FLASH_ATTN"] = "0"
    dpa_output_1 = attention_kernel(q, k, v, qkv_format='thd', attn_mask_type="no_mask", cu_seqlens_q=cu_seqlens_q, cu_seqlens_kv=cu_seqlens_kv)

    metrics = calculate_similarity(sage_int8_output, flash_output)
    print_metrics(f"Sage int8 vs Flash (T={config.total_seq}, H={config.num_heads}, D={config.head_dim})", metrics)

    metrics_e4m3 = calculate_similarity(sage_e4m3_output, flash_output)
    print_metrics(f"Sage e4m3 vs Flash (T={config.total_seq}, H={config.num_heads}, D={config.head_dim})", metrics_e4m3)

    metrics_e5m2 = calculate_similarity(sage_e5m2_output, flash_output)
    print_metrics(f"Sage e5m2 vs Flash (T={config.total_seq}, H={config.num_heads}, D={config.head_dim})", metrics_e5m2)

    metrics_2 = calculate_similarity(flash_ans, flash_output)
    print_metrics(f"Flash Kernel vs Flash (T={config.total_seq}, H={config.num_heads}, D={config.head_dim})", metrics_2)

    metrics_dpa = calculate_similarity(dpa_output_1, flash_output)
    print_metrics(f"DPA vs Flash (T={config.total_seq}, H={config.num_heads}, D={config.head_dim})", metrics_dpa)

    logger.add_result(config, 'int8', metrics)
    logger.add_result(config, 'e4m3', metrics_e4m3)
    logger.add_result(config, 'e5m2', metrics_e5m2)
    logger.add_result(config, 'flash', metrics_2)
    logger.add_result(config, 'dpa Sage', metrics_dpa)

def run_backward_tests_for_selected_configs():
    backward_configs = []
    
    for head_dim in [64]:
        for ranges in [(0.1, 0.1, 0.1)]:#, (1.0, 1.0, 1.0), (10.0, 10.0, 10.0),(0.1, 1.0, 10.0), (10.0, 1.0, 0.1), (0.1, 10.0, 0.1)]:
                       
            backward_configs.append(
                TestConfig(
                    batch_size=4,
                    num_heads=4,
                    seq_len=128,
                    head_dim=head_dim,
                    total_seq=4 * 128,
                    dtype=torch.float16,
                    q_range=ranges[0],
                    k_range=ranges[1],
                    v_range=ranges[2],
                    layout='bshd'
                )
            )
    
    for config in backward_configs:
        print(f"\n===== 运行反向传播测试 (head_dim={config.head_dim}, ranges={config.q_range}-{config.k_range}-{config.v_range}) =====")
        try:
            run_backward_test(config)
        except Exception as e:
            print(f"测试失败: {str(e)}")

def run_var_backward_tests_for_selected_configs():
    var_backward_configs = []
    
    for head_dim in [64]:
        # 添加不同范围组合
        for ranges in [(0.1, 0.1, 0.1)]:#, (1.0, 1.0, 1.0), (10.0, 10.0, 10.0)]:
            var_backward_configs.append(
                TestConfig(
                    batch_size=0,
                    num_heads=4,
                    seq_len=0,
                    head_dim=head_dim,
                    total_seq=2048,
                    dtype=torch.float16,
                    q_range=ranges[0],
                    k_range=ranges[1],
                    v_range=ranges[2],
                    layout='thd'
                )
            )
    
    for config in var_backward_configs:
        print(f"\n===== 运行可变长度反向传播测试 (head_dim={config.head_dim}, ranges={config.q_range}-{config.k_range}-{config.v_range}) =====")
        try:
            run_var_backward_test(config)
        except Exception as e:
            print(f"测试失败: {str(e)}")

def run_var_backward_test(config):
    """可变长度序列的前向和反向传播测试"""
    tols = {"atol": 1e-2, "rtol": 1e-2}
    scale = 1.0 / (config.head_dim ** 0.5)
    cu_seqlens_q = cu_seqlens_kv = torch.tensor([0, 300, 1100, 2048], device="cuda", dtype=torch.int32)
    
    os.environ["NVTE_FUSED_ATTN"] = "0"
    os.environ["NVTE_SAGE_ATTN"] = "0"
    os.environ["NVTE_UNFUSED_ATTN"] = "0"
    os.environ["NVTE_FLASH_ATTN"] = "1"
    
    torch.manual_seed(42)
    q_sage, k_sage, v_sage = create_tensors(config, config.q_range, config.k_range, config.v_range, requires_grad=True)
    
    torch.manual_seed(42)
    q_flash, k_flash, v_flash = create_tensors(config, config.q_range, config.k_range, config.v_range, requires_grad=True)
    
    os.environ["NVTE_FUSED_ATTN"] = "0"
    os.environ["NVTE_SAGE_ATTN"] = "1"
    os.environ["NVTE_UNFUSED_ATTN"] = "0"
    os.environ["NVTE_FLASH_ATTN"] = "0"
    
    sage_e4m3 = SageAttention(
        softmax_scale=scale,
        quantization_backend="triton",
        quantization_type="e4m3",
        smooth_k=True,
        return_lse=False,  # 添加返回lse以便更好地与flash_attn兼容
        attention_dropout=0.0,
    ).cuda()
    sage_e4m3.train()
    
    sage_output= sage_e4m3(
        q_sage, k_sage, v_sage,
        qkv_layout="thd",
        attn_mask_type="no_mask",
        cu_seqlens_q=cu_seqlens_q, 
        cu_seqlens_kv=cu_seqlens_kv,
        max_seqlen_q=config.total_seq,
        max_seqlen_kv=config.total_seq,
    )

    os.environ["NVTE_FUSED_ATTN"] = "0"
    os.environ["NVTE_SAGE_ATTN"] = "0"
    os.environ["NVTE_UNFUSED_ATTN"] = "0"
    os.environ["NVTE_FLASH_ATTN"] = "1"
    
    attention_flash = DotProductAttention(config.num_heads, config.head_dim, softmax_scale=scale)
    attention_flash.train()
    
    flash_output = attention_flash(
        q_flash, k_flash, v_flash,
        qkv_format='thd',
        attn_mask_type="no_mask",
        cu_seqlens_q=cu_seqlens_q, 
        cu_seqlens_kv=cu_seqlens_kv,
        max_seqlen_q=config.total_seq,
        max_seqlen_kv=config.total_seq,
    )
    
    # 直接使用flash_attn_varlen_func也可作为替代方案
    # flash_output = flash_attn_varlen_func(
    #     q_flash, k_flash, v_flash,
    #     cu_seqlens_q=cu_seqlens_q, 
    #     cu_seqlens_k=cu_seqlens_kv,
    #     max_seqlen_q=config.total_seq,
    #     max_seqlen_k=config.total_seq,
    #     causal=False,
    #     dropout_p=0.0,
    #     softmax_scale=scale,
    #     return_attn_probs=False,
    # )
    
    forward_metrics = calculate_similarity(sage_output, flash_output)
    print_metrics(f"前向传播 - Sage e4m3 vs Flash Varlen (T={config.total_seq}, H={config.num_heads}, D={config.head_dim})", forward_metrics)
    
    torch.manual_seed(42)
    print(flash_output.shape)
    print(sage_output.shape)
    flash_output.backward(torch.randn_like(flash_output))
    sage_output.backward(torch.randn_like(sage_output))

    
    grad_metrics = {
        'q': calculate_similarity(q_sage.grad, q_flash.grad),
        'k': calculate_similarity(k_sage.grad, k_flash.grad),
        'v': calculate_similarity(v_sage.grad, v_flash.grad)
    }
    
    print(f"\n===== 梯度统计信息 =====")
    print(f"Sage - Q梯度范围: {q_sage.grad.min().item():.6f} 到 {q_sage.grad.max().item():.6f}, 平均值: {q_sage.grad.mean().item():.6f}")
    print(f"Flash - Q梯度范围: {q_flash.grad.min().item():.6f} 到 {q_flash.grad.max().item():.6f}, 平均值: {q_flash.grad.mean().item():.6f}")
    
    print(f"Sage - K梯度范围: {k_sage.grad.min().item():.6f} 到 {k_sage.grad.max().item():.6f}, 平均值: {k_sage.grad.mean().item():.6f}")
    print(f"Flash - K梯度范围: {k_flash.grad.min().item():.6f} 到 {k_flash.grad.max().item():.6f}, 平均值: {k_flash.grad.mean().item():.6f}")
    
    print(f"Sage - V梯度范围: {v_sage.grad.min().item():.6f} 到 {v_sage.grad.max().item():.6f}, 平均值: {v_sage.grad.mean().item():.6f}")
    print(f"Flash - V梯度范围: {v_flash.grad.min().item():.6f} 到 {v_flash.grad.max().item():.6f}, 平均值: {v_flash.grad.mean().item():.6f}")
    
    print(f"\n===== 可变长度梯度对比 - SAGE vs FLASH =====")
    print_metrics("Q梯度", grad_metrics['q'])
    print_metrics("K梯度", grad_metrics['k'])
    print_metrics("V梯度", grad_metrics['v'])
    
    # logger.add_result(config, 'e4m3', forward_metrics, grad_metrics, "Varlen Forward+Backward")

if __name__ == "__main__":
    # for config in test_var_configs[:2]:  
    #     print(f"Running var test for config: {config}")
    #     run_var_test(config)
    
    # print("\n===== 开始反向传播测试 =====")
    # run_backward_tests_for_selected_configs()
    
    # print("\n===== 开始可变长度反向传播测试 =====")
    run_var_backward_tests_for_selected_configs()
    
    # logger.save("sage_attn_test_with_backward")