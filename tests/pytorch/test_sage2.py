# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
import torch
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
# 添加必要的导入
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import torch

from flash_attn import flash_attn_func


# shape = [28800, 15, 16, 72]
# device = torch.device("cuda:0")
# dtype = torch.bfloat16
# q = torch.randn(shape, device=device, dtype=dtype, requires_grad=True)
# k = torch.randn(shape, device=device, dtype=dtype, requires_grad=True)
# v = torch.randn(shape, device=device, dtype=dtype, requires_grad=True)
# grad = torch.randn(shape, device=device, dtype=dtype)

# print(f"{torch.cuda.memory_allocated()/1024**3}, {torch.cuda.max_memory_allocated()/1024**3}")

# o = flash_attn_func(
#     q,
#     k,
#     v,
#     dropout_p=0.0,
# )

# o.backward(grad)
# print(f"{torch.cuda.memory_allocated()}, {torch.cuda.max_memory_allocated()}")
def visualize_tensor(tensor, title, max_items=1000, figsize=(10, 6)):
    """Visualize distribution and values of a single tensor"""
    plt.figure(figsize=figsize)
    tensor_flat = tensor.detach().cpu().flatten().float().numpy()
    
    # Random sampling for large tensors
    if tensor_flat.size > max_items:
        indices = np.random.choice(tensor_flat.size, max_items, replace=False)
        tensor_flat = tensor_flat[indices]
    
    plt.subplot(1, 2, 1)
    plt.hist(tensor_flat, bins=50)
    plt.title(f"{title} - Distribution Histogram")
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.boxplot(tensor_flat)
    plt.title(f"{title} - Box Plot")
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f"{title.replace(' ', '_')}.png")
    plt.close()

def visualize_comparison(tensor1, tensor2, title, max_items=1000, figsize=(12, 8)):
    """Compare distributions and differences between two tensors"""
    plt.figure(figsize=figsize)
    
    # Convert to numpy arrays
    tensor1_flat = tensor1.detach().cpu().flatten().float().numpy()
    tensor2_flat = tensor2.detach().cpu().flatten().float().numpy()
    
    # Consistent sampling for comparison
    if tensor1_flat.size > max_items:
        indices = np.random.choice(tensor1_flat.size, max_items, replace=False)
        tensor1_flat = tensor1_flat[indices]
        tensor2_flat = tensor2_flat[indices]
    
    diff = tensor1_flat - tensor2_flat
    
    # Distribution comparison
    plt.subplot(2, 2, 1)
    plt.hist(tensor1_flat, bins=50, alpha=0.7, label='Tensor1')
    plt.hist(tensor2_flat, bins=50, alpha=0.7, label='Tensor2')
    plt.title(f"{title} - Distribution Comparison")
    plt.legend()
    plt.grid(True)
    
    # Scatter plot
    plt.subplot(2, 2, 2)
    plt.scatter(tensor1_flat, tensor2_flat, alpha=0.5, s=10)
    max_val = max(tensor1_flat.max(), tensor2_flat.max())
    min_val = min(tensor1_flat.min(), tensor2_flat.min())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    plt.title(f"{title} - Correlation Scatter")
    plt.xlabel('Tensor1')
    plt.ylabel('Tensor2')
    plt.grid(True)
    
    # Difference analysis
    plt.subplot(2, 2, 3)
    plt.hist(diff, bins=50)
    plt.title(f"{title} - Difference Histogram")
    plt.xlabel('Difference (Tensor1 - Tensor2)')
    plt.grid(True)
    
    plt.subplot(2, 2, 4)
    plt.boxplot([tensor1_flat, tensor2_flat, diff], labels=['Tensor1', 'Tensor2', 'Difference'])
    plt.title(f"{title} - Combined Box Plot")
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f"{title.replace(' ', '_')}_comparison.png")
    plt.close()

def visualize_heatmap(tensor, title, max_dim=64, figsize=(10, 8)):
    """Visualize tensor as heatmap"""
    plt.figure(figsize=figsize)
    
    # Tensor slicing
    if tensor.dim() >= 3:
        tensor_slice = tensor[0, :max_dim, :max_dim].detach().cpu().float().numpy()
    elif tensor.dim() == 2:
        tensor_slice = tensor[:max_dim, :max_dim].detach().cpu().float().numpy()
    else:
        tensor_1d = tensor.detach().cpu().float().numpy()
        size = min(max_dim*max_dim, tensor_1d.size)
        dim = int(np.sqrt(size))
        tensor_slice = tensor_1d[:dim*dim].reshape(dim, dim)
    
    sns.heatmap(tensor_slice, cmap='viridis')
    plt.title(f"{title} - Heatmap (Partial View)")
    plt.tight_layout()
    plt.savefig(f"{title.replace(' ', '_')}_heatmap.png")
    plt.close()

def visualize_qkv_and_gradients(q_sage, k_sage, v_sage, q_flash, k_flash, v_flash, 
                                sage_output, flash_output, 
                                config, output_dir="vis_results"):
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    original_dir = os.getcwd()
    os.chdir(output_dir)
    
    # Input comparisons
    visualize_comparison(q_sage, q_flash, f"Q Input Comparison (hd={config.head_dim})")
    visualize_comparison(k_sage, k_flash, f"K Input Comparison (hd={config.head_dim})")
    visualize_comparison(v_sage, v_flash, f"V Input Comparison (hd={config.head_dim})")
    
    # Output comparison
    visualize_comparison(sage_output, flash_output, 
                        f"Sage vs Flash Output (hd={config.head_dim})")
    
    # Gradient analysis
    if q_sage.grad is not None and q_flash.grad is not None:
        visualize_comparison(q_sage.grad, q_flash.grad, 
                            f"Q Gradients Comparison (hd={config.head_dim})")
        visualize_comparison(k_sage.grad, k_flash.grad, 
                            f"K Gradients Comparison (hd={config.head_dim})")
        visualize_comparison(v_sage.grad, v_flash.grad, 
                            f"V Gradients Comparison (hd={config.head_dim})")
    
    # Heatmap visualizations
    visualize_heatmap(q_sage, f"Q_Sage_Heatmap (hd={config.head_dim})")
    visualize_heatmap(q_flash, f"Q_Flash_Heatmap (hd={config.head_dim})")
    visualize_heatmap(sage_output, f"Sage_Output_Heatmap (hd={config.head_dim})")
    visualize_heatmap(flash_output, f"Flash_Output_Heatmap (hd={config.head_dim})")
    
    if q_sage.grad is not None:
        visualize_heatmap(q_sage.grad, f"Q_Sage_Grad_Heatmap (hd={config.head_dim})")
        visualize_heatmap(k_sage.grad, f"K_Sage_Grad_Heatmap (hd={config.head_dim})")
        visualize_heatmap(v_sage.grad, f"V_Sage_Grad_Heatmap (hd={config.head_dim})")
    
    if q_flash.grad is not None:
        visualize_heatmap(q_flash.grad, f"Q_Flash_Grad_Heatmap (hd={config.head_dim})")
        visualize_heatmap(k_flash.grad, f"K_Flash_Grad_Heatmap (hd={config.head_dim})")
        visualize_heatmap(v_flash.grad, f"V_Flash_Grad_Heatmap (hd={config.head_dim})")
    
    os.chdir(original_dir)
    print(f"Visualization results saved to: {os.path.join(os.getcwd(), output_dir)}")

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
    print("max_diff", torch.max(torch.abs(output1 - output2)))
    print("mean_diff", torch.mean(torch.abs(output1 - output2)))
    print("cos_sim", torch.nn.functional.cosine_similarity(output1, output2, dim=0))
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

def run_var_backward_test(config):
    tols = {"atol": 1e-2, "rtol": 1e-2}
    scale = 1.0 / (config.head_dim ** 0.5)
    cu_seqlens_q = cu_seqlens_kv = torch.tensor([0, 300, 1100, 2048], device="cuda", dtype=torch.int32)
    
    sage_e4m3 = SageAttention(
        softmax_scale=scale,
        quantization_backend="triton",
        quantization_type="e4m3",
        smooth_k=True,
        return_lse=False,
        attention_dropout=0.0,
    ).cuda()
    sage_e4m3.train()
    
    q_sage, k_sage, v_sage = create_tensors(config, config.q_range, config.k_range, config.v_range, requires_grad=True)
    q_flash = q_sage.copy()
    k_flash = k_sage.copy()
    v_flash = v_sage.copy()

    sage_output = sage_e4m3(
        q_sage, k_sage, v_sage,
        qkv_layout="thd",
        attn_mask_type="no_mask",
        cu_seqlens_q=cu_seqlens_q, 
        cu_seqlens_kv=cu_seqlens_kv,
        max_seqlen_q=config.total_seq,
        max_seqlen_kv=config.total_seq,
    )
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
    
    forward_metrics = calculate_similarity(sage_output, flash_output)
    print_metrics(f"前向传播 - Sage e4m3 vs Flash Varlen (T={config.total_seq}, H={config.num_heads}, D={config.head_dim})", forward_metrics)
    
    grad_output = torch.randn_like(sage_output)
    sage_output.backward(grad_output.clone())
    flash_output.backward(grad_output.clone())
    
    grad_metrics = {
        'q': calculate_similarity(q_sage.grad, q_flash.grad),
        'k': calculate_similarity(k_sage.grad, k_flash.grad),
        'v': calculate_similarity(v_sage.grad, v_flash.grad)
    }
    
    print(f"\n===== 可变长度梯度对比 - SAGE vs FLASH =====")
    print_metrics("Q梯度", grad_metrics['q'])
    print_metrics("K梯度", grad_metrics['k'])
    print_metrics("V梯度", grad_metrics['v'])
    
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
    # TODO 对齐 backwards开始时两者的输入是否能对齐

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

def run_var_backward_tests_for_selected_configs():
    var_backward_configs = []
    
    head_dims = [32, 64, 96, 128]
    num_heads = [1, 2, 4, 8, 16, 32, 64, 128]
    seq_configs = [256, 512, 1024]

    value_ranges = [
        (0.1, 0.1, 0.1),  # 小值范围
        (1.0, 1.0, 1.0),  # 中等值范围
        (10.0, 10.0, 10.0)  # 大值范围
    ]
    
    for head_dim in head_dims:
        for ranges in value_ranges:
            for num_head in num_heads:
                for seq_len in seq_configs:
                    var_backward_configs.append(
                        TestConfig(
                            batch_size=0,
                            num_heads=num_head,
                            seq_len=0, 
                            head_dim=head_dim,
                            total_seq=seq_len,
                            dtype=torch.float16,
                            q_range=ranges[0],
                            k_range=ranges[1],
                            v_range=ranges[2],
                            layout='thd'
                        )
                    )
    
    # import random
    # if len(var_backward_configs) > 20:
    #     print(f"配置总数: {len(var_backward_configs)}，随机采样20个进行测试")
    #     var_backward_configs = random.sample(var_backward_configs, 20)
    
    for i, config in enumerate(var_backward_configs):
        print(f"\n===== 测试 {i+1}/{len(var_backward_configs)}: 可变长度反向传播 (B={config.batch_size}, H={config.num_heads}, D={config.head_dim}, S={config.total_seq}, ranges={config.q_range}-{config.k_range}-{config.v_range}) =====")
        run_var_backward_test(config)


def validate_gradients(sage_grads, flash_grads, config):
    for i, (g1, g2) in enumerate(zip(sage_grads, flash_grads)):
        assert g1.shape == g2.shape, f"梯度维度不匹配: 层{i} {g1.shape} vs {g2.shape}"
        
    for grad in sage_grads + flash_grads:
        assert torch.isfinite(grad).all(), "梯度包含非法值(NaN/Inf)"

# 环境管理器
class EnvSwitcher:
    def __init__(self, env_vars):
        self.env_vars = env_vars
        self.original = {}

    def __enter__(self):
        for k, v in self.env_vars.items():
            self.original[k] = os.environ.get(k)
            os.environ[k] = str(v)

    def __exit__(self, *args):
        for k, v in self.original.items():
            if v is None:
                del os.environ[k]
            else:
                os.environ[k] = v

def run_var_backward_test(config):
    tols = {"atol": 1e-2, "rtol": 1e-2}
    scale = 1.0 / (config.head_dim ** 0.5)
    max_seqlen = config.total_seq
    cu_seqlens = torch.tensor(
        [0, max_seqlen//2, max_seqlen],  
        device="cuda", 
        dtype=torch.int32
    )
    base_tensor = torch.empty(config.total_seq, config.num_heads, config.head_dim,
                             device="cuda", dtype=torch.float16).normal_(0, 0.02)
    
    def clone_tensor(tensor):
        return tensor.clone().detach().requires_grad_(True)
    
    q_sage = clone_tensor(base_tensor)
    k_sage = clone_tensor(base_tensor)
    v_sage = clone_tensor(base_tensor)
    q_flash = clone_tensor(base_tensor)
    k_flash = clone_tensor(base_tensor)
    v_flash = clone_tensor(base_tensor)

    with EnvSwitcher({"NVTE_SAGE_ATTN": "1", "NVTE_FLASH_ATTN": "0"}):
        sage_dpa = DotProductAttention(config.num_heads, config.head_dim, softmax_scale=scale)
        # sage_dpa = SageAttention(
        #     softmax_scale=scale,
        #     quantization_backend="triton",
        #     quantization_type="e4m3",
        #     smooth_k=True,
        #     return_lse=False,
        #     attention_dropout=0.0,
        # ).cuda()
        sage_dpa.train()

        sage_output = sage_dpa(
            q_sage, k_sage, v_sage,
            # qkv_layout='thd_thd_thd',
            qkv_format='thd',
            attn_mask_type="no_mask",
            cu_seqlens_q=cu_seqlens, 
            cu_seqlens_kv=cu_seqlens,
            max_seqlen_q=max_seqlen, 
            max_seqlen_kv=max_seqlen,
        ).contiguous()

    reload(te_attention) #! 强制重载模块
    with EnvSwitcher({"NVTE_SAGE_ATTN": "0", "NVTE_FLASH_ATTN": "1", "NVTE_FUSED_ATTN": "0"}):
        flash_dpa = DotProductAttention(config.num_heads, config.head_dim, softmax_scale=scale)
        flash_dpa.train()

        flash_output = flash_dpa(
            q_flash, k_flash, v_flash,
            qkv_format='thd',
            attn_mask_type="no_mask",
            cu_seqlens_q=cu_seqlens, 
            cu_seqlens_kv=cu_seqlens,
            max_seqlen_q=max_seqlen, 
            max_seqlen_kv=max_seqlen,
        )


    forward_metrics = calculate_similarity(sage_output, flash_output)
    print_metrics(f"前向传播 - Sage e4m3 vs Flash Varlen \
                  (T={config.total_seq}, H={config.num_heads}, D={config.head_dim})", forward_metrics)

    grad_output = torch.randn_like(sage_output)
    # torch.cuda.synchronize() # blocking launch
    flash_output.backward(grad_output) 
    sage_output.backward(grad_output)

    print(q_sage.grad_fn)
    sage_grads=(q_sage.grad, k_sage.grad, v_sage.grad)
    flash_grads=(q_flash.grad, k_flash.grad, v_flash.grad)
    
    # validate_gradients(sage_grads, flash_grads, config)
    # validate_gradients(flash_grads, flash_api_grads, config)

    print("梯度范数 Sage VS Flash")
    print(f"Q: {sage_grads[0].norm():.4f} vs {flash_grads[0].norm():.4f}")
    print(f"K: {sage_grads[1].norm():.4f} vs {flash_grads[1].norm():.4f}")
    print(f"V: {sage_grads[2].norm():.4f} vs {flash_grads[2].norm():.4f}")

    # def validate_backward_inputs(model1, model2, in1, in2,  grad_out):
    #     """7维度反向传播输入验证"""
    #     # 维度1: 基础参数验证
    #     assert math.isclose(model1.softmax_scale, model2.softmax_scale, rel_tol=1e-6), "softmax_scale不一致"
    #     assert model1.attention_dropout == model2.attention_dropout, "dropout率不一致"
        
    # validate_backward_inputs(sage_dpa, flash_dpa, q_sage, q_flash_api, grad_output)
    
    grad_metrics = {
        'q': calculate_similarity(q_sage.grad, q_flash.grad),
        'k': calculate_similarity(k_sage.grad, k_flash.grad),
        'v': calculate_similarity(v_sage.grad, v_flash.grad)
    }
    # def scale_gradients(flash_grad, sage_grad):
    #     flash_range = flash_grad.max() - flash_grad.min()
    #     sage_range = sage_grad.max() - sage_grad.min()
    #     if flash_range > 0:
    #         scale_factor = sage_range / flash_range
    #         return flash_grad * scale_factor
    #     return flash_grad

    # q_flash_scaled = scale_gradients(q_flash.grad, q_sage.grad)
    # k_flash_scaled = scale_gradients(k_flash.grad, k_sage.grad)
    # v_flash_scaled = scale_gradients(v_flash.grad, v_sage.grad)


    # visualize_comparison(q_sage, q_flash, f"Q Input Comparison (hd={config.head_dim})")
    # visualize_comparison(k_sage, k_flash, f"K Input Comparison (hd={config.head_dim})")
    # visualize_comparison(v_sage, v_flash, f"V Input Comparison (hd={config.head_dim})")
    
    # visualize_comparison(sage_output, flash_output, 
    #                     f"Sage vs Flash Output (hd={config.head_dim})")
    
    if q_sage.grad is not None and q_flash.grad is not None:
        visualize_comparison(q_sage.grad, q_flash.grad, 
                            f"Q Gradients Original Comparison (hd={config.head_dim})")
        visualize_comparison(k_sage.grad, k_flash.grad, 
                            f"K Gradients Original Comparison (hd={config.head_dim})")
        visualize_comparison(v_sage.grad, v_flash.grad, 
                            f"V Gradients Original Comparison (hd={config.head_dim})")
        
        # visualize_comparison(q_sage.grad, q_flash_scaled, 
        #                     f"Q Gradients Scaled Comparison (hd={config.head_dim})")
        # visualize_comparison(k_sage.grad, k_flash_scaled, 
        #                     f"K Gradients Scaled Comparison (hd={config.head_dim})")
        # visualize_comparison(v_sage.grad, v_flash_scaled, 
        #                     f"V Gradients Scaled Comparison (hd={config.head_dim})")
    
    # # Heatmap visualizations
    # visualize_heatmap(q_sage, f"Q_Sage_Heatmap (hd={config.head_dim})")
    # visualize_heatmap(q_flash, f"Q_Flash_Heatmap (hd={config.head_dim})")
    # visualize_heatmap(sage_output, f"Sage_Output_Heatmap (hd={config.head_dim})")
    # visualize_heatmap(flash_output, f"Flash_Output_Heatmap (hd={config.head_dim})")
    
    # if q_sage.grad is not None:
    #     visualize_heatmap(q_sage.grad, f"Q_Sage_Grad_Heatmap (hd={config.head_dim})")
    #     visualize_heatmap(k_sage.grad, f"K_Sage_Grad_Heatmap (hd={config.head_dim})")
    #     visualize_heatmap(v_sage.grad, f"V_Sage_Grad_Heatmap (hd={config.head_dim})")
    
    # if q_flash.grad is not None:
    #     visualize_heatmap(q_flash.grad, f"Q_Flash_Grad_Heatmap (hd={config.head_dim})")
    #     visualize_heatmap(k_flash.grad, f"K_Flash_Grad_Heatmap (hd={config.head_dim})")
    #     visualize_heatmap(v_flash.grad, f"V_Flash_Grad_Heatmap (hd={config.head_dim})")
    
    
    logger.add_result(config, 'e4m3', forward_metrics, grad_metrics, "Varlen Forward+Backward")

if __name__ == "__main__":
    # for config in test_var_configs[:2]:  
    #     print(f"Running var test for config: {config}")
    #     run_var_test(config)
    
    # run_backward_tests_for_selected_configs()
    
    run_var_backward_tests_for_selected_configs()
    
    logger.save("sage_attn_test_with_backward")