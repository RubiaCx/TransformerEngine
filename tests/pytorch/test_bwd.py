# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
from transformer_engine.pytorch.attention import (
    SageAttention,
    FlashAttention,
    DotProductAttention, 
    _attention_backends,
)
import transformer_engine.pytorch.attention as te_attention
from importlib import reload

import torch.nn.functional as F
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

from flash_attn.flash_attn_interface import flash_attn_func, _flash_attn_forward, _flash_attn_varlen_backward
import torch
from termcolor import colored

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

torch.manual_seed(2025)
torch.set_printoptions(precision=4, threshold=None, edgeitems=None, linewidth=None, profile=None, sci_mode=False)

class TestConfig:
    def __init__(self, batch_size, num_heads, seq_len, head_dim, total_seq, 
                 dtype=torch.float16, q_range = 1.0, k_range = 1.0, v_range = 1.0, layout='bshd'):
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
        self.layout = layout

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

# def calculate_metrics(dict):
#     """以第一个元素为base进行对比"""
#     metrics = {}
#     # 获取第一个key作为基准
#     base_key = list(dict.keys())[0]
#     base_grad = dict[base_key].flatten().float()
    
#     for key, grad in dict.items():
#         if key == base_key:
#             continue  # 跳过基准自身
            
#         flat_grad = grad.flatten().float()
        
#         # 基础统计
#         l2_norm_base = base_grad.norm(p=2)
#         l2_norm_current = flat_grad.norm(p=2)
#         l2_ratio = (l2_norm_current / l2_norm_base).item()
#         cos_sim = F.cosine_similarity(base_grad, flat_grad, dim=0).item()
        
#         # 差异统计
#         abs_diff = torch.abs(base_grad - flat_grad)
#         max_diff = abs_diff.max().item()
#         mean_diff = abs_diff.mean().item()
        
#         # 记录关键指标
#         metrics[f"{base_key}_vs_{key}"] = {
#             'l2比率': l2_ratio,
#             '余弦相似度': cos_sim,
#             '最大差异': max_diff,
#             '平均差异': mean_diff
#         }
    
#     return metrics

# def print_comparison(metrics, title):
#     print(f"\n{title}:")
#     print("-" * 50)
    
#     for comp_key, data in metrics.items():
#         base, target = comp_key.split('_vs_')
#         l2_ratio = data['l2比率'] 
#         cos_sim = data['余弦相似度']
#         max_diff = data['最大差异']
#         mean_diff = data['平均差异']
        
#         # 显示结果状态
#         status = "✓" if cos_sim > 0.95 else "✗"
            
#         print(f"{base} → {target} [{status}]")
#         print(f"  L2比率:     {l2_ratio:.2%}")
#         print(f"  余弦相似度: {cos_sim:.6f}")
#         print(f"  最大差异:   {max_diff:.2e}")
#         print(f"  平均差异:   {mean_diff:.2e}")
#         print("-" * 30)

def compare_all_methods(tensor_dict, title):
    print(f"\n{title}:")
    print("-" * 80)
    
    methods = list(tensor_dict.keys())
    if len(methods) < 2:
        print("至少需要两种方法进行比较")
        return
    
    base_method = methods[0]  
    base_tensor = tensor_dict[base_method].flatten().float()
    base_norm = base_tensor.norm(p=2).item()
    
    print(f"{'指标':<12} | {base_method:<10}", end="")
    for method in methods[1:]:
        print(f" | {method:<10}", end="")
    print()
    print("-" * 80)
    
    print(f"{'最小值':<11}", end="")
    for method in methods:
        min_val = tensor_dict[method].min().item()
        print(f" | {min_val:10.2e}", end="")
    print()
    
    # 最大值
    print(f"{'最大值':<11}", end="")
    for method in methods:
        max_val = tensor_dict[method].max().item()
        print(f" | {max_val:10.2e}", end="")
    print()

    print(f"{'L2范数':<12} | {base_norm:.4e}", end="")
    for method in methods[1:]:
        curr_tensor = tensor_dict[method].flatten().float()
        curr_norm = curr_tensor.norm(p=2).item()
        print(f" | {curr_norm:10.2e}", end="")
    print()
    
    print(f"{'L2比率':<12} | {'--':<10}", end="")
    for method in methods[1:]:
        curr_tensor = tensor_dict[method].flatten().float()
        curr_norm = curr_tensor.norm(p=2).item()
        ratio = curr_norm / (base_norm + 1e-8)
        print(f" | {ratio:10.4f}", end="")
    print()
    
    print(f"{'余弦相似度':<9} | {'--':<10}", end="")
    for method in methods[1:]:
        curr_tensor = tensor_dict[method].flatten().float()
        cos_sim = F.cosine_similarity(base_tensor, curr_tensor, dim=0).item()
        print(f" | {cos_sim:10.4f}", end="")
    print()
    
    print(f"{'最大差异':<10} | {'--':<10}", end="")
    for method in methods[1:]:
        curr_tensor = tensor_dict[method].flatten().float()
        max_diff = torch.max(torch.abs(base_tensor - curr_tensor)).item()
        print(f" | {max_diff:10.2e}", end="")
    print()
    
    print(f"{'平均差异':<10} | {'--':<10}", end="")
    for method in methods[1:]:
        curr_tensor = tensor_dict[method].flatten().float()
        mean_diff = torch.mean(torch.abs(base_tensor - curr_tensor)).item()
        print(f" | {mean_diff:10.2e}", end="")
    print()
    print("-" * 80)

def visualize_distributions(tensor_dict, title, save_path=None):
    methods = list(tensor_dict.keys())
    all_values = []
    for method in methods:
        values = tensor_dict[method].detach().float().cpu().flatten().numpy()
        # Remove extreme outliers (beyond 3 std)
        mean, std = np.mean(values), np.std(values)
        mask = (values > mean - 3 * std) & (values < mean + 3 * std)
        filtered_values = values[mask]
        # Sample data to avoid processing too many points
        if len(filtered_values) > 10000:
            indices = np.random.choice(len(filtered_values), 10000, replace=False)
            filtered_values = filtered_values[indices]
        all_values.append(filtered_values)
    
    all_data = np.concatenate(all_values)
    if len(all_data) == 0:
        print(f"Warning: No valid data for visualization of {title}")
        return
        
    global_min, global_max = np.min(all_data), np.max(all_data)
    
    num_methods = len(methods)
    fig, axes = plt.subplots(2, num_methods, figsize=(15, 8))
    
    bins = np.linspace(global_min, global_max, 50)
    
    for i, method in enumerate(methods):
        values = tensor_dict[method].detach().float().cpu().flatten().numpy()
        if len(values) > 10000:
            indices = np.random.choice(len(values), 10000, replace=False)
            values = values[indices]
        
        mean_val = np.mean(values)
        std_val = np.std(values)
        min_val = np.min(values)
        max_val = np.max(values)
        
        axes[0, i].hist(values, bins=bins, alpha=0.7)
        axes[0, i].set_title(f"{method}", fontsize=12)
        axes[0, i].grid(True, linestyle='--', alpha=0.5)
        
        stat_text = f"Mean: {mean_val:.2e}\nStd: {std_val:.2e}\nMin: {min_val:.2e}\nMax: {max_val:.2e}"
        axes[0, i].text(0.95, 0.95, stat_text, transform=axes[0, i].transAxes,
                    verticalalignment='top', horizontalalignment='right',
                    bbox={'facecolor': 'white', 'alpha': 0.8, 'pad': 5})
        
        axes[0, i].set_xticklabels([])
    # Box plot on bottom-left
    box_data = []
    for method in methods:
        values = tensor_dict[method].detach().float().cpu().flatten().numpy()
        if len(values) > 10000:
            indices = np.random.choice(len(values), 10000, replace=False)
            values = values[indices]
        box_data.append(values)
        
    axes[1, 0].boxplot(box_data, labels=methods, vert=True, patch_artist=True, boxprops=dict(alpha=0.7), medianprops=dict(color='black'))
    axes[1, 0].set_title("Box Plot Comparison", fontsize=12)
    axes[1, 0].set_ylabel("Value", fontsize=10)
    axes[1, 0].grid(True, linestyle='--', alpha=0.7)
    axes[1, 0].tick_params(axis='x', rotation=30)
    
    # Overlay histogram for direct comparison in the remaining bottom cells
    if num_methods > 1:
        for i in range(1, num_methods):
            for j, method in enumerate(methods):
                values = tensor_dict[method].detach().float().cpu().flatten().numpy()
                if len(values) > 10000:
                    indices = np.random.choice(len(values), 10000, replace=False)
                    values = values[indices]
                
                if i == 1:  # First comparison: regular histograms
                    axes[1, i].hist(values, bins=bins, alpha=0.5, label=method)
                    axes[1, i].set_title("Overlay Comparison", fontsize=12)
                    axes[1, i].legend(fontsize=9)
                elif i == 2 and num_methods >= 3:  # Second comparison (if space): CDF
                    values = np.sort(values)
                    y = np.arange(1, len(values)+1) / len(values)
                    axes[1, i].plot(values, y, label=method, linewidth=2)
                    axes[1, i].set_title("CDF Comparison", fontsize=12)
                    axes[1, i].legend(fontsize=9)
            
            axes[1, i].grid(True, linestyle='--', alpha=0.5)
            axes[1, i].tick_params(axis='x', rotation=30)
    
    # Set common labels
    fig.text(0.5, 0.01, "Value", ha='center', fontsize=12)
    fig.text(0.01, 0.5, "Frequency", va='center', rotation='vertical', fontsize=12)
    
    plt.suptitle(f"Distribution Comparison of {title}", fontsize=16)
    plt.tight_layout(rect=[0.02, 0.03, 0.98, 0.95])
    
    # Save or display image
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

# bshd
def run_fix_backward_test(config):
    tols = {"atol": 1e-2, "rtol": 1e-2}
    scale = 1.0 / (config.head_dim ** 0.5)
    if config.layout == "bshd":
        base_tensor = torch.empty(config.batch_size, config.seq_len, config.num_heads, config.head_dim, device="cuda", dtype=config.dtype).normal_(0, 0.02) * config.q_range
        
    elif config.layout == "sbhd":
        base_tensor = torch.empty(config.seq_len, config.batch_size, config.num_heads, config.head_dim, device="cuda", dtype=config.dtype).normal_(0, 0.02) * config.q_range
    
    def clone_tensor(tensor):
        return tensor.clone().detach().requires_grad_(True)
    
    q_sage = clone_tensor(base_tensor)
    k_sage = clone_tensor(base_tensor)
    v_sage = clone_tensor(base_tensor)
    q_flash = clone_tensor(base_tensor)
    k_flash = clone_tensor(base_tensor)
    v_flash = clone_tensor(base_tensor)
    q_fused = clone_tensor(base_tensor)
    k_fused = clone_tensor(base_tensor)
    v_fused = clone_tensor(base_tensor)

    reload(te_attention) #! 强制重载模块
    with EnvSwitcher({"NVTE_SAGE_ATTN": "1", "NVTE_FLASH_ATTN": "0", "NVTE_FUSED_ATTN": "0"}):
        sage_dpa = DotProductAttention(config.num_heads, config.head_dim, softmax_scale=scale)
        sage_dpa.train()

        sage_output = sage_dpa(
            q_sage, k_sage, v_sage,
            qkv_format=config.layout,
            attn_mask_type="no_mask",
        )

    reload(te_attention) #! 强制重载模块
    with EnvSwitcher({"NVTE_SAGE_ATTN": "0", "NVTE_FLASH_ATTN": "1", "NVTE_FUSED_ATTN": "0"}):
        flash_dpa = DotProductAttention(config.num_heads, config.head_dim, softmax_scale=scale)
        flash_dpa.train()

        flash_output = flash_dpa(
            q_flash, k_flash, v_flash,
            qkv_format=config.layout,
            attn_mask_type="no_mask",
        )

    reload(te_attention) #! 强制重载模块
    with EnvSwitcher({"NVTE_SAGE_ATTN": "0", "NVTE_FLASH_ATTN": "0", "NVTE_FUSED_ATTN": "1"}):
        fused_dpa = DotProductAttention(config.num_heads, config.head_dim, softmax_scale=scale)
        fused_dpa.train()

        fused_output = fused_dpa(
            q_fused, k_fused, v_fused,
            qkv_format=config.layout,
            attn_mask_type="no_mask",
        )


    output = {
        "Flash": flash_output,
        "Sage": sage_output,
        "Fused": fused_output
    }
    compare_all_methods(output, "Forward")

    save_dir = "visualization_results"
    visualize_distributions(
        output, 
        f"Forward (B={config.batch_size}, S={config.seq_len}, H={config.num_heads}, D={config.head_dim})", 
        save_path=f"{save_dir}/forward.png"
    )

    grad_output = torch.randn_like(flash_output)
    # torch.cuda.synchronize() # blocking launch
    if config.layout == 'sbhd':
        grad_output_flash = grad_output.clone()  
        grad_output_sage = grad_output.clone()
        grad_output_fused = grad_output.clone()
        flash_output.backward(grad_output_flash)
        sage_output.backward(grad_output_sage)
        fused_output.backward(grad_output_fused)
    else:
        flash_output.backward(grad_output)
        grad_output_sage = grad_output.clone()
        sage_output = sage_output.view(*flash_output.shape)
        sage_output.backward(grad_output_sage)
        grad_output_fused = grad_output.clone()
        fused_output.backward(grad_output_fused)

    sage_grads=(q_sage.grad, k_sage.grad, v_sage.grad)
    flash_grads=(q_flash.grad, k_flash.grad, v_flash.grad)
    fused_grads=(q_fused.grad, k_fused.grad, v_fused.grad)
    
    # validate_gradients(sage_grads, flash_grads, config)
    # validate_gradients(flash_grads, flash_api_grads, config)
    grads_q = {
        "Flash": flash_grads[0],
        "Sage": sage_grads[0],
        "Fused": fused_grads[0]
    }
    grads_k = {
        "Flash": flash_grads[1],
        "Sage": sage_grads[1],
        "Fused": fused_grads[1]
    }
    grads_v = {
        "Flash": flash_grads[2],
        "Sage": sage_grads[2],
        "Fused": fused_grads[2]
    }

    compare_all_methods(grads_q, "dQ")
    compare_all_methods(grads_k, "dK")
    compare_all_methods(grads_v, "dV")
    
    visualize_distributions(
        grads_q, 
        f"Q Gradient (B={config.batch_size}, S={config.seq_len}, H={config.num_heads}, D={config.head_dim})",
        save_path=f"{save_dir}/dQ.png"
    )
    visualize_distributions(
        grads_k, 
        f"K Gradient (B={config.batch_size}, S={config.seq_len}, H={config.num_heads}, D={config.head_dim})",
        save_path=f"{save_dir}/dK.png"
    )
    visualize_distributions(
        grads_v, 
        f"V Gradient (B={config.batch_size}, S={config.seq_len}, H={config.num_heads}, D={config.head_dim})",
        save_path=f"{save_dir}/dV.png"
    )
        
if __name__ == "__main__":
    fix_backward_configs = []
    # seq_lens = [1024]
    # batch_sizes = [16]
    # num_heads = [16]
    # head_dims = [72]
    seq_lens = [1024]
    batch_sizes = [16]
    num_heads = [16]
    head_dims = [72]
    value_ranges = [
        (0.1, 0.1, 0.1), 
        # (1.0, 1.0, 1.0), 
        # (10.0, 10.0, 10.0), 
    ]
    
    for ranges in value_ranges:
        for batch_size in batch_sizes:
            for head_dim in head_dims:
                    for num_head in num_heads:
                        for seq_len in seq_lens:
                            for layout in ['sbhd']: # ,'bshd'
                                fix_backward_configs.append(
                                    TestConfig(
                                        batch_size=batch_size,
                                        num_heads=num_head,
                                        seq_len=seq_len, 
                                        head_dim=head_dim,
                                        total_seq=seq_len * batch_size,
                                        dtype= torch.float16,
                                        # dtype= torch.bfloat16,
                                        q_range=ranges[0],
                                        k_range=ranges[1],
                                        v_range=ranges[2],
                                        layout=layout,
                                    )
                                )
    
    for i, config in enumerate(fix_backward_configs):
        print(f"\n===== 测试 {i+1}/{len(fix_backward_configs)}: 定长反向传播 (B={config.batch_size}, S={config.seq_len}, H={config.num_heads}, D={config.head_dim}, layout={config.layout}, ranges={config.q_range}-{config.k_range}-{config.v_range}) =====")
        run_fix_backward_test(config)