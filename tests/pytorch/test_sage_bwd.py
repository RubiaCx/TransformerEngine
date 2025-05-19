# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
import torch
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

from flash_attn.flash_attn_interface import flash_attn_func, _flash_attn_forward, _flash_attn_varlen_backward

import os
import math
# 添加必要的导入
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import torch

from flash_attn import flash_attn_func

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
    
    tensor1_flat = tensor1.detach().cpu().flatten().float().numpy()
    tensor2_flat = tensor2.detach().cpu().flatten().float().numpy()
    
    if tensor1_flat.size > max_items:
        indices = np.random.choice(tensor1_flat.size, max_items, replace=False)
        tensor1_flat = tensor1_flat[indices]
        tensor2_flat = tensor2_flat[indices]
    
    diff = tensor1_flat - tensor2_flat
    
    plt.subplot(2, 2, 1)
    plt.hist(tensor1_flat, bins=50, alpha=0.7, label='Tensor1')
    plt.hist(tensor2_flat, bins=50, alpha=0.7, label='Tensor2')
    plt.title(f"{title} - Distribution Comparison")
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 2, 2)
    plt.scatter(tensor1_flat, tensor2_flat, alpha=0.5, s=10)
    max_val = max(tensor1_flat.max(), tensor2_flat.max())
    min_val = min(tensor1_flat.min(), tensor2_flat.min())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    plt.title(f"{title} - Correlation Scatter")
    plt.xlabel('Tensor1')
    plt.ylabel('Tensor2')
    plt.grid(True)
    
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
    plt.figure(figsize=figsize)
    
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

def validate_gradients(sage_grads, flash_grads, config):
    for i, (g1, g2) in enumerate(zip(sage_grads, flash_grads)):
        assert g1.shape == g2.shape, f"梯度维度不匹配: 层{i} {g1.shape} vs {g2.shape}"
        
    for grad in sage_grads + flash_grads:
        assert torch.isfinite(grad).all(), "梯度包含非法值(NaN/Inf)"

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

def calculate_metrics(dict):
    metrics = {}
    for key1, grad1 in dict.items():
        for key2, grad2 in dict.items():
            if key1 >= key2:
                continue
            flat1, flat2 = grad1.flatten().float(), grad2.flatten().float()
            
            # 基础统计
            l2_norm1 = flat1.norm(p=2)
            l2_norm2 = flat2.norm(p=2)
            l2_diff = (flat1 - flat2).norm(p=2)
            l2_div = l2_diff / (l2_norm1 + 1e-8)
            cos_sim = F.cosine_similarity(flat1, flat2, dim=0)
            
            # 差异统计
            abs_diff = torch.abs(flat1 - flat2)
            max_diff = abs_diff.max()
            mean_diff = abs_diff.mean()
            median_diff = abs_diff.median()
            p95_diff = abs_diff.kthvalue(int(0.95 * abs_diff.numel())).values
            
            # 相对差异
            rel_diff = abs_diff / (torch.abs(flat1) + 1e-8)
            max_rel = rel_diff.max()
            mean_rel = rel_diff.mean()
            
            metrics[f"{key1}_vs_{key2}"] = {
                'L2范数': (l2_norm1.item(), l2_norm2.item()),
                'L2散度': l2_div.item(),
                '余弦相似度': cos_sim.item(),
                '绝对差异': {
                    'max': max_diff.item(),
                    'mean': mean_diff.item(),
                    'median': median_diff.item(),
                    'p95': p95_diff.item()
                },
                '相对差异(%)': {
                    'max': max_rel.item() * 100,
                    'mean': mean_rel.item() * 100
                }
            }
    return metrics

from termcolor import colored
def print_comparison(metrics, threshold=0.95):
    """打印格式化的梯度对比报告"""
    
    for comp_key, data in metrics.items():
        name1, name2 = comp_key.split('_vs_')
        print(f"\n{colored('='*40 + f' {name1} vs {name2} ' + '='*40, 'cyan')}")
        
        # L2范数对比
        l2_1, l2_2 = data['L2范数']
        l2_ratio = l2_2 / (l2_1 + 1e-8)
        l2_status = colored(f"{l2_ratio:.2%}", 'green' if 0.9 < l2_ratio < 1.1 else 'red')
        print(f"{'L2范数比 (目标/参考)':<25}: {l2_1:.4e} vs {l2_2:.4e} → {l2_status}")
        
        # 余弦相似度
        cos_sim = data['余弦相似度']
        cos_color = 'green' if cos_sim >= threshold else 'red'
        print(f"{'余弦相似度':<25}: {colored(f'{cos_sim:.6f}', cos_color)}")
        
        # 差异统计
        print("\n绝对差异统计:")
        diff = data['绝对差异']
        print(f"  {'最大差异':<20}: {diff['max']:.4e}")
        print(f"  {'平均差异':<20}: {diff['mean']:.4e}")
        print(f"  {'中位数差异':<20}: {diff['median']:.4e}")
        print(f"  {'P95差异':<20}: {diff['p95']:.4e}")
        
        # 相对差异
        print("\n相对差异统计:")
        rel = data['相对差异(%)']
        print(f"  {'最大相对差异':<20}: {rel['max']:.2f}%")
        print(f"  {'平均相对差异':<20}: {rel['mean']:.2f}%")
        
        # L2散度
        print(f"\n{'L2散度':<25}: {data['L2散度']:.4e}")

# bshd
def run_fix_backward_test(config):
    tols = {"atol": 1e-2, "rtol": 1e-2}
    scale = 1.0 / (config.head_dim ** 0.5)
    if config.layout == "bshd":
        base_tensor = torch.empty(config.batch_size, config.seq_len, config.num_heads, config.head_dim,
                              device="cuda", dtype=torch.float16).normal_(0, 0.02) * config.q_range
        
    elif config.layout == "sbhd":
        base_tensor = torch.empty(config.seq_len, config.batch_size, config.num_heads, config.head_dim,
                              device="cuda", dtype=torch.float16).normal_(0, 0.02) * config.q_range
    
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

    reload(te_attention) 
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

    forward_1 = calculate_similarity(sage_output, flash_output)
    print_metrics(f"前向传播 - Sage e4m3 vs Flash \
                   (B={config.batch_size}, S={config.seq_len}, H={config.num_heads}, D={config.head_dim})", forward_1)
    forward_2 = calculate_similarity(sage_output, fused_output)
    print_metrics(f"前向传播 - Sage e4m3 vs Fused \
                   (B={config.batch_size}, S={config.seq_len}, H={config.num_heads}, D={config.head_dim})", forward_2)
    forward_3 = calculate_similarity(flash_output, fused_output)
    print_metrics(f"前向传播 - Flash vs Fused \
                   (B={config.batch_size}, S={config.seq_len}, H={config.num_heads}, D={config.head_dim})", forward_3)

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
        "Sage": sage_grads[0],
        "Flash": flash_grads[0],
        # "Fused": fused_grads[0]
    }
    grads_k = {
        "Sage": sage_grads[1],
        "Flash": flash_grads[1],
        # "Fused": fused_grads[1]
    }
    grads_v = {
        "Sage": sage_grads[2],
        "Flash": flash_grads[2],
        # "Fused": fused_grads[2]
    }
    metrics_q = calculate_metrics(grads_q)  
    metrics_k = calculate_metrics(grads_k)
    metrics_v = calculate_metrics(grads_v)
    print(f"\n{colored('='*40 + f' Q 梯度对比 ' + '='*40, 'red')}")
    print_comparison(metrics_q)
    print(f"\n{colored('='*40 + f' K 梯度对比 ' + '='*40, 'red')}")
    print_comparison(metrics_k)
    print(f"\n{colored('='*40 + f' V 梯度对比 ' + '='*40, 'red')}")
    print_comparison(metrics_v)


    # grad_metrics = {
    #     'q': calculate_similarity(q_sage.grad, q_flash.grad),
    #     'k': calculate_similarity(k_sage.grad, k_flash.grad),
    #     'v': calculate_similarity(v_sage.grad, v_flash.grad)
    # }
    # def scale_gradients(flash_grad, sage_grad):
    #     flash_range = flash_grad.max() - flash_grad.min()
    #     sage_range = sage_grad.max() - sage_grad.min()
    #     if flash_range > 0:
    #         scale_factor = sage_range / flash_range
    #         print(f"范围: {flash_range} vs {sage_range}")
    #         return flash_grad * scale_factor
    #     return flash_grad

    # q_flash_scaled = scale_gradients(q_flash.grad, q_sage.grad)
    # k_flash_scaled = scale_gradients(k_flash.grad, k_sage.grad)
    # v_flash_scaled = scale_gradients(v_flash.grad, v_sage.grad)

    # grad_metrics['q'] = calculate_similarity(q_sage.grad, q_flash_scaled)
    # grad_metrics['k'] = calculate_similarity(k_sage.grad, k_flash_scaled)
    # grad_metrics['v'] = calculate_similarity(v_sage.grad, v_flash_scaled)

    visualize_comparison(q_sage, q_flash, f"Q Input Comparison (hd={config.head_dim})")
    visualize_comparison(k_sage, k_flash, f"K Input Comparison (hd={config.head_dim})")
    visualize_comparison(v_sage, v_flash, f"V Input Comparison (hd={config.head_dim})")
    
    visualize_comparison(sage_output, flash_output, f"Sage vs Flash Output (hd={config.head_dim})")
    
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
    
    # Heatmap visualizations
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
    
    
    # logger.add_result(config, 'e4m3', forward_1, grad_metrics, "Varlen Forward+Backward")


def run_fix_backward_tests_for_selected_configs():
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
        (1.0, 1.0, 1.0), 
        (10.0, 10.0, 10.0), 
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
                                        q_range=ranges[0],
                                        k_range=ranges[1],
                                        v_range=ranges[2],
                                        layout=layout,
                                    )
                                )
    
    for i, config in enumerate(fix_backward_configs):
        print(f"\n===== 测试 {i+1}/{len(fix_backward_configs)}: 定长反向传播 (B={config.batch_size}, S={config.seq_len}, H={config.num_heads}, D={config.head_dim}, layout={config.layout}, ranges={config.q_range}-{config.k_range}-{config.v_range}) =====")
        run_fix_backward_test(config)

if __name__ == "__main__":
    
    run_fix_backward_tests_for_selected_configs()