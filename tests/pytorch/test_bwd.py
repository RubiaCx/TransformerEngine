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

from flash_attn.flash_attn_interface import flash_attn_func, _flash_attn_forward, _flash_attn_varlen_backward
import torch
import os
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
"""将所有方法的梯度指标并列显示，以第一个为基准"""
def compare_all_methods(grads_dict, title):
    print(f"\n{title}:")
    print("-" * 80)
    
    methods = list(grads_dict.keys())
    if len(methods) < 2:
        print("至少需要两种方法进行比较")
        return
    
    base_method = methods[0]  
    base_grad = grads_dict[base_method].flatten().float()
    base_norm = base_grad.norm(p=2).item()
    
    print(f"{'指标':<12} | {base_method:<10}", end="")
    for method in methods[1:]:
        print(f" | {method:<10}", end="")
    print()
    print("-" * 80)
    
    print(f"{'L2范数':<12} | {base_norm:.4e}", end="")
    for method in methods[1:]:
        curr_grad = grads_dict[method].flatten().float()
        curr_norm = curr_grad.norm(p=2).item()
        print(f" | {curr_norm:.2e}", end="")
    print()
    
    print(f"{'L2比率':<12} | {'--':<10}", end="")
    for method in methods[1:]:
        curr_grad = grads_dict[method].flatten().float()
        curr_norm = curr_grad.norm(p=2).item()
        ratio = curr_norm / (base_norm + 1e-8)
        status = "✓" if 0.9 < ratio < 1.1 else "✗"
        print(f" | {ratio:.4f} {status}", end="")
    print()
    
    print(f"{'余弦相似度':<9} | {'--':<10}", end="")
    for method in methods[1:]:
        curr_grad = grads_dict[method].flatten().float()
        cos_sim = F.cosine_similarity(base_grad, curr_grad, dim=0).item()
        status = "✓" if cos_sim > 0.95 else "✗"
        print(f" | {cos_sim:.4f} {status}", end="")
    print()
    
    print(f"{'最大差异':<10} | {'--':<10}", end="")
    for method in methods[1:]:
        curr_grad = grads_dict[method].flatten().float()
        max_diff = torch.max(torch.abs(base_grad - curr_grad)).item()
        print(f" | {max_diff:.2e}", end="")
    print()
    
    print(f"{'平均差异':<10} | {'--':<10}", end="")
    for method in methods[1:]:
        curr_grad = grads_dict[method].flatten().float()
        mean_diff = torch.mean(torch.abs(base_grad - curr_grad)).item()
        print(f" | {mean_diff:.2e}", end="")
    print()
    print("-" * 80)


# bshd
def run_fix_backward_test(config):
    tols = {"atol": 1e-2, "rtol": 1e-2}
    scale = 1.0 / (config.head_dim ** 0.5)
    if config.layout == "bshd":
        base_tensor = torch.empty(config.batch_size, config.seq_len, config.num_heads, config.head_dim,
                              device="cuda", dtype=torch.float16).normal_(0, 0.02)
        
    elif config.layout == "sbhd":
        base_tensor = torch.empty(config.seq_len, config.batch_size, config.num_heads, config.head_dim,
                              device="cuda", dtype=torch.float16).normal_(0, 0.02)
    
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

    with EnvSwitcher({"NVTE_SAGE_ATTN": "1", "NVTE_FLASH_ATTN": "0"}):
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