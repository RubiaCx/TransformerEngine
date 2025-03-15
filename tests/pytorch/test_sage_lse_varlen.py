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
from flash_attn.flash_attn_interface import _flash_attn_forward
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

def create_tensors(config, q_range, k_range, v_range):
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
        return t.to(config.dtype).contiguous()  

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

# base_configs = [
#     # (batch_size, num_heads, seq_len, head_dim, layout)
#     (1, 4, 512, 64, 'bhsd'),
#     (1, 4, 512, 64, 'bshd'), 
#     (4, 4, 512, 64, 'bhsd'),
#     (4, 4, 512, 64, 'bshd'),
#     (8, 4, 512, 64, 'bhsd'),
#     (8, 4, 512, 64, 'bshd'),
#     (16, 16, 2048, 128, 'bhsd'),
#     (16, 16, 2048, 128, 'bshd'),
#     (32, 8, 4096, 64, 'bhsd'),
#     (32, 8, 4096, 64, 'bshd'),
# ]
range_combinations = [
    # q_range, k_range, v_range
    (0.1, 0.1, 0.1),     
    (1.0, 1.0, 1.0),    
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
            'Max Diff', 'Mean Diff', 'Cos Sim'
        ]
        self.results = []
    
    def add_result(self, config, qtype, metrics):
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
            metrics['cos_sim'].item()
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

def run_var_test(config):
    tols = {"atol": 1e-2, "rtol": 1e-2}
    scale = 1.0 / (config.head_dim ** 0.5)
    cu_seqlens_q = cu_seqlens_kv = torch.tensor([0, 300, 1100, 2048], device="cuda", dtype=torch.int32)

    sage_int8 = SageAttention(
        softmax_scale=scale,
        quantization_backend="triton",
        quantization_type="int8",
        smooth_k=True,
        return_lse=True,
        attention_dropout=0.0,
    ).cuda()
    sage_int8.eval()
    sage_e4m3 = SageAttention(
        softmax_scale=scale,
        quantization_backend="triton",
        quantization_type="e4m3",
        smooth_k=True,
        return_lse=True,
        attention_dropout=0.0,
    ).cuda()
    sage_e4m3.eval()
    sage_e5m2 = SageAttention(
        softmax_scale=scale,
        quantization_backend="triton",
        quantization_type="e5m2",
        smooth_k=True,
        return_lse=True,
        attention_dropout=0.0,
    ).cuda()
    sage_e5m2.eval()

    q, k, v = create_tensors(config, config.q_range, config.k_range, config.v_range)
    
    sage_int8_output, sage_int8_lse = sage_int8(
        q, k, v,
        qkv_layout="thd",
        attn_mask_type="no_mask",
        cu_seqlens_q=cu_seqlens_q, 
        cu_seqlens_kv=cu_seqlens_kv,
        max_seqlen_q=config.total_seq,
        max_seqlen_kv=config.total_seq,
    )
    
    sage_e4m3_output, sage_e4m3_lse = sage_e4m3(
        q, k, v,
        qkv_layout="thd",
        attn_mask_type="no_mask",
        cu_seqlens_q=cu_seqlens_q, 
        cu_seqlens_kv=cu_seqlens_kv,
        max_seqlen_q=config.total_seq,
        max_seqlen_kv=config.total_seq,
    )

    sage_e5m2_output, sage_e5m2_lse = sage_e5m2(
        q, k, v,
        qkv_layout="thd",
        attn_mask_type="no_mask",
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
    
    # os.environ["NVTE_FUSED_ATTN"] = "0"
    # os.environ["NVTE_SAGE_ATTN"] = "1"
    # os.environ["NVTE_UNFUSED_ATTN"] = "0"
    # os.environ["NVTE_FLASH_ATTN"] = "0"
    # dpa_output_2 = attention_kernel(q, k, v, qkv_format='thd', attn_mask_type="no_mask", cu_seqlens_q=cu_seqlens_q, cu_seqlens_kv=cu_seqlens_kv)

    # flash_output = flash_output.reshape(flash_output.size(0), flash_output.size(1), -1).contiguous()

    metrics = calculate_similarity(sage_int8_output, flash_output)
    print_metrics(f"Sage int8 vs Flash (T={config.total_seq}, H={config.num_heads}, D={config.head_dim})", metrics)

    metrics_e4m3 = calculate_similarity(sage_e4m3_output, flash_output)
    print_metrics(f"Sage e4m3 vs Flash (T={config.total_seq}, H={config.num_heads}, D={config.head_dim})", metrics_e4m3)

    metrics_e5m2 = calculate_similarity(sage_e5m2_output, flash_output)
    print_metrics(f"Sage e5m2 vs Flash (T={config.total_seq}, H={config.num_heads}, D={config.head_dim})", metrics_e5m2)

    lse_int8 = calculate_similarity(sage_int8_lse, sage_e4m3_lse)
    print_metrics(f"Sage int8 lse vs Sage e4m3 lse (T={config.total_seq}, H={config.num_heads}, D={config.head_dim})", lse_int8)


    # metrics_dpa_2 = calculate_similarity(dpa_output_2, flash_output)
    # print_metrics(f"DPA 2 vs Flash (T={config.total_seq}, H={config.num_heads}, D={config.head_dim})", metrics_dpa_2)


for config in test_var_configs:
    print(f"Running var test for config: {config}")
    run_var_test(config)

# for config in test_configs:
#     print(f"Running fixlen test for config: {config}")
#     run_test(config)

# logger.save()

