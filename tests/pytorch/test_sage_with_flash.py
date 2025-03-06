import torch
from transformer_engine.pytorch.attention import (
    SageAttention,
    FlashAttention,
    UnfusedDotProductAttention,
)
import torch.nn.functional as F
from datetime import datetime
import pandas as pd

from flash_attn.flash_attn_interface import flash_attn_func
from flash_attn import flash_attn_varlen_func

torch.manual_seed(2025)
torch.set_printoptions(precision=4, threshold=None, edgeitems=None, linewidth=None, profile=None, sci_mode=False)

class TestConfig:
    def __init__(self, batch_size, num_heads, seq_len, head_dim, total_seq, 
                 dtype=torch.float16, value_range=1.0, layout='bhsd'):
        self.batch_size = batch_size
        self.num_heads = num_heads
        self.seq_len = seq_len
        self.head_dim = head_dim
        self.total_seq = total_seq
        self.dtype = dtype
        self.value_range = value_range
        self.layout = layout  

def create_tensors(config):
    if config.layout == 'bhsd':
        shape = (config.batch_size, config.num_heads, config.seq_len, config.head_dim)
    elif config.layout == 'bshd':
        shape = (config.batch_size, config.seq_len, config.num_heads, config.head_dim)
    elif config.layout == 'thd':
        shape = (config.total_seq, config.num_heads, config.head_dim)
    else:
        raise ValueError(f"Unsupported layout: {config.layout}")

    def generate_tensor():
        t = torch.randn(shape, device="cuda")
        t = t / t.std() * config.value_range
        return t.to(config.dtype).contiguous()  

    return generate_tensor(), generate_tensor(), generate_tensor()

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

base_configs = [
    # (batch_size, num_heads, seq_len, head_dim, layout)
    (1, 4, 512, 64, 'bshd'), 
    (4, 4, 512, 64, 'bshd'),
    (8, 4, 512, 64, 'bshd'),
    (16, 16, 2048, 128, 'bshd'),
    (32, 8, 4096, 64, 'bshd'),
]

test_configs = []
for bs, h, s, d, layout in base_configs:
    for dtype in [torch.float16, torch.bfloat16]:
        for value_range in [0.1, 1.0, 10.0, 50.0]:
            test_configs.append(
                TestConfig(
                    batch_size=bs,
                    num_heads=h,
                    seq_len=s,
                    head_dim=d,
                    total_seq=bs * s,
                    dtype=dtype,
                    value_range=value_range,
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
        for value_range in [0.1, 1.0, 10.0]:
            test_var_configs.append(
                TestConfig(
                    batch_size=0,
                    num_heads=h,
                    seq_len=0,
                    head_dim=d,
                    total_seq=t,
                    dtype=dtype,
                    value_range=value_range,
                    layout=layout
                )
            )

class ResultLogger:
    def __init__(self):
        self.columns = [
            'Quant Type', 
            'Batch Size', 'Num Heads', 'Seq Len', 'Head Dim', 'Total Seq',  
            'Data Type', 'Value Range', 'Layout', 
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
            config.value_range,
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

    sage_e4m3 = SageAttention(
        softmax_scale=scale,
        quantization_backend="triton",
        quantization_type="e4m3",
        smooth_k=True,
        return_lse=False,
        attention_dropout=0.0,
    ).cuda()

    sage_e5m2 = SageAttention(
        softmax_scale=scale,
        quantization_backend="triton",
        quantization_type="e5m2",
        smooth_k=True,
        return_lse=False,
        attention_dropout=0.0,
    ).cuda()

    flash = FlashAttention(
        softmax_scale=scale,
        attention_dropout=0.0,
    ).cuda()

    q, k, v = create_tensors(config)
    
    sage_int8_output = sage_int8(
        q, k, v,
        qkv_layout=f"{config.layout}_{config.layout}_{config.layout}",  # 动态生成布局参数
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

    metrics = calculate_similarity(sage_int8_output, flash_output)
    print_metrics(f"Sage int8 vs Flash (B={config.batch_size}, H={config.num_heads}, S={config.seq_len}, D={config.head_dim})", metrics)

    metrics_e4m3 = calculate_similarity(sage_e4m3_output, flash_output)
    print_metrics(f"Sage e4m3 vs Flash (B={config.batch_size}, H={config.num_heads}, S={config.seq_len}, D={config.head_dim})", metrics_e4m3)

    metrics_e5m2 = calculate_similarity(sage_e5m2_output, flash_output)
    print_metrics(f"Sage e5m2 vs Flash (B={config.batch_size}, H={config.num_heads}, S={config.seq_len}, D={config.head_dim})", metrics_e5m2)

    metrics_2 = calculate_similarity(flash_ans, flash_output)
    print_metrics(f"Flash vs Flash (B={config.batch_size}, H={config.num_heads}, S={config.seq_len}, D={config.head_dim})", metrics)

    logger.add_result(config, 'int8', metrics)
    logger.add_result(config, 'e4m3', metrics_e4m3)
    logger.add_result(config, 'e5m2', metrics_e5m2)
    logger.add_result(config, 'flash', metrics_2)

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

    sage_e4m3 = SageAttention(
        softmax_scale=scale,
        quantization_backend="triton",
        quantization_type="e4m3",
        smooth_k=True,
        return_lse=False,
        attention_dropout=0.0,
    ).cuda()

    sage_e5m2 = SageAttention(
        softmax_scale=scale,
        quantization_backend="triton",
        quantization_type="e5m2",
        smooth_k=True,
        return_lse=False,
        attention_dropout=0.0,
    ).cuda()

    flash = FlashAttention(
        softmax_scale=scale,
        attention_dropout=0.0,
    ).cuda()

    q, k, v = create_tensors(config)
    
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
    
    # flash_output = flash_output.reshape(flash_output.size(0), flash_output.size(1), -1).contiguous()

    metrics = calculate_similarity(sage_int8_output, flash_output)
    print_metrics(f"Sage int8 vs Flash (T={config.total_seq}, H={config.num_heads}, D={config.head_dim})", metrics)

    metrics_e4m3 = calculate_similarity(sage_e4m3_output, flash_output)
    print_metrics(f"Sage e4m3 vs Flash (T={config.total_seq}, H={config.num_heads}, D={config.head_dim})", metrics_e4m3)

    metrics_e5m2 = calculate_similarity(sage_e5m2_output, flash_output)
    print_metrics(f"Sage e5m2 vs Flash (T={config.total_seq}, H={config.num_heads}, D={config.head_dim})", metrics_e5m2)

    metrics_2 = calculate_similarity(flash_ans, flash_output)
    print_metrics(f"Flash vs FlashAPI (T={config.total_seq}, H={config.num_heads}, D={config.head_dim})", metrics)

    logger.add_result(config, 'int8', metrics)
    logger.add_result(config, 'e4m3', metrics_e4m3)
    logger.add_result(config, 'e5m2', metrics_e5m2)
    logger.add_result(config, 'flash', metrics_2)

# for config in test_var_configs:
#     print(f"Running var test for config: {config}")
#     run_var_test(config)

for config in test_configs:
    print(f"Running fixlen test for config: {config}")
    run_test(config)

logger.save()

