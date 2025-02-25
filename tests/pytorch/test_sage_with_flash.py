import torch
from transformer_engine.pytorch.attention import (
    SageAttention,
    FlashAttention,
    UnfusedDotProductAttention,
)
from sageattention import sageattn_qk_int8_pv_fp16_triton
import torch.nn.functional as F
from datetime import datetime
import pandas as pd

from flash_attn.flash_attn_interface import flash_attn_func

torch.manual_seed(2025)
torch.set_printoptions(precision=4, threshold=None, edgeitems=None, linewidth=None, profile=None, sci_mode=False)

class TestConfig:
    def __init__(self, batch_size, num_heads, seq_len, head_dim, 
                 dtype=torch.float16, value_range=1.0, layout='bhsd'):
        self.batch_size = batch_size
        self.num_heads = num_heads
        self.seq_len = seq_len
        self.head_dim = head_dim
        self.dtype = dtype
        self.value_range = value_range
        self.layout = layout  

def create_tensors(config):
    if config.layout == 'bhsd':
        shape = (config.batch_size, config.num_heads, config.seq_len, config.head_dim)
    elif config.layout == 'bshd':
        shape = (config.batch_size, config.seq_len, config.num_heads, config.head_dim)
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

base_configs = [
    # (batch_size, num_heads, seq_len, head_dim, layout)
    (1, 4, 512, 64, 'bhsd'),
    (1, 4, 512, 64, 'bshd'), 
    (4, 4, 512, 64, 'bhsd'),
    (4, 4, 512, 64, 'bshd'),
    (8, 4, 512, 64, 'bhsd'),
    (8, 4, 512, 64, 'bshd'),
    (16, 16, 2048, 128, 'bhsd'),
    (16, 16, 2048, 128, 'bshd'),
    (32, 8, 4096, 64, 'bhsd'),
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
                    dtype=dtype,
                    value_range=value_range,
                    layout=layout
                )
            )

class ResultLogger:
    def __init__(self):
        self.columns = [
            'Quant Type', 
            'Batch Size', 'Num Heads', 'Seq Len', 'Head Dim', 
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
        # df.to_excel(f"{filename}.xlsx", index=False)
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

    sage_output = sageattn_qk_int8_pv_fp16_triton(q, k, v, tensor_layout="HND", is_causal=False)

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
    
    if config.layout == 'bhsd':
        flash_output = flash_output.transpose(1, 2)  # [b,s,h,d] -> [b,h,s,d]

    metrics = calculate_similarity(sage_int8_output, flash_output)
    print_metrics(f"Sage int8 vs Flash (BS={config.batch_size}, Heads={config.num_heads})", metrics)

    metrics_e4m3 = calculate_similarity(sage_e4m3_output, flash_output)
    print_metrics(f"Sage e4m3 vs Flash (BS={config.batch_size}, Heads={config.num_heads})", metrics_e4m3)

    metrics_e5m2 = calculate_similarity(sage_e5m2_output, flash_output)
    print_metrics(f"Sage e5m2 vs Flash (BS={config.batch_size}, Heads={config.num_heads})", metrics_e5m2)

    # metrics_fp8 = calculate_similarity(sage_output, flash_output)
    # print_metrics(f"Sage fp8v1 vs Flash (BS={config.batch_size}, Heads={config.num_heads})", metrics_fp8)

    logger.add_result(config, 'int8', metrics)
    logger.add_result(config, 'e4m3', metrics_e4m3)
    logger.add_result(config, 'e5m2', metrics_e5m2)
    # logger.add_result(config, 'fp8v1', metrics_fp8)
    

for config in test_configs:
    run_test(config)

logger.save()

