import torch
from transformer_engine.pytorch.attention import (
    SageAttention,
    FlashAttention,
    UnfusedDotProductAttention,
)

from flash_attn.flash_attn_interface import _flash_attn_forward
import torch.nn.functional as F
from datetime import datetime
import pandas as pd

torch.backends.cuda.enable_flash_sdp(False)   
torch.backends.cuda.enable_math_sdp(True)  
from torch.nn.functional import scaled_dot_product_attention as sdpa

torch.manual_seed(2025)
torch.set_printoptions(precision=4, threshold=None, edgeitems=None, linewidth=None, profile=None, sci_mode=False)
#todo INT8 LSE 精度 diff
class TestConfig:
    def __init__(self, batch_size, num_heads, seq_len, head_dim, 
                 dtype=torch.float16, value_range=1.0, layout='bhsd', attn_mask_type='no_mask'):
        self.batch_size = batch_size
        self.num_heads = num_heads
        self.seq_len = seq_len
        self.head_dim = head_dim
        self.dtype = dtype
        self.value_range = value_range
        self.layout = layout 
        self.attn_mask_type = attn_mask_type

def create_tensors(config):
    if config.layout == 'bhsd':
        shape = (config.batch_size, config.num_heads, config.seq_len, config.head_dim)
    elif config.layout == 'bshd':
        shape = (config.batch_size, config.seq_len, config.num_heads, config.head_dim)
    elif config.layout == 'sbhd':
        shape = (config.seq_len, config.batch_size, config.num_heads, config.head_dim)
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
    # (batch_size, num_heads, seq_len, head_dim, layout, attn_mask_type)
    # (1, 24, 16000, 128, 'bhsd', 'no_mask'),
    # (1, 24, 72000, 128, 'bhsd', 'no_mask'),
    # (1, 32, 16000, 128, 'bhsd', 'no_mask'),
    # (1, 32, 72000, 128, 'bhsd', 'no_mask'),
    # (1, 24, 16000, 128, 'bhsd', 'causal'),
    # (1, 24, 72000, 128, 'bhsd', 'causal'),
    # (1, 32, 16000, 128, 'bhsd', 'causal'),
    # (1, 32, 72000, 128, 'bhsd', 'causal'),
    (1, 2, 64, 32, 'bhsd', 'no_mask'),
    # (1, 1, 64, 32, 'bhsd', 'causal'),
    # (1, 4, 512, 64, 'bhsd', 'no_mask'),
    # (1, 4, 512, 64, 'bhsd', 'causal'),
    # (8, 4, 512, 64, 'bhsd', 'no_mask'),
    # (16, 16, 2048, 128, 'bhsd', 'causal'),
    # (16, 16, 2048, 128, 'bhsd', 'no_mask'),
    # (32, 8, 4096, 64, 'bhsd', 'causal'),
    # (32, 8, 4096, 64, 'bhsd', 'no_mask'),
]

test_configs = []
for bs, h, s, d, layout, mask_type in base_configs:
    for dtype in [torch.float16, torch.bfloat16]:
        for value_range in [0.1]: #, 1.0, 10.0]:
            test_configs.append(
                TestConfig(
                    batch_size=bs,
                    num_heads=h,
                    seq_len=s,
                    head_dim=d,
                    dtype=dtype,
                    value_range=value_range,
                    layout=layout,
                    attn_mask_type=mask_type
                )
            )

class ResultLogger:
    def __init__(self):
        self.columns = [
            'Quant Type', 
            'Batch Size', 'Num Heads', 'Seq Len', 'Head Dim', 
            'Data Type', 
            'Layout', 
            'Mask Type', 
            'Value Range', 
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
            config.layout.upper(),  
            config.attn_mask_type,
            config.value_range,
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
        return_lse=True,
        attention_dropout=0.1,
    ).cuda()
    sage_int8.eval()
    sage_e4m3 = SageAttention(
        softmax_scale=scale,
        quantization_backend="triton",
        quantization_type="e4m3",
        smooth_k=True,
        return_lse=True,
        attention_dropout=0.1,
    ).cuda()
    sage_e4m3.eval()
    sage_e5m2 = SageAttention(
        softmax_scale=scale,
        quantization_backend="triton",
        quantization_type="e5m2",
        smooth_k=True,
        return_lse=True,
        attention_dropout=0.1,
    ).cuda()
    sage_e5m2.eval()
    q, k, v = create_tensors(config)
    
    sage_int8_output, lse_int8 = sage_int8(
        q, k, v,
        qkv_layout=f"{config.layout}_{config.layout}_{config.layout}",
        attn_mask_type=config.attn_mask_type
    )
    
    sage_e4m3_output, lse_e4m3 = sage_e4m3(
        q, k, v,
        qkv_layout=f"{config.layout}_{config.layout}_{config.layout}",
        attn_mask_type=config.attn_mask_type
    )

    sage_e5m2_output, lse_e5m2 = sage_e5m2(
        q, k, v,
        qkv_layout=f"{config.layout}_{config.layout}_{config.layout}",
        attn_mask_type=config.attn_mask_type
    )

    if config.layout == 'bhsd': # [b,h,s,d] -> [b,s,h,d]
        q_flash = q.transpose(1, 2).contiguous()  
        k_flash = k.transpose(1, 2).contiguous()
        v_flash = v.transpose(1, 2).contiguous()
    else:  
        q_flash = q.contiguous()  
        k_flash = k.contiguous()
        v_flash = v.contiguous()

    flash_output, _, _, _, _, softmax_lse, _, _ = _flash_attn_forward(
        q, k, v,
        dropout_p=0.1,
        softmax_scale=scale,
        causal=bool("causal" in config.attn_mask_type),
        window_size=(-1, -1),
        alibi_slopes=None,
        return_softmax=True,
    )
    flash_output = flash_output.permute(0, 2, 1, 3)
    flash_output = flash_output.reshape(flash_output.size(0), flash_output.size(1), -1).contiguous()
    print("flash_output", flash_output.shape)
    print("sage_int8_output", sage_int8_output.shape)

    print("softmax_lse", softmax_lse.shape)
    print("lse_int8", lse_int8.shape)
    
    metrics4 = calculate_similarity(lse_int8, lse_e4m3)
    metrics5 = calculate_similarity(lse_int8, softmax_lse)
    print("metrics4", metrics4)
    print("metrics5", metrics5)
    logger.add_result(config, 'int8 vs e4m3', metrics4)
    logger.add_result(config, 'int8 vs flash', metrics5)

    metrics6 = calculate_similarity(sage_int8_output, flash_output)
    metrics7 = calculate_similarity(sage_e4m3_output, flash_output)
    metrics8 = calculate_similarity(sage_e5m2_output, flash_output)
    print("metrics6", metrics6)
    print("metrics7", metrics7)
    print("metrics8", metrics8)
    logger.add_result(config, 'int8 vs flash', metrics6)
    logger.add_result(config, 'e4m3 vs flash', metrics7)
    logger.add_result(config, 'e5m2 vs flash', metrics8)
for config in test_configs:
    run_test(config)

# logger.save()

