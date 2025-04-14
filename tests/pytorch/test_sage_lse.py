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
import numpy as np

torch.backends.cuda.enable_flash_sdp(False)   
torch.backends.cuda.enable_math_sdp(True)  
from torch.nn.functional import scaled_dot_product_attention as sdpa

torch.manual_seed(2025)
torch.set_printoptions(precision=4, threshold=None, edgeitems=None, linewidth=None, profile=None, sci_mode=False)
#todo INT8 LSE 精度 diff
class TestConfig:
    def __init__(self, batch_size, num_heads, seq_len, head_dim, 
                 dtype=torch.float16, layout='bhsd', attn_mask_type='no_mask'):
        self.batch_size = batch_size
        self.num_heads = num_heads
        self.seq_len = seq_len
        self.head_dim = head_dim
        self.dtype = dtype
        self.layout = layout 
        self.attn_mask_type = attn_mask_type
        self.value_min = -3.0  
        self.value_max = 3.0 

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
        if hasattr(config, 'value_min') and hasattr(config, 'value_max'):
            t = (t - t.min()) / (t.max() - t.min()) * (config.value_max - config.value_min) + config.value_min
        return t.to(config.dtype).contiguous()

    return generate_tensor(), generate_tensor(), generate_tensor()

def compute_statistics(tensor):
    return {
        'min': tensor.min().item(),
        'max': tensor.max().item(),
        'mean': tensor.mean().item(),
        'std': tensor.std().item()
    }

def calculate_similarity(output1, output2):
    output1 = output1.flatten().float()
    output2 = output2.flatten().float()
    if torch.isinf(output1).any() or torch.isnan(output1).any():
        print(f"Warning: First tensor contains inf/nan values")
    if torch.isinf(output2).any() or torch.isnan(output2).any():
        print(f"Warning: Second tensor contains inf/nan values")

    mask = ~(torch.isinf(output1) | torch.isnan(output1) | torch.isinf(output2) | torch.isnan(output2))
    if mask.sum() == 0: 
        return {
            'max_diff': float('inf'),
            'mean_diff': float('inf'),
            'cos_sim': float('nan')
        }
    
    valid_output1 = output1[mask]
    valid_output2 = output2[mask]
    
    return {
        'max_diff': torch.max(torch.abs(valid_output1 - valid_output2)).item(),
        'mean_diff': torch.mean(torch.abs(valid_output1 - valid_output2)).item(),
        'cos_sim': torch.nn.functional.cosine_similarity(valid_output1, valid_output2, dim=0).item()
        if valid_output1.numel() > 0 else float('nan')
    }

base_configs = [
    # (batch_size, num_heads, seq_len, head_dim, layout, attn_mask_type)
    # (1, 4, 128, 64, 'bshd', 'no_mask'),
    # (8, 8, 256, 64, 'sbhd', 'no_mask'),
    
    # (4, 8, 1024, 64, 'bshd', 'no_mask'),
    # (4, 16, 1024, 64, 'sbhd', 'no_mask'),
    
    # (1, 32, 2048, 128, 'bshd', 'no_mask'),
    (1, 32, 2048, 128, 'sbhd', 'no_mask'),
    
    # (2, 8, 256, 80, 'bshd', 'no_mask'),
    # (2, 8, 256, 80, 'sbhd', 'no_mask'),
    # (16, 16, 1024,  72, 'sbhd', 'no_mask'),
    # (16, 16, 1024,  72, 'bshd', 'no_mask'),
]

test_configs = []
for bs, h, s, d, layout, mask_type in base_configs:
    for dtype in [torch.float16, torch.bfloat16]:
        config = TestConfig(
            batch_size=bs,
            num_heads=h,
            seq_len=s,
            head_dim=d,
            dtype=dtype,
            layout=layout,
            attn_mask_type=mask_type
        )
        test_configs.append(config)

        ranges = [
            # (-7.0, 7.0),  
            # (-10.0, 10.0), 
            # (-10.0, 10.0), 
            # (0.0, 5.0),  
            # (-5.0, 0.0) 
            (0.0, 50.0)
        ]
        
        for min_val, max_val in ranges:
            range_config = TestConfig(
                batch_size=bs,
                num_heads=h,
                seq_len=s,
                head_dim=d,
                dtype=dtype,
                layout=layout,
                attn_mask_type=mask_type
            )
            range_config.value_min = min_val
            range_config.value_max = max_val
            test_configs.append(range_config)

class ResultLogger:
    def __init__(self):
        self.columns = [
            'Quant Type', 
            'Batch Size', 'Num Heads', 'Seq Len', 'Head Dim', 
            'Data Type', 'Layout', 'Mask Type', 
            'Value Range', 
            'LSE Min', 'LSE Max',
            'Flash Min', 'Flash Max',
            'LSE Max Diff', 'LSE Mean Diff', 'LSE Cosine Sim',
            'Out Max Diff', 'Out Mean Diff', 'Out Cosine Sim'
        ]
        self.results = []
    
    def add_result(self, config, qtype, lse_stats, flash_stats, lse_similarity, output_similarity):
        value_range = f"{config.value_min:.1f}~{config.value_max:.1f}"
            
        self.results.append([
            qtype, 
            config.batch_size,
            config.num_heads,
            config.seq_len,
            config.head_dim,
            str(config.dtype).split('.')[-1],
            config.layout,  
            config.attn_mask_type,
            value_range,
            lse_stats['min'],
            lse_stats['max'],
            flash_stats['min'],
            flash_stats['max'],
            lse_similarity['max_diff'],
            lse_similarity['mean_diff'],
            lse_similarity['cos_sim'],
            output_similarity['max_diff'],
            output_similarity['mean_diff'],
            output_similarity['cos_sim']
        ])
    
    def save(self, filename="sage_lse_results"):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{filename}_{timestamp}"
        df = pd.DataFrame(self.results, columns=self.columns)
        df.to_excel(f"{filename}.xlsx", index=False)
        with open(f"{filename}.txt", "w", encoding='utf-8') as f:
            f.write(df.to_string(index=False))
        print(f"Results saved to {filename}.[xlsx|txt]")

logger = ResultLogger()

def run_test(config):
    """运行测试"""
    print(f"\nTest Config: {config.layout} {config.batch_size}x{config.num_heads}x{config.seq_len}x{config.head_dim} "
          f"{config.dtype} {config.attn_mask_type} Value Range:[{config.value_min:.1f}, {config.value_max:.1f}]")
    
    scale = 1.0 / (config.head_dim ** 0.5)
    
    sage_models = {
        "int8": SageAttention(
            softmax_scale=scale,
            quantization_backend="triton",
            quantization_type="int8",
            smooth_k=True,
            return_lse=True,
            attention_dropout=0.1,
        ).cuda().eval(),
        
        "e4m3": SageAttention(
            softmax_scale=scale,
            quantization_backend="triton",
            quantization_type="e4m3",
            smooth_k=True,
            return_lse=True,
            attention_dropout=0.1,
        ).cuda().eval(),
        
        "e5m2": SageAttention(
            softmax_scale=scale,
            quantization_backend="triton",
            quantization_type="e5m2",
            smooth_k=True,
            return_lse=True,
            attention_dropout=0.1,
        ).cuda().eval(),
        
        "none": SageAttention(
            softmax_scale=scale,
            quantization_backend="triton",
            quantization_type="none",
            smooth_k=True,
            return_lse=True,
            attention_dropout=0.1,
        ).cuda().eval()
    }
    
    q, k, v = create_tensors(config)
    
    sage_results = {}
    for qtype, model in sage_models.items():
        output, lse = model(
            q, k, v,
            qkv_layout=f"{config.layout}_{config.layout}_{config.layout}",
            attn_mask_type=config.attn_mask_type
        )
        print(f"lse type: {lse.dtype}")
        if torch.isinf(lse).any() or torch.isnan(lse).any():
            print(f"Warning: {qtype} LSE contains inf/nan values")
        if torch.isinf(output).any() or torch.isnan(output).any():
            print(f"Warning: {qtype} output contains inf/nan values")
        sage_results[qtype] = {"output": output, "lse": lse}

    
    # 转换布局以适配Flash Attention
    if config.layout == 'bhsd':
        q_flash = q.transpose(1, 2).contiguous()  
        k_flash = k.transpose(1, 2).contiguous()
        v_flash = v.transpose(1, 2).contiguous()
    elif config.layout == 'bshd':
        q_flash = q.contiguous()  
        k_flash = k.contiguous()
        v_flash = v.contiguous()
    elif config.layout == 'sbhd':
        q_flash = q.permute(1, 0, 2, 3).contiguous()
        k_flash = k.permute(1, 0, 2, 3).contiguous()
        v_flash = v.permute(1, 0, 2, 3).contiguous()
    
    # 运行Flash Attention作为参考
    flash_output, _, _, _, _, softmax_lse, _, _ = _flash_attn_forward(
        q_flash, k_flash, v_flash,
        dropout_p=0.1,
        softmax_scale=scale,
        causal=bool("causal" in config.attn_mask_type),
        window_size=(-1, -1),
        alibi_slopes=None,
        return_softmax=True,
    )
    if torch.isinf(softmax_lse).any() or torch.isnan(softmax_lse).any():
        print(f"Warning: Flash LSE contains inf/nan values")
    if torch.isinf(flash_output).any() or torch.isnan(flash_output).any():
        print(f"Warning: flash output contains inf/nan values")
    print("softmax_lse: ", softmax_lse)
    print(f"softmax_lse type: {softmax_lse.dtype}")

    if config.layout == 'bhsd':
        flash_output = flash_output.permute(0, 2, 1, 3)
    elif config.layout == 'sbhd':
        flash_output = flash_output.permute(1, 0, 2, 3)
        
    if config.layout == 'sbhd' and softmax_lse.shape != sage_results["int8"]["lse"].shape:
        softmax_lse = softmax_lse.permute(2, 0, 1)
    
    flash_stats = compute_statistics(softmax_lse)
    
    for qtype, result in sage_results.items():
        lse = result["lse"]
        output = result["output"]
        
        lse_stats = compute_statistics(lse)
        
        # Compute similarity for both LSE and output
        lse_similarity = calculate_similarity(lse, softmax_lse)
        output_similarity = calculate_similarity(output, flash_output)
        
        logger.add_result(config, qtype, lse_stats, flash_stats, lse_similarity, output_similarity)
        
        print(f"{qtype.upper()} - LSE cosine sim: {lse_similarity['cos_sim']:.6f}, max diff: {lse_similarity['max_diff']:.6f} | "
              f"OUT cosine sim: {output_similarity['cos_sim']:.6f}, max diff: {output_similarity['max_diff']:.6f}")

for i, config in enumerate(test_configs):
    print(f"\nRunning test {i+1}/{len(test_configs)}")
    try:
        run_test(config)
    except Exception as e:
        print(f"Test failed: {e}")

logger.save("sage_lse_comparison")

