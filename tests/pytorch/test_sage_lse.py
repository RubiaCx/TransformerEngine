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
        if hasattr(config, 'value_min') and hasattr(config, 'value_max'):
            t = (t - t.min()) / (t.max() - t.min()) * (config.value_max - config.value_min) + config.value_min
        else:
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
    # (1, 1, 64, 32, 'bhsd', 'causal'),
    # (1, 4, 512, 64, 'bhsd', 'no_mask'),
    # (1, 4, 512, 64, 'bhsd', 'causal'),
    # (8, 4, 512, 64, 'bhsd', 'no_mask'),
    # (16, 16, 2048, 128, 'bhsd', 'causal'),
    # (16, 16, 2048, 128, 'bhsd', 'no_mask'),
    # (32, 8, 4096, 64, 'bhsd', 'causal'),
    # (32, 8, 4096, 64, 'bhsd', 'no_mask'),
    (1, 2, 64, 32, 'bshd', 'no_mask'),
    (1, 4, 128, 64, 'bshd', 'no_mask'),
    (8, 8, 256, 64, 'bshd', 'no_mask'),
    (2, 16, 512, 128, 'bshd', 'causal'),
    (1, 2, 64, 32, 'sbhd', 'no_mask'),
    (1, 4, 128, 64, 'sbhd', 'no_mask'),
    (8, 8, 256, 64, 'sbhd', 'no_mask'),
    (2, 16, 512, 128, 'sbhd', 'causal'),
    
    # 添加更多小批量高头数配置
    (1, 16, 128, 64, 'bshd', 'no_mask'),
    (1, 32, 128, 64, 'bshd', 'no_mask'),
    (1, 16, 128, 64, 'sbhd', 'no_mask'),
    (1, 32, 128, 64, 'sbhd', 'no_mask'),
    
    # 添加更多中等规模配置
    (4, 8, 512, 64, 'bshd', 'causal'),
    (4, 8, 512, 64, 'sbhd', 'causal'),
    (4, 16, 1024, 64, 'bshd', 'no_mask'),
    (4, 16, 1024, 64, 'sbhd', 'no_mask'),
    
    # 添加更多大规模配置
    (2, 24, 1024, 128, 'bshd', 'causal'),
    (2, 24, 1024, 128, 'sbhd', 'causal'),
    
    # 添加一些非典型形状配置 (不同的head_dim)
    (2, 8, 256, 40, 'bshd', 'no_mask'),
    (2, 8, 256, 40, 'sbhd', 'no_mask'),
    (2, 8, 256, 80, 'bshd', 'causal'),
    (2, 8, 256, 80, 'sbhd', 'causal'),
    
    # 添加一些更长序列的配置
    (1, 8, 2048, 64, 'bshd', 'no_mask'),
    (1, 8, 2048, 64, 'sbhd', 'no_mask'),
]

test_configs = []
for bs, h, s, d, layout, mask_type in base_configs:
    for dtype in [torch.float16, torch.bfloat16]:
        # 基本值范围测试
        test_configs.append(
            TestConfig(
                batch_size=bs,
                num_heads=h,
                seq_len=s,
                head_dim=d,
                dtype=dtype,
                value_range=0.1,
                layout=layout,
                attn_mask_type=mask_type
            )
        )
        
        # -3到3值范围测试
        config = TestConfig(
            batch_size=bs,
            num_heads=h,
            seq_len=s,
            head_dim=d,
            dtype=dtype,
            value_range=0.1,
            layout=layout,
            attn_mask_type=mask_type
        )
        config.value_min = -3.0
        config.value_max = 3.0
        test_configs.append(config)
        
        # -7到7值范围测试
        config = TestConfig(
            batch_size=bs,
            num_heads=h,
            seq_len=s,
            head_dim=d,
            dtype=dtype,
            value_range=0.1,
            layout=layout,
            attn_mask_type=mask_type
        )
        config.value_min = -7.0
        config.value_max = 7.0
        test_configs.append(config)
        
        # 添加额外的值范围测试: -1到1 (更窄范围)
        config = TestConfig(
            batch_size=bs,
            num_heads=h,
            seq_len=s,
            head_dim=d,
            dtype=dtype,
            value_range=0.1,
            layout=layout,
            attn_mask_type=mask_type
        )
        config.value_min = -1.0
        config.value_max = 1.0
        test_configs.append(config)
        
        # 添加额外的值范围测试: -10到10 (更宽范围)
        config = TestConfig(
            batch_size=bs,
            num_heads=h,
            seq_len=s,
            head_dim=d,
            dtype=dtype,
            value_range=0.1,
            layout=layout,
            attn_mask_type=mask_type
        )
        config.value_min = -10.0
        config.value_max = 10.0
        test_configs.append(config)
        
        # 添加额外的值范围测试: 0到5 (非对称正范围)
        config = TestConfig(
            batch_size=bs,
            num_heads=h,
            seq_len=s,
            head_dim=d,
            dtype=dtype,
            value_range=0.1,
            layout=layout,
            attn_mask_type=mask_type
        )
        config.value_min = 0.0
        config.value_max = 5.0
        test_configs.append(config)
        
        # 添加额外的值范围测试: -5到0 (非对称负范围)
        config = TestConfig(
            batch_size=bs,
            num_heads=h,
            seq_len=s,
            head_dim=d,
            dtype=dtype,
            value_range=0.1,
            layout=layout,
            attn_mask_type=mask_type
        )
        config.value_min = -5.0
        config.value_max = 0.0
        test_configs.append(config)

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
        if hasattr(config, 'value_min') and hasattr(config, 'value_max'):
            value_range_display = f"{config.value_min:.1f} to {config.value_max:.1f}"
        else:
            value_range_display = f"{config.value_range:.1f}"
            
        self.results.append([
            qtype.upper(),
            config.batch_size,
            config.num_heads,
            config.seq_len,
            config.head_dim,
            str(config.dtype).split('.')[-1],
            config.layout.upper(),  
            config.attn_mask_type,
            value_range_display,
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
    elif config.layout == 'bshd':
        q_flash = q.contiguous()  
        k_flash = k.contiguous()
        v_flash = v.contiguous()
    elif config.layout == 'sbhd':
        q_flash = q.permute(1, 0, 2, 3).contiguous()
        k_flash = k.permute(1, 0, 2, 3).contiguous()
        v_flash = v.permute(1, 0, 2, 3).contiguous()

    flash_output, _, _, _, _, softmax_lse, _, _ = _flash_attn_forward(
        q_flash, k_flash, v_flash,
        dropout_p=0.1,
        softmax_scale=scale,
        causal=bool("causal" in config.attn_mask_type),
        window_size=(-1, -1),
        alibi_slopes=None,
        return_softmax=True,
    )
    
    if config.layout == 'bhsd':
        flash_output = flash_output.permute(0, 2, 1, 3)
    elif config.layout == 'bshd':
        flash_output = flash_output.permute(1, 0, 2, 3)
    elif config.layout == 'sbhd':
        flash_output = flash_output.permute(1, 0, 2, 3)
    
    if config.layout == 'bhsd':
        pass
    elif config.layout == 'bshd':
        if softmax_lse.shape != lse_int8.shape:
            pass
    elif config.layout == 'sbhd':
        if softmax_lse.shape != lse_int8.shape:
            softmax_lse = softmax_lse.permute(2, 0, 1)
    
    if flash_output.shape != sage_int8_output.shape:
        flash_output = flash_output.reshape(sage_int8_output.shape)
    
    if hasattr(config, 'value_min') and hasattr(config, 'value_max'):
        value_range_display = f"Value range: {config.value_min:.1f} to {config.value_max:.1f}"
        value_range_str = f"{config.value_min:.1f}~{config.value_max:.1f}"
    else:
        value_range_display = f"Value range: {config.value_range:.1f}"
        value_range_str = f"{config.value_range:.1f}"
    
    print(value_range_display)
    
    metrics4 = calculate_similarity(lse_int8, softmax_lse)
    metrics5 = calculate_similarity(lse_e4m3, softmax_lse)
    logger.add_result(config, f'int8 vs flash ({value_range_str})', metrics4)
    logger.add_result(config, f'e4m3 vs flash ({value_range_str})', metrics5)

    # metrics6 = calculate_similarity(sage_int8_output, flash_output)
    # metrics7 = calculate_similarity(sage_e4m3_output, flash_output)
    # metrics8 = calculate_similarity(sage_e5m2_output, flash_output)
    # logger.add_result(config, f'int8 vs flash ({value_range_str})', metrics6)
    # logger.add_result(config, f'e4m3 vs flash ({value_range_str})', metrics7)
    # logger.add_result(config, f'e5m2 vs flash ({value_range_str})', metrics8)

for config in test_configs:
    run_test(config)

logger.save()

