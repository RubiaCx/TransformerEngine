import torch
import numpy as np
from transformer_engine.pytorch.attention import SageAttention
from flash_attn.flash_attn_interface import _flash_attn_forward
import matplotlib.pyplot as plt
import seaborn as sns

torch.backends.cuda.enable_flash_sdp(False)   
torch.backends.cuda.enable_math_sdp(True)  

torch.manual_seed(2025)
torch.set_printoptions(precision=4, sci_mode=False)

def make_tensors(batch_size=1, num_heads=32, seq_len=2048, head_dim=128, 
                 value_min=-50.0, value_max=50.0, dtype=torch.float16, layout='sbhd'):
    if layout == 'bhsd':
        shape = (batch_size, num_heads, seq_len, head_dim)
    elif layout == 'bshd':
        shape = (batch_size, seq_len, num_heads, head_dim)
    elif layout == 'sbhd':
        shape = (seq_len, batch_size, num_heads, head_dim)
    
    print(f"Creating tensors with shape {shape} in range [{value_min}, {value_max}]")
    
    def create_tensor():
        t = torch.randn(shape, device="cuda")
        t = (t - t.min()) / (t.max() - t.min()) * (value_max - value_min) + value_min
        return t.to(dtype).contiguous()
    
    q = create_tensor()
    k = create_tensor()
    v = create_tensor()
    
    return q, k, v

def debug_test(q, k, v, scale, layout, qtype="int8"):
    print(f"\n===== Testing {qtype} quantization =====")
    
    sage = SageAttention(
        softmax_scale=scale,
        quantization_backend="triton",
        quantization_type=qtype,
        smooth_k=True,
        return_lse=True,
        attention_dropout=0.0,
    ).cuda().eval()
    
    try:
        output, lse = sage(q, k, v, qkv_layout=f"{layout}_{layout}_{layout}", attn_mask_type="no_mask")
        
        has_inf_lse = torch.isinf(lse).any().item()
        has_nan_lse = torch.isnan(lse).any().item()
        
        print(f"LSE shape: {lse.shape}")
        print(f"LSE stats: min={lse.min().item():.4f}, max={lse.max().item():.4f}, "
              f"mean={lse.mean().item():.4f}, std={lse.std().item():.4f}")
        print(f"LSE has inf: {has_inf_lse}, has nan: {has_nan_lse}")
        
        has_inf_out = torch.isinf(output).any().item()
        has_nan_out = torch.isnan(output).any().item()
        
        print(f"Output shape: {output.shape}")
        print(f"Output stats: min={output.min().item():.4f}, max={output.max().item():.4f}, "
              f"mean={output.mean().item():.4f}, std={output.std().item():.4f}")
        print(f"Output has inf: {has_inf_out}, has nan: {has_nan_out}")
        
        return {
            "lse": lse, 
            "output": output, 
            "lse_stats": {
                "min": lse.min().item(),
                "max": lse.max().item(),
                "has_inf": has_inf_lse,
                "has_nan": has_nan_lse
            }
        }
    except Exception as e:
        print(f"Error running SAGE: {e}")
        return None

def compute_qk_stats(q, k, scale, layout):
    print("\n===== Computing Q·K^T statistics =====")
    
    if layout == 'bhsd':
        q_flat = q.transpose(1, 2)  # [B, S, H, D]
        k_flat = k.transpose(1, 2)  # [B, S, H, D]
    elif layout == 'bshd':
        q_flat = q  # [B, S, H, D]
        k_flat = k  # [B, S, H, D]
    elif layout == 'sbhd':
        q_flat = q.permute(1, 0, 2, 3)  # [B, S, H, D]
        k_flat = k.permute(1, 0, 2, 3)  # [B, S, H, D]
    
    B, S, H, D = q_flat.shape
    
    q_flat = q_flat.float()
    k_flat = k_flat.float()
    
    q_flat = q_flat.reshape(B, S, H, D)
    k_flat = k_flat.reshape(B, S, H, D)
    
    batch_idx = 0
    head_idx = 0
    
    sample_size = min(100, S)
    
    qk = torch.matmul(q_flat[batch_idx, :sample_size, head_idx], 
                      k_flat[batch_idx, :sample_size, head_idx].transpose(0, 1))
    qk *= scale
    
    qk_min = qk.min().item()
    qk_max = qk.max().item()
    qk_mean = qk.mean().item()
    qk_std = qk.std().item()
    
    print(f"Q·K^T shape: {qk.shape}")
    print(f"Q·K^T stats: min={qk_min:.4f}, max={qk_max:.4f}, mean={qk_mean:.4f}, std={qk_std:.4f}")
    
    if qk_max > 15:
        print(f"WARNING: Q·K^T max value ({qk_max:.4f}) exceeds 15, which may cause exp() overflow in float16")
    
    try:
        exp_qk = torch.exp(qk)
        print(f"exp(Q·K^T) stats: min={exp_qk.min().item():.4e}, max={exp_qk.max().item():.4e}")
        if torch.isinf(exp_qk).any():
            print("WARNING: exp(Q·K^T) contains inf values")
    except Exception as e:
        print(f"Error computing exp(Q·K^T): {e}")
    
    try:
        max_qk = torch.max(qk, dim=1, keepdim=True)[0]
        stable_qk = qk - max_qk
        exp_stable = torch.exp(stable_qk)
        sum_exp = torch.sum(exp_stable, dim=1, keepdim=True)
        lse = max_qk + torch.log(sum_exp)
        print(f"Manually computed LSE: shape={lse.shape}, min={lse.min().item():.4f}, max={lse.max().item():.4f}")
    except Exception as e:
        print(f"Error computing manual LSE: {e}")
    
    qk_flat = qk.flatten().cpu().numpy()
    
    return {
        "qk_min": qk_min,
        "qk_max": qk_max,
        "qk_mean": qk_mean,
        "qk_std": qk_std,
        "qk_distribution": qk_flat
    }

def run_tests():
    batch_size = 1
    num_heads = 32
    seq_len = 2048
    head_dim = 128
    layout = 'sbhd'
    dtype = torch.float16
    scale = 1.0 / (head_dim ** 0.5)
    
    print("\n===== TESTING POSITIVE VALUES (0.0 ~ 50.0) =====")
    q_pos, k_pos, v_pos = make_tensors(batch_size, num_heads, seq_len, head_dim, 
                                      0.0, 50.0, dtype, layout)
    
    qk_stats_pos = compute_qk_stats(q_pos, k_pos, scale, layout)
    
    sage_results_pos = {}
    for qtype in ["int8", "e4m3", "e5m2", "none"]:
        sage_results_pos[qtype] = debug_test(q_pos, k_pos, v_pos, scale, layout, qtype)
    
    print("\n===== TESTING NEGATIVE VALUES (-50.0 ~ 0.0) =====")
    q_neg, k_neg, v_neg = make_tensors(batch_size, num_heads, seq_len, head_dim, 
                                      -50.0, 0.0, dtype, layout)
    
    qk_stats_neg = compute_qk_stats(q_neg, k_neg, scale, layout)
    
    sage_results_neg = {}
    for qtype in ["int8", "e4m3", "e5m2", "none"]:
        sage_results_neg[qtype] = debug_test(q_neg, k_neg, v_neg, scale, layout, qtype)
    
    print("\n===== TESTING NORMAL RANGE (-10.0 ~ 10.0) =====")
    q_norm, k_norm, v_norm = make_tensors(batch_size, num_heads, seq_len, head_dim, 
                                       -10.0, 10.0, dtype, layout)
    
    qk_stats_norm = compute_qk_stats(q_norm, k_norm, scale, layout)
    
    sage_results_norm = {}
    for qtype in ["int8", "e4m3", "e5m2", "none"]:
        sage_results_norm[qtype] = debug_test(q_norm, k_norm, v_norm, scale, layout, qtype)
    
    plot_distributions(qk_stats_pos["qk_distribution"], qk_stats_neg["qk_distribution"], 
                       qk_stats_norm["qk_distribution"])

def plot_distributions(qk_pos, qk_neg, qk_norm):
    """绘制Q·K^T分布图，帮助理解溢出问题"""
    plt.figure(figsize=(15, 10))
    
    plt.subplot(3, 1, 1)
    sns.histplot(qk_pos, bins=100, kde=True)
    plt.title('Q·K^T Distribution for Range 0.0 ~ 50.0')
    # plt.axvline(x=15, color='r', linestyle='--', label='Float16 Exp Limit')
    # plt.legend()
    
    # 绘制负值范围
    plt.subplot(3, 1, 2)
    sns.histplot(qk_neg, bins=100, kde=True)
    plt.title('Q·K^T Distribution for Range -50.0 ~ 0.0')
    # plt.axvline(x=15, color='r', linestyle='--', label='Float16 Exp Limit')
    # plt.legend()
    
    # 绘制正常范围
    plt.subplot(3, 1, 3)
    sns.histplot(qk_norm, bins=100, kde=True)
    plt.title('Q·K^T Distribution for Range -10.0 ~ 10.0')
    # plt.axvline(x=15, color='r', linestyle='--', label='Float16 Exp Limit')
    # plt.legend()
    
    plt.tight_layout()
    plt.savefig('qk_distributions.png')
    print("Distribution plots saved to 'qk_distributions.png'")

if __name__ == "__main__":
    run_tests()