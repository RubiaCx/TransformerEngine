import pytest
import torch
import triton.tools.experimental_descriptor

import triton
import triton.language as tl

import torch.nn.functional as F
from flash_attn.flash_attn_interface import _flash_attn_backward
from flash_attn.flash_attn_interface import _flash_attn_forward 

# 计算 Delta = rowsum(O ⊙ dO)

@triton.jit
def _attn_bwd_preprocess(O, DO, 
                         Delta,  
                         BS, HEAD_NUM, SEQ_LEN, 
                         BLOCK_M: tl.constexpr, HEAD_DIM: tl.constexpr 
                         ):
    off_m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    off_hz = tl.program_id(1)
    off_n = tl.arange(0, HEAD_DIM)
    # load
    o = tl.load(O + off_hz * HEAD_DIM * SEQ_LEN + off_m[:, None] * HEAD_DIM + off_n[None, :])
    do = tl.load(DO + off_hz * HEAD_DIM * SEQ_LEN + off_m[:, None] * HEAD_DIM + off_n[None, :]).to(tl.float32)
    delta = tl.sum(o * do, axis=1)
    # write-back
    tl.store(Delta + off_hz * SEQ_LEN + off_m, delta)


# The main inner-loop logic for computing dK and dV.
@triton.jit
def _attn_bwd_dkdv(dk, dv, 
                   Q, k, v, sm_scale, 
                   DO, LSE, D, 
                   # shared by Q/K/V/DO.
                   stride_tok, stride_d, 
                   HEAD_NUM, SEQ_LEN, 
                   BLOCK_M1: tl.constexpr, 
                   BLOCK_N1: tl.constexpr, 
                   HEAD_DIM: tl.constexpr, 
                   # Filled in by the wrapper.
                   start_n, start_m, num_steps, 
                   MASK: tl.constexpr):
    offs_m = start_m + tl.arange(0, BLOCK_M1)
    offs_n = start_n + tl.arange(0, BLOCK_N1)
    offs_k = tl.arange(0, HEAD_DIM)
    qT_ptrs = Q + offs_m[None, :] * stride_tok + offs_k[:, None] * stride_d
    do_ptrs = DO + offs_m[:, None] * stride_tok + offs_k[None, :] * stride_d
    # BLOCK_N1 must be a multiple of BLOCK_M1, otherwise the code wouldn't work.
    tl.static_assert(BLOCK_N1 % BLOCK_M1 == 0)
    curr_m = start_m
    step_m = BLOCK_M1
    for blk_idx in range(num_steps):
        qT = tl.load(qT_ptrs)
        # Load m before computing qk to reduce pipeline stall.
        offs_m = curr_m + tl.arange(0, BLOCK_M1)
        m = tl.load(LSE + offs_m)
        qkT = tl.dot(k, qT)
        pT = tl.math.exp2(qkT - m[None, :])
        # Autoregressive masking.
        if MASK:
            mask = (offs_m[None, :] >= offs_n[:, None])
            pT = tl.where(mask, pT, 0.0)
        do = tl.load(do_ptrs)
        # Compute dV.
        ppT = pT
        ppT = ppT.to(tl.float16)
        dv += tl.dot(ppT, do)
        # D (= delta) is pre-divided by ds_scale.
        Di = tl.load(D + offs_m)
        # Compute dP and dS.
        dpT = tl.dot(v, tl.trans(do)).to(tl.float32)
        dsT = pT * (dpT - Di[None, :])
        dsT = dsT.to(tl.float16)
        dk += tl.dot(dsT, tl.trans(qT))
        # Increment pointers.
        curr_m += step_m
        qT_ptrs += step_m * stride_tok
        do_ptrs += step_m * stride_tok
    return dk, dv


# the main inner-loop logic for computing dQ
@triton.jit
def _attn_bwd_dq(dq, q, K, V, 
                 do, m, D,
                 # shared by Q/K/V/DO.
                 stride_tok, stride_d, 
                 HEAD_NUM, SEQ_LEN, 
                 BLOCK_M2: tl.constexpr, 
                 BLOCK_N2: tl.constexpr, 
                 HEAD_DIM: tl.constexpr,
                 # Filled in by the wrapper.
                 start_m, start_n, num_steps, 
                 MASK: tl.constexpr):
    offs_m = start_m + tl.arange(0, BLOCK_M2)
    offs_n = start_n + tl.arange(0, BLOCK_N2)
    offs_k = tl.arange(0, HEAD_DIM)
    kT_ptrs = K + offs_n[None, :] * stride_tok + offs_k[:, None] * stride_d
    vT_ptrs = V + offs_n[None, :] * stride_tok + offs_k[:, None] * stride_d
    # D (= delta) is pre-divided by ds_scale.
    Di = tl.load(D + offs_m)
    # BLOCK_M2 must be a multiple of BLOCK_N2, otherwise the code wouldn't work.
    tl.static_assert(BLOCK_M2 % BLOCK_N2 == 0)
    curr_n = start_n
    step_n = BLOCK_N2
    for blk_idx in range(num_steps):
        kT = tl.load(kT_ptrs)
        vT = tl.load(vT_ptrs)
        qk = tl.dot(q, kT)
        p = tl.math.exp2(qk - m)
        # Autoregressive masking.
        if MASK:
            offs_n = curr_n + tl.arange(0, BLOCK_N2)
            mask = (offs_m[:, None] >= offs_n[None, :])
            p = tl.where(mask, p, 0.0)
        # Compute dP and dS.
        dp = tl.dot(do, vT).to(tl.float32)
        ds = p * (dp - Di[:, None])
        ds = ds.to(tl.float16)
        # Compute dQ.
        # NOTE: We need to de-scale dq in the end, because kT was pre-scaled.
        dq += tl.dot(ds, tl.trans(kT))
        # Increment pointers.
        curr_n += step_n
        kT_ptrs += step_n * stride_tok
        vT_ptrs += step_n * stride_tok
    return dq


@triton.jit
def _attn_bwd(Q, K, V, sm_scale, 
              DO, 
              DQ, DK, DV, 
              LSE, D,
              # shared by Q/K/V/DO.
              stride_z, stride_h, stride_tok, stride_d,  #
              HEAD_NUM, SEQ_LEN,  #
              BLOCK_M1: tl.constexpr,  #
              BLOCK_N1: tl.constexpr,  #
              BLOCK_M2: tl.constexpr,  #
              BLOCK_N2: tl.constexpr,  #
              BLK_SLICE_FACTOR: tl.constexpr,  #
              HEAD_DIM: tl.constexpr):
    LN2: tl.constexpr = 0.6931471824645996  # = ln(2)
    pid = tl.program_id(0)
    bhid = tl.program_id(2)
    off_chz = (bhid * SEQ_LEN).to(tl.int64)
    adj = (stride_h * (bhid % HEAD_NUM) + stride_z * (bhid // HEAD_NUM)).to(tl.int64)

    # offset pointers for batch/head
    Q += adj
    K += adj
    V += adj
    DO += adj
    DQ += adj
    DK += adj
    DV += adj
    LSE += off_chz
    D += off_chz

    # load scales
    offs_k = tl.arange(0, HEAD_DIM)

    start_n = pid * BLOCK_N1
    start_m = start_n

    MASK_BLOCK_M1: tl.constexpr = BLOCK_M1 // BLK_SLICE_FACTOR
    offs_n = start_n + tl.arange(0, BLOCK_N1)

    dv = tl.zeros([BLOCK_N1, HEAD_DIM], dtype=tl.float32)
    dk = tl.zeros([BLOCK_N1, HEAD_DIM], dtype=tl.float32)

    # load K and V: they stay in SRAM throughout the inner loop.
    k = tl.load(K + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d)
    v = tl.load(V + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d)

    num_steps = BLOCK_N1 // MASK_BLOCK_M1

    dk, dv = _attn_bwd_dkdv(dk, dv, 
                            Q, k, v, sm_scale, 
                            DO, 
                            LSE, D, 
                            stride_tok, stride_d, 
                            HEAD_NUM, SEQ_LEN, 
                            MASK_BLOCK_M1, BLOCK_N1, HEAD_DIM, 
                            start_n, start_m, num_steps, 
                            MASK=True 
                            )

    start_m += num_steps * MASK_BLOCK_M1
    num_steps = (SEQ_LEN - start_m) // BLOCK_M1

    # Compute dK and dV for non-masked blocks.
    dk, dv = _attn_bwd_dkdv( 
        dk, dv, 
        Q, k, v, sm_scale, 
        DO, 
        LSE, D, 
        stride_tok, stride_d, 
        HEAD_NUM, SEQ_LEN, 
        BLOCK_M1, BLOCK_N1, HEAD_DIM, 
        start_n, start_m, num_steps, 
        MASK=False
    )

    dv_ptrs = DV + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d
    tl.store(dv_ptrs, dv)

    # Write back dK.
    dk *= sm_scale
    dk_ptrs = DK + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d
    tl.store(dk_ptrs, dk)

    # THIS BLOCK DOES DQ:
    start_m = pid * BLOCK_M2
    end_n = start_m + BLOCK_M2

    MASK_BLOCK_N2: tl.constexpr = BLOCK_N2 // BLK_SLICE_FACTOR
    offs_m = start_m + tl.arange(0, BLOCK_M2)

    q = tl.load(Q + offs_m[:, None] * stride_tok + offs_k[None, :] * stride_d)
    dq = tl.zeros([BLOCK_M2, HEAD_DIM], dtype=tl.float32)
    do = tl.load(DO + offs_m[:, None] * stride_tok + offs_k[None, :] * stride_d)

    m = tl.load(LSE + offs_m)
    m = m[:, None]

    # Compute dQ for masked (diagonal) blocks.
    # NOTE: This code scans each row of QK^T backward (from right to left,
    # but inside each call to _attn_bwd_dq, from left to right), but that's
    # not due to anything important.  I just wanted to reuse the loop
    # structure for dK & dV above as much as possible.
    num_steps = BLOCK_M2 // MASK_BLOCK_N2
    dq = _attn_bwd_dq(dq, q, K, V,  #
                      do, m, D,  #
                      stride_tok, stride_d,  #
                      HEAD_NUM, SEQ_LEN,  #
                      BLOCK_M2, MASK_BLOCK_N2, HEAD_DIM,  #
                      start_m, end_n - num_steps * MASK_BLOCK_N2, num_steps,  #
                      MASK=True  #
                      )
    end_n -= num_steps * MASK_BLOCK_N2
    # stage 2
    num_steps = end_n // BLOCK_N2
    dq = _attn_bwd_dq(dq, q, K, V, 
                      do, m, D, 
                      stride_tok, stride_d, 
                      HEAD_NUM, SEQ_LEN, 
                      BLOCK_M2, BLOCK_N2, HEAD_DIM, 
                      start_m, end_n - num_steps * BLOCK_N2, num_steps, 
                      MASK=False 
                      )
    # Write back dQ.
    dq_ptrs = DQ + offs_m[:, None] * stride_tok + offs_k[None, :] * stride_d
    dq *= LN2
    tl.store(dq_ptrs, dq)

class _attention(torch.autograd.Function):
    @staticmethod
    def forward(ctx,
                q, k, v, 
                softmax_scale,
                output_dtype=torch.float16, 
                causal=False,
                quant_type="int8"):
        from flash_attn.flash_attn_interface import _flash_attn_forward 
        out, q, k, v, out_padded, softmax_lse, S_dmask, rng_state = _flash_attn_forward(
                q,
                k,
                v,
                0.0,
                softmax_scale,
                causal=causal,
                window_size=(-1, -1),
                alibi_slopes=None,
                return_softmax=False,
            )

        ctx.save_for_backward(q, k, v, out_padded, softmax_lse, rng_state)
        ctx.dropout_p = 0.0
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        ctx.window_size = (-1, -1)
        ctx.alibi_slopes = None
        ctx.deterministic = False
        return out

    @staticmethod
    def backward(ctx, do):
        q, k, v, o, lse, rng_state = ctx.saved_tensors
        q = q.transpose(1, 2).contiguous()
        k = k.transpose(1, 2).contiguous()
        v = v.transpose(1, 2).contiguous()
        o = o.transpose(1, 2).contiguous()
        do = do.transpose(1, 2).contiguous()
        assert do.is_contiguous()
        # assert q.stride() == k.stride() == v.stride() == o.stride() == do.stride()
        # print("SHAPE: ", q.shape, k.shape, v.shape, o.shape, do.shape)

        dq = torch.empty_like(q)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)

        BATCH, HEAD_N_Q, SEQ_Q, HEAD_DIM_Q = q.shape
        _, HEAD_N_K, SEQ_KV, HEAD_DIM_K = k.shape
        PRE_BLOCK = 128
        GROUPS =  max(HEAD_N_Q // HEAD_N_K, 1)
        #! 前向反向的quant scale block size 对齐
        BLOCK_M1, BLOCK_N1, BLOCK_M2, BLOCK_N2 = 32, 128, 128, 32
        BLK_SLICE_FACTOR = 2

        # q = q * (ctx.softmax_scale * RCP_LN2) 
        arg_k = k
        arg_k = arg_k * (ctx.softmax_scale * 1.4426950408889634) 
        # v = v * (ctx.softmax_scale * RCP_LN2) 
        assert SEQ_Q % PRE_BLOCK == 0
        pre_grid = (SEQ_Q // PRE_BLOCK, BATCH * HEAD_N_Q)
        delta = torch.empty_like(lse)
        _attn_bwd_preprocess[pre_grid](
            o, do, 
            delta, 
            BATCH, HEAD_N_Q, SEQ_Q, 
            BLOCK_M=PRE_BLOCK, HEAD_DIM=HEAD_DIM_K
        )
        # NUM_WARPS, NUM_STAGES = 4, 5
        grid = (SEQ_Q // BLOCK_N1, 1, BATCH * HEAD_N_Q)
        _attn_bwd[grid](
            q, arg_k, v, 
            ctx.softmax_scale, 
            do, dq, dk, dv, 
            lse, delta, 
            q.stride(0), q.stride(1), q.stride(2), q.stride(3), 
            HEAD_N_Q, SEQ_Q, 
            BLOCK_M1=BLOCK_M1, BLOCK_N1=BLOCK_N1, 
            BLOCK_M2=BLOCK_M2, BLOCK_N2=BLOCK_N2, 
            BLK_SLICE_FACTOR=BLK_SLICE_FACTOR, 
            HEAD_DIM=HEAD_DIM_K, 
            num_warps=4 if HEAD_DIM_K == 64 else 5,
            num_stages=3 if HEAD_DIM_K == 64 else 2
        )
        dq = dq.transpose(1, 2)
        dk = dk.transpose(1, 2)
        dv = dv.transpose(1, 2)
        dq = dq[..., : do.shape[-1]].contiguous()
        dk = dk[..., : do.shape[-1]].contiguous()
        dv = dv[..., : do.shape[-1]].contiguous()
        print(f"dk.isnan().any(): {dk.isnan().any()}, dk.isinf().any(): {dk.isinf().any()}, dv.isnan().any(): {dv.isnan().any()}")
        print(f"dq.isnan().any(): {dq.isnan().any()}, dq.isinf().any(): {dq.isinf().any()}, dv.isinf().any(): {dv.isinf().any()}")
        return dq, dk, dv, None, None, None, None

class _attention_v2(torch.autograd.Function):
    @staticmethod
    def forward(ctx,
                q, k, v, 
                softmax_scale,
                output_dtype=torch.float16, 
                causal=False,
                quant_type="int8"):
        out, q, k, v, out_padded, softmax_lse, S_dmask, rng_state = _flash_attn_forward(
                q,
                k,
                v,
                0.0,
                softmax_scale,
                causal=causal,
                window_size=(-1, -1),
                alibi_slopes=None,
                return_softmax=False,
            )

        ctx.save_for_backward(q, k, v, out_padded, softmax_lse, rng_state)
        ctx.dropout_p = 0.0
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        ctx.window_size = (-1, -1)
        ctx.alibi_slopes = None
        ctx.deterministic = False
        return out

    @staticmethod
    def backward(ctx, do):

        q, k, v, o, lse, rng_state = ctx.saved_tensors
        maybe_contiguous = lambda x: x.contiguous() if x.is_contiguous() == False else x
        do, q, k, v, o = [maybe_contiguous(x) for x in (do, q, k, v, o)]
        assert do.is_contiguous()
        # assert q.stride() == k.stride() == v.stride() == o.stride() == do.stride()
        # print("SHAPE: ", q.shape, k.shape, v.shape, o.shape, do.shape)

        dq = torch.empty_like(q)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)
        _flash_attn_backward(
                do,
                q,
                k,
                v,
                o,
                lse,
                dq,
                dk,
                dv,
                ctx.dropout_p,
                ctx.softmax_scale,
                ctx.causal,
                ctx.window_size,
                ctx.alibi_slopes,
                ctx.deterministic,
                rng_state=rng_state,
            )
        return dq, dk, dv, None, None, None, None, None

# TODO Flash FWD + Sage BWD / Flash BWD
def sage_attention(q, k, v, softmax_scale=None, output_dtype=torch.float16, causal=False, quant_type="int8"):
    return _attention.apply(q, k, v, softmax_scale, output_dtype, causal, quant_type)

def sage_attention_v2(q, k, v, softmax_scale=None, output_dtype=torch.float16, causal=False, quant_type="int8"):
    return _attention_v2.apply(q, k, v, softmax_scale, output_dtype, causal, quant_type)


def test_sage_attention(batch_size=1, seq_len=128, num_heads=8, head_dim=64, causal=True, dtype=torch.float16):
    import time
    import torch.nn.functional as F
    from flash_attn.flash_attn_interface import flash_attn_func
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 创建输入张量 - 使用BSHD格式，这是FlashAttention和sage_attention期望的格式
    torch.manual_seed(20)
    q_bshd = torch.empty((batch_size, seq_len, num_heads, head_dim), device=device, dtype=dtype).normal_(mean=0.0, std=0.5).requires_grad_()
    k_bshd = torch.empty((batch_size, seq_len, num_heads, head_dim), device=device, dtype=dtype).normal_(mean=0.0, std=0.5).requires_grad_()
    v_bshd = torch.empty((batch_size, seq_len, num_heads, head_dim), device=device, dtype=dtype).normal_(mean=0.0, std=0.5).requires_grad_()
    
    # 转换为BHSD格式用于标准点积注意力
    q_bhsd = q_bshd.transpose(1, 2).contiguous()
    k_bhsd = k_bshd.transpose(1, 2).contiguous()
    v_bhsd = v_bshd.transpose(1, 2).contiguous()
    
    softmax_scale = head_dim ** -0.5
    dout_bshd = torch.randn_like(q_bshd)
    dout_bhsd = dout_bshd.transpose(1, 2).contiguous()
    
    # 标准点积注意力（SDPA）实现 - 使用BHSD格式
    def sdpa_attention(q, k, v, softmax_scale, causal=False):
        # 确保输入是BHSD格式
        p = torch.matmul(q, k.transpose(2, 3)) * softmax_scale
        if causal:
            causal_mask = torch.triu(torch.ones((seq_len, seq_len), device=device), diagonal=1)
            p = p.masked_fill(causal_mask.bool().unsqueeze(0).unsqueeze(0), float("-inf"))
        p = torch.softmax(p.float(), dim=-1).to(dtype)
        return torch.matmul(p, v)
    
    # 准备比较的实现
    implementations = {
        "sdpa_ref": lambda: sdpa_attention(q_bhsd, k_bhsd, v_bhsd, softmax_scale, causal),
        "flash_sage": lambda: sage_attention(q_bshd, k_bshd, v_bshd, softmax_scale, causal=causal, quant_type="none"),
        "flash_flash": lambda: sage_attention_v2(q_bshd, k_bshd, v_bshd, softmax_scale, causal=causal),
        "flash_attn_ref": lambda: flash_attn_func(q_bshd, k_bshd, v_bshd, dropout_p=0.0, softmax_scale=softmax_scale, causal=causal)
    }
    
    # 预热
    print("预热中...")
    for name, impl in implementations.items():
        try:
            _ = impl()
            print(f"  {name} 预热成功")
        except Exception as e:
            print(f"  {name} 预热失败: {e}")
            implementations.pop(name)
    torch.cuda.synchronize()
    
    # 性能测试
    print("\n性能测试:")
    results = {}
    iterations = 10
    for name, impl in implementations.items():
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(iterations):
            output = impl()
            torch.cuda.synchronize()
        end = time.time()
        avg_time = (end - start) / iterations * 1000  # 转换为毫秒
        results[name] = {"output": output, "time": avg_time}
        print(f"{name}: {avg_time:.3f} ms/iter")
    
    # 精度比较
    print("\n精度比较:")
    # 使用flash_attn_ref作为主要参考
    reference_name = "flash_attn_ref"
    reference = results[reference_name]["output"]
    
    for name, result in results.items():
        if name == reference_name:
            continue
        output = result["output"]
        
        # 确保格式一致再比较
        if name == "sdpa_ref":
            # 将BHSD转换为BSHD以与参考比较
            output = output.transpose(1, 2).contiguous()
        
        max_diff = (output - reference).abs().max().item()
        mean_diff = (output - reference).abs().mean().item()
        cos_sim = F.cosine_similarity(output.flatten(), reference.flatten(), dim=0).item()
        print(f"{name} vs {reference_name}:")
        print(f"  max_diff: {max_diff:.6f}")
        print(f"  mean_diff: {mean_diff:.6f}")
        print(f"  cosine_similarity: {cos_sim:.6f}")
    
    # 反向传播测试
    print("\n反向传播测试:")
    
    # 首先为flash_attn_ref实现单独运行反向传播
    q_flash = q_bshd.detach().requires_grad_(True)
    k_flash = k_bshd.detach().requires_grad_(True)
    v_flash = v_bshd.detach().requires_grad_(True)
    
    # 运行flash_attn_ref并获取梯度
    torch.cuda.synchronize()
    start = time.time()
    flash_output = flash_attn_func(q_flash, k_flash, v_flash, dropout_p=0.0, softmax_scale=softmax_scale, causal=causal)
    flash_output.backward(dout_bshd)
    torch.cuda.synchronize()
    end = time.time()
    
    flash_grad_time = (end - start) * 1000  # 转换为毫秒
    print(f"flash_attn_ref backward: {flash_grad_time:.3f} ms")
    
    # 为其他实现运行反向传播，并与flash_attn_ref比较
    for name, impl in implementations.items():
        if name == "flash_attn_ref":
            continue
            
        # 使用可训练的副本
        if name == "sdpa_ref":
            q_copy = q_bhsd.detach().requires_grad_(True)
            k_copy = k_bhsd.detach().requires_grad_(True)
            v_copy = v_bhsd.detach().requires_grad_(True)
            dout_copy = dout_bhsd
        else:
            q_copy = q_bshd.detach().requires_grad_(True)
            k_copy = k_bshd.detach().requires_grad_(True)
            v_copy = v_bshd.detach().requires_grad_(True)
            dout_copy = dout_bshd
        
        torch.cuda.synchronize()
        start = time.time()
        
        # 前向传播
        if name == "sdpa_ref":
            output = sdpa_attention(q_copy, k_copy, v_copy, softmax_scale, causal)
        elif name == "flash_sage":
            output = sage_attention(q_copy, k_copy, v_copy, softmax_scale, causal=causal, quant_type="none")
        elif name == "flash_flash":
            output = sage_attention_v2(q_copy, k_copy, v_copy, softmax_scale, causal=causal)
        
        # 反向传播
        output.backward(dout_copy)
        
        torch.cuda.synchronize()
        end = time.time()
        
        grad_time = (end - start) * 1000  # 转换为毫秒
        print(f"{name} backward: {grad_time:.3f} ms")
        
        # 计算与flash_attn_ref梯度的差异
        print(f"  与flash_attn_ref梯度比较:")
        
        # 确保梯度不为None并且格式一致
        if q_copy.grad is not None and q_flash.grad is not None:
            # 对于SDPA，需要转换梯度格式
            if name == "sdpa_ref":
                q_grad = q_copy.grad.transpose(1, 2).contiguous()
                k_grad = k_copy.grad.transpose(1, 2).contiguous()
                v_grad = v_copy.grad.transpose(1, 2).contiguous()
            else:
                q_grad = q_copy.grad
                k_grad = k_copy.grad
                v_grad = v_copy.grad
                
            q_grad_diff = (q_grad - q_flash.grad).abs()
            print(f"  dQ max_diff: {q_grad_diff.max().item():.6f}, mean_diff: {q_grad_diff.mean().item():.6f}")
            print(f"  dQ cos_sim: {F.cosine_similarity(q_grad.flatten(), q_flash.grad.flatten(), dim=0).item():.6f}")
            print(f"  dQ scale_factor: {q_grad.abs().max().item() / max(q_flash.grad.abs().max().item(), 1e-8):.6f}")
            
            k_grad_diff = (k_grad - k_flash.grad).abs()
            print(f"  dK max_diff: {k_grad_diff.max().item():.6f}, mean_diff: {k_grad_diff.mean().item():.6f}")
            print(f"  dK cos_sim: {F.cosine_similarity(k_grad.flatten(), k_flash.grad.flatten(), dim=0).item():.6f}")
            print(f"  dK scale_factor: {k_grad.abs().max().item() / max(k_flash.grad.abs().max().item(), 1e-8):.6f}")
            
            v_grad_diff = (v_grad - v_flash.grad).abs()
            print(f"  dV max_diff: {v_grad_diff.max().item():.6f}, mean_diff: {v_grad_diff.mean().item():.6f}")
            print(f"  dV cos_sim: {F.cosine_similarity(v_grad.flatten(), v_flash.grad.flatten(), dim=0).item():.6f}")
            print(f"  dV scale_factor: {v_grad.abs().max().item() / max(v_flash.grad.abs().max().item(), 1e-8):.6f}")
        else:
            print("  梯度为None，无法比较")

if __name__ == "__main__":
    # 默认测试
    test_sage_attention()
    
    # 不同规模的测试
    print("\n=== 测试更长序列 ===")
    test_sage_attention(batch_size=2, seq_len=512, num_heads=16)
    
    # # 非因果测试
    # print("\n=== 非因果测试 ===")
    test_sage_attention(causal=False)
    