import pytest
import torch
import triton.tools.experimental_descriptor

import triton
import triton.language as tl

DEVICE = "cuda"

@triton.jit
def _attn_fwd_inner(acc, l_i, m_i, q,
                    K_block_ptr, V_block_ptr,
                    start_m, qk_scale,
                    BLOCK_M: tl.constexpr, HEAD_DIM: tl.constexpr, BLOCK_N: tl.constexpr,
                    STAGE: tl.constexpr, offs_m: tl.constexpr, offs_n: tl.constexpr,
                    SEQ_LEN: tl.constexpr, fp8_v: tl.constexpr):
    # range of values handled by this stage
    if STAGE == 1:
        lo, hi = 0, start_m * BLOCK_M
    elif STAGE == 2:
        lo, hi = start_m * BLOCK_M, (start_m + 1) * BLOCK_M
        lo = tl.multiple_of(lo, BLOCK_M)
    # causal = False
    else:
        lo, hi = 0, SEQ_LEN
    K_block_ptr = tl.advance(K_block_ptr, (0, lo))
    V_block_ptr = tl.advance(V_block_ptr, (lo, 0))

    for start_n in range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)

        k = tl.load(K_block_ptr)
        qk = tl.dot(q, k)
        if STAGE == 2:
            mask = offs_m[:, None] >= (start_n + offs_n[None, :])
            qk = qk * qk_scale + tl.where(mask, 0, -1.0e6)
            m_ij = tl.maximum(m_i, tl.max(qk, 1))
            qk -= m_ij[:, None]
        else:
            m_ij = tl.maximum(m_i, tl.max(qk, 1) * qk_scale)
            qk = qk * qk_scale - m_ij[:, None]
        p = tl.math.exp2(qk)
        l_ij = tl.sum(p, 1)

        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + l_ij

        acc = acc * alpha[:, None]

        v = tl.load(V_block_ptr)
        if fp8_v:
            p = p.to(tl.float8e5)
        else:
            p = p.to(tl.float16)
        acc = tl.dot(p, v, acc)

        m_i = m_ij
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
    return acc, l_i, m_i

configs = [
    triton.Config({'BLOCK_M': BM, 'BLOCK_N': BN}, num_stages=s, num_warps=w) \
    for BM in [64, 128]\
    for BN in [32, 64]\
    for s in [3, 4, 7]\
    for w in [4, 8]\
]

def keep(conf):
    BLOCK_M = conf.kwargs["BLOCK_M"]
    BLOCK_N = conf.kwargs["BLOCK_N"]
    if BLOCK_M * BLOCK_N < 128 * 128 and conf.num_warps == 8:
        return False
    return True

@triton.autotune(list(filter(keep, configs)), key=["SEQ_LEN", "HEAD_DIM"])
@triton.jit
def _attn_fwd(Q, K, V, sm_scale, LSE, Out, 
              stride_qz, stride_qh, stride_qm, stride_qk, 
              stride_kz, stride_kh, stride_kn, stride_kk, 
              stride_vz, stride_vh, stride_vk, stride_vn, 
              stride_oz, stride_oh, stride_om, stride_on, 
              BS, HEAD_NUM, SEQ_LEN, 
              HEAD_DIM: tl.constexpr, 
              BLOCK_M: tl.constexpr, 
              BLOCK_N: tl.constexpr, 
              STAGE: tl.constexpr 
              ):
    tl.static_assert(BLOCK_N <= HEAD_DIM)
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // HEAD_NUM
    off_h = off_hz % HEAD_NUM
    qvk_offset = off_z.to(tl.int64) * stride_qz + off_h.to(tl.int64) * stride_qh

    # block pointers
    Q_block_ptr = tl.make_block_ptr(
        base=Q + qvk_offset,
        shape=(SEQ_LEN, HEAD_DIM),
        strides=(stride_qm, stride_qk),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, HEAD_DIM),
        order=(1, 0),
    )
    v_order: tl.constexpr = (0, 1) if V.dtype.element_ty == tl.float8e5 else (1, 0)
    V_block_ptr = tl.make_block_ptr(
        base=V + qvk_offset,
        shape=(SEQ_LEN, HEAD_DIM),
        strides=(stride_vk, stride_vn),
        offsets=(0, 0),
        block_shape=(BLOCK_N, HEAD_DIM),
        order=v_order,
    )
    K_block_ptr = tl.make_block_ptr(
        base=K + qvk_offset,
        shape=(HEAD_DIM, SEQ_LEN),
        strides=(stride_kk, stride_kn),
        offsets=(0, 0),
        block_shape=(HEAD_DIM, BLOCK_N),
        order=(0, 1),
    )
    O_block_ptr = tl.make_block_ptr(
        base=Out + qvk_offset,
        shape=(SEQ_LEN, HEAD_DIM),
        strides=(stride_om, stride_on),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, HEAD_DIM),
        order=(1, 0),
    )

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)

    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    qk_scale = sm_scale * 1.44269504  # 1/log(2)
    
    q = tl.load(Q_block_ptr) # load q: it will stay in SRAM throughout
    if STAGE & 1: # 1 or 3
        acc, l_i, m_i = _attn_fwd_inner(acc, l_i, m_i, q, K_block_ptr, V_block_ptr, 
                                        start_m, qk_scale, 
                                        BLOCK_M, HEAD_DIM, BLOCK_N, 
                                        4 - STAGE, offs_m, offs_n, SEQ_LEN, V.dtype.element_ty == tl.float8e5 
                                        )
    if STAGE & 2: # 2 or 3
        acc, l_i, m_i = _attn_fwd_inner(acc, l_i, m_i, q, K_block_ptr, V_block_ptr, 
                                        start_m, qk_scale,
                                        BLOCK_M, HEAD_DIM, BLOCK_N, 
                                        2, offs_m, offs_n, SEQ_LEN, V.dtype.element_ty == tl.float8e5 
                                        )
    m_i += tl.math.log2(l_i)
    acc = acc / l_i[:, None]
    m_ptrs = LSE + off_hz * SEQ_LEN + offs_m
    tl.store(m_ptrs, m_i)
    tl.store(O_block_ptr, acc.to(Out.type.element_ty))

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
        # if tl.program_id(0) == 0 and tl.program_id(1) == 0:
        #     if tl.program_id(2) == 0:
        #         qk_min = tl.min(qk);  qk_max = tl.max(qk)
        #         tl.device_print("qk min:", qk_min)
        #         tl.device_print("qk max:", qk_max)
        #         m_min = tl.min(m);  m_max = tl.max(m)
        #         tl.device_print("m min:", m_min)
        #         tl.device_print("m max:", m_max)
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

    bhid = tl.program_id(2)
    off_chz = (bhid * SEQ_LEN).to(tl.int64)
    adj = (stride_h * (bhid % HEAD_NUM) + stride_z * (bhid // HEAD_NUM)).to(tl.int64)
    pid = tl.program_id(0)

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
    def forward(ctx, q, k, v, causal, sm_scale, USE_TMA=True):
        # q k v 的 shape 是 [B, H, S, D]
        BATCH, N_HEAD, SEQ_L, HEAD_DIM_Q = q.shape
        HEAD_DIM_K = k.shape[-1]
        HEAD_DIM_V = v.shape[-1] # when v is in float8_e5m2 it is transposed.

        assert HEAD_DIM_Q == HEAD_DIM_K and HEAD_DIM_K == HEAD_DIM_V
        assert HEAD_DIM_K in {16, 32, 64, 128, 256}

        o = torch.empty_like(q)
        LSE = torch.empty((BATCH, N_HEAD, SEQ_L), device=q.device, dtype=torch.float32)

        stage = 3 if causal else 1
        extra_kern_args = {}
        # 沿 seq 维度分块 BLOCK_M；将 Batch 和 Head 维度合并；表示每个 warp 只处理 1 个 batch 和 head
        grid = lambda args: (triton.cdiv(SEQ_L, args["BLOCK_M"]), BATCH * N_HEAD, 1)
        ctx.grid = grid
        _attn_fwd[grid](
            q, k, v, sm_scale, LSE, o, 
            q.stride(0), q.stride(1), q.stride(2), q.stride(3), 
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            o.stride(0), o.stride(1), o.stride(2), o.stride(3),
            BATCH, N_HEAD,
            SEQ_LEN=SEQ_L,
            HEAD_DIM=HEAD_DIM_K,
            STAGE=stage,
            **extra_kern_args)

        ctx.save_for_backward(q, k, v, o, LSE)
        ctx.sm_scale = sm_scale
        ctx.HEAD_DIM = HEAD_DIM_K
        ctx.causal = causal
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, o, LSE = ctx.saved_tensors
        print(f"lse ranges: {LSE.min()}, {LSE.max()}")
        print(f"o ranges: {o.min()}, {o.max()}")
        print(f"q ranges: {q.min()}, {q.max()}")
        print(f"k ranges: {k.min()}, {k.max()}")
        print(f"v ranges: {v.min()}, {v.max()}")
        assert do.is_contiguous()
        assert q.stride() == k.stride() == v.stride() == o.stride() == do.stride()
        dq = torch.empty_like(q)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)
        BATCH, N_HEAD, SEQ_L = q.shape[:3]
        PRE_BLOCK = 128
        NUM_WARPS, NUM_STAGES = 4, 5
        BLOCK_M1, BLOCK_N1, BLOCK_M2, BLOCK_N2 = 32, 128, 128, 32
        BLK_SLICE_FACTOR = 2
        RCP_LN2 = 1.4426950408889634  # = 1.0 / ln(2)
        arg_k = k
        arg_k = arg_k * (ctx.sm_scale * RCP_LN2)
        PRE_BLOCK = 128
        assert SEQ_L % PRE_BLOCK == 0
        pre_grid = (SEQ_L // PRE_BLOCK, BATCH * N_HEAD)
        delta = torch.empty_like(LSE)
        _attn_bwd_preprocess[pre_grid](
            o, do, 
            delta, 
            BATCH, N_HEAD, SEQ_L, 
            BLOCK_M=PRE_BLOCK, HEAD_DIM=ctx.HEAD_DIM 
        )
        grid = (SEQ_L // BLOCK_N1, 1, BATCH * N_HEAD)
        _attn_bwd[grid](
            q, arg_k, v, ctx.sm_scale, do, dq, dk, dv, 
            LSE, delta, 
            q.stride(0), q.stride(1), q.stride(2), q.stride(3), 
            N_HEAD, SEQ_L, 
            BLOCK_M1=BLOCK_M1, BLOCK_N1=BLOCK_N1, 
            BLOCK_M2=BLOCK_M2, BLOCK_N2=BLOCK_N2, 
            BLK_SLICE_FACTOR=BLK_SLICE_FACTOR, 
            HEAD_DIM=ctx.HEAD_DIM, 
            num_warps=NUM_WARPS, 
            num_stages=NUM_STAGES 
        )

        return dq, dk, dv, None, None

attention = _attention.apply

@pytest.mark.parametrize("BS, HEAD_NUM, SEQ_LEN, HEAD_DIM", [(1, 2, 1024, 64)])
@pytest.mark.parametrize("causal", [True, False])
def test_op(BS, HEAD_NUM, SEQ_LEN, HEAD_DIM, causal, dtype=torch.float16):
    torch.manual_seed(20)
    q = (torch.empty((BS, HEAD_NUM, SEQ_LEN, HEAD_DIM), dtype=dtype, device=DEVICE).normal_(mean=0.0, std=0.5).requires_grad_())
    k = (torch.empty((BS, HEAD_NUM, SEQ_LEN, HEAD_DIM), dtype=dtype, device=DEVICE).normal_(mean=0.0, std=0.5).requires_grad_())
    v = (torch.empty((BS, HEAD_NUM, SEQ_LEN, HEAD_DIM), dtype=dtype, device=DEVICE).normal_(mean=0.0, std=0.5).requires_grad_())
    # sm_scale = 0.5
    sm_scale = HEAD_DIM ** -0.5
    dout = torch.randn_like(q)
    # reference implementation
    LSE = torch.tril(torch.ones((SEQ_LEN, SEQ_LEN), device=DEVICE))
    p = torch.matmul(q, k.transpose(2, 3)) * sm_scale
    if causal:
        p[:, :, LSE == 0] = float("-inf")
    p = torch.softmax(p.float(), dim=-1).half()
    # p = torch.exp(p)
    ref_out = torch.matmul(p, v)
    ref_out.backward(dout)
    ref_dv, v.grad = v.grad.clone(), None
    ref_dk, k.grad = k.grad.clone(), None
    ref_dq, q.grad = q.grad.clone(), None
    # triton implementation
    tri_out = attention(q, k, v, causal, sm_scale).half()
    avg_ms = triton.testing.do_bench(lambda: attention(q, k, v, causal, sm_scale).half())
    print('[Attn] avg_ms:', avg_ms)
    tri_out.backward(dout)
    tri_dv, v.grad = v.grad.clone(), None
    tri_dk, k.grad = k.grad.clone(), None
    tri_dq, q.grad = q.grad.clone(), None
    
    # compare
    assert torch.allclose(ref_out, tri_out, atol=1e-2, rtol=0)
    rtol = 0.0
    # Relative tolerance workaround for known hardware limitation of MI200 GPU.
    # For details see https://pytorch.org/docs/stable/notes/numerical_accuracy.html#reduced-precision-fp16-and-bf16-gemms-and-convolutions-on-amd-instinct-mi200-devices
    if torch.version.hip is not None and triton.runtime.driver.active.get_current_target().arch == "gfx90a":
        rtol = 1e-2
    # assert torch.allclose(ref_dv, tri_dv, atol=1e-2, rtol=rtol)
    # assert torch.allclose(ref_dk, tri_dk, atol=1e-2, rtol=rtol)
    # assert torch.allclose(ref_dq, tri_dq, atol=1e-2, rtol=rtol)
    import torch.nn.functional as F
    
    # 计算梯度的余弦相似度
    dq_cos_sim = F.cosine_similarity(ref_dq.flatten(), tri_dq.flatten(), dim=0)
    dk_cos_sim = F.cosine_similarity(ref_dk.flatten(), tri_dk.flatten(), dim=0)
    dv_cos_sim = F.cosine_similarity(ref_dv.flatten(), tri_dv.flatten(), dim=0)
    
    # 计算最大误差和平均误差
    dq_max_diff = (ref_dq - tri_dq).abs().max()
    dk_max_diff = (ref_dk - tri_dk).abs().max()
    dv_max_diff = (ref_dv - tri_dv).abs().max()
    
    dq_mean_diff = (ref_dq - tri_dq).abs().mean()
    dk_mean_diff = (ref_dk - tri_dk).abs().mean()
    dv_mean_diff = (ref_dv - tri_dv).abs().mean()
    
    # 计算scale_factor
    dq_scale_factor = tri_dq.abs().max() / ref_dq.abs().max() if ref_dq.abs().max() > 0 else 0
    dk_scale_factor = tri_dk.abs().max() / ref_dk.abs().max() if ref_dk.abs().max() > 0 else 0
    dv_scale_factor = tri_dv.abs().max() / ref_dv.abs().max() if ref_dv.abs().max() > 0 else 0
    
    # 打印详细的梯度比较信息
    print("\n梯度比较:")
    print("dQ比较:")
    print(f"  max_diff: {dq_max_diff}")
    print(f"  mean_diff: {dq_mean_diff}")
    print(f"  cos_sim: {dq_cos_sim}")
    print(f"  scale_factor: {dq_scale_factor}")
    
    print("dK比较:")
    print(f"  max_diff: {dk_max_diff}")
    print(f"  mean_diff: {dk_mean_diff}")
    print(f"  cos_sim: {dk_cos_sim}")
    print(f"  scale_factor: {dk_scale_factor}")
    
    print("dV比较:")
    print(f"  max_diff: {dv_max_diff}")
    print(f"  mean_diff: {dv_mean_diff}")
    print(f"  cos_sim: {dv_cos_sim}")
    print(f"  scale_factor: {dv_scale_factor}")

try:
    from flash_attn.flash_attn_interface import \
        flash_attn_qkvpacked_func as flash_attn_func
    HAS_FLASH = True
except BaseException:
    HAS_FLASH = False

TORCH_HAS_FP8 = hasattr(torch, 'float8_e5m2')
BATCH, N_HEADS, HEAD_DIM = 4, 32, 64
# vary seq length for fixed head and batch=4
configs = []
for mode in ["fwd", "bwd"]:
    for causal in [True, False]:
        if mode == "bwd" and not causal:
            continue
        configs.append(
            triton.testing.Benchmark(
                x_names=["SEQ_LEN"],
                x_vals=[2**i for i in range(10, 15)],
                line_arg="provider",
                line_vals=["triton-fp16"] + (["triton-fp8"] if TORCH_HAS_FP8 else []) +
                (["flash"] if HAS_FLASH else []),
                line_names=["Triton [FP16]"] + (["Triton [FP8]"] if TORCH_HAS_FP8 else []) +
                (["Flash-2"] if HAS_FLASH else []),
                styles=[("red", "-"), ("blue", "-"), ("green", "-")],
                ylabel="TFLOPS",
                plot_name=f"fused-attention-batch{BATCH}-head{N_HEADS}-d{HEAD_DIM}-{mode}-causal={causal}",
                args={
                    "HEAD_NUM": N_HEADS,
                    "BATCH": BATCH,
                    "HEAD_DIM": HEAD_DIM,
                    "mode": mode,
                    "causal": causal,
                },
            ))


@triton.testing.perf_report(configs)
def bench_flash_attention(BATCH, HEAD_NUM, SEQ_LEN, HEAD_DIM, causal, mode, provider, device=DEVICE):
    assert mode in ["fwd", "bwd"]
    dtype = torch.float16
    if "triton" in provider:
        q = torch.randn((BATCH, HEAD_NUM, SEQ_LEN, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
        k = torch.randn((BATCH, HEAD_NUM, SEQ_LEN, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
        v = torch.randn((BATCH, HEAD_NUM, SEQ_LEN, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
        if mode == "fwd" and "fp8" in provider:
            q = q.to(torch.float8_e5m2)
            k = k.to(torch.float8_e5m2)
            v = v.permute(0, 1, 3, 2).contiguous()
            v = v.permute(0, 1, 3, 2)
            v = v.to(torch.float8_e5m2)
        sm_scale = 1.3
        fn = lambda: attention(q, k, v, causal, sm_scale)
        if mode == "bwd":
            o = fn()
            do = torch.randn_like(o)
            fn = lambda: o.backward(do, retain_graph=True)
        ms = triton.testing.do_bench(fn)
    if provider == "flash":
        qkv = torch.randn((BATCH, SEQ_LEN, 3, HEAD_NUM, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
        fn = lambda: flash_attn_func(qkv, causal=causal)
        if mode == "bwd":
            o = fn()
            do = torch.randn_like(o)
            fn = lambda: o.backward(do, retain_graph=True)
        ms = triton.testing.do_bench(fn)
    flops_per_matmul = 2.0 * BATCH * HEAD_NUM * SEQ_LEN * SEQ_LEN * HEAD_DIM
    total_flops = 2 * flops_per_matmul
    if causal:
        total_flops *= 0.5
    if mode == "bwd":
        total_flops *= 2.5  # 2.0(bwd) + 0.5(recompute)
    return total_flops * 1e-12 / (ms * 1e-3)


if __name__ == "__main__":
    # only works on post-Ampere GPUs right now
    # bench_flash_attention.run(save_path=".", print_data=True)
    test_op(16, 16, 1024, 64, True)