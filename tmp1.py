@triton.jit
def _attn_bwd_causal(Q, K, V,
                    sm_scale, 
                    DO, DQ, DK, DV, 
                    LSE, D,
                    # shared by Q/K/V/DO.
                    stride_qb, stride_qh, stride_qs, stride_qd,
                    HEAD_NUM, SEQ_LEN, GROUPS,
                    BLOCK_Q1: tl.constexpr, 
                    BLOCK_KV1: tl.constexpr, 
                    BLOCK_Q2: tl.constexpr, 
                    BLOCK_KV2: tl.constexpr, 
                    BLK_SLICE_FACTOR: tl.constexpr, 
                    HEAD_DIM: tl.constexpr):
    pid = tl.program_id(0)
    bhid = tl.program_id(1)
    off_chz = (bhid * SEQ_LEN).to(tl.int64)
    adj = (stride_qh * (bhid % HEAD_NUM) + stride_qb * (bhid // HEAD_NUM)).to(tl.int64)

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

    start_n = pid * BLOCK_KV1
    start_m = start_n

    MASK_BLOCK_Q1: tl.constexpr = BLOCK_Q1 // BLK_SLICE_FACTOR
    offs_n = start_n + tl.arange(0, BLOCK_KV1)

    dv = tl.zeros([BLOCK_KV1, HEAD_DIM], dtype=tl.float32)
    dk = tl.zeros([BLOCK_KV1, HEAD_DIM], dtype=tl.float32)

    # load K and V: they stay in SRAM throughout the inner loop.
    k = tl.load(K + offs_n[:, None] * stride_qs + offs_k[None, :] * stride_qd)
    v = tl.load(V + offs_n[:, None] * stride_qs + offs_k[None, :] * stride_qd)

    num_steps = BLOCK_KV1 // MASK_BLOCK_Q1

    dk, dv = _attn_bwd_dkdv(dk, dv, 
                            Q, k, v, 
                            sm_scale, 
                            DO, LSE, D, 
                            stride_qs, stride_qd, 
                            HEAD_NUM, SEQ_LEN, 
                            MASK_BLOCK_Q1, BLOCK_KV1, HEAD_DIM, 
                            start_n, start_m, num_steps, 
                            MASK=True 
                            )

    start_m += num_steps * MASK_BLOCK_Q1
    num_steps = (SEQ_LEN - start_m) // BLOCK_Q1

    # Compute dK and dV for non-masked blocks.
    dk, dv = _attn_bwd_dkdv( 
        dk, dv, 
        Q, k, v, 
        sm_scale, 
        DO, LSE, D, 
        stride_qs, stride_qd, 
        HEAD_NUM, SEQ_LEN,
        BLOCK_Q1, BLOCK_KV1, HEAD_DIM, 
        start_n, start_m, num_steps, 
        MASK=False
    )
    dv_ptrs = DV + offs_n[:, None] * stride_qs + offs_k[None, :] * stride_qd
    tl.store(dv_ptrs, dv)

    dk_ptrs = DK + offs_n[:, None] * stride_qs + offs_k[None, :] * stride_qd
    dk = dk * sm_scale
    tl.store(dk_ptrs, dk)

    # THIS BLOCK DOES DQ:
    start_m = pid * BLOCK_Q2
    end_n = start_m + BLOCK_Q2

    MASK_BLOCK_Q2: tl.constexpr = BLOCK_KV2 // BLK_SLICE_FACTOR
    offs_m = start_m + tl.arange(0, BLOCK_Q2)

    q = tl.load(Q + offs_m[:, None] * stride_qs + offs_k[None, :] * stride_qd)
    dq = tl.zeros([BLOCK_Q2, HEAD_DIM], dtype=tl.float32)
    do = tl.load(DO + offs_m[:, None] * stride_qs + offs_k[None, :] * stride_qd)

    m = tl.load(LSE + offs_m)
    m = m[:, None]

    # Compute dQ for masked (diagonal) blocks.
    # NOTE: This code scans each row of QK^T backward (from right to left,
    # but inside each call to _attn_bwd_dq, from left to right), but that's
    # not due to anything important.  I just wanted to reuse the loop
    # structure for dK & dV above as much as possible.
    num_steps = BLOCK_Q2 // MASK_BLOCK_Q2
    dq = _attn_bwd_dq(dq, q, K, V, 
                      do, m, D, 
                      stride_qs, stride_qd, 
                      HEAD_NUM, SEQ_LEN, 
                      BLOCK_Q2, MASK_BLOCK_Q2, HEAD_DIM, 
                      start_m, end_n - num_steps * MASK_BLOCK_Q2, num_steps, 
                      MASK=True 
                      )
    end_n -= num_steps * MASK_BLOCK_Q2
    # stage 2
    num_steps = end_n // BLOCK_KV2
    dq = _attn_bwd_dq(dq, q, K, V, 
                      do, m, D, 
                      stride_qs, stride_qd, 
                      HEAD_NUM, SEQ_LEN, 
                      BLOCK_Q2, BLOCK_KV2, HEAD_DIM, 
                      start_m, end_n - num_steps * BLOCK_KV2, num_steps, 
                      MASK=False 
                      )
    # Write back dQ.
    dq_ptrs = DQ + offs_m[:, None] * stride_qs + offs_k[None, :] * stride_qd
    dq = dq * LN2
    tl.store(dq_ptrs, dq)
