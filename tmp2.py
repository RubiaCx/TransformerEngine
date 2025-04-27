
@triton.jit
def _attn_bwd_nomask(Q, K, V,
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
    offs_n = start_n + tl.arange(0, BLOCK_KV1)

    dv = tl.zeros([BLOCK_KV1, HEAD_DIM], dtype=tl.float32)
    dk = tl.zeros([BLOCK_KV1, HEAD_DIM], dtype=tl.float32)

    # load K and V
    k = tl.load(K + offs_n[:, None] * stride_qs + offs_k[None, :] * stride_qd)
    v = tl.load(V + offs_n[:, None] * stride_qs + offs_k[None, :] * stride_qd)

    num_steps = SEQ_LEN // BLOCK_Q1
    dk, dv = _attn_bwd_dkdv( 
        dk, dv, 
        Q, k, v, 
        sm_scale, 
        DO, LSE, D, 
        stride_qs, stride_qd, 
        HEAD_NUM, SEQ_LEN,
        BLOCK_Q1, BLOCK_KV1, HEAD_DIM, 
        start_n, 0, num_steps, 
        MASK=False
    )

    dv_ptrs = DV + offs_n[:, None] * stride_qs + offs_k[None, :] * stride_qd
    tl.store(dv_ptrs, dv)

    dk_ptrs = DK + offs_n[:, None] * stride_qs + offs_k[None, :] * stride_qd
    dk = dk * sm_scale
    tl.store(dk_ptrs, dk)

    start_m = pid * BLOCK_Q2
    offs_m = start_m + tl.arange(0, BLOCK_Q2)

    q = tl.load(Q + offs_m[:, None] * stride_qs + offs_k[None, :] * stride_qd)
    dq = tl.zeros([BLOCK_Q2, HEAD_DIM], dtype=tl.float32)
    do = tl.load(DO + offs_m[:, None] * stride_qs + offs_k[None, :] * stride_qd)

    m = tl.load(LSE + offs_m)
    m = m[:, None]

    num_steps = SEQ_LEN // BLOCK_KV2
    dq = _attn_bwd_dq(dq, q, K, V, 
                      do, m, D, 
                      stride_qs, stride_qd, 
                      HEAD_NUM, SEQ_LEN, 
                      BLOCK_Q2, BLOCK_KV2, HEAD_DIM, 
                      start_m, 0, num_steps,  
                      MASK=False 
                      )

    dq_ptrs = DQ + offs_m[:, None] * stride_qs + offs_k[None, :] * stride_qd
    dq = dq * LN2
    tl.store(dq_ptrs, dq)