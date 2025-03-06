# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import functools
import logging
import math
import os
from importlib.metadata import version
from typing import Any, Dict, List, Tuple, Union, Optional
from contextlib import contextmanager

import pytest
import torch

from transformer_engine.common import recipe
from transformer_engine.pytorch import TransformerLayer, fp8_autocast, fp8_model_init
from transformer_engine.pytorch.attention import (
    DotProductAttention,
    MultiheadAttention,
    RotaryPositionEmbedding,
    get_attention_backend,
    _flash_attn_2_plus,
    _flash_attn_2_3_plus,
    check_set_window_size,
    AttentionParams,
    _attention_backends,
)
from transformer_engine.pytorch.constants import TE_DType
import transformer_engine.pytorch.cpp_extensions as ext
from transformer_engine.pytorch.cpp_extensions.fused_attn import (
    AttnBiasType,
    AttnMaskType,
    FusedAttnBackend,
    QKVLayout,
    fused_attn_bwd,
    fused_attn_fwd,
)
from transformer_engine.pytorch.distributed import CudaRNGStatesTracker
import transformer_engine.pytorch.fp8 as fp8
from transformer_engine.pytorch.module.base import TransformerEngineBaseModule
from transformer_engine.pytorch.utils import (
    get_device_compute_capability,
    init_method_normal,
    scaled_init_method_normal,
    is_bf16_compatible,
)
from transformer_engine.pytorch.utils import get_cudnn_version

# Initialize RNG state
seed = 1234
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
_cpu_rng_state = torch.get_rng_state()
_cuda_rng_state = torch.cuda.get_rng_state()

def reset_rng_states() -> None:
    """Revert back to initial RNG state"""
    torch.set_rng_state(_cpu_rng_state)
    torch.cuda.set_rng_state(_cuda_rng_state)


@pytest.fixture(autouse=True)
def reset_global_fp8_state():
    yield
    fp8.FP8GlobalStateManager.reset()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

class ModelConfig:
    def __init__(
        self,
        batch_size: int,
        num_heads: int,
        num_gqa_groups: int,
        head_dim_qk: int,
        max_seqlen_q: int,
        max_seqlen_kv: int,
        dropout_p: float,
        attn_mask_type: str,
        attn_bias_type: str,
        head_dim_v: int = None,
        alibi_type: str = "none",
        num_layers: int = 1,
        bias_shape: str = "1hss",
        window_size: Tuple[int, int] = (-1, -1),
    ):
        self.batch_size = batch_size
        self.num_heads = num_heads
        self.num_gqa_groups = num_gqa_groups
        self.head_dim_qk = head_dim_qk
        self.head_dim_v = head_dim_qk if head_dim_v is None else head_dim_v
        self.hidden_size = num_heads * head_dim_qk
        self.hidden_size_kv = num_gqa_groups * self.head_dim_v
        self.max_seqlen_q = max_seqlen_q
        self.max_seqlen_kv = max_seqlen_kv
        self.dropout_p = dropout_p
        self.attn_mask_type = attn_mask_type
        self.attn_bias_type = attn_bias_type
        self.alibi_type = alibi_type
        self.attn_type = "self" if (max_seqlen_q == max_seqlen_kv) else "cross"
        self.num_layers = num_layers
        self.bias_shape = bias_shape
        self.window_size = window_size


def _run_dot_product_attention(
    dtype: torch.dtype,
    config: ModelConfig,
    backend: str,
    ckpt_attn: bool,
    qkv_layout: str,
    workspace_opt: bool,
    pad_between_seqs: bool,
    is_training: bool,
) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Run DotProductAttention module with one forward pass and one backward pass"""

    # Set RNG and environment varables
    reset_rng_states()
    os.environ["NVTE_FLASH_ATTN"] = "0"
    os.environ["NVTE_FUSED_ATTN"] = "0"
    if backend == "FlashAttention":
        os.environ["NVTE_FLASH_ATTN"] = "1"
    if backend == "FusedAttention":
        os.environ["NVTE_FUSED_ATTN"] = "1"
        os.environ["NVTE_FUSED_ATTN_FORCE_WORKSPACE_OPT"] = "1" if workspace_opt else "0"
    if backend == "SageAttention":
        os.environ["NVTE_SAGE_ATTN"] = "1"
    global _attention_backends
    _attention_backends["backend_selection_requires_update"] = True

    # Create seqlens
    qkv_format = "".join([i for i in qkv_layout.split("_")[0] if i.isalpha()])
    if "padding" in config.attn_mask_type or qkv_format == "thd":
        if config.attn_type == "self":
            seqlens_q = torch.randint(
                1, config.max_seqlen_q, [config.batch_size], dtype=torch.int32, device="cuda"
            )
            seqlens_kv = seqlens_q
        if config.attn_type == "cross":
            seqlens_q = torch.randint(
                1, config.max_seqlen_q, [config.batch_size], dtype=torch.int32, device="cuda"
            )
            seqlens_kv = torch.randint(
                1, config.max_seqlen_kv, [config.batch_size], dtype=torch.int32, device="cuda"
            )
    else:
        seqlens_q = torch.full(
            [config.batch_size], config.max_seqlen_q, dtype=torch.int32, device="cuda"
        )
        seqlens_kv = torch.full(
            [config.batch_size], config.max_seqlen_kv, dtype=torch.int32, device="cuda"
        )
    cu_seqlens_q = torch.zeros(config.batch_size + 1, dtype=torch.int32, device="cuda")
    cu_seqlens_kv = torch.zeros(config.batch_size + 1, dtype=torch.int32, device="cuda")
    cu_seqlens_q[1:] = torch.cumsum(seqlens_q, dim=0)
    cu_seqlens_kv[1:] = torch.cumsum(seqlens_kv, dim=0)

    seqlens_q_after_pad = seqlens_q.clone()
    seqlens_kv_after_pad = seqlens_kv.clone()
    cu_seqlens_q_after_pad = cu_seqlens_q.clone()
    cu_seqlens_kv_after_pad = cu_seqlens_kv.clone()
    pad_len = [0] * config.batch_size
    if pad_between_seqs:
        max_pad_len = 3
        pad_len = torch.randint(0, max_pad_len + 1, [config.batch_size], device="cuda")  # 3
        seqlens_q_after_pad = seqlens_q + pad_len
        seqlens_kv_after_pad = seqlens_kv + pad_len
        cu_seqlens_q_after_pad[1:] = torch.cumsum(seqlens_q_after_pad, dim=0)
        cu_seqlens_kv_after_pad[1:] = torch.cumsum(seqlens_kv_after_pad, dim=0)

    # Create attention mask if padding
    attention_mask = None
    if "padding" in config.attn_mask_type:
        if config.attn_type == "self":
            attention_mask_q = torch.Tensor([]).to(dtype=torch.bool)
            for i in range(config.batch_size):
                attention_mask_q = torch.cat(
                    [
                        attention_mask_q,
                        torch.Tensor(
                            [False] * seqlens_q[i] + [True] * (config.max_seqlen_q - seqlens_q[i])
                        )
                        .to(dtype=torch.bool)
                        .unsqueeze(0)
                        .unsqueeze(0)
                        .unsqueeze(0),
                    ],
                    dim=0,
                )
            attention_mask = attention_mask_q.to(device="cuda")
        if config.attn_type == "cross":
            attention_mask_q = torch.Tensor([]).to(dtype=torch.bool)
            attention_mask_kv = torch.Tensor([]).to(dtype=torch.bool)
            for i in range(config.batch_size):
                attention_mask_q = torch.cat(
                    [
                        attention_mask_q,
                        torch.Tensor(
                            [False] * seqlens_q[i] + [True] * (config.max_seqlen_q - seqlens_q[i])
                        )
                        .to(dtype=torch.bool)
                        .unsqueeze(0)
                        .unsqueeze(0)
                        .unsqueeze(0),
                    ],
                    dim=0,
                )
                attention_mask_kv = torch.cat(
                    [
                        attention_mask_kv,
                        torch.Tensor(
                            [False] * seqlens_kv[i]
                            + [True] * (config.max_seqlen_kv - seqlens_kv[i])
                        )
                        .to(dtype=torch.bool)
                        .unsqueeze(0)
                        .unsqueeze(0)
                        .unsqueeze(0),
                    ],
                    dim=0,
                )
            attention_mask = (
                attention_mask_q.to(device="cuda"),
                attention_mask_kv.to(device="cuda"),
            )

    alibi_slopes = None
    if config.attn_bias_type == "alibi" and config.alibi_type == "custom":
        if config.bias_shape == "1hss":
            alibi_slopes = (
                torch.randn(config.num_heads).abs().to(dtype=torch.float32, device="cuda")
            )
        if config.bias_shape == "bhss":
            alibi_slopes = (
                torch.randn(config.batch_size, config.num_heads)
                .abs()
                .to(dtype=torch.float32, device="cuda")
            )

    # Create input tensors
    dim_to_num = {
        "b": config.batch_size,
        "sq": config.max_seqlen_q,
        "skv": config.max_seqlen_kv,
        "h": config.num_heads,
        "hg": config.num_gqa_groups,
        "dqk": config.head_dim_qk,
        "dv": config.head_dim_v,
        "t": cu_seqlens_q_after_pad[-1],
        "tg": cu_seqlens_kv_after_pad[-1],
        "3": 3,
        "2": 2,
        "1": 1,
    }
    inp = []
    inp_orig = []
    for i, layout in enumerate(qkv_layout.split("_")):
        layout = "_".join(layout)
        if i == 0:
            layout = layout.replace("s", "sq")
        else:
            layout = layout.replace("s", "skv")
            layout = layout.replace("h", "hg")
            layout = layout.replace("t", "tg")
        if i == 2:
            layout = layout.replace("d", "dv")
        else:
            layout = layout.replace("d", "dqk")
        tensor_shape = [dim_to_num[j] for j in layout.split("_")]
        tensor = 0.1 * torch.randn(tensor_shape, dtype=dtype, device="cuda")
        tensor_orig = tensor
        if qkv_format == "thd" and pad_between_seqs:
            tensor_orig = torch.Tensor([]).to(device="cuda", dtype=dtype)
            if layout in ["t_h_dqk", "t_3_h_dqk", "t_h_3_dqk"]:
                for i in range(1, config.batch_size + 1):
                    valid_range = (
                        cu_seqlens_q_after_pad[i - 1],
                        cu_seqlens_q_after_pad[i] - pad_len[i - 1],
                    )
                    pad_range = (
                        cu_seqlens_q_after_pad[i] - pad_len[i - 1],
                        cu_seqlens_q_after_pad[i],
                    )
                    tensor[pad_range[0] : pad_range[1]] = 0.0
                    tensor_orig = torch.cat(
                        [tensor_orig, tensor[valid_range[0] : valid_range[1]]], dim=0
                    )
            if layout in ["tg_hg_dqk", "tg_2_hg_dqk", "tg_hg_2_dqk", "tg_hg_dv"]:
                for i in range(1, config.batch_size + 1):
                    valid_range = (
                        cu_seqlens_kv_after_pad[i - 1],
                        cu_seqlens_kv_after_pad[i] - pad_len[i - 1],
                    )
                    pad_range = (
                        cu_seqlens_kv_after_pad[i] - pad_len[i - 1],
                        cu_seqlens_kv_after_pad[i],
                    )
                    tensor[pad_range[0] : pad_range[1]] = 0.0
                    tensor_orig = torch.cat(
                        [tensor_orig, tensor[valid_range[0] : valid_range[1]]], dim=0
                    )
        tensor_count = 1
        split_dim = 0
        for dim, l in enumerate(layout.split("_")):
            if l.isdigit():
                tensor_count = int(l)
                split_dim = dim
                break
        tensors = torch.split(tensor, 1, dim=split_dim) if split_dim != 0 else [tensor]
        tensors_orig = (
            torch.split(tensor_orig, 1, dim=split_dim) if split_dim != 0 else [tensor_orig]
        )
        for j in range(tensor_count):
            if split_dim != 0:
                inp.append(tensors[j].squeeze(split_dim))
                inp_orig.append(tensors_orig[j].squeeze(split_dim))
            else:
                inp.append(tensors[j])
                inp_orig.append(tensors_orig[j])
    for i in range(3):
        inp[i].requires_grad = True
        inp_orig[i].requires_grad = True

    # Create output gradient
    qkv_format_kv = "_".join(qkv_format)
    qkv_format_kv = qkv_format_kv.replace("s", "sq")
    qkv_format_kv = qkv_format_kv.replace("d", "dv")
    out_grad_shape = [dim_to_num[i] for i in qkv_format_kv.split("_")]
    out_grad_shape_new = [*out_grad_shape[:-2], out_grad_shape[-2] * out_grad_shape[-1]]
    out_grad = 0.001 * torch.randint(0, 200, out_grad_shape_new, dtype=dtype, device="cuda")
    out_grad_orig = out_grad
    if qkv_format == "thd" and pad_between_seqs:
        out_grad_orig = torch.Tensor([]).to(device="cuda", dtype=dtype)
        if qkv_format_kv == "t_h_dv":
            for i in range(1, config.batch_size + 1):
                valid_range = (
                    cu_seqlens_q_after_pad[i - 1],
                    cu_seqlens_q_after_pad[i] - pad_len[i - 1],
                )
                pad_range = (cu_seqlens_q_after_pad[i] - pad_len[i - 1], cu_seqlens_q_after_pad[i])
                out_grad[pad_range[0] : pad_range[1]] = 0.0
                out_grad_orig = torch.cat(
                    [out_grad_orig, out_grad[valid_range[0] : valid_range[1]]], dim=0
                )

    # Create bias
    if config.attn_bias_type in ["no_bias", "alibi"]:
        bias = None
    if config.attn_bias_type == "post_scale_bias":
        shape = "_".join(config.bias_shape)
        shape = shape.replace("_s_s", "_sq_skv")
        tensor_shape = [dim_to_num[j] for j in shape.split("_")]
        bias = torch.randn(tensor_shape, dtype=dtype, device="cuda")
        if config.bias_shape != "1hss":
            bias.requires_grad = False

    # Create RNG
    _DUMMY_CUDA_RNG_STATE_TRACKER = CudaRNGStatesTracker()
    _DUMMY_CUDA_RNG_STATE_TRACKER.add("model-parallel-rng", seed)

    def get_dummy_cuda_rng_tracker() -> CudaRNGStatesTracker:
        """Get cuda rng tracker."""
        return _DUMMY_CUDA_RNG_STATE_TRACKER

    # Set up model
    block = DotProductAttention(
        config.num_heads,
        (config.head_dim_qk, config.head_dim_v),
        num_gqa_groups=config.num_gqa_groups,
        qkv_format=qkv_format,
        attn_mask_type=config.attn_mask_type,
        sequence_parallel=False,
        tp_size=1,
        get_rng_state_tracker=get_dummy_cuda_rng_tracker,
        tp_group=None,
        layer_number=1,
        attention_type=config.attn_type,
    ).to(dtype=dtype, device="cuda")
    block.train()
    for p in block.parameters():
        p.requires_grad = True

    # Run a forward and backward pass
    if backend in ["FlashAttention", "UnfusedDotProductAttention", "SageAttention"]:
        q = inp_orig[0]
        k = inp_orig[1]
        v = inp_orig[2]
        d_out = out_grad_orig
    if backend == "FusedAttention":
        q = inp[0]
        k = inp[1]
        v = inp[2]
        d_out = out_grad
    out = block(
        q,
        k,
        v,
        window_size=config.window_size,
        attention_mask=attention_mask,
        qkv_format=qkv_format,
        max_seqlen_q=config.max_seqlen_q,
        max_seqlen_kv=config.max_seqlen_kv,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_kv=cu_seqlens_kv,
        cu_seqlens_q_padded=cu_seqlens_q_after_pad if backend == "FusedAttention" else None,
        cu_seqlens_kv_padded=cu_seqlens_kv_after_pad if backend == "FusedAttention" else None,
        attn_mask_type=config.attn_mask_type,
        checkpoint_core_attention=ckpt_attn,
        core_attention_bias_type=config.attn_bias_type,
        core_attention_bias=bias,
        alibi_slopes=alibi_slopes,
        fast_zero_fill=True,
    )
    if is_training:
        out.backward(d_out)

    # 验证梯度计算图
    # assert out.requires_grad, "输出未启用梯度"
    # assert q.grad_fn is not None, "查询层未参与计算图"

    if backend in ["FlashAttention", "UnfusedDotProductAttention"]:
        if is_training:
            return out, (q.grad, k.grad, v.grad)
        else:
            return out, (None, None, None)
    if backend == "FusedAttention":
        if qkv_format == "thd" and pad_between_seqs:
            out_orig = torch.Tensor([]).to(device="cuda", dtype=dtype)
            if is_training:
                q_grad_orig = torch.Tensor([]).to(device="cuda", dtype=dtype)
                k_grad_orig = torch.Tensor([]).to(device="cuda", dtype=dtype)
                v_grad_orig = torch.Tensor([]).to(device="cuda", dtype=dtype)
            for i in range(1, config.batch_size + 1):
                valid_range_q = (
                    cu_seqlens_q_after_pad[i - 1],
                    cu_seqlens_q_after_pad[i] - pad_len[i - 1],
                )
                valid_range_kv = (
                    cu_seqlens_kv_after_pad[i - 1],
                    cu_seqlens_kv_after_pad[i] - pad_len[i - 1],
                )
                out_orig = torch.cat([out_orig, out[valid_range_q[0] : valid_range_q[1]]], dim=0)
                if is_training:
                    q_grad_orig = torch.cat(
                        [q_grad_orig, q.grad[valid_range_q[0] : valid_range_q[1]]], dim=0
                    )
                    k_grad_orig = torch.cat(
                        [k_grad_orig, k.grad[valid_range_kv[0] : valid_range_kv[1]]], dim=0
                    )
                    v_grad_orig = torch.cat(
                        [v_grad_orig, v.grad[valid_range_kv[0] : valid_range_kv[1]]], dim=0
                    )
            if is_training:
                return out_orig, (q_grad_orig, k_grad_orig, v_grad_orig)
            else:
                return out_orig, (None, None, None)
        else:
            if is_training:
                return out, (q.grad, k.grad, v.grad)
            else:
                return out, (None, None, None)

@contextmanager
def logging_context(highest_level=logging.WARNING):
    previous_level = logging.root.manager.disable
    logging.disable(highest_level)
    try:
        yield
    finally:
        logging.disable(previous_level)

def _get_attention_backends(
    config: ModelConfig,
    qkv_dtype: torch.dtype,
    qkv_layout: str,
    window_size: Tuple[int, int] = (-1, -1),
    pad_between_seqs: bool = False,
    context_parallel: bool = False,
    deterministic: bool = False,
    fp8: bool = False,
    fp8_meta: Optional[Dict[str, Any]] = None,
) -> Tuple[List, List]:
    """Check if what attention backends support a model configuration"""

    os.environ["NVTE_FLASH_ATTN"] = "1"
    os.environ["NVTE_FUSED_ATTN"] = "1"
    os.environ["NVTE_UNFUSED_ATTN"] = "1"
    os.environ["NVTE_SAGE_ATTN"] = "1"
    global _attention_backends
    _attention_backends["backend_selection_requires_update"] = True

    alibi_slopes_shape = None
    if config.attn_bias_type == "alibi" and config.alibi_type == "custom":
        if config.bias_shape == "1hss":
            alibi_slopes_shape = [config.num_heads]
        if config.bias_shape == "bhss":
            alibi_slopes_shape = [config.batch_size, config.num_heads]

    core_attention_bias_shape = (
        config.bias_shape if config.attn_bias_type == "post_scale_bias" else None
    )
    core_attention_bias_requires_grad = False
    # d=256 is supported by cuDNN 9.0+ for inference but not training
    if (
        config.attn_bias_type == "post_scale_bias"
        and config.head_dim_qk <= 128
        and config.head_dim_v <= 128
    ):
        core_attention_bias_requires_grad = True

    fused_attn_backends = []
    available_backends = None
    fused_attention_backend = None

    def test():
        attention_params = AttentionParams(
            qkv_dtype=qkv_dtype,
            qkv_layout=qkv_layout,
            batch_size=config.batch_size,
            num_heads=config.num_heads,
            num_gqa_groups=config.num_gqa_groups,
            max_seqlen_q=config.max_seqlen_q,
            max_seqlen_kv=config.max_seqlen_kv,
            head_dim_qk=config.head_dim_qk,
            head_dim_v=config.head_dim_v,
            attn_mask_type=config.attn_mask_type,
            window_size=window_size,
            alibi_slopes_shape=alibi_slopes_shape,
            core_attention_bias_type=config.attn_bias_type,
            core_attention_bias_shape=core_attention_bias_shape,
            core_attention_bias_requires_grad=core_attention_bias_requires_grad,
            pad_between_seqs=pad_between_seqs,
            attention_dropout=config.dropout_p,
            context_parallel=context_parallel,
            deterministic=deterministic,
            fp8=fp8,
            fp8_meta=fp8_meta,
        )
        _, _, fused_attention_backend, _, _, available_backends = get_attention_backend(
            attention_params
        )
        return available_backends, fused_attention_backend

    backends = {0: "F16_max512_seqlen", 1: "F16_arbitrary_seqlen", 2: "FP8"}
    with logging_context():
        for i in range(3):
            os.environ["NVTE_FUSED_ATTN_BACKEND"] = str(i)
            _attention_backends["backend_selection_requires_update"] = True
            available_backends, fused_attention_backend = test()
            if fused_attention_backend == FusedAttnBackend[backends[i]]:
                fused_attn_backends.append(fused_attention_backend)
    return available_backends, fused_attn_backends


model_configs_base = {
    "base": ModelConfig(
        batch_size=8,
        num_heads=16,
        num_gqa_groups=16,
        max_seqlen_q=128,
        max_seqlen_kv=128,
        head_dim_qk=64,  # ≤128
        head_dim_v=64,
        attn_mask_type="causal",  # 支持的掩码类型
        attn_bias_type="no_bias",
        dropout_p=0.0,
        window_size=(-1, -1),  # 禁用滑动窗口
    ),
}


param_types = [torch.float16]
if is_bf16_compatible():  # bf16 requires sm_80 or higher
    param_types.append(torch.bfloat16)
param_types_lean = [torch.bfloat16]


@pytest.mark.skipif(get_cudnn_version() < (8, 9, 1), reason="cuDNN 8.9.1+ is required.")
@pytest.mark.parametrize("dtype", param_types)
@pytest.mark.parametrize("model_configs", [model_configs_base])
@pytest.mark.parametrize("model", model_configs_base.keys())
@pytest.mark.parametrize("ckpt_attn", [False])
@pytest.mark.parametrize("workspace_opt", [True, False])
@pytest.mark.parametrize("qkv_layout", [None])
@pytest.mark.parametrize("swa", [False])
@pytest.mark.parametrize("pad_between_seqs", [False])
def test_dot_product_attention(
    dtype, model_configs, model, ckpt_attn, workspace_opt, qkv_layout, swa, pad_between_seqs
):
    """Test DotProductAttention module"""
    # Get configs
    tols = dict(atol=5e-3, rtol=5e-3)
    if dtype == torch.bfloat16:
        tols = dict(atol=2.5e-2, rtol=2.5e-2)
    config = model_configs[model]
    is_mla = config.head_dim_qk != config.head_dim_v
    if qkv_layout is None:
        if config.attn_type == "self":
            qkv_layout = "sb3hd" if not is_mla else "sbhd_sbhd_sbhd"
        else:
            qkv_layout = "bshd_bs2hd" if not is_mla else "bshd_bshd_bshd"
    if "3" in qkv_layout and config.attn_type == "cross":
        pytest.skip("No need to test this layout for cross attention")

    # Test backend availability
    window_size = (-1, -1)
    if swa:
        window_size = tuple(torch.randint(0, config.max_seqlen_kv, [2], dtype=torch.int32).tolist())
    config.window_size = check_set_window_size(config.attn_mask_type, window_size)
    available_backends, fused_attn_backends = _get_attention_backends(
        config,
        qkv_dtype=dtype,
        qkv_layout=qkv_layout,
        window_size=config.window_size,
        pad_between_seqs=pad_between_seqs,
    )
    flash_attn_supported, fused_attn_supported, unfused_attn_supported, sage_attn_supported = available_backends
    # FlashAttention does not support pad_between_seqs, but _run_dot_product_attention
    # mannually pads and unpads the input and output of FlashAttention for testing purposes
    if pad_between_seqs and not (
        config.max_seqlen_q != config.max_seqlen_kv
        and config.attn_mask_type in ["causal", "padding_causal"]
    ):
        flash_attn_supported = True

    # Skip if only unfused backend is supported
    if (len(fused_attn_backends) + flash_attn_supported + unfused_attn_supported + sage_attn_supported) < 2:
        pytest.skip("Less than two backends to compare.")

    is_training = False # config.head_dim_qk <= 128 and config.head_dim_v <= 128
    # UnfusedDotProductAttention backend
    if unfused_attn_supported:
        unfused_attn_fwd, unfused_attn_bwd = _run_dot_product_attention(
            dtype,
            config,
            "UnfusedDotProductAttention",
            ckpt_attn,
            qkv_layout,
            workspace_opt,
            pad_between_seqs,
            is_training,
        )

    # FusedAttention backend
    if fused_attn_supported:
        if len(fused_attn_backends) == 1:
            fused_attn_fwd, fused_attn_bwd = _run_dot_product_attention(
                dtype,
                config,
                "FusedAttention",
                ckpt_attn,
                qkv_layout,
                workspace_opt,
                pad_between_seqs,
                is_training,
            )
        if len(fused_attn_backends) == 2:
            os.environ["NVTE_FUSED_ATTN_BACKEND"] = "0"
            fused_attn_fwd, fused_attn_bwd = _run_dot_product_attention(
                dtype,
                config,
                "FusedAttention",
                ckpt_attn,
                qkv_layout,
                workspace_opt,
                pad_between_seqs,
                is_training,
            )
            os.environ["NVTE_FUSED_ATTN_BACKEND"] = "1"
            fused_attn_fwd_1, fused_attn_bwd_1 = _run_dot_product_attention(
                dtype,
                config,
                "FusedAttention",
                ckpt_attn,
                qkv_layout,
                workspace_opt,
                pad_between_seqs,
                is_training,
            )

    # FlashAttention backend
    if flash_attn_supported:
        flash_attn_fwd, flash_attn_bwd = _run_dot_product_attention(
            dtype,
            config,
            "FlashAttention",
            ckpt_attn,
            qkv_layout,
            workspace_opt,
            pad_between_seqs,
            is_training,
        )
    if sage_attn_supported:
        sage_attn_fwd = _run_dot_product_attention(
            dtype,
            config,
            "SageAttention",
            ckpt_attn,
            qkv_layout,
            workspace_opt,
            pad_between_seqs,
            is_training,
        )
    if unfused_attn_supported and fused_attn_supported:
        logging.info("[test_dot_product_attention]: unfused attn vs fused attn")
        torch.testing.assert_close(fused_attn_fwd, unfused_attn_fwd, **tols)
        for i, _ in enumerate(unfused_attn_bwd):
            torch.testing.assert_close(fused_attn_bwd[i], unfused_attn_bwd[i], **tols)
    if unfused_attn_supported and flash_attn_supported:
        logging.info("[test_dot_product_attention]: unfused attn vs flash attn")
        torch.testing.assert_close(flash_attn_fwd, unfused_attn_fwd, **tols)
        for i, _ in enumerate(flash_attn_bwd):
            torch.testing.assert_close(unfused_attn_bwd[i], flash_attn_bwd[i], **tols)
    if fused_attn_supported and flash_attn_supported:
        logging.info("[test_dot_product_attention]: fused attn vs flash attn")
        torch.testing.assert_close(fused_attn_fwd, flash_attn_fwd, **tols)
        for i, _ in enumerate(flash_attn_bwd):
            torch.testing.assert_close(fused_attn_bwd[i], flash_attn_bwd[i], **tols)
    if fused_attn_supported and len(fused_attn_backends) == 2:
        logging.info("[test_dot_product_attention]: fused attn backend 0 vs 1")
        torch.testing.assert_close(fused_attn_fwd, fused_attn_fwd_1, **tols)
        for i, _ in enumerate(fused_attn_bwd):
            torch.testing.assert_close(fused_attn_bwd[i], fused_attn_bwd_1[i], **tols)
    if sage_attn_supported and flash_attn_supported:
        logging.info("[test_dot_product_attention]: sage attn vs flash attn")
        torch.testing.assert_close(sage_attn_fwd, flash_attn_fwd, **tols)


model_configs_mask = {
    #     test:             b,  h, hg,   d,   sq,  skv,   p,             mask,      bias
    "mask_1_0": ModelConfig(8, 16, 16, 64, 128, 128, 0.0, "causal", "no_bias"),
    "mask_1_1": ModelConfig(4, 16, 16, 64, 128, 256, 0.0, "causal", "no_bias"),
    "mask_2_0": ModelConfig(2, 24, 24, 128, 2048, 2048, 0.0, "causal", "no_bias"),
    "mask_2_1": ModelConfig(1, 24, 24, 128, 2048, 4096, 0.0, "causal", "no_bias"),
    "mask_3_0": ModelConfig(8, 16, 16, 64, 128, 128, 0.0, "padding", "no_bias"),
    "mask_3_1": ModelConfig(4, 16, 16, 64, 128, 256, 0.0, "padding", "no_bias"),
    "mask_4_0": ModelConfig(2, 24, 24, 128, 2048, 2048, 0.0, "padding", "no_bias"),
    "mask_4_1": ModelConfig(1, 24, 24, 128, 2048, 4096, 0.0, "padding", "no_bias"),
    "mask_5_0": ModelConfig(8, 16, 16, 64, 128, 128, 0.0, "padding_causal", "no_bias"),
    "mask_5_1": ModelConfig(4, 16, 16, 64, 128, 256, 0.0, "padding_causal", "no_bias"),
    "mask_6_0": ModelConfig(2, 24, 24, 128, 2048, 2048, 0.0, "padding_causal", "no_bias"),
    "mask_6_1": ModelConfig(1, 24, 24, 128, 2048, 4096, 0.0, "padding_causal", "no_bias"),
    "mask_7_0": ModelConfig(2, 24, 24, 128, 2048, 2048, 0.0, "causal_bottom_right", "no_bias"),
    "mask_7_1": ModelConfig(1, 24, 24, 128, 2048, 4096, 0.0, "causal_bottom_right", "no_bias"),
    "mask_8_0": ModelConfig(
        2, 24, 24, 128, 2048, 2048, 0.0, "padding_causal_bottom_right", "no_bias"
    ),
    "mask_8_1": ModelConfig(
        1, 24, 24, 128, 2048, 4096, 0.0, "padding_causal_bottom_right", "no_bias"
    ),
}


@pytest.mark.skipif(get_cudnn_version() < (8, 9, 1), reason="cuDNN 8.9.1+ is required.")
@pytest.mark.parametrize("dtype", param_types_lean)
@pytest.mark.parametrize("model_configs", [model_configs_mask])
@pytest.mark.parametrize("model", model_configs_mask.keys())
def test_dpa_mask(dtype, model_configs, model):
    """Test DotProductAttention module with different mask types"""
    test_dot_product_attention(dtype, model_configs, model, False, True, None, False, False)


qkv_layouts = [
    "sb3hd",
    "sbh3d",
    "sbhd_sb2hd",
    "sbhd_sbh2d",
    "sbhd_sbhd_sbhd",
    "bs3hd",
    "bsh3d",
    "bshd_bs2hd",
    "bshd_bsh2d",
    "bshd_bshd_bshd",
]


model_configs_layout = {
    #       test:             b,  h, hg,   d,   sq,  skv,   p,             mask,             bias
    "layout_0_0": ModelConfig(2, 16, 16, 64, 128, 128, 0.0, "no_mask", "no_bias"),
    "layout_0_1": ModelConfig(2, 16, 16, 64, 128, 128, 0.0, "causal", "post_scale_bias"),
    "layout_0_2": ModelConfig(1, 16, 16, 64, 128, 256, 0.0, "padding", "no_bias"),
    "layout_0_3": ModelConfig(1, 16, 16, 64, 128, 256, 0.0, "padding_causal", "post_scale_bias"),
    "layout_1_0": ModelConfig(2, 24, 24, 128, 2048, 2048, 0.0, "no_mask", "no_bias"),
    "layout_1_1": ModelConfig(2, 24, 24, 128, 2048, 2048, 0.0, "causal", "post_scale_bias"),
    "layout_1_2": ModelConfig(1, 24, 24, 128, 2048, 4096, 0.0, "padding", "no_bias"),
    "layout_1_3": ModelConfig(1, 24, 24, 128, 2048, 4096, 0.0, "padding_causal", "post_scale_bias"),
    "layout_2_0": ModelConfig(2, 16, 16, 256, 1, 2048, 0.0, "no_mask", "no_bias"),
    "layout_2_1": ModelConfig(2, 24, 24, 256, 2048, 2048, 0.0, "causal", "post_scale_bias"),
}


@pytest.mark.skipif(get_cudnn_version() < (8, 9, 5), reason="cuDNN 8.9.5+ is required.")
@pytest.mark.parametrize("dtype", param_types_lean)
@pytest.mark.parametrize("model_configs", [model_configs_layout])
@pytest.mark.parametrize("model", model_configs_layout.keys())
@pytest.mark.parametrize("qkv_layout", qkv_layouts)
def test_dpa_qkv_layout(dtype, model_configs, model, qkv_layout):
    """Test DotProductAttention module with different QKV layouts"""
    test_dot_product_attention(dtype, model_configs, model, False, True, qkv_layout, False, False)


qkv_layouts_thd = ["t3hd", "th3d", "thd_t2hd", "thd_th2d", "thd_thd_thd"]
model_configs_layout_thd = {
    #       test:             b,  h, hg,   d,   sq,  skv,   p,             mask,             bias
    "layout_0_1": ModelConfig(1, 16, 16, 64, 128, 128, 0.0, "padding", "no_bias"),
    "layout_0_2": ModelConfig(8, 16, 16, 64, 128, 128, 0.0, "padding", "no_bias"),
    "layout_0_3": ModelConfig(1, 16, 16, 64, 128, 128, 0.0, "padding_causal", "no_bias"),
    "layout_0_4": ModelConfig(8, 16, 16, 64, 128, 128, 0.0, "padding_causal", "no_bias"),
    "layout_1_1": ModelConfig(1, 16, 16, 64, 2048, 2048, 0.0, "padding", "no_bias"),
    "layout_1_2": ModelConfig(8, 16, 16, 64, 2048, 2048, 0.0, "padding", "no_bias"),
    "layout_1_3": ModelConfig(1, 16, 16, 64, 2048, 2048, 0.0, "padding_causal", "no_bias"),
    "layout_1_4": ModelConfig(8, 16, 16, 64, 2048, 2048, 0.0, "padding_causal", "no_bias"),
    "layout_2_1": ModelConfig(1, 16, 16, 128, 128, 128, 0.0, "padding", "no_bias"),
    "layout_2_2": ModelConfig(1, 16, 16, 64, 128, 256, 0.0, "padding", "no_bias"),
    "layout_2_3": ModelConfig(1, 16, 16, 128, 2048, 2048, 0.0, "padding_causal", "no_bias"),
    "layout_2_4": ModelConfig(8, 16, 16, 64, 2048, 4096, 0.0, "padding_causal", "no_bias"),
}


@pytest.mark.skipif(get_cudnn_version() < (9, 0, 0), reason="cuDNN 9.0.0+ is required.")
@pytest.mark.skipif(
    get_device_compute_capability() < (9, 0), reason="THD is only supported on Hopper+."
)
@pytest.mark.parametrize("dtype", param_types_lean)
@pytest.mark.parametrize("model_configs", [model_configs_layout_thd])
@pytest.mark.parametrize("model", model_configs_layout_thd.keys())
@pytest.mark.parametrize("qkv_layout", qkv_layouts_thd)
def test_dpa_qkv_layout_thd(dtype, model_configs, model, qkv_layout):
    """Test DotProductAttention module with different QKV layouts"""
    pad_between_seqs = True
    test_dot_product_attention(
        dtype, model_configs, model, False, True, qkv_layout, False, pad_between_seqs
    )
    if get_cudnn_version() >= (9, 3, 0):
        # cuDNN 9.3.0+ is required to run pad_between_seqs = False/True in the same run
        pad_between_seqs = False
        test_dot_product_attention(
            dtype, model_configs, model, False, True, qkv_layout, False, pad_between_seqs
        )

model_configs_te_layer = {
    #   test:             b,  h, hg,   d,   sq,  skv,   p,      mask,             bias
    "te_1_0": ModelConfig(2, 16, 16, 64, 128, 128, 0.0, "no_mask", "post_scale_bias"),
    "te_1_1": ModelConfig(4, 16, 16, 64, 128, 128, 0.0, "causal", "post_scale_bias"),
    "te_1_2": ModelConfig(2, 16, 16, 64, 128, 128, 0.0, "padding", "post_scale_bias"),
    "te_2_0": ModelConfig(1, 16, 16, 64, 2048, 2048, 0.0, "causal", "no_bias"),
    "te_2_1": ModelConfig(2, 16, 16, 64, 2048, 2048, 0.0, "no_mask", "no_bias"),
    "te_2_2": ModelConfig(1, 16, 16, 64, 2048, 2048, 0.0, "padding", "no_bias"),
    "te_3_0": ModelConfig(4, 16, 16, 64, 128, 128, 0.0, "causal", "alibi"),
    "te_3_1": ModelConfig(4, 16, 16, 64, 2048, 2048, 0.0, "causal", "alibi"),
}


@pytest.mark.skipif(get_cudnn_version() < (8, 9, 1), reason="cuDNN 8.9.1+ is required.")
@pytest.mark.parametrize("dtype", param_types)
@pytest.mark.parametrize("model_configs", [model_configs_te_layer])
@pytest.mark.parametrize("model", model_configs_te_layer.keys())
@pytest.mark.parametrize("ckpt_attn", [False])
@pytest.mark.parametrize("qkv_format", ["sbhd"])
@pytest.mark.parametrize("fused_qkv_params", [False])
@pytest.mark.parametrize("RoPE", [False])
def test_transformer_layer(
    dtype, model_configs, model, ckpt_attn, qkv_format, fused_qkv_params, RoPE
):
    """Test TransformerLayer module"""

    # Get configs
    config = model_configs[model]
    tols = dict(atol=5e-1, rtol=5e-2)
    workspace_opt = True

    # Test backend availability
    available_backends, fused_attn_backends = _get_attention_backends(
        config,
        qkv_dtype=dtype,
        qkv_layout="sbh3d" if fused_qkv_params else "sb3hd",
    )
    flash_attn_supported, fused_attn_supported, unfused_attn_supported, sage_attn_supported = available_backends

    # Skip if only unfused backend is supported
    if (len(fused_attn_backends) + flash_attn_supported + unfused_attn_supported + sage_attn_supported) < 3:
        pytest.skip("Less than three backends to compare.")

    # UnfusedDotProductAttention backend
    if unfused_attn_supported:
        unfused_attn_fwd, unfused_attn_bwd = _run_transformer_layer(
            dtype,
            config,
            "UnfusedDotProductAttention",
            ckpt_attn,
            qkv_format,
            workspace_opt,
            fused_qkv_params,
            RoPE,
        )

    # FusedAttention backend
    if fused_attn_supported:
        fused_attn_fwd, fused_attn_bwd = _run_transformer_layer(
            dtype,
            config,
            "FusedAttention",
            ckpt_attn,
            qkv_format,
            workspace_opt,
            fused_qkv_params,
            RoPE,
        )

    # FlashAttention backend
    if flash_attn_supported:
        flash_attn_fwd, flash_attn_bwd = _run_transformer_layer(
            dtype,
            config,
            "FlashAttention",
            ckpt_attn,
            qkv_format,
            workspace_opt,
            fused_qkv_params,
            RoPE,
        )

    if unfused_attn_supported and fused_attn_supported:
        logging.info("[test_transformer_layer]: unfused attn vs fused attn")
        torch.testing.assert_close(fused_attn_fwd, unfused_attn_fwd, **tols)
        torch.testing.assert_close(fused_attn_bwd, unfused_attn_bwd, **tols)
    if unfused_attn_supported and flash_attn_supported:
        logging.info("[test_transformer_layer]: unfused attn vs flash attn")
        torch.testing.assert_close(flash_attn_fwd, unfused_attn_fwd, **tols)
        torch.testing.assert_close(flash_attn_bwd, unfused_attn_bwd, **tols)
    if fused_attn_supported and flash_attn_supported:
        logging.info("[test_transformer_layer]: fused attn vs flash attn")
        torch.testing.assert_close(fused_attn_fwd, flash_attn_fwd, **tols)
        torch.testing.assert_close(fused_attn_bwd, flash_attn_bwd, **tols)


@pytest.mark.skipif(get_cudnn_version() < (8, 9, 1), reason="cuDNN 8.9.1+ is required.")
@pytest.mark.parametrize("dtype", param_types_lean)
@pytest.mark.parametrize("model_configs", [model_configs_te_layer])
@pytest.mark.parametrize("model", ["te_1_2", "te_2_0"])
@pytest.mark.parametrize("qkv_format", ["bshd", "sbhd"])
def test_te_layer_misc(dtype, model_configs, model, qkv_format):
    """Test TransformerLayer module with miscellaneous settings"""
    ckpt_attn = True
    fused_qkv_params = True
    RoPE = True
    test_transformer_layer(
        dtype, model_configs, model, ckpt_attn, qkv_format, fused_qkv_params, RoPE
    )


@pytest.mark.skipif(get_cudnn_version() < (8, 9, 1), reason="cuDNN 8.9.1+ is required.")
@pytest.mark.parametrize("dtype", param_types_lean)
@pytest.mark.parametrize("model_configs", [model_configs_te_layer])
@pytest.mark.parametrize("model", ["te_2_0", "te_2_1", "te_2_2"])
def test_te_layer_mqa_gqa(dtype, model_configs, model):
    """Test TransformerLayer module with MQA/GQA"""

    def find_factors(x):
        f = []
        for i in range(2, x + 1):
            if x % i == 0:
                f.append(i)
        return f

    ckpt_attn = True
    qkv_format = "bshd"
    fused_qkv_params = True
    RoPE = True
    config = model_configs[model]
    num_querys_per_gqa_group = find_factors(config.num_heads)

    for num_q_per_gqa_group in num_querys_per_gqa_group:
        config.num_gqa_groups = config.num_heads // num_q_per_gqa_group
        test_transformer_layer(
            dtype, model_configs, model, ckpt_attn, qkv_format, fused_qkv_params, RoPE
        )


def _run_transformer_layer(
    dtype: torch.dtype,
    config: ModelConfig,
    backend: str,
    ckpt_attn: bool,
    qkv_format: str,
    workspace_opt: bool,
    fused_qkv_params: bool,
    RoPE: bool,
) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Run TransformerLayer module with one forward pass and one backward pass"""

    # Set RNG and environment variables
    reset_rng_states()
    os.environ["NVTE_FLASH_ATTN"] = "0"
    os.environ["NVTE_FUSED_ATTN"] = "0"
    if backend == "FlashAttention":
        os.environ["NVTE_FLASH_ATTN"] = "1"
    if backend == "FusedAttention":
        os.environ["NVTE_FUSED_ATTN"] = "1"
    global _attention_backends
    _attention_backends["backend_selection_requires_update"] = True

    # Create input tensor
    inp = torch.randn(
        config.max_seqlen_q,
        config.batch_size,
        config.hidden_size,
        dtype=dtype,
        device="cuda",
        requires_grad=False, #True,
    )
    # In case the format to be tested is batch-first, need to transpose the
    # input tensor.
    if qkv_format == "bshd":
        inp = inp.transpose(0, 1)

    # Create seqlens
    if "padding" in config.attn_mask_type:
        seqlens_q = torch.randint(
            1, config.max_seqlen_q, [config.batch_size], dtype=torch.int32, device="cuda"
        )
    else:
        seqlens_q = torch.full(
            [config.batch_size], config.max_seqlen_q, dtype=torch.int32, device="cuda"
        )

    # Create attention mask if padding
    attention_mask = None
    if "padding" in config.attn_mask_type:
        attention_mask_q = torch.Tensor([]).to(dtype=torch.bool)
        for i in range(config.batch_size):
            attention_mask_q = torch.cat(
                [
                    attention_mask_q,
                    torch.Tensor(
                        [False] * seqlens_q[i] + [True] * (config.max_seqlen_q - seqlens_q[i])
                    )
                    .to(torch.bool)
                    .unsqueeze(0)
                    .unsqueeze(0)
                    .unsqueeze(0),
                ],
                dim=0,
            )
        attention_mask = attention_mask_q.to(device="cuda")

    sigma = 0.02
    init_method = init_method_normal(sigma)
    output_layer_init_method = scaled_init_method_normal(sigma, config.num_layers)

    layer_number = 1
    drop_path_rate = 0.0
    drop_path_rates = [rate.item() for rate in torch.linspace(0, drop_path_rate, config.num_layers)]

    # Create bias
    bias = None
    if config.attn_bias_type == "post_scale_bias":
        bias = torch.randn(
            1,
            config.num_heads,
            config.max_seqlen_q,
            config.max_seqlen_kv,
            dtype=dtype,
            device="cuda",
        )

    # Create RoPE
    rotary_pos_emb = None
    if RoPE:
        PE = RotaryPositionEmbedding(dim=config.head_dim_qk)
        rotary_pos_emb = PE(config.max_seqlen_q).to(device="cuda")

    # Set up model
    block = TransformerLayer(
        config.hidden_size,
        4 * config.hidden_size,
        config.num_heads,
        num_gqa_groups=config.num_gqa_groups,
        layernorm_epsilon=1e-5,
        hidden_dropout=0.0,
        attention_dropout=config.dropout_p,
        init_method=init_method,
        output_layer_init_method=output_layer_init_method,
        layer_number=layer_number,
        kv_channels=config.head_dim_qk,
        self_attn_mask_type=config.attn_mask_type,
        tp_group=None,
        tp_size=1,
        params_dtype=dtype,
        get_rng_state_tracker=None,
        fuse_wgrad_accumulation=False,
        seq_length=config.max_seqlen_q,
        micro_batch_size=config.batch_size,
        sequence_parallel=False,
        apply_residual_connection_post_layernorm=False,
        output_layernorm=False,
        layer_type="encoder",
        drop_path_rate=drop_path_rates[layer_number - 1],
        set_parallel_mode=True,
        fuse_qkv_params=fused_qkv_params,
        zero_centered_gamma=False,
        qkv_weight_interleaved=False,
        ub_tp_comm_overlap=False,
        bias=True,
        attn_input_format=qkv_format,
    ).to(dtype=dtype, device="cuda")
    block.train()
    for p in block.parameters():
        p.requires_grad = True

    # Create ALiBi slopes
    alibi_slopes = None
    if config.attn_bias_type == "alibi" and config.alibi_type == "custom":
        alibi_slopes = torch.randn(config.num_heads).abs().to(dtype=torch.float32, device="cuda")

    # Run a forward and backward pass
    out = block(
        inp,
        attention_mask=attention_mask,
        self_attn_mask_type=config.attn_mask_type,
        checkpoint_core_attention=False,
        rotary_pos_emb=rotary_pos_emb,
        core_attention_bias_type=config.attn_bias_type,
        core_attention_bias=bias,
        alibi_slopes=alibi_slopes,
    )
    loss = out.sum()
    loss.backward()

    # 验证梯度计算图
    assert out.requires_grad, "输出未启用梯度"
    assert inp.grad_fn is not None, "输入张量未参与梯度计算"

    return out, inp.grad


class AttentionTestRunner:
    def __init__(self):
        self.results = {'total': 0, 'passed': 0, 'failed': 0}

    def run_test_case(self, config: ModelConfig, dtype: torch.dtype, backend: str) -> bool:
        self.results['total'] += 1
        try:
            # 设置环境变量
            os.environ.update({
                "NVTE_FLASH_ATTN": "1" if backend == "FlashAttention" else "0",
                "NVTE_FUSED_ATTN": "1" if backend == "FusedAttention" else "0",
                "NVTE_SAGE_ATTN": "1" if backend == "SageAttention" else "0",
                "NVTE_UNFUSED_ATTN": "1" if backend == "UnfusedDotProductAttention" else "0"
            })
            
            # 执行核心测试逻辑
            _run_dot_product_attention(
                dtype, config, backend, False, "sb3hd", True, False, True
            )
            
            self.results['passed'] += 1
            return True
        except Exception as e:
            logging.error(f"{backend}测试失败: {str(e)}")
            self.results['failed'] += 1
            return False

    def compare_backends(self, config: ModelConfig, dtype: torch.dtype):
        backends = ["FusedAttention", "FlashAttention", "UnfusedDotProductAttention", "SageAttention"]
        status = {}
        
        for backend in backends:
            status[backend] = self.run_test_case(config, dtype, backend)
            print(f"backend: {backend}, status: {status[backend]}")
        
        print(f"\n【{config.attn_type}注意力】后端一致性检查:")
        for a, b in [("FusedAttention", "FlashAttention"), 
                     ("FlaseAttention", "SageAttention")]:
            if status.get(a) and status.get(b):
                print(f"  {a} ↔ {b}: 一致")
            else:
                print(f"  {a} ↔ {b}: 不一致")

    def run(self):
        """主测试流程"""
        test_cases = [
            (ModelConfig(8, 16, 16, 64, 128, 128, 0.0, "no_mask", "no_bias"), torch.float16),
        ]

        for config, dtype in test_cases:
            print(f"\n=== 测试配置 ===")
            print(f"类型: {config.attn_type}")
            print(f"形状: {config.batch_size}×{config.max_seqlen_q}×{config.num_heads}")
            self.compare_backends(config, dtype)

        # 最终报告
        print("\n=== 测试汇总 ===")
        print(f"总用例: {self.results['total']}")
        print(f"成功: {self.results['passed']}")
        print(f"失败: {self.results['failed']}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    runner = AttentionTestRunner()
    runner.run()

# 重置环境变量，确保所有后端都可能被激活
def test_attention_backends():
    print("测试不同配置下可用的Attention后端...")
    
    # 启用所有可能的后端
    os.environ["NVTE_FLASH_ATTN"] = "1"
    os.environ["NVTE_FUSED_ATTN"] = "1" 
    os.environ["NVTE_UNFUSED_ATTN"] = "1"
    os.environ["NVTE_SAGE_ATTN"] = "1"
    
    # 确保重新选择后端
    _attention_backends["backend_selection_requires_update"] = True
    
    # 测试不同的配置组合
    configs = [
        # 名称, 数据类型, 批次大小, 头数, 序列长度, 头维度, mask类型
        ("短序列(FP16)", torch.float16, 8, 16, 128, 64, "no_mask"),
        ("长序列(FP16)", torch.float16, 2, 16, 2048, 64, "no_mask"),
        ("短序列(BF16)", torch.bfloat16, 8, 16, 128, 64, "no_mask"),
        ("长序列(BF16)", torch.bfloat16, 2, 16, 2048, 64, "no_mask"),
        ("Causal短序列", torch.float16, 8, 16, 128, 64, "causal"),
        ("Causal长序列", torch.float16, 2, 16, 2048, 64, "causal"),
        ("Padding短序列", torch.float16, 8, 16, 128, 64, "padding"),
        ("Padding长序列", torch.float16, 2, 16, 2048, 64, "padding"),
    ]
    
    results = {}
    compute_capability = torch.cuda.get_device_capability()
    print(f"GPU计算能力: {compute_capability[0]}.{compute_capability[1]}")
    
    for name, dtype, batch_size, num_heads, seq_len, head_dim, mask_type in configs:
        # 创建AttentionParams
        params = AttentionParams(
            qkv_dtype=dtype,
            qkv_layout="sbh3d",
            batch_size=batch_size,
            num_heads=num_heads,
            num_gqa_groups=num_heads,
            max_seqlen_q=seq_len,
            max_seqlen_kv=seq_len,
            head_dim_qk=head_dim,
            head_dim_v=head_dim,
            attn_mask_type=mask_type,
        )
        
        # 获取后端支持情况
        _, _, _, _, _, available_backends = get_attention_backend(params)
        
        # 存储结果
        flash_attn, fused_attn, unfused_attn, sage_attn = available_backends
        results[name] = {
            "FlashAttention": "✓" if flash_attn else "✗",
            "FusedAttention": "✓" if fused_attn else "✗",
            "UnfusedAttention": "✓" if unfused_attn else "✗",
            "SageAttention": "✓" if sage_attn else "✗",
        }
    
    # 打印结果表格
    print("\n后端兼容性测试结果:")
    print("-" * 80)
    headers = ["配置", "FlashAttention", "FusedAttention", "UnfusedAttention", "SageAttention"]
    print(f"{headers[0]:<20} {headers[1]:<15} {headers[2]:<15} {headers[3]:<18} {headers[4]:<15}")
    print("-" * 80)
    
    for name, backends in results.items():
        print(f"{name:<20} {backends['FlashAttention']:<15} {backends['FusedAttention']:<15} {backends['UnfusedAttention']:<18} {backends['SageAttention']:<15}")
    
    # 检查环境变量
    print("\n当前环境变量设置:")
    env_vars = ["NVTE_FLASH_ATTN", "NVTE_FUSED_ATTN", "NVTE_UNFUSED_ATTN", "NVTE_SAGE_ATTN"]
    for var in env_vars:
        print(f"{var}: {os.environ.get(var, '未设置')}")

if __name__ == "__main__":
    test_attention_backends()