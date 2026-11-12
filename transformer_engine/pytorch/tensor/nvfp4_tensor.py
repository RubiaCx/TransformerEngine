# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Tensor class with NVFP4 data"""

from __future__ import annotations
from collections.abc import Iterable
import math
import warnings
from typing import Any, Dict, Optional, Tuple, Union
import functools

import torch
import transformer_engine_torch as tex

from transformer_engine.pytorch.device import te_device_type
from transformer_engine.common.recipe import NVFP4BlockScaling, Recipe
from ..constants import NVFP4_BLOCK_SCALING_SIZE, dist_group_type, DType
from ..utils import (
    canonicalize_process_group,
    ceil_div,
    devices_match,
    round_up_to_nearest_multiple,
)

from .storage.nvfp4_tensor_storage import NVFP4TensorStorage, _FromNVFP4Func
from ..quantized_tensor import QuantizedTensor, QuantizedTensorStorage, Quantizer
from ..dynamo import register_value_opaque_quantizer
from ._quantization_helpers import _IdentityFunc, safe_quantized_repr

aten = torch.ops.aten


def get_no_random_sign_vector(device: int) -> torch.Tensor:
    """Non-random sign vector for Hadamard transform."""
    return torch.tensor([1], dtype=torch.float32, device=device)


def get_sign_from_vector(vector: torch.Tensor) -> int:
    """Convert sign vector to bitmask.

    Used for random Hadamard transform.

    """
    mask = 0
    for i, v in enumerate(vector):
        mask |= (v == -1) << i
    return mask.item()


def get_wgrad_sign_vector(device: int) -> torch.Tensor:
    """Hard-coded random signs for Hadamard transform.

    https://xkcd.com/221/

    """
    return torch.tensor(
        [1, 1, 1, -1, 1, -1, -1, -1, -1, -1, -1, 1, -1, 1, -1, -1],
        dtype=torch.float32,
        device=device,
    )


def get_hadamard_matrix(hadamard_dimension: int, device: int) -> torch.Tensor:
    """Construct a 16x16 Hadamard matrix."""
    assert hadamard_dimension == 16, "Only hadamard dimension 16 is supported."
    hadamard_scale = 1 / math.sqrt(hadamard_dimension)
    return (
        torch.tensor(
            [
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1],
                [1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1],
                [1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1],
                [1, 1, 1, 1, -1, -1, -1, -1, 1, 1, 1, 1, -1, -1, -1, -1],
                [1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, -1, -1, 1, -1, 1],
                [1, 1, -1, -1, -1, -1, 1, 1, 1, 1, -1, -1, -1, -1, 1, 1],
                [1, -1, -1, 1, -1, 1, 1, -1, 1, -1, -1, 1, -1, 1, 1, -1],
                [1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1],
                [1, -1, 1, -1, 1, -1, 1, -1, -1, 1, -1, 1, -1, 1, -1, 1],
                [1, 1, -1, -1, 1, 1, -1, -1, -1, -1, 1, 1, -1, -1, 1, 1],
                [1, -1, -1, 1, 1, -1, -1, 1, -1, 1, 1, -1, -1, 1, 1, -1],
                [1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1],
                [1, -1, 1, -1, -1, 1, -1, 1, -1, 1, -1, 1, 1, -1, 1, -1],
                [1, 1, -1, -1, -1, -1, 1, 1, -1, -1, 1, 1, 1, 1, -1, -1],
                [1, -1, -1, 1, -1, 1, 1, -1, -1, 1, 1, -1, 1, -1, -1, 1],
            ],
            dtype=torch.float32,
            device=device,
        )
        * hadamard_scale
    )


@functools.lru_cache(maxsize=None)
def get_rht_matrix(with_random_sign_mask: bool, device: int) -> torch.Tensor:
    """Construct matrix used in random Hadamard transform."""
    hadamard_dimension = 16
    if with_random_sign_mask:
        signs = get_wgrad_sign_vector(device=device)
    else:
        signs = get_no_random_sign_vector(device=device)
    sign_matrix = signs * torch.eye(hadamard_dimension, dtype=torch.float32, device=device)
    rht_matrix = sign_matrix @ get_hadamard_matrix(hadamard_dimension, device=device)
    return rht_matrix.to(dtype=torch.bfloat16)


@functools.lru_cache(maxsize=None)
def get_random_sign_mask_for_rht(with_random_sign_mask: bool, device: int) -> int:
    """Sign mask for random Hadamard transform."""
    if with_random_sign_mask:
        return get_sign_from_vector(get_wgrad_sign_vector(device=device))
    return 0


_RHT_DIMS = (16, 128)
_RHT_SIGN_POLICIES = ("static", "resampled")
_SPLITMIX64_MASK = (1 << 64) - 1

# Global step counter that drives resampled RHT sign vectors. All quantizers with
# ``rht_sign_policy="resampled"`` derive their sign vector from
# ``(rht_sign_seed, step // rht_resample_interval, rht_dim)``, so the quantizers of the
# two operands of a backward GEMM (e.g. the forward input and the output gradient for
# WGRAD) use the same rotation as long as they share those options and the step is not
# advanced between the forward pass and its backward pass.
_rht_sign_step = 0  # pylint: disable=invalid-name
_rht_auto_step_handle = None  # pylint: disable=invalid-name


def nvfp4_rht_get_step() -> int:
    """Current step of the resampled-RHT sign schedule."""
    return _rht_sign_step


def nvfp4_rht_set_step(step: int) -> None:
    """Set the step of the resampled-RHT sign schedule (e.g. when resuming training)."""
    global _rht_sign_step  # pylint: disable=global-statement
    if step < 0:
        raise ValueError("step must be non-negative.")
    _rht_sign_step = int(step)


def nvfp4_rht_step(num_steps: int = 1) -> int:
    """Advance the resampled-RHT sign schedule.

    Call this once per optimizer step, after the backward pass. Advancing the step
    between a forward pass and its backward pass breaks the pairing of the rotations
    of the two WGRAD operands.
    """
    nvfp4_rht_set_step(_rht_sign_step + int(num_steps))
    return _rht_sign_step


def nvfp4_rht_enable_auto_step(enabled: bool = True) -> None:
    """Advance the resampled-RHT sign schedule after every ``torch.optim`` optimizer step.

    Registers (or removes) a global optimizer post-step hook. With several optimizers
    the schedule advances once per optimizer step call.
    """
    global _rht_auto_step_handle  # pylint: disable=global-statement
    if _rht_auto_step_handle is not None:
        _rht_auto_step_handle.remove()
        _rht_auto_step_handle = None
    if enabled:
        from torch.optim.optimizer import (  # pylint: disable=import-outside-toplevel
            register_optimizer_step_post_hook,
        )

        def _hook(optimizer, args, kwargs):  # pylint: disable=unused-argument
            nvfp4_rht_step()

        _rht_auto_step_handle = register_optimizer_step_post_hook(_hook)


def _splitmix64(state: int) -> int:
    """SplitMix64 output function (pure Python, reproducible across platforms)."""
    z = (state + 0x9E3779B97F4A7C15) & _SPLITMIX64_MASK
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & _SPLITMIX64_MASK
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & _SPLITMIX64_MASK
    return z ^ (z >> 31)


# Sign-vector streams: the two backward GEMMs rotate along different dimensions and
# use independent sign vectors.
RHT_SIGN_STREAM_WGRAD = 0
RHT_SIGN_STREAM_DGRAD = 1


@functools.lru_cache(maxsize=64)
def get_rht_sign_bits(
    rht_dim: int, seed: int, generation: int, stream: int = RHT_SIGN_STREAM_WGRAD
) -> Tuple[int, ...]:
    """Pseudo-random sign bits (1 means -1) for a dimension-``rht_dim`` RHT.

    Deterministic function of ``(rht_dim, seed, generation, stream)``.
    """
    state = _splitmix64((seed & _SPLITMIX64_MASK) ^ 0x4E564650345F5248)  # "NVFP4_RH"
    state = _splitmix64(state ^ (generation & _SPLITMIX64_MASK))
    state = _splitmix64(state ^ rht_dim)
    if stream != RHT_SIGN_STREAM_WGRAD:
        state = _splitmix64(state ^ (stream << 32))
    bits = []
    while len(bits) < rht_dim:
        state = _splitmix64(state)
        bits.extend((state >> i) & 1 for i in range(64))
    return tuple(bits[:rht_dim])


@functools.lru_cache(maxsize=16)
def get_rht_matrix_from_sign_bits(sign_bits: Tuple[int, ...], device: int) -> torch.Tensor:
    """Sign-folded Hadamard blocks used by the RHT kernels.

    Returns a BF16 tensor of shape ``[rht_dim, 16]`` whose row ``k`` is
    ``sign[k] * H16[k % 16, :] / 4``. For ``rht_dim == 16`` this is the same matrix as
    :func:`get_rht_matrix`. For ``rht_dim == 128`` the kernels complete the rotation
    with an 8-point Hadamard butterfly across the eight 16-element blocks (scaled by
    ``1 / sqrt(8)``), i.e. ``H128 = H8 (x) H16`` in Sylvester ordering.
    """
    rht_dim = len(sign_bits)
    signs = torch.tensor(
        [-1.0 if bit else 1.0 for bit in sign_bits], dtype=torch.float32, device=device
    )
    hadamard_16 = get_hadamard_matrix(16, device=device)
    blocks = hadamard_16.repeat(rht_dim // 16, 1)
    return (signs.unsqueeze(1) * blocks).to(dtype=torch.bfloat16)


def rht_rotate_high_precision(
    tensor: torch.Tensor, rht_matrix: torch.Tensor, dim: int
) -> torch.Tensor:
    """Rotate a high-precision tensor with the tiled RHT described by ``rht_matrix``.

    Debug/validation helper (not a quantization path): applies the same rotation as
    the tiled-RHT kernels, computed in FP64, without quantizing. ``dim=0`` rotates
    along the flattened leading dimensions (as the columnwise copies), ``dim=-1``
    along the last dimension (as the rotated rowwise copy). ``rht_matrix`` is the
    ``[rht_dim, 16]`` matrix handed to the kernels.
    """
    rht_dim = rht_matrix.shape[0]
    blocks_per_chunk = rht_dim // 16
    work = tensor.detach().reshape(-1, tensor.shape[-1]).to(torch.float64)
    if dim == 0:
        work = work.t()
    rows, length = work.shape
    if length % rht_dim != 0:
        raise ValueError(f"Rotated dimension ({length}) must be divisible by {rht_dim}.")
    matrix = rht_matrix.to(device=work.device, dtype=torch.float64)
    matrix = matrix.reshape(blocks_per_chunk, 16, 16)
    chunks = work.reshape(rows, length // rht_dim, blocks_per_chunk, 16)
    rotated = torch.einsum("rcjk,jkn->rcjn", chunks, matrix)
    if blocks_per_chunk > 1:
        idx = torch.arange(blocks_per_chunk, device=work.device)
        anded = idx.unsqueeze(1) & idx.unsqueeze(0)
        parity = torch.zeros_like(anded)
        for bit in range(blocks_per_chunk.bit_length()):
            parity ^= (anded >> bit) & 1
        hadamard = (1 - 2 * parity).to(torch.float64) / math.sqrt(blocks_per_chunk)
        rotated = torch.einsum("ij,rcjn->rcin", hadamard, rotated)
    rotated = rotated.reshape(rows, length)
    if dim == 0:
        rotated = rotated.t()
    return rotated.reshape(tensor.shape).to(tensor.dtype).contiguous()


def nvfp4_dgrad_rotated_weight(
    grad_output_quantizer: "NVFP4Quantizer", weight_quantized, weight_high_precision
):
    """Rotated columnwise weight copy for a rotated DGRAD GEMM.

    The source is the high-precision weight or the de-quantized forward weight,
    depending on ``grad_output_quantizer.dgrad_weight_source``. The copy is rotated
    along the output-channel dimension with the sign vector of the rotated rowwise
    grad-output copy.
    """
    if grad_output_quantizer.dgrad_weight_source == "dequantized":
        if not isinstance(weight_quantized, QuantizedTensorStorage):
            raise RuntimeError(
                "dgrad_rht_weight_source='dequantized' requires the quantized forward weight."
            )
        weight = weight_quantized.dequantize(dtype=torch.bfloat16)
    else:
        weight = weight_high_precision
        if isinstance(weight, QuantizedTensorStorage):
            weight = weight.dequantize(dtype=torch.bfloat16)
    weight = weight.detach().to(dtype=torch.bfloat16)
    return grad_output_quantizer.dgrad_weight_quantizer()(weight)


class NVFP4Quantizer(Quantizer):
    """Builder class for NVFP4 tensors with NV block scaling"""

    dtype: DType
    """Scale dtype (e4m3 block scaling factors or ue5m3 for wider dynamic range)"""
    scale_dtype: DType
    """Random Hadamard Transform"""
    with_rht: bool
    with_post_rht_amax: bool
    """amax reduction options"""
    with_amax_reduction: bool

    """2D block scaling, only applicable for weights."""
    with_2d_quantization: bool

    """Stochastic rounding, only applicable for gradients."""
    stochastic_rounding: bool

    """Whether emitted NVFP4 tensors store one FP32 amax per row."""
    row_scaled_nvfp4: bool
    """Whether to use NVFP4 4over6 map-to-4/map-to-6 block selection."""
    nvfp4_use_4over6: bool
    """Global E4M3 scale bound used by emitted NVFP4 tensors."""
    nvfp4_e4m3_max: int
    """Whether the columnwise copy also uses 4over6 (only applies to non-RHT columnwise copies)."""
    nvfp4_4over6_columnwise: bool
    """Whether the columnwise copy is quantized from the dequantized rowwise copy."""
    columnwise_from_rowwise: bool
    """NVFP4 4over6 candidate-selection error mode."""
    nvfp4_4over6_err_mode: str
    """Whether to disable the global (second-level) NVFP4 scale."""
    disable_second_level_scale: bool

    """RHT sign mask (0 when sign randomization is disabled)"""
    rht_matrix_random_sign_mask_t: int

    """RHT dimension along the rotated (inner) dimension: 16 or 128."""
    rht_dim: int
    """MS-EDEN for the columnwise (rotated) copy: round-to-nearest FP4 codes plus
    EDEN-corrected, stochastically rounded micro-scales (arXiv:2601.22813). When
    set, ``stochastic_rounding`` only applies to the rowwise (un-rotated) copy."""
    ms_eden: bool
    """RHT sign-vector policy: 'static' or 'resampled'."""
    rht_sign_policy: str
    """Seed of the RHT sign vector (ignored for the static dimension-16 vector)."""
    rht_sign_seed: int
    """Number of steps (see ``nvfp4_rht_step``) between sign-vector resamplings."""
    rht_resample_interval: int
    """Grid parameter ``s`` of the MS-EDEN round-to-nearest quantizer: the micro-scale
    of a group is ``group_amax / s`` before rounding. 6 uses the full E2M1 range;
    larger values clip elements above ``6 / s`` of the group amax."""
    ms_eden_grid_max: float
    """Sign stream of the columnwise (rotated) copy: 0 for WGRAD operands, 1 for the
    rotated weight copy consumed by DGRAD."""
    rht_sign_stream: int
    """Also rotate the rowwise copy along the last dimension (inner dimension of DGRAD)
    with the DGRAD sign stream. Used for the output gradient when DGRAD is rotated.
    Toggled with ``set_rowwise_rht`` by modules that also rotate the weight operand."""
    rht_rowwise: bool
    """Whether the recipe requests a rotated DGRAD for the tensor of this quantizer.
    Modules that implement it (``Linear``) enable ``rht_rowwise``; elsewhere the
    rowwise copy stays un-rotated."""
    dgrad_rht: bool
    """Source of the rotated weight copy for DGRAD: 'high_precision' or 'dequantized'
    (only meaningful when ``rht_rowwise`` is set)."""
    dgrad_weight_source: str
    """Whether the rotated DGRAD weight copy uses MS-EDEN (MS-EDEN operand scope
    'all'); otherwise it is a range-aligned round-to-nearest copy."""
    dgrad_weight_ms_eden: bool

    def __init__(
        self,
        fp4_dtype: Union[DType, tex.DType] = DType.kFloat4E2M1,
        scale_dtype: Union[DType, tex.DType] = DType.kFloat8E4M3,
        rowwise: bool = True,
        columnwise: bool = True,
        with_amax_reduction: bool = False,
        amax_reduction_group: Optional[dist_group_type] = None,
        with_rht: bool = False,
        with_post_rht_amax: bool = False,
        with_2d_quantization: bool = False,
        stochastic_rounding: bool = False,
        row_scaled_nvfp4: bool = False,
        nvfp4_use_4over6: bool = False,
        nvfp4_e4m3_max: int = 0,
        nvfp4_4over6_err_mode: str = "MAE",
        with_random_sign_mask: bool = True,
        disable_second_level_scale: bool = False,
        nvfp4_4over6_columnwise: bool = True,
        columnwise_from_rowwise: bool = False,
        rht_dim: int = 16,
        ms_eden: bool = False,
        rht_sign_policy: str = "static",
        rht_sign_seed: int = 0,
        rht_resample_interval: int = 1,
        ms_eden_grid_max: float = 6.0,
        rht_sign_stream: int = RHT_SIGN_STREAM_WGRAD,
        rht_rowwise: bool = False,
        dgrad_weight_source: str = "high_precision",
        dgrad_weight_ms_eden: bool = False,
        dgrad_rht: bool = False,
    ) -> None:
        super().__init__(rowwise=rowwise, columnwise=columnwise)
        if ms_eden_grid_max <= 0:
            raise ValueError("ms_eden_grid_max must be positive.")
        if dgrad_weight_source not in ("high_precision", "dequantized"):
            raise ValueError("dgrad_weight_source must be 'high_precision' or 'dequantized'.")
        if rht_rowwise and not with_rht:
            raise ValueError("rht_rowwise=True requires with_rht=True.")
        if (rht_dim != 16 or rht_rowwise) and (
            nvfp4_use_4over6
            or with_2d_quantization
            or row_scaled_nvfp4
            or disable_second_level_scale
        ):
            raise ValueError(
                "rht_dim=128 and rht_rowwise=True are incompatible with 4over6, 2D quantization, "
                "row-scaled NVFP4 and disable_second_level_scale."
            )
        self.dgrad_weight_ms_eden = dgrad_weight_ms_eden
        self.dgrad_rht = dgrad_rht
        self.ms_eden_grid_max = float(ms_eden_grid_max)
        self.rht_sign_stream = int(rht_sign_stream)
        self.rht_rowwise = rht_rowwise
        self.dgrad_weight_source = dgrad_weight_source
        if rht_dim not in _RHT_DIMS:
            raise ValueError(f"rht_dim must be one of {_RHT_DIMS}, got {rht_dim}.")
        if rht_sign_policy not in _RHT_SIGN_POLICIES:
            raise ValueError(
                f"rht_sign_policy must be one of {_RHT_SIGN_POLICIES}, got {rht_sign_policy!r}."
            )
        if rht_resample_interval < 1:
            raise ValueError("rht_resample_interval must be a positive integer.")
        if (
            rht_dim != 16
            or rht_sign_policy != "static"
            or rht_rowwise
            or rht_sign_stream != RHT_SIGN_STREAM_WGRAD
        ) and not with_random_sign_mask:
            raise ValueError(
                "with_random_sign_mask=False is only supported with rht_dim=16, "
                "rht_sign_policy='static' and the default sign stream."
            )
        if ms_eden:
            if not with_rht:
                raise ValueError("ms_eden=True requires with_rht=True.")
            if nvfp4_use_4over6:
                raise ValueError("ms_eden=True is incompatible with nvfp4_use_4over6.")
            if with_2d_quantization or row_scaled_nvfp4 or disable_second_level_scale:
                raise ValueError(
                    "ms_eden=True is incompatible with 2D quantization, row-scaled NVFP4 "
                    "and disable_second_level_scale."
                )
            if DType.cast(scale_dtype) != DType.kFloat8E4M3:
                raise ValueError("ms_eden=True requires E4M3 scales.")
        self.rht_dim = rht_dim
        self.ms_eden = ms_eden
        self.rht_sign_policy = rht_sign_policy
        self.rht_sign_seed = int(rht_sign_seed)
        self.rht_resample_interval = int(rht_resample_interval)
        self.dtype = DType.cast(fp4_dtype)
        self.scale_dtype = DType.cast(scale_dtype)
        if self.scale_dtype not in (DType.kFloat8E4M3, DType.kFloat8UE5M3):
            raise ValueError("scale_dtype must be DType.kFloat8E4M3 or DType.kFloat8UE5M3.")
        self.with_rht = with_rht
        self.with_post_rht_amax = with_post_rht_amax
        self.with_amax_reduction = with_amax_reduction
        self.amax_reduction_group = amax_reduction_group
        self.with_2d_quantization = with_2d_quantization
        self.stochastic_rounding = stochastic_rounding
        if row_scaled_nvfp4 and disable_second_level_scale:
            warnings.warn(
                "Row-scaled NVFP4 requires second-level scaling; disabling "
                "row_scaled_nvfp4 because disable_second_level_scale=True.",
                UserWarning,
                stacklevel=2,
            )
            row_scaled_nvfp4 = False
        self.row_scaled_nvfp4 = row_scaled_nvfp4
        self.nvfp4_use_4over6 = nvfp4_use_4over6
        if nvfp4_use_4over6 and self.scale_dtype == DType.kFloat8UE5M3:
            raise ValueError(
                "nvfp4_use_4over6 is incompatible with scale_dtype=DType.kFloat8UE5M3."
            )
        self.nvfp4_e4m3_max = nvfp4_e4m3_max
        # 4over6 always applies to the rowwise copy (un-rotated, round-to-nearest).
        # The columnwise copy uses 4over6 only if this flag is set and the
        # columnwise copy is not an RHT copy (see ``columnwise_uses_4over6``).
        self.nvfp4_4over6_columnwise = bool(nvfp4_4over6_columnwise)
        # Source of the columnwise (backward) copy: the high-precision input (default)
        # or the dequantized rowwise (forward) copy.
        self.columnwise_from_rowwise = bool(columnwise_from_rowwise)
        self.nvfp4_4over6_err_mode = nvfp4_4over6_err_mode.upper()
        if self.nvfp4_4over6_err_mode not in ("MAE", "MSE"):
            raise ValueError("nvfp4_4over6_err_mode must be 'MAE' or 'MSE'.")
        self.disable_second_level_scale = disable_second_level_scale
        self.rht_matrix_random_sign_mask_t = get_random_sign_mask_for_rht(
            with_random_sign_mask, torch.cuda.current_device()
        )
        self._rebuild_derived_state()

    @property
    def columnwise_uses_4over6(self) -> bool:
        """Whether the columnwise copy emitted by this quantizer uses 4over6.

        With RHT, the columnwise copy is the rotated wgrad operand and is
        always standard NVFP4.
        """
        return self.nvfp4_use_4over6 and self.nvfp4_4over6_columnwise and not self.with_rht

    @property
    def nvfp4_columnwise_e4m3_max(self) -> int:
        """Explicit global scale bound of the columnwise copy of emitted tensors.

        ``0`` means the columnwise copy uses the same bound as the rowwise
        copy (``nvfp4_e4m3_max``). A non-zero value is returned when the
        rowwise copy is 4over6 with a reduced bound and the columnwise copy
        is standard NVFP4, which uses the full range of the scale dtype.
        """
        if not self.nvfp4_use_4over6 or self.columnwise_uses_4over6 or self.nvfp4_e4m3_max == 0:
            return 0
        return 114688 if self.scale_dtype == DType.kFloat8UE5M3 else 448

    def __getstate__(self):
        """Exclude unpicklable process group from serialized state."""
        state = self.__dict__.copy()
        state["amax_reduction_group"] = None
        return state

    def _rebuild_derived_state(self) -> None:
        """Build the derived ``rht_matrix`` (also used after value-key reconstruction).

        ``rht_matrix`` is a ``torch.Tensor`` derived from the sign mask, so it
        cannot be part of the (hashable) value key. ``__init__`` and
        ``_rebuild_quantizer`` both call this hook; the ``lru_cache`` on
        :func:`get_rht_matrix` makes an already-seen (flag, device) pair a
        cheap hit.
        """
        # Fields added after the first value-key format: default them when absent.
        self.rht_dim = getattr(self, "rht_dim", 16)
        self.ms_eden = getattr(self, "ms_eden", False)
        self.rht_sign_policy = getattr(self, "rht_sign_policy", "static")
        self.rht_sign_seed = getattr(self, "rht_sign_seed", 0)
        self.rht_resample_interval = getattr(self, "rht_resample_interval", 1)
        self.ms_eden_grid_max = getattr(self, "ms_eden_grid_max", 6.0)
        self.rht_sign_stream = getattr(self, "rht_sign_stream", RHT_SIGN_STREAM_WGRAD)
        self.rht_rowwise = getattr(self, "rht_rowwise", False)
        self.dgrad_weight_source = getattr(self, "dgrad_weight_source", "high_precision")
        self.dgrad_weight_ms_eden = getattr(self, "dgrad_weight_ms_eden", False)
        self.dgrad_rht = getattr(self, "dgrad_rht", False)
        self.rht_matrix_rowwise = None
        self._dgrad_weight_quantizer = None

        if self._rht_uses_legacy_signs() and not self.rht_rowwise:
            # Default: hard-coded sign vector, nothing to refresh.
            self.rht_dynamic_signs = False
            self._rht_sign_generation = 0
            self.rht_matrix = get_rht_matrix(
                self.rht_matrix_random_sign_mask_t != 0, torch.cuda.current_device()
            )
            return

        # Seeded sign vectors. The C++ quantizer calls ``refresh_rht_signs`` before each
        # quantization when ``rht_dynamic_signs`` is set.
        self.rht_dynamic_signs = self.rht_sign_policy == "resampled"
        self._rht_sign_generation = None
        self.refresh_rht_signs()

    def _rht_uses_legacy_signs(self) -> bool:
        """Whether the columnwise copy uses the hard-coded dimension-16 sign vector."""
        return (
            self.rht_dim == 16
            and self.rht_sign_policy == "static"
            and self.rht_sign_stream == RHT_SIGN_STREAM_WGRAD
        )

    def rht_sign_generation(self) -> int:
        """Index of the sign vector in use for the current step."""
        if self.rht_sign_policy == "resampled":
            return nvfp4_rht_get_step() // self.rht_resample_interval
        return 0

    def rht_sign_bits(self, rowwise: bool = False) -> Tuple[int, ...]:
        """Sign bits (1 means -1) of the RHT sign vector for the current step.

        ``rowwise=False``: sign vector of the columnwise copy. ``rowwise=True``: sign
        vector of the rotated rowwise copy (DGRAD stream). Two tensors rotate
        identically (and their rotations cancel in a GEMM over the rotated dimension)
        iff this tuple is identical.
        """
        if rowwise:
            return get_rht_sign_bits(
                self.rht_dim, self.rht_sign_seed, self.rht_sign_generation(), RHT_SIGN_STREAM_DGRAD
            )
        if self._rht_uses_legacy_signs():
            mask = self.rht_matrix_random_sign_mask_t
            return tuple((mask >> i) & 1 for i in range(16))
        return get_rht_sign_bits(
            self.rht_dim, self.rht_sign_seed, self.rht_sign_generation(), self.rht_sign_stream
        )

    def refresh_rht_signs(self) -> None:
        """Rebuild the RHT matrices (and dimension-16 sign mask) if the generation changed."""
        legacy = self._rht_uses_legacy_signs()
        if legacy and not self.rht_rowwise:
            return
        generation = self.rht_sign_generation()
        if generation == self._rht_sign_generation:
            return
        device = torch.cuda.current_device()
        if legacy:
            self.rht_matrix = get_rht_matrix(self.rht_matrix_random_sign_mask_t != 0, device)
        else:
            sign_bits = self.rht_sign_bits()
            self.rht_matrix = get_rht_matrix_from_sign_bits(sign_bits, device)
            if self.rht_dim == 16:
                # The dimension-16 kernels that do not take the matrix use this mask.
                self.rht_matrix_random_sign_mask_t = sum(
                    bit << i for i, bit in enumerate(sign_bits)
                )
        if self.rht_rowwise:
            self.rht_matrix_rowwise = get_rht_matrix_from_sign_bits(
                self.rht_sign_bits(rowwise=True), device
            )
        self._rht_sign_generation = generation

    def set_rowwise_rht(self, enabled: bool) -> None:
        """Enable/disable the rotation of the rowwise copy (rotated DGRAD operand)."""
        if enabled == self.rht_rowwise:
            return
        if enabled and not self.with_rht:
            raise ValueError("A rotated rowwise copy requires with_rht=True.")
        self.rht_rowwise = enabled
        self._rebuild_derived_state()

    def dgrad_weight_quantizer(self) -> "NVFP4Quantizer":
        """Quantizer of the rotated columnwise weight copy consumed by a rotated DGRAD.

        Rotates along the output-channel dimension with the DGRAD sign stream, i.e.
        with the same sign vector as the rotated rowwise copy of the output gradient
        produced by this quantizer.
        """
        if not self.rht_rowwise:
            raise RuntimeError("dgrad_weight_quantizer requires rht_rowwise=True.")
        if self._dgrad_weight_quantizer is None:
            quantizer = NVFP4Quantizer(
                fp4_dtype=self.dtype,
                scale_dtype=self.scale_dtype,
                rowwise=False,
                columnwise=True,
                with_rht=True,
                with_post_rht_amax=True,
                stochastic_rounding=False,
                rht_dim=self.rht_dim,
                ms_eden=self.dgrad_weight_ms_eden,
                rht_sign_policy=self.rht_sign_policy,
                rht_sign_seed=self.rht_sign_seed,
                rht_resample_interval=self.rht_resample_interval,
                ms_eden_grid_max=self.ms_eden_grid_max,
                rht_sign_stream=RHT_SIGN_STREAM_DGRAD,
            )
            quantizer.internal = True
            quantizer.optimize_for_gemm = self.optimize_for_gemm
            self._dgrad_weight_quantizer = quantizer
        return self._dgrad_weight_quantizer

    def update_quantized(
        self,
        src: torch.Tensor,
        dst: QuantizedTensor,
        *,
        noop_flag: Optional[torch.Tensor] = None,
    ) -> QuantizedTensor:

        assert isinstance(dst, NVFP4Tensor), f"Cannot store quantized NVFP4 in {type(dst)} type."

        # Make sure input is in expected format
        if not devices_match(src.device, dst.device):
            src = src.to(device=dst.device)
        if not src.is_contiguous():
            src = src.contiguous()

        # Apply the destination tensor's amax reduction group on a throwaway copy
        quantizer = self
        group = getattr(dst, "amax_reduction_group", None)
        if group is not None:
            quantizer = self.copy()
            quantizer.with_amax_reduction = True
            quantizer.amax_reduction_group = group

        # Launch cast kernel
        tex.quantize(src, quantizer, dst, noop_flag)

        return dst

    def copy(self) -> NVFP4Quantizer:
        """Create shallow copy"""

        quantizer = NVFP4Quantizer(
            fp4_dtype=self.dtype,
            scale_dtype=self.scale_dtype,
            rowwise=self.rowwise_usage,
            columnwise=self.columnwise_usage,
            with_amax_reduction=self.with_amax_reduction,
            # Absent on quantizers rebuilt from a value key (deprecated field).
            amax_reduction_group=getattr(self, "amax_reduction_group", None),
            with_rht=self.with_rht,
            with_post_rht_amax=self.with_post_rht_amax,
            with_2d_quantization=self.with_2d_quantization,
            stochastic_rounding=self.stochastic_rounding,
            row_scaled_nvfp4=self.row_scaled_nvfp4,
            nvfp4_use_4over6=self.nvfp4_use_4over6,
            nvfp4_e4m3_max=self.nvfp4_e4m3_max,
            nvfp4_4over6_err_mode=self.nvfp4_4over6_err_mode,
            with_random_sign_mask=(
                self.rht_matrix_random_sign_mask_t != 0
                or self.rht_dim != 16
                or self.rht_sign_policy != "static"
                or self.rht_rowwise
                or self.rht_sign_stream != RHT_SIGN_STREAM_WGRAD
            ),
            disable_second_level_scale=self.disable_second_level_scale,
            nvfp4_4over6_columnwise=self.nvfp4_4over6_columnwise,
            columnwise_from_rowwise=self.columnwise_from_rowwise,
            rht_dim=self.rht_dim,
            ms_eden=self.ms_eden,
            rht_sign_policy=self.rht_sign_policy,
            rht_sign_seed=self.rht_sign_seed,
            rht_resample_interval=self.rht_resample_interval,
            ms_eden_grid_max=self.ms_eden_grid_max,
            rht_sign_stream=self.rht_sign_stream,
            rht_rowwise=self.rht_rowwise,
            dgrad_weight_source=self.dgrad_weight_source,
            dgrad_weight_ms_eden=self.dgrad_weight_ms_eden,
            dgrad_rht=self.dgrad_rht,
        )
        quantizer.internal = self.internal
        quantizer.optimize_for_gemm = self.optimize_for_gemm

        return quantizer

    def quantize_impl(self, tensor: torch.Tensor) -> QuantizedTensor:
        """Quantize tensor implementation"""
        return tex.quantize(tensor, self)

    def is_requantization_safe(self) -> bool:
        """NVFP4 quantization is replay-safe unless stochastic rounding is enabled.

        MS-EDEN stochastically rounds the micro-scales and resampled RHT signs change
        between steps, so neither is replay-safe.
        """
        return not (self.stochastic_rounding or self.ms_eden or self.rht_sign_policy == "resampled")

    def supports_quantized_allgather(self, inp: torch.Tensor) -> bool:
        """Whether tensor shape supports quantized all-gather.

        For distributed all-gather with columnwise scaling, the first
        dimension must be aligned to the block size so that no scaling
        block spans across GPU boundaries.
        """
        if self.row_scaled_nvfp4:
            return False
        if inp.ndim < 2:
            return False
        if inp.shape[-1] % 32 != 0:
            return False
        if self.columnwise_usage and math.prod(inp.shape[:-1]) % NVFP4_BLOCK_SCALING_SIZE != 0:
            return False
        return True

    def get_scale_shape(self, shape: Iterable[int], columnwise: bool) -> Tuple[int, int]:
        """Calculate the shape of the scaling tensor for NVFP4 1D blockwise quantization.

        This method determines the shape of the scaling tensor needed for blockwise quantization,
        taking into account the input tensor shape and whether columnwise scaling is used.

        Parameters
        ----------
        shape : Iterable[int]
            Shape of the input tensor to be quantized
        columnwise : bool
            Whether to use columnwise scaling (True) or rowwise scaling (False)

        Returns
        -------
        Tuple[int, int]
            Shape of the scaling tensor as (outer_dim, inner_dim)
            For NVFP4 1D blockwise quantization, blocksize is 16
            - If columnwise: (round_to_multiple(K, 128), round_to_multiple(roundup(M / 16), 4))
            - If rowwise: (round_to_multiple(M, 128), round_to_multiple(roundup(K / 16), 4))
        Swizzle kernel will be performed before GEMM to suit the need of CuBLAS.
        CuBLAS doc: https://docs.nvidia.com/cuda/cublas/index.html#d-block-scaling-factors-layout
        """
        M, K = 1, 1
        M = math.prod(shape[:-1])
        K = shape[-1]

        if columnwise:
            outer = round_up_to_nearest_multiple(K, 128)
            inner = round_up_to_nearest_multiple(ceil_div(M, NVFP4_BLOCK_SCALING_SIZE), 4)
            return (outer, inner)
        # rowwise
        outer = round_up_to_nearest_multiple(M, 128)
        inner = round_up_to_nearest_multiple(ceil_div(K, NVFP4_BLOCK_SCALING_SIZE), 4)
        return (outer, inner)

    @staticmethod
    def get_columnwise_shape(shape: Iterable[int]) -> Tuple[int, ...]:
        """Return the flattened columnwise shape before FP4 packing."""
        shape = tuple(shape)
        if not shape:
            return ()
        return (shape[-1], math.prod(shape[:-1]))

    @staticmethod
    def convert_shape_for_fp4(shape: Iterable[int]) -> Tuple[int, ...]:
        """Convert shape for FP4 data by dividing the last dimension by 2"""
        shape = list(shape)
        if shape[-1] % 2 != 0:
            raise ValueError(f"FP4 packing requires an even last dimension, got shape {shape}")
        shape[-1] = shape[-1] // 2
        return tuple(shape)

    def calibrate(self, tensor: torch.Tensor) -> None:
        pass  # Calibration is no-op

    def _canonicalized_amax_reduction_group(self) -> dist_group_type:
        """Get process group for amax reduction"""
        return canonicalize_process_group(self.amax_reduction_group)

    def _get_compatible_recipe(self) -> Union[type[Recipe], None]:
        return NVFP4BlockScaling

    # ----- TensorSpec / pure-Python allocation -----

    def storage_metadata(self, fake_dtype: torch.dtype) -> Dict[str, Any]:
        return {
            "cls": NVFP4TensorStorage if self.internal else NVFP4Tensor,
            "nontensor_kwargs": {
                "fp4_dtype": self.dtype,
                "scale_dtype": self.scale_dtype,
                "quantizer": self,
                "with_gemm_swizzled_scales": self.optimize_for_gemm,
                "row_scaled_nvfp4": self.row_scaled_nvfp4,
                "nvfp4_use_4over6": self.nvfp4_use_4over6,
                "nvfp4_e4m3_max": self.nvfp4_e4m3_max,
                "nvfp4_columnwise_e4m3_max": self.nvfp4_columnwise_e4m3_max,
                "fake_dtype": fake_dtype,
            },
        }

    def inner_tensor_specs(
        self, shape: Tuple[int, ...]
    ) -> Dict[str, Tuple[Tuple[int, ...], torch.dtype]]:
        shape = tuple(shape)
        specs: Dict[str, Tuple[Tuple[int, ...], torch.dtype]] = {}
        # FP4 data packs 2 values per byte (uint8); block scales are E4M3 stored
        # as uint8; amax inner tensors are FP32 (per-row when row-scaled, else scalar).
        # Order matches NVFP4TensorStorage._INNER_TENSORS (the canonical
        # __tensor_flatten__ order): data + scale_inv per usage first, amax last.
        # Workaround: call @staticmethods via the class, not the instance --
        # instance access breaks torch.compile guard generation (pytorch #182741).
        if self.rowwise_usage:
            specs["_rowwise_data"] = (type(self).convert_shape_for_fp4(shape), torch.uint8)
            specs["_rowwise_scale_inv"] = (
                tuple(self.get_scale_shape(shape, columnwise=False)),
                torch.uint8,
            )
        if self.columnwise_usage:
            specs["_columnwise_data"] = (
                type(self).convert_shape_for_fp4(type(self).get_columnwise_shape(shape)),
                torch.uint8,
            )
            specs["_columnwise_scale_inv"] = (
                tuple(self.get_scale_shape(shape, columnwise=True)),
                torch.uint8,
            )
        if self.rowwise_usage:
            amax_rowwise_shape = (math.prod(shape[:-1]),) if self.row_scaled_nvfp4 else (1,)
            specs["_amax_rowwise"] = (amax_rowwise_shape, torch.float32)
        if self.columnwise_usage:
            amax_columnwise_shape = (shape[-1],) if self.row_scaled_nvfp4 else (1,)
            specs["_amax_columnwise"] = (amax_columnwise_shape, torch.float32)
        return specs


register_value_opaque_quantizer(NVFP4Quantizer)


class NVFP4Tensor(NVFP4TensorStorage, QuantizedTensor):
    """Quantized tensor class with FP4 data

    The tensor presents as having a standard, higher-precision dtype,
    but the data itself is (scaled) FP4. For most tensor operations,
    the data will be cast to the nominal dtype before performing the
    operation.

    Parameters
    ----------
    rowwise_data : torch.Tensor
        Raw FP4 data in a uint8 tensor (rowwise layout).
    rowwise_scale_inv : torch.Tensor
        Reciprocal of the scaling factor applied when
        casting to FP4, i.e. the scaling factor that must
        be applied when casting from FP4 to higher
        precision (rowwise).
    columnwise_data : torch.Tensor, optional
        Raw FP4 data in a uint8 tensor (columnwise layout).
    columnwise_scale_inv : torch.Tensor, optional
        Reciprocal of the scaling factor for columnwise FP4 data.
    amax_rowwise : torch.Tensor, optional
        Rowwise amax tracking tensor.
    amax_columnwise : torch.Tensor, optional
        Columnwise amax tracking tensor.
    fp4_dtype : DType
        The FP4 data type used for quantization.
    scale_dtype: DType
        The FP8 scale factor data type used for quantization.
    quantizer : Quantizer
        The quantizer instance used for this tensor.
    dtype : torch.dtype, default = torch.float32
        Nominal tensor datatype, used in dequantize.
    """

    # Optional amax all-reduce group, set by FSDP2 in ``fsdp_pre_all_gather``
    amax_reduction_group: Optional[dist_group_type] = None

    # NOTE: We reorder the *args so that we can instantiate a NVFP4TensorStorage with positional args,
    # which significantly reduces the Pybind11 overhead when calling the constructor from C++.
    def __new__(
        cls,
        *args,
        rowwise_data: Optional[torch.Tensor],
        rowwise_scale_inv: Optional[torch.Tensor],
        columnwise_data: Optional[torch.Tensor],
        columnwise_scale_inv: Optional[torch.Tensor],
        amax_rowwise: Optional[torch.Tensor],
        amax_columnwise: Optional[torch.Tensor],
        fp4_dtype: DType,
        scale_dtype: DType = DType.kFloat8E4M3,
        quantizer: Quantizer,
        with_gemm_swizzled_scales: bool,
        row_scaled_nvfp4: bool = False,
        nvfp4_use_4over6: bool = False,
        nvfp4_e4m3_max: int = 0,
        nvfp4_columnwise_e4m3_max: int = 0,
        **kwargs,
    ):
        instance = super().__new__(
            cls,
            rowwise_data,
            rowwise_scale_inv,
            columnwise_data,
            columnwise_scale_inv,
            amax_rowwise,
            amax_columnwise,
            fp4_dtype,
            scale_dtype,
            quantizer,
            with_gemm_swizzled_scales,
            *args,
            row_scaled_nvfp4=row_scaled_nvfp4,
            nvfp4_use_4over6=nvfp4_use_4over6,
            nvfp4_e4m3_max=nvfp4_e4m3_max,
            nvfp4_columnwise_e4m3_max=nvfp4_columnwise_e4m3_max,
            **kwargs,
        )
        return instance

    def __repr__(self, *, tensor_contents=None):
        try:
            return f"NVFP4Tensor, data={self.dequantize()})"
        except Exception as exc:  # pylint: disable=broad-except
            return safe_quantized_repr(self, "NVFP4Tensor", error=exc)

    def dequantize(self, *, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        """
        Construct plain PyTorch tensor from NVFP4Tensor

        By default the resulting tensor's dtype is the
        NVFP4Tensor's nominal dtype.
        """
        # Convert PyTorch dtype to TE dtype
        if dtype is None:
            dtype = self.dtype

        if torch.is_grad_enabled():
            return _FromNVFP4Func.apply(self, dtype)
        return _FromNVFP4Func.forward(None, self, dtype)

    def _get_quantizer(self) -> Quantizer:
        """Get builder for quantized tensor

        Quantizer can be used for in-place operations.

        """
        if self._quantizer is not None:
            return self._quantizer
        return NVFP4Quantizer()

    def quantize_(
        self,
        tensor: torch.Tensor,
        *,
        noop_flag: Optional[torch.Tensor] = None,
    ) -> NVFP4Tensor:
        """Update FP8 data

        Parameters
        ----------
        tensor: torch.Tensor
            Tensor to copy from
        noop_flag: torch.Tensor, optional
            float32 flag indicating whether to avoid performing update

        """
        if isinstance(tensor, QuantizedTensor):
            return self.quantize_(tensor.dequantize())
        self._get_quantizer().update_quantized(tensor, self, noop_flag=noop_flag)
        return self

    def clone(self) -> NVFP4Tensor:
        # pylint: disable=missing-function-docstring
        assert self._rowwise_data is not None
        rowwise_data = self._rowwise_data.detach().clone()
        columnwise_data = None
        if self._columnwise_data is not None:
            columnwise_data = self._columnwise_data.detach().clone()
        return _IdentityFunc.apply(
            self,
            {
                "rowwise_data": rowwise_data,
                "columnwise_data": columnwise_data,
            },
        )

    def view(self, *shape: Tuple[int]) -> NVFP4Tensor:
        # pylint: disable=missing-function-docstring
        return _ViewFunc.apply(self, shape)

    def reshape(self, *shape: Tuple[int]) -> NVFP4Tensor:
        # pylint: disable=missing-function-docstring
        return _ReshapeFunc.apply(self, shape)

    def contiguous(
        self,
        memory_format: torch.memory_format = torch.contiguous_format,
    ) -> NVFP4Tensor:
        """Returns tensor with data in provided memory format

        Returns `self` if data is already in correct memory format.

        """
        if self._rowwise_data is not None and self._rowwise_data.is_contiguous(
            memory_format=memory_format
        ):
            return self
        if self._columnwise_data is not None and self._columnwise_data.is_contiguous(
            memory_format=memory_format
        ):
            return self
        raise ValueError("NVFP4Tensor does not support different memory formats!")

    def get_usages(self) -> Dict[str, bool]:
        return {
            "rowwise": self._rowwise_data is not None,
            "columnwise": self._columnwise_data is not None,
        }

    def fsdp_pre_all_gather(self, mesh, orig_size, contiguous_orig_stride, module, mp_policy):
        """Called by FSDP2 before all-gather of weights.

        Only all-gathers rowwise data and scales. Columnwise data is derived
        locally in post_all_gather via _create_columnwise(), halving the
        all-gather communication volume.
        """
        # pylint: disable=unused-argument

        if self._with_gemm_swizzled_scales:
            raise NotImplementedError(
                "FSDP2 is not supported for NVFP4Tensors with GEMM-swizzled scales."
            )

        if mesh is not None:
            # Reduce amax across the mesh so all weight shards get the same scale
            self.amax_reduction_group = mesh.get_group()

        shard_M = math.prod(self.shape[:-1])

        assert shard_M % NVFP4_BLOCK_SCALING_SIZE == 0, (
            f"FSDP2 requires shard_M ({shard_M}) to be a multiple of "
            f"NVFP4_BLOCK_SCALING_SIZE ({NVFP4_BLOCK_SCALING_SIZE}). "
            "Adjust model dimensions or world size."
        )

        assert self._rowwise_data is not None, (
            "FSDP2 requires rowwise data, but _rowwise_data is None. "
            "Ensure the NVFP4Quantizer was created with rowwise=True."
        )

        # Rowwise data: (shard_M, K//2) — M in dim0, pass as-is
        rowwise_data = self._rowwise_data
        # Rowwise scale: (round_up(shard_M, 128), inner) — unpad dim0 to shard_M
        rowwise_scale_inv = self._rowwise_scale_inv
        if rowwise_scale_inv is not None:
            rowwise_scale_inv = rowwise_scale_inv[:shard_M, :]

        columnwise_usage = self._quantizer.columnwise_usage
        if columnwise_usage:
            assert self._quantizer.with_2d_quantization, (
                "FSDP2 columnwise usage requires 2D quantization to be enabled. "
                "Ensure the NVFP4Quantizer was created with with_2d_quantization=True."
            )

        # Only all-gather rowwise tensors; columnwise will be derived locally
        # via _create_columnwise() in post_all_gather.
        sharded_tensors = (rowwise_data, rowwise_scale_inv)

        # Pass amax via metadata (scalar, same on all ranks — not all-gathered)
        metadata = (
            self._fp4_dtype,
            self._scale_dtype,
            columnwise_usage,
            self._amax_rowwise,
            self._amax_columnwise,
            self._row_scaled_nvfp4,
            self._nvfp4_use_4over6,
            self._nvfp4_e4m3_max,
            self.shape[-1],
        )
        return sharded_tensors, metadata

    def fsdp_post_all_gather(
        self,
        all_gather_outputs: Tuple[torch.Tensor, ...],
        metadata,
        param_dtype: torch.dtype,
        *,
        out: Optional[NVFP4Tensor] = None,
    ):
        """Called by FSDP2 after all-gather of weights.

        Repads rowwise scales and constructs the full NVFP4Tensor from
        all-gathered rowwise data. Columnwise data is derived locally
        via _create_columnwise() instead of being all-gathered.
        """
        (
            fp4_dtype,
            scale_dtype,
            columnwise_usage,
            amax_rowwise,
            amax_columnwise,
            row_scaled_nvfp4,
            nvfp4_use_4over6,
            nvfp4_e4m3_max,
            K,
        ) = metadata

        # Only rowwise data+scales were all-gathered
        rowwise_data, rowwise_scale_inv = all_gather_outputs[:2]
        full_M = rowwise_data.shape[0]

        # Repad rowwise scale dim0 to round_up(full_M, 128)
        if rowwise_scale_inv is not None:
            target_m = round_up_to_nearest_multiple(full_M, 128)
            current_m = rowwise_scale_inv.shape[0]
            if current_m < target_m:
                rowwise_scale_inv = torch.nn.functional.pad(
                    rowwise_scale_inv, (0, 0, 0, target_m - current_m)
                )

        logical_shape = (full_M, K)

        if out is not None:
            # Update existing tensor in-place (subsequent iterations)
            out._rowwise_data = rowwise_data
            out._rowwise_scale_inv = rowwise_scale_inv
            out._amax_rowwise = amax_rowwise
            out._amax_columnwise = amax_columnwise
            out._row_scaled_nvfp4 = row_scaled_nvfp4
            out._nvfp4_use_4over6 = nvfp4_use_4over6
            out._nvfp4_e4m3_max = nvfp4_e4m3_max
            # The columnwise copy is derived locally from the all-gathered rowwise
            # copy (see _create_columnwise), so it uses the rowwise scale bound.
            out._nvfp4_columnwise_e4m3_max = 0
        else:
            # Construct new tensor (first iteration)
            out = NVFP4Tensor(
                shape=logical_shape,
                dtype=param_dtype,
                fp4_dtype=fp4_dtype,
                scale_dtype=scale_dtype,
                rowwise_data=rowwise_data,
                rowwise_scale_inv=rowwise_scale_inv,
                columnwise_data=None,
                columnwise_scale_inv=None,
                amax_rowwise=amax_rowwise,
                amax_columnwise=amax_columnwise,
                quantizer=self._quantizer,
                requires_grad=False,
                with_gemm_swizzled_scales=False,
                device=rowwise_data.device,
                row_scaled_nvfp4=row_scaled_nvfp4,
                nvfp4_use_4over6=nvfp4_use_4over6,
                nvfp4_e4m3_max=nvfp4_e4m3_max,
            )

        # Derive columnwise data locally via transpose instead of all-gathering it
        if columnwise_usage:
            out._create_columnwise()

        out._quantizer.set_usage(rowwise=True, columnwise=columnwise_usage)
        return out, all_gather_outputs

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs=None):

        # View op
        if func == aten.view.default:
            if len(args) != 2:
                raise RuntimeError(
                    f"Unexpected args for view op (expected 2 args, got {len(args)})"
                )
            tensor = args[0]
            shape = args[1]
            if shape == list(tensor.size()):
                return tensor.detach()
            return tensor.view(shape)

        # as_strided — FSDP2 applies this on the unsharded param.
        # Only the identity case (same shape, contiguous strides, zero offset) is supported.
        # Non-identity as_strided cannot fall through because NVFP4 does not support
        # dequantization, so we raise explicitly rather than producing undefined behavior.
        if func == aten.as_strided.default:
            tensor = args[0]
            shape = args[1]
            strides = args[2]
            storage_offset = args[3] if len(args) > 3 else 0
            if (
                len(shape) == len(strides) == 2
                and tuple(strides) == (shape[-1], 1)
                and tuple(shape) == tuple(tensor.size())
                and storage_offset == 0
            ):
                return NVFP4Tensor.make_like(tensor)
            raise NotImplementedError(
                "NVFP4Tensor does not support non-identity as_strided "
                f"(shape={shape}, strides={strides}, storage_offset={storage_offset}, "
                f"tensor.size()={tuple(tensor.size())})"
            )

        # slice — FSDP2 applies this for shard unpadding.
        # When the slice covers the full dimension, return self.
        if func == aten.slice.Tensor:
            tensor = args[0]
            dim = args[1] if len(args) > 1 else 0
            start = args[2] if len(args) > 2 else None
            end = args[3] if len(args) > 3 else None
            step = args[4] if len(args) > 4 else 1
            if (
                step == 1
                and (start is None or start == 0)
                and (end is None or end >= tensor.size(dim))
            ):
                return NVFP4Tensor.make_like(tensor)
            raise NotImplementedError(
                "NVFP4Tensor does not support partial slicing "
                f"(dim={dim}, start={start}, end={end}, "
                f"tensor.size(dim)={tensor.size(dim)})"
            )

        # record_stream — FSDP2 records streams on all-gathered tensors.
        if func == torch.ops.aten.record_stream.default:
            qt, stream = args
            for t in (
                qt._rowwise_data,
                qt._columnwise_data,
                qt._rowwise_scale_inv,
                qt._columnwise_scale_inv,
                qt._amax_rowwise,
                qt._amax_columnwise,
            ):
                if t is not None and t.device.type == te_device_type():
                    t.record_stream(stream)
            return None

        # copy_ — FSDP2 may call this during resharding or parameter writeback.
        if func == aten.copy_.default:
            dst, src = args[0], args[1]
            if isinstance(src, NVFP4Tensor) and isinstance(dst, NVFP4Tensor):
                if dst._rowwise_data is not None and src._rowwise_data is not None:
                    dst._rowwise_data.copy_(src._rowwise_data.detach())
                    dst._rowwise_scale_inv.copy_(src._rowwise_scale_inv.detach())
                if dst._columnwise_data is not None and src._columnwise_data is not None:
                    dst._columnwise_data.copy_(src._columnwise_data.detach())
                    dst._columnwise_scale_inv.copy_(src._columnwise_scale_inv.detach())
                if dst._amax_rowwise is not None and src._amax_rowwise is not None:
                    dst._amax_rowwise.copy_(src._amax_rowwise.detach())
                if dst._amax_columnwise is not None and src._amax_columnwise is not None:
                    dst._amax_columnwise.copy_(src._amax_columnwise.detach())
                return dst

        # NVFP4 dequantize not supported. Add manual support for needed funcs.
        if func in (aten.empty_like.default, aten.zero_.default):
            tensor = args[0]
            data_init_func = torch.zeros_like if func == aten.zero_.default else torch.empty_like
            scale_inv_init_func = (
                torch.ones_like if func == aten.zero_.default else torch.empty_like
            )

            if tensor._rowwise_data is not None:
                rowwise_data = data_init_func(tensor._rowwise_data, *args[1:], **kwargs)
                rowwise_scale_inv = scale_inv_init_func(
                    tensor._rowwise_scale_inv, *args[1:], **kwargs
                )
                amax_rowwise = (
                    None
                    if tensor._amax_rowwise is None
                    else torch.zeros_like(tensor._amax_rowwise, *args[1:], **kwargs)
                )
            else:
                rowwise_data, rowwise_scale_inv, amax_rowwise = None, None, None

            if tensor._columnwise_data is not None:
                columnwise_data = data_init_func(tensor._columnwise_data, *args[1:], **kwargs)
                columnwise_scale_inv = scale_inv_init_func(
                    tensor._columnwise_scale_inv, *args[1:], **kwargs
                )
                amax_columnwise = (
                    None
                    if tensor._amax_columnwise is None
                    else torch.zeros_like(tensor._amax_columnwise, *args[1:], **kwargs)
                )
            else:
                columnwise_data, columnwise_scale_inv, amax_columnwise = (
                    None,
                    None,
                    None,
                )

            return NVFP4Tensor(
                shape=tensor.shape,
                dtype=tensor.dtype,
                fp4_dtype=tensor._fp4_dtype,
                scale_dtype=tensor._scale_dtype,
                rowwise_data=rowwise_data,
                rowwise_scale_inv=rowwise_scale_inv,
                columnwise_data=columnwise_data,
                columnwise_scale_inv=columnwise_scale_inv,
                amax_rowwise=amax_rowwise,
                amax_columnwise=amax_columnwise,
                quantizer=tensor._quantizer,
                requires_grad=tensor.requires_grad,
                with_gemm_swizzled_scales=tensor._with_gemm_swizzled_scales,
                device=tensor.device,
                row_scaled_nvfp4=tensor._row_scaled_nvfp4,
                nvfp4_use_4over6=tensor._nvfp4_use_4over6,
                nvfp4_e4m3_max=tensor._nvfp4_e4m3_max,
                nvfp4_columnwise_e4m3_max=tensor._nvfp4_columnwise_e4m3_max,
            )

        # Default case
        return super().__torch_dispatch__(func, types, args, kwargs)

    def __reduce_ex__(self, protocol: int) -> tuple:
        """Custom pickling"""
        return (
            _make_nvfp4_tensor_in_reduce_ex,
            (
                self.shape,
                self._rowwise_data,
                self._rowwise_scale_inv,
                self._columnwise_data,
                self._columnwise_scale_inv,
                self._amax_rowwise,
                self._amax_columnwise,
                self._fp4_dtype,
                self.dtype,
                self._quantizer,
                self._with_gemm_swizzled_scales,
                self._row_scaled_nvfp4,
                self._nvfp4_use_4over6,
                self._nvfp4_e4m3_max,
                self._scale_dtype,
                self._nvfp4_columnwise_e4m3_max,
            ),
        )

    @classmethod
    def _make_in_reduce_ex(
        cls,
        shape: torch.Size,
        rowwise_data: torch.Tensor,
        rowwise_scale_inv: torch.Tensor,
        columnwise_data: torch.Tensor,
        columnwise_scale_inv: torch.Tensor,
        amax_rowwise: torch.Tensor,
        amax_columnwise: torch.Tensor,
        fp4_dtype: DType,
        dtype: torch.dtype,
        quantizer: Quantizer,
        with_gemm_swizzled_scales: bool = False,
    ) -> NVFP4Tensor:
        """This classmethod is kept for backward compatibility only.
        ``__reduce_ex__`` used to point at this classmethod, but bound
        classmethods pickle as ``(getattr, (cls, name))`` which adds an
        extra reduction step to the pickle stream. The current
        ``__reduce_ex__`` references the module-level
        ``_make_nvfp4_tensor_in_reduce_ex`` instead so the pickle stream
        uses a single ``GLOBAL`` opcode. This classmethod is retained so
        that previously pickled ``NVFP4Tensor`` payloads (which still
        reference ``NVFP4Tensor._make_in_reduce_ex``) can still be
        unpickled.
        """
        return _make_nvfp4_tensor_in_reduce_ex(
            shape,
            rowwise_data,
            rowwise_scale_inv,
            columnwise_data,
            columnwise_scale_inv,
            amax_rowwise,
            amax_columnwise,
            fp4_dtype,
            dtype,
            quantizer,
            with_gemm_swizzled_scales,
        )

    def _get_data(self) -> NVFP4Tensor:
        """Get tensor data property"""
        return super().data

    @torch.no_grad()
    def _set_data(self, tensor: torch.Tensor) -> None:
        """Set tensor data property

        Just takes FP8 data if setting from a NVFP4Tensor. Otherwise
        casts to FP8.

        """

        # Tensor device
        new_device = tensor.device if tensor.device.type == te_device_type() else self.device
        if not devices_match(new_device, tensor.device):
            tensor = tensor.to(device=new_device)

        # Just copy FP8 data if other tensor is NVFP4Tensor
        if isinstance(tensor, NVFP4Tensor):
            if (  # pylint: disable=too-many-boolean-expressions
                self.size() != tensor.size()
                or self.stride() != tensor.stride()
                or self.storage_offset() != tensor.storage_offset()
                or self.dtype != tensor.dtype
                or self.layout != tensor.layout
                or not devices_match(self.device, new_device)
            ):
                dummy_tensor = torch.Tensor._make_wrapper_subclass(
                    NVFP4Tensor,
                    tensor.size(),
                    strides=tensor.stride(),
                    storage_offset=tensor.storage_offset(),
                    dtype=tensor.dtype,
                    layout=tensor.layout,
                    requires_grad=tensor.requires_grad,
                    device=new_device,
                )
                # pylint: disable=unnecessary-dunder-call
                super(NVFP4Tensor, type(self)).data.__set__(self, dummy_tensor)

            self._rowwise_data = tensor._rowwise_data
            self._columnwise_data = tensor._columnwise_data
            self._quantizer = tensor._quantizer
            self._rowwise_scale_inv = tensor._rowwise_scale_inv
            self._columnwise_scale_inv = tensor._columnwise_scale_inv
            self._amax_rowwise = tensor._amax_rowwise
            self._amax_columnwise = tensor._amax_columnwise
            self._with_gemm_swizzled_scales = tensor._with_gemm_swizzled_scales
            self._row_scaled_nvfp4 = tensor._row_scaled_nvfp4
            self._nvfp4_use_4over6 = tensor._nvfp4_use_4over6
            self._nvfp4_e4m3_max = tensor._nvfp4_e4m3_max
            self._nvfp4_columnwise_e4m3_max = tensor._nvfp4_columnwise_e4m3_max
            return

        # Quantize to FP8
        assert self._quantizer is not None, "Can't quantize without a quantizer"
        self._quantizer.update_quantized(tensor, self)
        if self.requires_grad != tensor.requires_grad:
            self.requires_grad_(requires_grad=tensor.requires_grad)

    # Cast to FP8 when setting NVFP4Tensor.data
    data = property(_get_data, _set_data)

    @property
    def device(self):
        """Return the device of the tensor. Define this to avoid expensive PyObject lookups."""
        if self._rowwise_data is not None:
            return self._rowwise_data.device
        if self._columnwise_data is not None:
            return self._columnwise_data.device
        raise RuntimeError("NVFP4Tensor has no data!")

    @property
    def shape(self):
        """Return the shape of the tensor. Define this to avoid expensive PyObject lookups."""
        if self._rowwise_data is not None:
            byte_shape = self._rowwise_data.shape
            return torch.Size(byte_shape[:-1] + (byte_shape[-1] * 2,))
        if self._columnwise_data is not None:
            byte_shape = self._columnwise_data.shape
            return torch.Size(byte_shape[1:-1] + (byte_shape[-1] * 2, byte_shape[0]))
        return torch.Tensor.size(self)

    @property
    def is_cuda(self):
        """Return whether the tensor is on a CUDA device."""
        if self._rowwise_data is not None:
            return self._rowwise_data.is_cuda
        if self._columnwise_data is not None:
            return self._columnwise_data.is_cuda
        raise RuntimeError("NVFP4Tensor has no data!")


def _make_nvfp4_tensor_in_reduce_ex(
    shape: torch.Size,
    rowwise_data: torch.Tensor,
    rowwise_scale_inv: torch.Tensor,
    columnwise_data: torch.Tensor,
    columnwise_scale_inv: torch.Tensor,
    amax_rowwise: torch.Tensor,
    amax_columnwise: torch.Tensor,
    fp4_dtype: DType,
    dtype: torch.dtype,
    quantizer: Quantizer,
    with_gemm_swizzled_scales: bool,
    row_scaled_nvfp4: bool = False,
    nvfp4_use_4over6: bool = False,
    nvfp4_e4m3_max: int = 0,
    scale_dtype: DType = DType.kFloat8E4M3,
    nvfp4_columnwise_e4m3_max: int = 0,
) -> NVFP4Tensor:
    """Reconstruct an ``NVFP4Tensor`` from its ``__reduce_ex__`` payload."""
    # Infer device from whichever inner buffer is populated so the wrapper
    # subclass stays consistent with its data buffers (e.g. CPU after DCP
    # async-staging deserialize, CUDA after the usual quantize path).
    device = None
    if rowwise_data is not None:
        device = rowwise_data.device
    elif columnwise_data is not None:
        device = columnwise_data.device
    return NVFP4Tensor(
        shape=shape,
        dtype=dtype,
        fp4_dtype=fp4_dtype,
        scale_dtype=scale_dtype,
        rowwise_data=rowwise_data,
        rowwise_scale_inv=rowwise_scale_inv,
        columnwise_data=columnwise_data,
        columnwise_scale_inv=columnwise_scale_inv,
        amax_rowwise=amax_rowwise,
        amax_columnwise=amax_columnwise,
        quantizer=quantizer,
        requires_grad=False,
        with_gemm_swizzled_scales=with_gemm_swizzled_scales,
        device=device,
        row_scaled_nvfp4=row_scaled_nvfp4,
        nvfp4_use_4over6=nvfp4_use_4over6,
        nvfp4_e4m3_max=nvfp4_e4m3_max,
        nvfp4_columnwise_e4m3_max=nvfp4_columnwise_e4m3_max,
    )


class _ViewFunc(torch.autograd.Function):
    """View function

    View the NVFP4Tensor using the provided shape.

    """

    @staticmethod
    def forward(
        ctx,
        tensor: NVFP4Tensor,
        shape: Optional[list[int]] = None,
    ) -> NVFP4Tensor:
        # pylint: disable=missing-function-docstring

        # Return input tensor if shape is not provided
        cur_shape = tensor.shape
        if ctx is not None:
            ctx.shape = cur_shape
        if shape is None:
            return tensor

        # Canonicalize shape
        if not isinstance(shape, Iterable):
            shape = [shape]
        elif len(shape) == 1 and isinstance(shape[0], Iterable):
            shape = shape[0]
        if -1 in shape:
            shape = list(shape)
            d_inferred = -math.prod(cur_shape) // math.prod(shape)
            for i, d in enumerate(shape):
                if d == -1:
                    shape[i] = d_inferred
                    break
        if shape[-1] != cur_shape[-1]:
            warnings.warn(
                "NVFP4Tensor does not support reshaping inner dimension "
                f"(attempted to reshape dims={tuple(tensor.shape)} to {tuple(shape)}). "
                "If you are using this for FSDP2 without compiled_autograd_enabled, "
                "then ignore this warning since this view is not going to be used anywhere.",
                stacklevel=2,
            )
            return tensor.dequantize().view(*shape)

        # Reshape data
        new_rowwise_data = None
        new_columnwise_data = None
        if tensor._rowwise_data is not None:
            if shape[-1] % 2 != 0:
                raise ValueError(
                    "Cannot represent row-wise data for NVFP4 tensor "
                    f"with shape={shape} as byte array."
                )
            byte_shape = list(shape[:-1]) + [shape[-1] // 2]
            new_rowwise_data = tensor._rowwise_data.view(byte_shape)
        if tensor._columnwise_data is not None:
            columnwise_shape = (shape[-1], math.prod(shape[:-1]))
            if columnwise_shape[-1] % 2 != 0:
                raise ValueError(
                    "Cannot represent column-wise data for NVFP4 tensor "
                    f"with shape={shape} as byte array."
                )
            byte_shape = (columnwise_shape[0], columnwise_shape[1] // 2)
            new_columnwise_data = tensor._columnwise_data.view(byte_shape)

        # Construct tensor
        return NVFP4Tensor(
            shape,
            tensor.dtype,
            rowwise_data=new_rowwise_data,
            rowwise_scale_inv=tensor._rowwise_scale_inv,
            columnwise_data=new_columnwise_data,
            columnwise_scale_inv=tensor._columnwise_scale_inv,
            amax_rowwise=tensor._amax_rowwise,
            amax_columnwise=tensor._amax_columnwise,
            quantizer=tensor._quantizer,
            fp4_dtype=tensor._fp4_dtype,
            scale_dtype=tensor._scale_dtype,
            requires_grad=tensor.requires_grad,
            with_gemm_swizzled_scales=tensor._with_gemm_swizzled_scales,
            device=tensor.device,
            row_scaled_nvfp4=tensor._row_scaled_nvfp4,
            nvfp4_use_4over6=tensor._nvfp4_use_4over6,
            nvfp4_e4m3_max=tensor._nvfp4_e4m3_max,
            nvfp4_columnwise_e4m3_max=tensor._nvfp4_columnwise_e4m3_max,
        )

    @staticmethod
    def backward(
        ctx,
        grad: torch.Tensor,
    ) -> Tuple[Optional[torch.Tensor], ...]:
        # pylint: disable=missing-function-docstring

        if isinstance(grad, NVFP4Tensor):
            new_rowwise_data = None
            new_columnwise_data = None
            if grad._rowwise_data is not None:
                if ctx.shape[-1] % 2 != 0:
                    raise ValueError(
                        "Cannot represent row-wise data for NVFP4 tensor "
                        f"with shape={ctx.shape} as byte array."
                    )
                byte_shape = list(ctx.shape[:-1]) + [ctx.shape[-1] // 2]
                new_rowwise_data = grad._rowwise_data.view(byte_shape)
            if grad._columnwise_data is not None:
                columnwise_shape = (ctx.shape[-1], math.prod(ctx.shape[:-1]))
                if columnwise_shape[-1] % 2 != 0:
                    raise ValueError(
                        "Cannot represent column-wise data for NVFP4 tensor "
                        f"with shape={ctx.shape} as byte array."
                    )
                byte_shape = (columnwise_shape[0], columnwise_shape[1] // 2)
                new_columnwise_data = grad._columnwise_data.view(byte_shape)
            dgrad = NVFP4Tensor(
                ctx.shape,
                grad.dtype,
                rowwise_data=new_rowwise_data,
                rowwise_scale_inv=grad._rowwise_scale_inv,
                columnwise_data=new_columnwise_data,
                columnwise_scale_inv=grad._columnwise_scale_inv,
                amax_rowwise=grad._amax_rowwise,
                amax_columnwise=grad._amax_columnwise,
                quantizer=grad._quantizer,
                fp4_dtype=grad._fp4_dtype,
                scale_dtype=grad._scale_dtype,
                requires_grad=grad.requires_grad,
                with_gemm_swizzled_scales=grad._with_gemm_swizzled_scales,
                device=grad.device,
                row_scaled_nvfp4=grad._row_scaled_nvfp4,
                nvfp4_use_4over6=grad._nvfp4_use_4over6,
                nvfp4_e4m3_max=grad._nvfp4_e4m3_max,
                nvfp4_columnwise_e4m3_max=grad._nvfp4_columnwise_e4m3_max,
            )
            return dgrad, None
        return grad.view(ctx.shape), None


class _ReshapeFunc(torch.autograd.Function):
    """Reshape function

    Reshape the NVFP4Tensor using the provided shape.

    """

    @staticmethod
    def forward(
        ctx,
        tensor: NVFP4Tensor,
        shape: Optional[list[int]] = None,
    ) -> NVFP4Tensor:
        # pylint: disable=missing-function-docstring

        # Return input tensor if shape is not provided
        cur_shape = tensor.shape
        if ctx is not None:
            ctx.shape = cur_shape
        if shape is None:
            return tensor

        # Canonicalize shape
        if not isinstance(shape, Iterable):
            shape = [shape]
        elif len(shape) == 1 and isinstance(shape[0], Iterable):
            shape = shape[0]
        if -1 in shape:
            shape = list(shape)
            d_inferred = -math.prod(cur_shape) // math.prod(shape)
            for i, d in enumerate(shape):
                if d == -1:
                    shape[i] = d_inferred
                    break
        if shape[-1] != cur_shape[-1]:
            warnings.warn(
                "NVFP4Tensor does not support reshaping inner dimension "
                f"(attempted to reshape dims={tuple(tensor.shape)} to {tuple(shape)}). "
                "If you are using this for FSDP2 without compiled_autograd_enabled, "
                "then ignore this warning since this view is not going to be used anywhere.",
                stacklevel=2,
            )
            return tensor.dequantize().reshape(*shape)

        # Reshape data
        new_rowwise_data = None
        new_columnwise_data = None
        if tensor._rowwise_data is not None:
            if shape[-1] % 2 != 0:
                raise ValueError(
                    "Cannot represent row-wise data for NVFP4 tensor "
                    f"with shape={shape} as byte array."
                )
            byte_shape = list(shape[:-1]) + [shape[-1] // 2]
            new_rowwise_data = tensor._rowwise_data.reshape(byte_shape)
        if tensor._columnwise_data is not None:
            columnwise_shape = (shape[-1], math.prod(shape[:-1]))
            if columnwise_shape[-1] % 2 != 0:
                raise ValueError(
                    "Cannot represent column-wise data for NVFP4 tensor "
                    f"with shape={shape} as byte array."
                )
            byte_shape = (columnwise_shape[0], columnwise_shape[1] // 2)
            new_columnwise_data = tensor._columnwise_data.reshape(byte_shape)

        # Construct tensor
        return NVFP4Tensor(
            shape,
            tensor.dtype,
            rowwise_data=new_rowwise_data,
            rowwise_scale_inv=tensor._rowwise_scale_inv,
            columnwise_data=new_columnwise_data,
            columnwise_scale_inv=tensor._columnwise_scale_inv,
            amax_rowwise=tensor._amax_rowwise,
            amax_columnwise=tensor._amax_columnwise,
            quantizer=tensor._quantizer,
            fp4_dtype=tensor._fp4_dtype,
            scale_dtype=tensor._scale_dtype,
            requires_grad=tensor.requires_grad,
            with_gemm_swizzled_scales=tensor._with_gemm_swizzled_scales,
            device=tensor.device,
            row_scaled_nvfp4=tensor._row_scaled_nvfp4,
            nvfp4_use_4over6=tensor._nvfp4_use_4over6,
            nvfp4_e4m3_max=tensor._nvfp4_e4m3_max,
            nvfp4_columnwise_e4m3_max=tensor._nvfp4_columnwise_e4m3_max,
        )

    @staticmethod
    def backward(
        ctx,
        grad: torch.Tensor,
    ) -> Tuple[Optional[torch.Tensor], ...]:
        # pylint: disable=missing-function-docstring

        if isinstance(grad, NVFP4Tensor):
            new_rowwise_data = None
            new_columnwise_data = None
            if grad._rowwise_data is not None:
                if ctx.shape[-1] % 2 != 0:
                    raise ValueError(
                        "Cannot represent row-wise data for NVFP4 tensor "
                        f"with shape={ctx.shape} as byte array."
                    )
                byte_shape = list(ctx.shape[:-1]) + [ctx.shape[-1] // 2]
                new_rowwise_data = grad._rowwise_data.reshape(byte_shape)
            if grad._columnwise_data is not None:
                columnwise_shape = (ctx.shape[-1], math.prod(ctx.shape[:-1]))
                if columnwise_shape[-1] % 2 != 0:
                    raise ValueError(
                        "Cannot represent column-wise data for NVFP4 tensor "
                        f"with shape={ctx.shape} as byte array."
                    )
                byte_shape = (columnwise_shape[0], columnwise_shape[1] // 2)
                new_columnwise_data = grad._columnwise_data.reshape(byte_shape)
            dgrad = NVFP4Tensor(
                ctx.shape,
                grad.dtype,
                rowwise_data=new_rowwise_data,
                rowwise_scale_inv=grad._rowwise_scale_inv,
                columnwise_data=new_columnwise_data,
                columnwise_scale_inv=grad._columnwise_scale_inv,
                amax_rowwise=grad._amax_rowwise,
                amax_columnwise=grad._amax_columnwise,
                quantizer=grad._quantizer,
                fp4_dtype=grad._fp4_dtype,
                scale_dtype=grad._scale_dtype,
                requires_grad=grad.requires_grad,
                with_gemm_swizzled_scales=grad._with_gemm_swizzled_scales,
                device=grad.device,
                row_scaled_nvfp4=grad._row_scaled_nvfp4,
                nvfp4_use_4over6=grad._nvfp4_use_4over6,
                nvfp4_e4m3_max=grad._nvfp4_e4m3_max,
                nvfp4_columnwise_e4m3_max=grad._nvfp4_columnwise_e4m3_max,
            )
            return dgrad, None
        return grad.view(ctx.shape), None
