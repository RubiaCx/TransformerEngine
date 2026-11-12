# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Pure-PyTorch reference of MS-EDEN (Quartet II, arXiv:2601.22813, Algorithm 1).

Test oracle only. It mirrors the operation order of the CUDA kernels in
``transformer_engine/common/hadamard_transform/ms_eden_hadamard_transform_cast_fusion.cu``
(post hoc range alignment variant, Section 7 of the paper):

1. Tiled randomized Hadamard transform along the last dimension of ``x^T``:
   16-point Hadamard with folded signs (FP32 accumulation, scale 1/4), then for
   ``rht_dim == 128`` an 8-point butterfly across the eight 16-element blocks and a
   multiplication by ``fl32(1 / sqrt(8))``.
2. Per 16-element group: ``d = E8M3(group_amax * fl32(1/6))`` (3 mantissa bits,
   round-to-nearest-even, FP32 exponent range), FP4 values ``RTN(x * min(1/d, FLT_MAX))``.
3. Per rotation chunk: ``S = <x, x> / <x, x_rtn>``.
4. Per group: ``t = (S * d) * encode_scale`` with
   ``encode_scale = scale_max * 6 / (amax * headroom)``; the stored E4M3 scale is the
   stochastic rounding of ``t``; the stored amax is ``amax * headroom``.

This module only depends on ``torch`` so that the reference itself can be checked on CPU.
"""

import math
from typing import Dict, Sequence, Tuple

import torch

FP4_MAGNITUDES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
E4M3_MAX = 448.0
FLT_MAX = torch.finfo(torch.float32).max


def hadamard_matrix(dim: int, dtype=torch.float64) -> torch.Tensor:
    """Unnormalized Sylvester Hadamard matrix: H[r, c] = (-1)^popcount(r & c)."""
    idx = torch.arange(dim)
    anded = idx.unsqueeze(1) & idx.unsqueeze(0)
    popcount = torch.zeros_like(anded)
    for bit in range(dim.bit_length()):
        popcount += (anded >> bit) & 1
    return (1 - 2 * (popcount & 1)).to(dtype)


def sign_vector(sign_bits: Sequence[int], device=None, dtype=torch.float64) -> torch.Tensor:
    """Convert sign bits (1 means -1) to a +-1 vector."""
    return torch.tensor([-1.0 if b else 1.0 for b in sign_bits], dtype=dtype, device=device)


def rht_rotate(xt: torch.Tensor, sign_bits: Sequence[int]) -> torch.Tensor:
    """Tiled RHT along the last dimension of ``xt`` in the kernels' operation order.

    ``xt`` has shape [C, R] with R divisible by ``len(sign_bits)``. Returns FP32.
    """
    rht_dim = len(sign_bits)
    assert rht_dim in (16, 128)
    num_rows, length = xt.shape
    assert length % rht_dim == 0
    signs = sign_vector(sign_bits, device=xt.device).repeat(length // rht_dim)
    # Tensor-core stage: FP32 accumulation of exact BF16 products with +-0.25.
    h16 = hadamard_matrix(16).to(xt.device)
    stage1 = (xt.to(torch.float64) * signs).reshape(num_rows, length // 16, 16) @ h16
    out = (stage1 * 0.25).to(torch.float32)
    if rht_dim == 16:
        return out.reshape(num_rows, length)
    # Register stage: in-place 8-point fast Walsh-Hadamard butterfly in FP32.
    out = out.reshape(num_rows, length // 128, 8, 16).clone()
    span = 1
    while span < 8:
        for base in range(0, 8, 2 * span):
            for j in range(base, base + span):
                low = out[:, :, j, :].clone()
                high = out[:, :, j + span, :].clone()
                out[:, :, j, :] = low + high
                out[:, :, j + span, :] = low - high
        span *= 2
    out = out * torch.tensor(0.35355339059327373, dtype=torch.float32, device=xt.device)
    return out.reshape(num_rows, length)


def rht_rotate_exact(xt: torch.Tensor, sign_bits: Sequence[int]) -> torch.Tensor:
    """Tiled RHT along the last dimension in float64: (x * signs) @ H / sqrt(dim)."""
    rht_dim = len(sign_bits)
    num_rows, length = xt.shape
    signs = sign_vector(sign_bits, device=xt.device).repeat(length // rht_dim)
    had = hadamard_matrix(rht_dim).to(xt.device) / math.sqrt(rht_dim)
    out = (xt.to(torch.float64) * signs).reshape(num_rows, length // rht_dim, rht_dim) @ had
    return out.reshape(num_rows, length)


def rht_rotate_inverse(yt: torch.Tensor, sign_bits: Sequence[int]) -> torch.Tensor:
    """Inverse of :func:`rht_rotate_exact` (float64)."""
    rht_dim = len(sign_bits)
    num_rows, length = yt.shape
    signs = sign_vector(sign_bits, device=yt.device).repeat(length // rht_dim)
    had = hadamard_matrix(rht_dim).to(yt.device) / math.sqrt(rht_dim)
    out = yt.to(torch.float64).reshape(num_rows, length // rht_dim, rht_dim) @ had.t()
    return out.reshape(num_rows, length) * signs


def round_to_e8m3(x: torch.Tensor) -> torch.Tensor:
    """Round non-negative FP32 values to 3 mantissa bits (round-to-nearest-even)."""
    bits = x.to(torch.float32).contiguous().view(torch.int32).to(torch.int64)
    lsb = (bits >> 20) & 1
    bits = (bits + 0x7FFFF + lsb) & 0xFFF00000
    return bits.to(torch.int32).view(torch.float32)


def fp4_rtn_abs(a: torch.Tensor) -> torch.Tensor:
    """Magnitude that round-to-nearest-even to E2M1 produces for ``a >= 0``."""
    step_inv = torch.where(a < 2.0, 2.0, torch.where(a < 4.0, 1.0, 0.5)).to(a.dtype)
    step = torch.where(a < 2.0, 0.5, torch.where(a < 4.0, 1.0, 2.0)).to(a.dtype)
    return torch.clamp(torch.round(a * step_inv) * step, max=6.0)


def fp4_codes_to_values(codes: torch.Tensor) -> torch.Tensor:
    """Decode E2M1 nibbles (sign in bit 3) to FP32 values."""
    lut = torch.tensor(FP4_MAGNITUDES, dtype=torch.float32, device=codes.device)
    codes = codes.to(torch.int64)
    mags = lut[codes & 0x7]
    return torch.where((codes & 0x8) != 0, -mags, mags)


def unpack_fp4(x: torch.Tensor) -> torch.Tensor:
    """Unpack FP4 bytes ([rows, cols / 2]) into nibbles ([rows, cols]), low nibble first."""
    repeated = x.repeat_interleave(2, dim=1)
    repeated[:, 0::2] &= 0x0F
    repeated[:, 1::2] >>= 4
    return repeated


def e4m3_decode(codes: torch.Tensor) -> torch.Tensor:
    """Decode non-negative E4M3 bytes to FP32."""
    codes = codes.to(torch.int64)
    exponent = (codes >> 3) & 0xF
    mantissa = codes & 0x7
    normal = torch.ldexp((8 + mantissa).to(torch.float32), (exponent - 7 - 3).to(torch.int32))
    subnormal = mantissa.to(torch.float32) * 2.0**-9
    return torch.where(exponent == 0, subnormal, normal)


def e4m3_neighbours(t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Largest E4M3 value <= t and smallest E4M3 value >= t (saturating at 448)."""
    grid = e4m3_decode(torch.arange(0, 0x7F, device=t.device))  # 0 ... 448, ascending
    t = torch.clamp(t.to(torch.float32), min=0.0, max=E4M3_MAX)
    hi_idx = torch.searchsorted(grid, t.contiguous())
    hi = grid[torch.clamp(hi_idx, max=grid.numel() - 1)]
    lo = torch.where(hi == t, hi, grid[torch.clamp(hi_idx - 1, min=0)])
    return lo, hi


def e4m3_stochastic_round(t: torch.Tensor, uniform: torch.Tensor) -> torch.Tensor:
    """Stochastically round ``t >= 0`` to E4M3 values using uniforms in [0, 1)."""
    lo, hi = e4m3_neighbours(t)
    t = torch.clamp(t.to(torch.float32), min=0.0, max=E4M3_MAX)
    frac = torch.where(hi > lo, (t - lo) / (hi - lo), torch.zeros_like(t))
    return torch.where(uniform < frac, hi, lo)


def e4m3_round_to_nearest(t: torch.Tensor) -> torch.Tensor:
    """Round ``t >= 0`` to the nearest E4M3 value (ties to the even code, saturating)."""
    grid = e4m3_decode(torch.arange(0, 0x7F, device=t.device))
    t = torch.clamp(t.to(torch.float32), min=0.0, max=E4M3_MAX)
    hi_idx = torch.clamp(torch.searchsorted(grid, t.contiguous()), max=grid.numel() - 1)
    lo_idx = torch.clamp(hi_idx - 1, min=0)
    lo, hi = grid[lo_idx], grid[hi_idx]
    below, above = t - lo, hi - t
    pick_hi = (above < below) | ((above == below) & (hi_idx % 2 == 0))
    return torch.where(pick_hi, hi, lo)


def aligned_reference(
    x: torch.Tensor, sign_bits: Sequence[int], scale_max: float = E4M3_MAX
) -> Dict[str, torch.Tensor]:
    """Range-aligned round-to-nearest NVFP4 quantization of the rotated ``x^T``.

    Mirrors the two-pass (amax, then aligned cast) mode of the tiled-RHT kernel for a
    2D input ``x`` of shape [R, C]; outputs use the transposed [C, R] layout.
    ``scaled`` are the values handed to the FP4 rounding (for stochastic-rounding
    adjacency checks).
    """
    xt = x.to(torch.float32).t().contiguous()
    num_rows, length = xt.shape
    rotated = rht_rotate(xt, sign_bits)
    groups = rotated.reshape(num_rows, length // 16, 16)
    amax = rotated.abs().max().to(torch.float32)
    f32 = lambda v: torch.tensor(v, dtype=torch.float32, device=x.device)
    encode = torch.clamp(f32(scale_max * 6.0) / amax, max=FLT_MAX)
    if amax == 0 or encode == 0:
        encode = f32(1.0)
    decode = f32(1.0) / encode
    multiplier = encode * f32(1.0 / 6.0)
    scales = e4m3_round_to_nearest(groups.abs().amax(dim=-1) * multiplier)
    block_inv = torch.clamp(f32(1.0) / (scales * decode), max=FLT_MAX)
    scaled = groups * block_inv.unsqueeze(-1)
    fp4_abs = fp4_rtn_abs(scaled.abs())
    return {
        "rotated": rotated,
        "scaled": scaled.reshape(num_rows, length),
        "fp4_values": torch.where(scaled < 0, -fp4_abs, fp4_abs).reshape(num_rows, length),
        "scales": scales,
        "amax": amax,
    }


def ms_eden_reference(
    x: torch.Tensor,
    sign_bits: Sequence[int],
    scale_max: float = E4M3_MAX,
    grid_max: float = 6.0,
) -> Dict[str, torch.Tensor]:
    """Deterministic part of columnwise MS-EDEN for a 2D input ``x`` of shape [R, C].

    ``grid_max`` is the grid parameter ``s``: pseudo-scale ``d = E8M3(group_amax / s)``
    and stored amax ``= amax * (448 / 256) * (6 / s)``.

    Returns a dict with (all for the transposed [C, R] layout):

    * ``rotated``: rotated FP32 values.
    * ``fp4_values``: round-to-nearest E2M1 values (FP32, signed).
    * ``pseudo_scales``: E8M3 scales ``d`` per group, shape [C, R / 16].
    * ``eden_factors``: ``S`` per group (constant within a rotation chunk).
    * ``scale_targets``: ``t`` per group, the value stochastically rounded to E4M3.
    * ``amax_raw`` / ``amax_stored``: post-rotation amax and the amax stored in the tensor.
    """
    rht_dim = len(sign_bits)
    xt = x.to(torch.float32).t().contiguous()
    num_rows, length = xt.shape
    assert length % 128 == 0
    rotated = rht_rotate(xt, sign_bits)
    groups = rotated.reshape(num_rows, length // 16, 16)
    group_amax = groups.abs().amax(dim=-1)
    f32 = lambda v: torch.tensor(v, dtype=torch.float32, device=x.device)
    grid_max_inv = f32(1.0) / f32(grid_max)
    headroom = f32(448.0 / 256.0) * (f32(6.0) / f32(grid_max))
    pseudo = round_to_e8m3(group_amax * grid_max_inv)
    pseudo_inv = torch.clamp(1.0 / pseudo, max=FLT_MAX)
    scaled = groups * pseudo_inv.unsqueeze(-1)
    fp4_abs = fp4_rtn_abs(scaled.abs())
    fp4_values = torch.where(scaled < 0, -fp4_abs, fp4_abs)
    sum_xx = (groups * groups).sum(dim=-1)
    sum_xq = (groups.abs() * fp4_abs).sum(dim=-1) * pseudo
    groups_per_chunk = rht_dim // 16
    num = sum_xx.reshape(num_rows, -1, groups_per_chunk).sum(dim=-1, keepdim=True)
    den = sum_xq.reshape(num_rows, -1, groups_per_chunk).sum(dim=-1, keepdim=True)
    factor = torch.where(den > 0, num / den, torch.ones_like(num))
    factor = torch.where(torch.isfinite(factor), factor, torch.ones_like(factor))
    factor = factor.expand(-1, -1, groups_per_chunk).reshape(num_rows, length // 16)
    corrected = pseudo * factor
    amax_raw = rotated.abs().max().to(torch.float32)
    amax_stored = amax_raw * headroom
    encode_scale = torch.tensor(scale_max * 6.0, dtype=torch.float32, device=x.device) / amax_stored
    encode_scale = torch.clamp(encode_scale, max=FLT_MAX)
    if amax_stored == 0 or encode_scale == 0:
        encode_scale = torch.ones_like(encode_scale)
    return {
        "rotated": rotated,
        "fp4_values": fp4_values.reshape(num_rows, length),
        "pseudo_scales": pseudo,
        "eden_factors": factor,
        "scale_targets": corrected * encode_scale,
        "amax_raw": amax_raw,
        "amax_stored": amax_stored,
    }


def dequantize_columnwise(
    fp4_values: torch.Tensor,
    scales: torch.Tensor,
    amax_stored: torch.Tensor,
    scale_max: float = E4M3_MAX,
) -> torch.Tensor:
    """Standard NVFP4 decode: value * scale * amax / (6 * scale_max). Returns float64 [C, R]."""
    num_rows, length = fp4_values.shape
    decode = amax_stored.to(torch.float64) / (6.0 * scale_max)
    block = scales.to(torch.float64).reshape(num_rows, length // 16, 1) * decode
    return (fp4_values.to(torch.float64).reshape(num_rows, length // 16, 16) * block).reshape(
        num_rows, length
    )


def ms_eden_quantize_reference(
    x: torch.Tensor,
    sign_bits: Sequence[int],
    generator: torch.Generator = None,
    grid_max: float = 6.0,
) -> torch.Tensor:
    """Full MS-EDEN (with stochastic scale rounding). Returns the dequantized rotated
    tensor in the transposed [C, R] layout (float64)."""
    ref = ms_eden_reference(x, sign_bits, grid_max=grid_max)
    uniform = torch.rand(
        ref["scale_targets"].shape, generator=generator, device=ref["scale_targets"].device
    )
    scales = e4m3_stochastic_round(ref["scale_targets"], uniform)
    return dequantize_columnwise(ref["fp4_values"], scales, ref["amax_stored"])
