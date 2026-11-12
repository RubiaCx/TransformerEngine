# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""NVFP4 4over6 in the pre-training configuration (RHT + stochastic rounding enabled).

Semantics under test:

* 4over6 applies to the rowwise copy of forward tensors: the un-rotated,
  round-to-nearest operands of the fprop GEMM (input X and weight W).
* The columnwise RHT copy of X (wgrad operand) and all gradient tensors
  (RHT + stochastic rounding) are standard NVFP4, identical to the recipe
  without 4over6.
* The columnwise copy of W (dgrad operand) is standard NVFP4 unless
  ``nvfp4_4over6_columnwise`` selects weights.
* The rowwise 4over6 copy and the standard columnwise copy carry separate
  global scale bounds (256 / 448 by default).

NOTE: This file depends on test_nvfp4_quantize_exact.py (standalone 4over6
kernel versus NVFP4QuantizerRef) and test_nvfp4_rht_quantize_exact.py
(standard RHT quantization versus NVFP4QuantizerRef).
"""

import os
from contextlib import contextmanager

import pytest
import torch

import transformer_engine.pytorch as te
import transformer_engine_torch as tex
from transformer_engine.common import recipe as te_recipe
from transformer_engine.common.recipe import NVFP4BlockScaling
from transformer_engine.pytorch import NVFP4Quantizer
from transformer_engine.pytorch.custom_recipes import reference_utils
from transformer_engine.pytorch.custom_recipes.reference_nvfp4 import NVFP4QuantizerRef
from transformer_engine.pytorch.quantization import NVFP4BlockScalingRecipeState

from nvfp4_utils import get_nvfp4_scale_shape_no_padding

recipe_available, reason_for_no_recipe = te.is_nvfp4_available(return_reason=True)

DEVICE = "cuda"
FP4_DTYPE = te.DType.kFloat4E2M1

# FP4 E2M1 magnitudes indexed by the low three code bits; bit 3 is the sign.
_FP4_MAGNITUDES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)

# Shapes eligible for the fused RHT cast-fusion kernel (rows % 64 == 0, cols % 128 == 0).
FUSED_SHAPES = [
    (64, 128),
    (128, 128),
    (256, 1024),
    (1024, 256),
    (2048, 2048),
    (8192, 5120),
]
# Shapes that take the unfused RHT path (not 64/128 aligned; still multiples of 16).
UNFUSED_SHAPES = [
    (256, 272),
    (304, 304),
    (320, 144),
    (96, 128),
    (32, 128),
]


@contextmanager
def _env(name: str, value):
    old_value = os.environ.get(name)
    if value is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = value
    try:
        yield
    finally:
        if old_value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = old_value


def _seed(seed: int = 0) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def _unpack_fp4(x: torch.Tensor) -> torch.Tensor:
    """Unpack two FP4 codes per byte (low nibble first) into one code per byte."""
    x = x.view(dtype=torch.uint8)
    repeated = x.repeat_interleave(2, dim=1)
    repeated[:, 0::2] &= 0x0F
    repeated[:, 1::2] >>= 4
    return repeated


def _fp4_codes_to_values(codes: torch.Tensor) -> torch.Tensor:
    lut = torch.tensor(_FP4_MAGNITUDES, dtype=torch.float32, device=codes.device)
    values = lut[(codes & 0x7).long()]
    return torch.where((codes & 0x8) != 0, -values, values)


def _e4m3_bytes_to_float(scale: torch.Tensor) -> torch.Tensor:
    return scale.contiguous().view(dtype=torch.float8_e4m3fn).to(torch.float32)


def _quantize(quantizer: NVFP4Quantizer, x: torch.Tensor, use_cpp_allocator: bool = True):
    if use_cpp_allocator:
        return quantizer(x)
    out = quantizer.make_empty(x.shape, dtype=x.dtype, device=x.device, requires_grad=False)
    return quantizer.update_quantized(x, out)


def _valid_scale(scale: torch.Tensor, shape, columnwise: bool) -> torch.Tensor:
    valid_shape = get_nvfp4_scale_shape_no_padding(shape, columnwise)
    return scale[: valid_shape[0], : valid_shape[1]]


def _assert_rowwise_matches_reference(
    sut,
    x: torch.Tensor,
    *,
    tile_shape,
    e4m3_max: int,
    err_mode: str,
    err_use_fast_math: bool = False,
) -> None:
    """Bit-exact comparison of the rowwise copy against the PyTorch 4over6 reference."""
    ref_quantizer = NVFP4QuantizerRef(
        dtype=reference_utils.Fp4Formats.E2M1,
        rowwise=True,
        columnwise=False,
        pow_2_scales=False,
        eps=0.0,
        quant_tile_shape=tile_shape,
        nvfp4_use_4over6=True,
        nvfp4_e4m3_max=e4m3_max,
        nvfp4_4over6_err_mode=err_mode,
        nvfp4_4over6_err_use_fast_math=err_use_fast_math,
        with_rht=False,
    )
    ref = ref_quantizer.quantize(x)
    qx = _unpack_fp4(sut._rowwise_data)
    qx_ref = _unpack_fp4(ref.data)
    torch.testing.assert_close(qx, qx_ref, atol=0.0, rtol=0.0)
    sx_ref = ref.scale.view(dtype=torch.uint8)
    sx = sut._rowwise_scale_inv[: sx_ref.shape[0], : sx_ref.shape[1]]
    torch.testing.assert_close(sx, sx_ref, atol=0.0, rtol=0.0)
    torch.testing.assert_close(sut._amax_rowwise, ref.global_amax_row, atol=0.0, rtol=0.0)


def _assert_columnwise_identical(sut, baseline, shape) -> None:
    """Bit-exact comparison of the columnwise copies of two NVFP4 tensors."""
    torch.testing.assert_close(
        sut._columnwise_data.view(dtype=torch.uint8),
        baseline._columnwise_data.view(dtype=torch.uint8),
        atol=0.0,
        rtol=0.0,
    )
    torch.testing.assert_close(
        _valid_scale(sut._columnwise_scale_inv, shape, True),
        _valid_scale(baseline._columnwise_scale_inv, shape, True),
        atol=0.0,
        rtol=0.0,
    )
    torch.testing.assert_close(sut._amax_columnwise, baseline._amax_columnwise, atol=0.0, rtol=0.0)


# ---------------------------------------------------------------------------
# Independent pure-PyTorch 4over6 reference (1x16 blocks, input-domain error).
# ---------------------------------------------------------------------------


def _round_to_fp4(v: torch.Tensor) -> torch.Tensor:
    """Round-to-nearest-even onto the FP4 E2M1 grid with saturation. Returns values."""
    a = v.abs().clamp(max=6.0)
    # Midpoints between consecutive magnitudes. A tie goes to the even code, i.e. up
    # when the lower code is odd.
    midpoints = (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)
    code = torch.zeros_like(a, dtype=torch.long)
    for lower_code, midpoint in enumerate(midpoints):
        if lower_code % 2 == 1:
            code += (a >= midpoint).long()
        else:
            code += (a > midpoint).long()
    lut = torch.tensor(_FP4_MAGNITUDES, dtype=torch.float32, device=v.device)
    return torch.copysign(lut[code], v)


def _reference_4over6_1d(x: torch.Tensor, e4m3_max: int, err_mode: str):
    """Both 4over6 candidates for every 1x16 block of ``x`` (2D, cols % 16 == 0).

    Returns (map-to-4 FP4 values, map-to-6 FP4 values, scale4, scale6, err4, err6,
    global amax) with per-block errors in float64. Scales are computed with the same
    float32 operation order as the kernels.
    """
    rows, cols = x.shape
    xf = x.to(torch.float32).reshape(rows, cols // 16, 16)
    fp4_max = torch.tensor(6.0, dtype=torch.float32, device=x.device)
    flt_max = torch.tensor(torch.finfo(torch.float32).max, device=x.device)
    scale_max = torch.tensor(float(e4m3_max), dtype=torch.float32, device=x.device)
    e4m3_type_max = torch.tensor(448.0, dtype=torch.float32, device=x.device)

    global_amax = xf.abs().max()
    s_enc = torch.minimum(scale_max * fp4_max / global_amax, flt_max)
    if global_amax == 0 or s_enc == 0:
        s_enc = torch.ones_like(s_enc)
    s_dec = 1.0 / s_enc

    block_amax = xf.abs().amax(dim=-1, keepdim=True)
    base = block_amax / fp4_max * s_enc

    def candidate(scale_f32: torch.Tensor):
        scale = torch.minimum(scale_f32, e4m3_type_max).to(torch.float8_e4m3fn)
        scale = scale.to(torch.float32)
        inv = torch.minimum(1.0 / (scale * s_dec), flt_max)
        q = _round_to_fp4(xf * inv)
        dequant = (q * scale) * global_amax / (fp4_max * scale_max)
        diff = dequant.to(torch.float64) - xf.to(torch.float64)
        err = diff.abs().sum(dim=-1) if err_mode == "MAE" else (diff * diff).sum(dim=-1)
        return q, scale.squeeze(-1), err

    q4, scale4, err4 = candidate(base * 1.5)
    q6, scale6, err6 = candidate(base)
    return q4, q6, scale4, scale6, err4, err6, global_amax


def _assert_rowwise_matches_independent_reference(
    sut, x: torch.Tensor, e4m3_max: int, err_mode: str
) -> None:
    """Check the rowwise copy block-by-block against an independent 4over6 reference.

    Every block must equal one of the two candidates (FP4 values and block scale), and must be the lower-error candidate unless the two errors are within a
    relative 1e-4 (the kernels accumulate the error in FP32).
    """
    rows, cols = x.shape
    q4, q6, scale4, scale6, err4, err6, global_amax = _reference_4over6_1d(x, e4m3_max, err_mode)
    torch.testing.assert_close(sut._amax_rowwise.view(()), global_amax, atol=0.0, rtol=0.0)

    codes = _unpack_fp4(sut._rowwise_data).reshape(rows, cols // 16, 16)
    scale = _e4m3_bytes_to_float(sut._rowwise_scale_inv[:rows, : cols // 16])
    values = _fp4_codes_to_values(codes)

    is_map4 = (scale == scale4) & (values == q4).all(dim=-1)
    is_map6 = (scale == scale6) & (values == q6).all(dim=-1)
    assert bool((is_map4 | is_map6).all()), "A block matches neither 4over6 candidate"

    near_tie = (err4 - err6).abs() <= 1e-4 * torch.maximum(err4, err6)
    expect_map4 = err4 < err6
    wrong = ~near_tie & ((expect_map4 & ~is_map4) | (~expect_map4 & ~is_map6))
    assert not bool(wrong.any()), f"{int(wrong.sum())} blocks did not pick the lower error"
    # 4over6 must actually be exercised.
    assert bool((is_map4 & ~is_map6).any()), "No block selected the map-to-4 candidate"


# ---------------------------------------------------------------------------
# (a) Quantizer-level: rowwise 4over6 + columnwise RHT standard NVFP4.
# ---------------------------------------------------------------------------


def _check_rht_rowwise_4over6(
    M: int,
    N: int,
    *,
    e4m3_max: int,
    err_mode: str,
    err_use_fast_math: bool,
    return_rowwise: bool,
    return_transpose: bool,
    use_cpp_allocator: bool,
) -> None:
    _seed()
    x = torch.randn((M, N), dtype=torch.bfloat16, device=DEVICE)

    common = dict(
        fp4_dtype=FP4_DTYPE,
        rowwise=return_rowwise,
        columnwise=return_transpose,
        with_amax_reduction=False,
        amax_reduction_group=None,
        with_rht=True,
        with_post_rht_amax=True,
    )
    sut_quantizer = NVFP4Quantizer(
        nvfp4_use_4over6=True,
        nvfp4_e4m3_max=e4m3_max,
        nvfp4_4over6_err_mode=err_mode,
        **common,
    )
    baseline_quantizer = NVFP4Quantizer(**common)

    with _env("NVTE_NVFP4_4OVER6_ERR_USE_FAST_MATH", "1" if err_use_fast_math else "0"):
        sut = _quantize(sut_quantizer, x, use_cpp_allocator)
    baseline = _quantize(baseline_quantizer, x, use_cpp_allocator)

    # Metadata: the 4over6 bound applies to the rowwise copy only.
    assert sut._nvfp4_use_4over6
    assert sut._nvfp4_e4m3_max == e4m3_max
    assert sut._nvfp4_columnwise_e4m3_max == 448
    assert baseline._nvfp4_e4m3_max == 0
    assert baseline._nvfp4_columnwise_e4m3_max == 0

    if return_rowwise:
        _assert_rowwise_matches_reference(
            sut,
            x,
            tile_shape=(1, 16),
            e4m3_max=e4m3_max,
            err_mode=err_mode,
            err_use_fast_math=err_use_fast_math,
        )
        if not err_use_fast_math:
            _assert_rowwise_matches_independent_reference(sut, x, e4m3_max, err_mode)
        # The rowwise copy must differ from standard NVFP4.
        assert not torch.equal(
            sut._rowwise_data.view(dtype=torch.uint8),
            baseline._rowwise_data.view(dtype=torch.uint8),
        )
    else:
        assert sut._rowwise_data is None

    if return_transpose:
        _assert_columnwise_identical(sut, baseline, x.shape)
    else:
        assert sut._columnwise_data is None


@pytest.mark.skipif(not recipe_available, reason=reason_for_no_recipe)
@pytest.mark.parametrize("M, N", FUSED_SHAPES + UNFUSED_SHAPES)
@pytest.mark.parametrize("err_mode", ["MAE", "MSE"])
@pytest.mark.parametrize("e4m3_max", [256, 448], ids=["e4m3_256", "e4m3_448"])
@pytest.mark.parametrize("err_use_fast_math", [False, True], ids=["err_exact", "err_fast"])
@pytest.mark.parametrize("quantize_mode", ["both_directions", "rowwise_only", "columnwise_only"])
@pytest.mark.parametrize(
    "use_cpp_allocator", [True, False], ids=["cpp_allocator", "python_allocator"]
)
def test_rht_rowwise_4over6_versus_reference(
    M, N, err_mode, e4m3_max, err_use_fast_math, quantize_mode, use_cpp_allocator
):
    _check_rht_rowwise_4over6(
        M,
        N,
        e4m3_max=e4m3_max,
        err_mode=err_mode,
        err_use_fast_math=err_use_fast_math,
        return_rowwise=quantize_mode != "columnwise_only",
        return_transpose=quantize_mode != "rowwise_only",
        use_cpp_allocator=use_cpp_allocator,
    )


@pytest.mark.skipif(not recipe_available, reason=reason_for_no_recipe)
@pytest.mark.parametrize("M, N", FUSED_SHAPES[:4])
@pytest.mark.parametrize("err_mode", ["MAE", "MSE"])
def test_rht_rowwise_4over6_fused_matches_unfused(M, N, err_mode):
    """The fused kernel and the unfused path must produce identical tensors."""
    _seed()
    x = torch.randn((M, N), dtype=torch.bfloat16, device=DEVICE)
    quantizer = NVFP4Quantizer(
        fp4_dtype=FP4_DTYPE,
        with_rht=True,
        with_post_rht_amax=True,
        nvfp4_use_4over6=True,
        nvfp4_e4m3_max=256,
        nvfp4_4over6_err_mode=err_mode,
    )
    fused = quantizer(x)
    with _env("NVTE_NVFP4_DISABLE_RHT_CAST_FUSION", "1"):
        unfused = quantizer(x)
    torch.testing.assert_close(
        fused._rowwise_data.view(dtype=torch.uint8),
        unfused._rowwise_data.view(dtype=torch.uint8),
        atol=0.0,
        rtol=0.0,
    )
    torch.testing.assert_close(
        _valid_scale(fused._rowwise_scale_inv, x.shape, False),
        _valid_scale(unfused._rowwise_scale_inv, x.shape, False),
        atol=0.0,
        rtol=0.0,
    )
    _assert_columnwise_identical(fused, unfused, x.shape)


@pytest.mark.skipif(not recipe_available, reason=reason_for_no_recipe)
@pytest.mark.parametrize("M, N", [(256, 1024), (1024, 256)])
@pytest.mark.parametrize("err_mode", ["MAE", "MSE"])
def test_rht_rowwise_4over6_swizzled_scales(M, N, err_mode):
    """With optimize_for_gemm, the fused kernel emits GEMM-swizzled scales directly."""
    if torch.cuda.get_device_capability() in ((12, 0), (12, 1)):
        pytest.skip("SM120/121 RHT quantization emits compact scales")
    _seed()
    x = torch.randn((M, N), dtype=torch.bfloat16, device=DEVICE)
    kwargs = dict(
        fp4_dtype=FP4_DTYPE,
        with_rht=True,
        with_post_rht_amax=True,
        nvfp4_use_4over6=True,
        nvfp4_e4m3_max=256,
        nvfp4_4over6_err_mode=err_mode,
    )
    compact_quantizer = NVFP4Quantizer(**kwargs)
    swizzled_quantizer = NVFP4Quantizer(**kwargs)
    swizzled_quantizer.optimize_for_gemm = True
    compact = compact_quantizer(x)
    swizzled = swizzled_quantizer(x)
    assert not compact._with_gemm_swizzled_scales
    assert swizzled._with_gemm_swizzled_scales
    torch.testing.assert_close(
        compact._rowwise_data.view(dtype=torch.uint8),
        swizzled._rowwise_data.view(dtype=torch.uint8),
        atol=0.0,
        rtol=0.0,
    )
    torch.testing.assert_close(
        compact._columnwise_data.view(dtype=torch.uint8),
        swizzled._columnwise_data.view(dtype=torch.uint8),
        atol=0.0,
        rtol=0.0,
    )
    # Dequantization of the rowwise copy must not depend on the scale layout.
    torch.testing.assert_close(
        compact.dequantize(dtype=torch.float32),
        swizzled.dequantize(dtype=torch.float32),
        atol=0.0,
        rtol=0.0,
    )


@pytest.mark.skipif(not recipe_available, reason=reason_for_no_recipe)
@pytest.mark.parametrize("M, N", [(256, 1024), (256, 272)])
@pytest.mark.parametrize("err_mode", ["MAE", "MSE"])
@pytest.mark.parametrize("e4m3_max", [256, 448], ids=["e4m3_256", "e4m3_448"])
def test_rht_rowwise_4over6_dequantize(M, N, err_mode, e4m3_max):
    """Dequantization uses the rowwise copy and the rowwise scale bound."""
    _seed()
    x = torch.randn((M, N), dtype=torch.bfloat16, device=DEVICE)
    quantizer = NVFP4Quantizer(
        fp4_dtype=FP4_DTYPE,
        with_rht=True,
        with_post_rht_amax=True,
        nvfp4_use_4over6=True,
        nvfp4_e4m3_max=e4m3_max,
        nvfp4_4over6_err_mode=err_mode,
    )
    sut = quantizer(x)
    dequant = sut.dequantize(dtype=torch.float32)

    codes = _unpack_fp4(sut._rowwise_data).reshape(M, N // 16, 16)
    scale = _e4m3_bytes_to_float(sut._rowwise_scale_inv[:M, : N // 16])
    values = _fp4_codes_to_values(codes)
    amax = sut._amax_rowwise.view(())
    expected = ((values * scale.unsqueeze(-1)) * amax / (6.0 * float(e4m3_max))).reshape(M, N)
    torch.testing.assert_close(dequant, expected, rtol=1e-5, atol=0.0)

    # 4over6 should not be worse than standard NVFP4 on the selected error metric.
    baseline = NVFP4Quantizer(fp4_dtype=FP4_DTYPE, with_rht=True, with_post_rht_amax=True)(x)
    xf = x.to(torch.float64)

    def total_error(t):
        diff = t.dequantize(dtype=torch.float32).to(torch.float64) - xf
        return diff.abs().sum() if err_mode == "MAE" else (diff * diff).sum()

    assert total_error(sut) <= total_error(baseline) * (1.0 + 1e-3)


# ---------------------------------------------------------------------------
# (a) Split quantization (GroupedLinear host-aware path).
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not recipe_available, reason=reason_for_no_recipe)
@pytest.mark.parametrize(
    "split_sections, N",
    [
        # All splits 128-aligned: fused grouped kernel on SM100/SM110.
        ([256, 256, 256, 256], 512),
        ([128, 384, 0, 512], 256),
        ([0, 256, 512, 256], 1024),
        # Not all 128-aligned: rowwise 4over6 per split + grouped columnwise RHT kernel.
        ([64, 192, 320, 448], 256),
        ([64, 0, 192, 768], 512),
    ],
    ids=["even", "zero_middle", "zero_front", "unaligned", "unaligned_zero"],
)
@pytest.mark.parametrize("err_mode", ["MAE", "MSE"])
@pytest.mark.parametrize("e4m3_max", [256, 448], ids=["e4m3_256", "e4m3_448"])
@pytest.mark.parametrize("quantize_mode", ["both_directions", "rowwise_only", "columnwise_only"])
def test_split_quantize_rht_rowwise_4over6(split_sections, N, err_mode, e4m3_max, quantize_mode):
    return_rowwise = quantize_mode != "columnwise_only"
    return_transpose = quantize_mode != "rowwise_only"
    M = sum(split_sections)
    _seed()
    x = torch.randn((M, N), dtype=torch.bfloat16, device=DEVICE)
    chunks = torch.split(x, split_sections)

    def make_quantizer(use_4over6: bool) -> NVFP4Quantizer:
        return NVFP4Quantizer(
            fp4_dtype=FP4_DTYPE,
            rowwise=return_rowwise,
            columnwise=return_transpose,
            with_rht=True,
            with_post_rht_amax=True,
            nvfp4_use_4over6=use_4over6,
            nvfp4_e4m3_max=e4m3_max if use_4over6 else 0,
            nvfp4_4over6_err_mode=err_mode,
        )

    sut_outputs = tex.split_quantize(
        x, split_sections, [make_quantizer(True) for _ in split_sections]
    )
    baseline_outputs = tex.split_quantize(
        x, split_sections, [make_quantizer(False) for _ in split_sections]
    )

    for chunk, section, sut, baseline in zip(chunks, split_sections, sut_outputs, baseline_outputs):
        assert sut._nvfp4_use_4over6
        assert sut._nvfp4_e4m3_max == e4m3_max
        assert sut._nvfp4_columnwise_e4m3_max == 448
        if section == 0:
            continue
        if return_rowwise:
            _assert_rowwise_matches_reference(
                sut, chunk, tile_shape=(1, 16), e4m3_max=e4m3_max, err_mode=err_mode
            )
        if return_transpose:
            _assert_columnwise_identical(sut, baseline, chunk.shape)


# ---------------------------------------------------------------------------
# (a) Weights: no RHT, 1D and 2D block scaling.
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not recipe_available, reason=reason_for_no_recipe)
@pytest.mark.parametrize("M, N", [(128, 128), (256, 1024), (1024, 256), (256, 272), (304, 304)])
@pytest.mark.parametrize("with_2d_quantization", [False, True], ids=["1d", "2d"])
@pytest.mark.parametrize("err_mode", ["MAE", "MSE"])
@pytest.mark.parametrize("e4m3_max", [256, 448], ids=["e4m3_256", "e4m3_448"])
@pytest.mark.parametrize(
    "columnwise_4over6", [False, True], ids=["colwise_standard", "colwise_4over6"]
)
def test_weight_4over6_rowwise_and_columnwise(
    M, N, with_2d_quantization, err_mode, e4m3_max, columnwise_4over6
):
    """W (rowwise, fprop) uses 4over6. W.T (columnwise, dgrad) is standard NVFP4 unless
    nvfp4_4over6_columnwise is set."""
    if with_2d_quantization and (M % 32 != 0 or N % 32 != 0):
        pytest.skip("NVFP4 2D 4over6 exact tests require the optimized BF16 kernel path")
    _seed()
    w = torch.randn((M, N), dtype=torch.bfloat16, device=DEVICE)
    tile_shape = (16, 16) if with_2d_quantization else (1, 16)

    sut_quantizer = NVFP4Quantizer(
        fp4_dtype=FP4_DTYPE,
        with_rht=False,
        with_2d_quantization=with_2d_quantization,
        nvfp4_use_4over6=True,
        nvfp4_e4m3_max=e4m3_max,
        nvfp4_4over6_err_mode=err_mode,
        nvfp4_4over6_columnwise=columnwise_4over6,
    )
    baseline_quantizer = NVFP4Quantizer(
        fp4_dtype=FP4_DTYPE, with_rht=False, with_2d_quantization=with_2d_quantization
    )
    sut = sut_quantizer(w)
    baseline = baseline_quantizer(w)

    assert sut._nvfp4_e4m3_max == e4m3_max
    _assert_rowwise_matches_reference(
        sut, w, tile_shape=tile_shape, e4m3_max=e4m3_max, err_mode=err_mode
    )

    if columnwise_4over6:
        # Both copies use 4over6 with the same bound.
        assert sut._nvfp4_columnwise_e4m3_max == 0
        ref = NVFP4QuantizerRef(
            dtype=reference_utils.Fp4Formats.E2M1,
            rowwise=True,
            columnwise=True,
            pow_2_scales=False,
            eps=0.0,
            quant_tile_shape=tile_shape,
            nvfp4_use_4over6=True,
            nvfp4_e4m3_max=e4m3_max,
            nvfp4_4over6_err_mode=err_mode,
        ).quantize(w)
        torch.testing.assert_close(
            _unpack_fp4(sut._columnwise_data), _unpack_fp4(ref.data_t), atol=0.0, rtol=0.0
        )
        sx_t_ref = ref.scale_t.view(dtype=torch.uint8)
        torch.testing.assert_close(
            sut._columnwise_scale_inv[: sx_t_ref.shape[0], : sx_t_ref.shape[1]],
            sx_t_ref,
            atol=0.0,
            rtol=0.0,
        )
    else:
        # The columnwise copy is unchanged by enabling 4over6 and keeps the 448 bound.
        assert sut._nvfp4_columnwise_e4m3_max == 448
        _assert_columnwise_identical(sut, baseline, w.shape)


# ---------------------------------------------------------------------------
# (b) 4over6 disabled: unchanged outputs and metadata.
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not recipe_available, reason=reason_for_no_recipe)
@pytest.mark.parametrize("M, N", [(128, 128), (256, 1024), (256, 272), (304, 304)])
@pytest.mark.parametrize("quantize_mode", ["both_directions", "rowwise_only", "columnwise_only"])
def test_4over6_disabled_rht_matches_existing_reference(M, N, quantize_mode):
    """Default RHT quantization still matches the existing reference bit-exactly.

    The full coverage is the existing exact suites (test_nvfp4_quantize_exact.py,
    test_nvfp4_rht_quantize_exact.py, test_nvfp4_group_quantize.py,
    test_nvfp4_sr_quantize.py, test_nvfp4_module_exact.py), which must pass unchanged.
    """
    from test_nvfp4_rht_quantize_exact import (
        check_quantization_nvfp4_versus_reference as check_rht_reference,
    )

    check_rht_reference(
        x_dtype=torch.bfloat16,
        M=M,
        N=N,
        contiguous=True,
        return_rowwise=quantize_mode != "columnwise_only",
        return_transpose=quantize_mode != "rowwise_only",
        use_cpp_allocator=True,
    )


@pytest.mark.skipif(not recipe_available, reason=reason_for_no_recipe)
def test_4over6_disabled_metadata_defaults():
    _seed()
    x = torch.randn((256, 1024), dtype=torch.bfloat16, device=DEVICE)
    for with_rht in (False, True):
        quantizer = NVFP4Quantizer(
            fp4_dtype=FP4_DTYPE, with_rht=with_rht, with_post_rht_amax=with_rht
        )
        assert not quantizer.columnwise_uses_4over6
        assert quantizer.nvfp4_columnwise_e4m3_max == 0
        out = quantizer(x)
        assert not out._nvfp4_use_4over6
        assert out._nvfp4_e4m3_max == 0
        assert out._nvfp4_columnwise_e4m3_max == 0


# ---------------------------------------------------------------------------
# Recipe wiring and guards.
# ---------------------------------------------------------------------------


def _make_quantizers(recipe: NVFP4BlockScaling, mode: str, num_quantizers: int):
    return NVFP4BlockScalingRecipeState(
        recipe, mode=mode, num_quantizers=num_quantizers
    ).make_quantizers()


@pytest.mark.skipif(not recipe_available, reason=reason_for_no_recipe)
@pytest.mark.parametrize("scope", ["weights", "activations", "all"])
@pytest.mark.parametrize("err_mode", ["MAE", "MSE"])
def test_pretrain_recipe_quantizers(scope, err_mode):
    """4over6 with RHT + stochastic rounding enabled is a supported recipe."""
    recipe = NVFP4BlockScaling(nvfp4_4over6=scope, nvfp4_4over6_err_mode=err_mode)
    assert not recipe.disable_rht and not recipe.disable_stochastic_rounding

    inp, weight, out = _make_quantizers(recipe, "forward", 3)
    act_4over6 = scope in ("activations", "all")
    weight_4over6 = scope in ("weights", "all")

    # Forward input: rowwise 4over6, columnwise RHT standard.
    assert inp.with_rht and not inp.stochastic_rounding
    assert inp.nvfp4_use_4over6 == act_4over6
    assert not inp.columnwise_uses_4over6
    assert inp.nvfp4_e4m3_max == (256 if act_4over6 else 0)
    assert inp.nvfp4_columnwise_e4m3_max == (448 if act_4over6 else 0)
    assert out.nvfp4_use_4over6 == act_4over6

    # Weight: rowwise 4over6 (2D), columnwise standard by default.
    assert not weight.with_rht and not weight.stochastic_rounding
    assert weight.with_2d_quantization
    assert weight.nvfp4_use_4over6 == weight_4over6
    assert not weight.columnwise_uses_4over6
    assert weight.nvfp4_e4m3_max == (256 if weight_4over6 else 0)
    assert weight.nvfp4_columnwise_e4m3_max == (448 if weight_4over6 else 0)

    # Gradients: standard NVFP4 with RHT + stochastic rounding.
    for grad in _make_quantizers(recipe, "backward", 2):
        assert not grad.nvfp4_use_4over6
        assert grad.with_rht and grad.stochastic_rounding
        assert grad.nvfp4_e4m3_max == 0
        assert grad.nvfp4_columnwise_e4m3_max == 0


@pytest.mark.skipif(not recipe_available, reason=reason_for_no_recipe)
@pytest.mark.parametrize(
    "columnwise_scope, disable_rht, expect_inp, expect_weight",
    [
        # auto: off with RHT (pre-training), on without RHT (previous behavior).
        ("auto", False, False, False),
        ("auto", True, True, True),
        ("none", True, False, False),
        ("weights", False, False, True),
        ("weights", True, False, True),
        # A columnwise RHT copy is never 4over6.
        ("activations", False, False, False),
        ("activations", True, True, False),
        ("all", False, False, True),
        ("all", True, True, True),
    ],
)
def test_recipe_columnwise_scope(columnwise_scope, disable_rht, expect_inp, expect_weight):
    recipe = NVFP4BlockScaling(
        nvfp4_4over6="all",
        nvfp4_4over6_columnwise=columnwise_scope,
        disable_rht=disable_rht,
    )
    inp, weight, _ = _make_quantizers(recipe, "forward", 3)
    assert inp.columnwise_uses_4over6 == expect_inp
    assert weight.columnwise_uses_4over6 == expect_weight
    assert inp.nvfp4_columnwise_e4m3_max == (0 if expect_inp else 448)
    assert weight.nvfp4_columnwise_e4m3_max == (0 if expect_weight else 448)


@pytest.mark.skipif(not recipe_available, reason=reason_for_no_recipe)
def test_recipe_unsupported_combinations_still_raise():
    # 4over6 is never combined with stochastic rounding.
    recipe = NVFP4BlockScaling(nvfp4_4over6="all")
    recipe.fp4_quant_fwd_inp = te_recipe.QParams(
        random_hadamard_transform=True, stochastic_rounding=True
    )
    with pytest.raises(ValueError, match="stochastic rounding"):
        _make_quantizers(recipe, "forward", 3)

    # 4over6 requires E4M3 block scales.
    recipe = NVFP4BlockScaling(nvfp4_4over6="all", fp8_format=te_recipe.Format.UE5M3)
    with pytest.raises(ValueError, match="UE5M3"):
        _make_quantizers(recipe, "forward", 3)

    # The recipe is a pydantic dataclass: a failed assert surfaces as a ValidationError.
    with pytest.raises((AssertionError, ValueError)):
        NVFP4BlockScaling(nvfp4_4over6="all", nvfp4_4over6_columnwise="invalid")


@pytest.mark.skipif(not recipe_available, reason=reason_for_no_recipe)
def test_4over6_with_stochastic_rounding_quantize_raises():
    x = torch.randn((256, 1024), dtype=torch.bfloat16, device=DEVICE)
    quantizer = NVFP4Quantizer(
        fp4_dtype=FP4_DTYPE,
        with_rht=True,
        with_post_rht_amax=True,
        stochastic_rounding=True,
        nvfp4_use_4over6=True,
        nvfp4_e4m3_max=256,
    )
    with pytest.raises(RuntimeError, match="stochastic rounding"):
        quantizer(x)


# ---------------------------------------------------------------------------
# (c) Operand-level and module-level: only the fprop operands change.
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not recipe_available, reason=reason_for_no_recipe)
@pytest.mark.parametrize("err_mode", ["MAE", "MSE"])
@pytest.mark.parametrize("disable_2d_quantization", [False, True], ids=["w2d", "w1d"])
@pytest.mark.parametrize(
    "tokens, in_features, out_features",
    [(1024, 512, 256), (352, 288, 160)],
    ids=["aligned", "unaligned"],
)
def test_pretrain_recipe_operands(
    err_mode, disable_2d_quantization, tokens, in_features, out_features
):
    """With the recipe quantizers, enabling 4over6 changes X and W rowwise copies only."""
    recipe_kwargs = dict(disable_2d_quantization=disable_2d_quantization)
    recipe_4over6 = NVFP4BlockScaling(
        nvfp4_4over6="all", nvfp4_4over6_err_mode=err_mode, **recipe_kwargs
    )
    recipe_baseline = NVFP4BlockScaling(**recipe_kwargs)

    _seed()
    x = torch.randn((tokens, in_features), dtype=torch.bfloat16, device=DEVICE)
    w = torch.randn((out_features, in_features), dtype=torch.bfloat16, device=DEVICE)
    dy = torch.randn((tokens, out_features), dtype=torch.bfloat16, device=DEVICE)

    inp_q, weight_q, _ = _make_quantizers(recipe_4over6, "forward", 3)
    inp_b, weight_b, _ = _make_quantizers(recipe_baseline, "forward", 3)
    grad_q = _make_quantizers(recipe_4over6, "backward", 2)[0]
    grad_b = _make_quantizers(recipe_baseline, "backward", 2)[0]

    # X: rowwise (fprop) is 4over6, columnwise RHT (wgrad) is unchanged.
    xq, xb = inp_q(x), inp_b(x)
    _assert_rowwise_matches_reference(xq, x, tile_shape=(1, 16), e4m3_max=256, err_mode=err_mode)
    assert not torch.equal(
        xq._rowwise_data.view(dtype=torch.uint8), xb._rowwise_data.view(dtype=torch.uint8)
    )
    _assert_columnwise_identical(xq, xb, x.shape)
    assert (xq._nvfp4_e4m3_max, xq._nvfp4_columnwise_e4m3_max) == (256, 448)

    # W: rowwise (fprop) is 4over6, columnwise (dgrad) is unchanged.
    wq, wb = weight_q(w), weight_b(w)
    tile_shape = (1, 16) if disable_2d_quantization else (16, 16)
    _assert_rowwise_matches_reference(wq, w, tile_shape=tile_shape, e4m3_max=256, err_mode=err_mode)
    assert not torch.equal(
        wq._rowwise_data.view(dtype=torch.uint8), wb._rowwise_data.view(dtype=torch.uint8)
    )
    _assert_columnwise_identical(wq, wb, w.shape)
    assert (wq._nvfp4_e4m3_max, wq._nvfp4_columnwise_e4m3_max) == (256, 448)

    # dY: standard NVFP4 with RHT + stochastic rounding in both recipes.
    _seed(1)
    dyq = grad_q(dy)
    _seed(1)
    dyb = grad_b(dy)
    assert not dyq._nvfp4_use_4over6
    torch.testing.assert_close(
        dyq._rowwise_data.view(dtype=torch.uint8),
        dyb._rowwise_data.view(dtype=torch.uint8),
        atol=0.0,
        rtol=0.0,
    )
    torch.testing.assert_close(
        _valid_scale(dyq._rowwise_scale_inv, dy.shape, False),
        _valid_scale(dyb._rowwise_scale_inv, dy.shape, False),
        atol=0.0,
        rtol=0.0,
    )
    _assert_columnwise_identical(dyq, dyb, dy.shape)


def _run_linear(recipe, weight, x, dy, seed):
    in_features = weight.shape[1]
    out_features = weight.shape[0]
    module = te.Linear(
        in_features=in_features,
        out_features=out_features,
        bias=False,
        device=DEVICE,
        params_dtype=torch.bfloat16,
    )
    with torch.no_grad():
        module.weight.copy_(weight)
    x = x.clone().detach().requires_grad_(True)
    with te.autocast(enabled=True, recipe=recipe):
        y = module(x)
    # Gradient quantization uses stochastic rounding: use the same RNG state for both runs.
    _seed(seed)
    y.backward(dy)
    return y.detach().clone(), x.grad.detach().clone(), module.weight.grad.detach().clone()


@pytest.mark.skipif(not recipe_available, reason=reason_for_no_recipe)
@pytest.mark.parametrize("scope", ["all", "activations", "weights"])
@pytest.mark.parametrize("err_mode", ["MAE", "MSE"])
@pytest.mark.parametrize("disable_2d_quantization", [False, True], ids=["w2d", "w1d"])
@pytest.mark.parametrize(
    "batch, seq, in_features, out_features",
    [(8, 128, 512, 256), (4, 88, 288, 160)],
    ids=["aligned", "unaligned"],
)
def test_linear_pretrain_recipe_4over6(
    scope, err_mode, disable_2d_quantization, batch, seq, in_features, out_features
):
    """te.Linear with 4over6 + RHT + stochastic rounding: forward and backward run, the
    fprop result uses the 4over6 operands, and dgrad/wgrad are identical to the recipe
    without 4over6 for the same dY."""
    recipe_kwargs = dict(disable_2d_quantization=disable_2d_quantization)
    recipe_4over6 = NVFP4BlockScaling(
        nvfp4_4over6=scope, nvfp4_4over6_err_mode=err_mode, **recipe_kwargs
    )
    recipe_baseline = NVFP4BlockScaling(**recipe_kwargs)
    assert not recipe_4over6.disable_rht and not recipe_4over6.disable_stochastic_rounding

    _seed()
    weight = torch.randn((out_features, in_features), dtype=torch.bfloat16, device=DEVICE) * 0.05
    x = torch.randn((batch, seq, in_features), dtype=torch.bfloat16, device=DEVICE)
    dy = torch.randn((batch, seq, out_features), dtype=torch.bfloat16, device=DEVICE)

    y_q, dx_q, dw_q = _run_linear(recipe_4over6, weight, x, dy, seed=7)
    y_b, dx_b, dw_b = _run_linear(recipe_baseline, weight, x, dy, seed=7)

    for t in (y_q, dx_q, dw_q):
        assert torch.isfinite(t.float()).all()

    # fprop operands changed.
    assert not torch.equal(y_q, y_b)
    # dgrad (dY, W.T) and wgrad (RHT(X), RHT(dY)) operands are unchanged.
    torch.testing.assert_close(dx_q, dx_b, atol=0.0, rtol=0.0)
    torch.testing.assert_close(dw_q, dw_b, atol=0.0, rtol=0.0)

    # fprop reference: GEMM of the dequantized rowwise operands produced by the recipe
    # quantizers. This also checks that GEMM applies the rowwise scale bounds.
    inp_q, weight_q, _ = _make_quantizers(recipe_4over6, "forward", 3)
    x_2d = x.reshape(-1, in_features)
    x_deq = inp_q(x_2d).dequantize(dtype=torch.float32)
    w_deq = weight_q(weight).dequantize(dtype=torch.float32)
    y_ref = (x_deq.double() @ w_deq.double().t()).reshape(batch, seq, out_features)
    rel_err = (y_q.double() - y_ref).norm() / y_ref.norm()
    assert rel_err < 1e-2, f"fprop relative error {rel_err:.3e}"


@pytest.mark.skipif(not recipe_available, reason=reason_for_no_recipe)
@pytest.mark.parametrize("disable_2d_quantization", [False, True], ids=["w2d", "w1d"])
def test_linear_pretrain_recipe_4over6_columnwise_weights(disable_2d_quantization):
    """With nvfp4_4over6_columnwise='weights', W.T (dgrad) also uses 4over6: dgrad changes,
    wgrad is still unchanged."""
    recipe_kwargs = dict(disable_2d_quantization=disable_2d_quantization)
    recipe_4over6 = NVFP4BlockScaling(
        nvfp4_4over6="all", nvfp4_4over6_columnwise="weights", **recipe_kwargs
    )
    recipe_baseline = NVFP4BlockScaling(**recipe_kwargs)

    batch, seq, in_features, out_features = 8, 128, 512, 256
    _seed()
    weight = torch.randn((out_features, in_features), dtype=torch.bfloat16, device=DEVICE) * 0.05
    x = torch.randn((batch, seq, in_features), dtype=torch.bfloat16, device=DEVICE)
    dy = torch.randn((batch, seq, out_features), dtype=torch.bfloat16, device=DEVICE)

    _, dx_q, dw_q = _run_linear(recipe_4over6, weight, x, dy, seed=7)
    _, dx_b, dw_b = _run_linear(recipe_baseline, weight, x, dy, seed=7)

    assert torch.isfinite(dx_q.float()).all()
    assert not torch.equal(dx_q, dx_b)
    torch.testing.assert_close(dw_q, dw_b, atol=0.0, rtol=0.0)

    # dgrad reference with the dequantized 4over6 W. dY is quantized with stochastic
    # rounding, so only a loose tolerance applies.
    _, weight_q, _ = _make_quantizers(recipe_4over6, "forward", 3)
    assert weight_q.columnwise_uses_4over6
    w_deq = weight_q(weight).dequantize(dtype=torch.float32)
    dx_ref = (dy.reshape(-1, out_features).double() @ w_deq.double()).reshape(
        batch, seq, in_features
    )
    rel_err = (dx_q.double() - dx_ref).norm() / dx_ref.norm()
    assert rel_err < 0.25, f"dgrad relative error {rel_err:.3e}"


# ---------------------------------------------------------------------------
# Backward copies from the forward-quantized operands
# (nvfp4_backward_weight_source / nvfp4_backward_activation_source).
# ---------------------------------------------------------------------------


def _reference_dequantize_rowwise(q, shape, e4m3_max: int, dtype: torch.dtype) -> torch.Tensor:
    """PyTorch dequantization of the rowwise copy, same operation order as the TE kernel."""
    rows, cols = shape
    codes = _unpack_fp4(q._rowwise_data).reshape(rows, cols // 16, 16)
    scale = _e4m3_bytes_to_float(q._rowwise_scale_inv[:rows, : cols // 16])
    values = _fp4_codes_to_values(codes)
    amax = q._amax_rowwise.view(())
    factor_inv = torch.tensor(1.0 / (6.0 * float(e4m3_max)), dtype=torch.float32, device=DEVICE)
    final_scale = scale * amax * factor_inv
    return (values * final_scale.unsqueeze(-1)).reshape(rows, cols).to(dtype)


@pytest.mark.skipif(not recipe_available, reason=reason_for_no_recipe)
@pytest.mark.parametrize("M, N", [(256, 1024), (1024, 256), (256, 272), (304, 304)])
@pytest.mark.parametrize(
    "use_4over6, columnwise_4over6",
    [(False, False), (True, False), (True, True)],
    ids=["standard", "row4over6_colstandard", "row4over6_col4over6"],
)
def test_weight_columnwise_from_rowwise_1d(M, N, use_4over6, columnwise_4over6):
    """1D weights: the columnwise copy equals quantize(dequantize(rowwise copy))."""
    _seed()
    w = torch.randn((M, N), dtype=torch.bfloat16, device=DEVICE)
    e4m3_max = 256 if use_4over6 else 0
    kwargs = dict(
        fp4_dtype=FP4_DTYPE,
        with_rht=False,
        with_2d_quantization=False,
        nvfp4_use_4over6=use_4over6,
        nvfp4_e4m3_max=e4m3_max,
        nvfp4_4over6_columnwise=columnwise_4over6,
    )
    sut = NVFP4Quantizer(columnwise_from_rowwise=True, **kwargs)(w)
    default = NVFP4Quantizer(**kwargs)(w)

    # The rowwise (forward) copy does not depend on the option.
    torch.testing.assert_close(
        sut._rowwise_data.view(dtype=torch.uint8),
        default._rowwise_data.view(dtype=torch.uint8),
        atol=0.0,
        rtol=0.0,
    )
    torch.testing.assert_close(
        _valid_scale(sut._rowwise_scale_inv, w.shape, False),
        _valid_scale(default._rowwise_scale_inv, w.shape, False),
        atol=0.0,
        rtol=0.0,
    )
    torch.testing.assert_close(sut._amax_rowwise, default._amax_rowwise, atol=0.0, rtol=0.0)

    # TE dequantization of the rowwise copy matches the PyTorch dequantization.
    w_deq = _reference_dequantize_rowwise(sut, w.shape, e4m3_max or 448, torch.bfloat16)
    torch.testing.assert_close(sut.dequantize(dtype=torch.bfloat16), w_deq, atol=0.0, rtol=0.0)

    # Columnwise copy versus the PyTorch reference applied to the dequantized rowwise copy.
    col_4over6 = use_4over6 and columnwise_4over6
    ref = NVFP4QuantizerRef(
        dtype=reference_utils.Fp4Formats.E2M1,
        rowwise=True,
        columnwise=True,
        pow_2_scales=False,
        eps=0.0,
        quant_tile_shape=(1, 16),
        nvfp4_use_4over6=col_4over6,
        nvfp4_e4m3_max=e4m3_max if col_4over6 else 0,
    ).quantize(w_deq)
    # The dequantized input contains -0.0 (negative values that quantized to zero). The kernel
    # keeps the sign bit of a zero code and the reference does not, so compare FP4 values.
    sut_codes = _unpack_fp4(sut._columnwise_data)
    ref_codes = _unpack_fp4(ref.data_t)
    torch.testing.assert_close(
        _fp4_codes_to_values(sut_codes), _fp4_codes_to_values(ref_codes), atol=0.0, rtol=0.0
    )
    assert bool((((sut_codes ^ ref_codes) & 0x7) == 0).all())
    sx_t_ref = ref.scale_t.view(dtype=torch.uint8)
    torch.testing.assert_close(
        sut._columnwise_scale_inv[: sx_t_ref.shape[0], : sx_t_ref.shape[1]],
        sx_t_ref,
        atol=0.0,
        rtol=0.0,
    )
    amax_deq = w_deq.abs().max().to(torch.float32).view(1)
    torch.testing.assert_close(sut._amax_columnwise, amax_deq, atol=0.0, rtol=0.0)
    # The option must change the columnwise copy.
    assert not torch.equal(
        sut._columnwise_data.view(dtype=torch.uint8),
        default._columnwise_data.view(dtype=torch.uint8),
    )


@pytest.mark.skipif(not recipe_available, reason=reason_for_no_recipe)
@pytest.mark.parametrize("M, N", [(256, 1024), (1024, 256)])
@pytest.mark.parametrize(
    "use_4over6, columnwise_4over6, expect_noop",
    [(False, False, True), (True, True, True), (True, False, False)],
    ids=["standard", "row4over6_col4over6", "row4over6_colstandard"],
)
def test_weight_columnwise_from_rowwise_2d(M, N, use_4over6, columnwise_4over6, expect_noop):
    """2D weights: no-op when both copies use the same scheme, otherwise re-quantized."""
    _seed()
    w = torch.randn((M, N), dtype=torch.bfloat16, device=DEVICE)
    e4m3_max = 256 if use_4over6 else 0
    kwargs = dict(
        fp4_dtype=FP4_DTYPE,
        with_rht=False,
        with_2d_quantization=True,
        nvfp4_use_4over6=use_4over6,
        nvfp4_e4m3_max=e4m3_max,
        nvfp4_4over6_columnwise=columnwise_4over6,
    )
    sut = NVFP4Quantizer(columnwise_from_rowwise=True, **kwargs)(w)
    default = NVFP4Quantizer(**kwargs)(w)
    torch.testing.assert_close(
        sut._rowwise_data.view(dtype=torch.uint8),
        default._rowwise_data.view(dtype=torch.uint8),
        atol=0.0,
        rtol=0.0,
    )
    if expect_noop:
        _assert_columnwise_identical(sut, default, w.shape)
        return
    # Standard 2D quantization of the dequantized 4over6 rowwise copy.
    w_deq = sut.dequantize(dtype=torch.bfloat16)
    expected = NVFP4Quantizer(
        fp4_dtype=FP4_DTYPE, rowwise=False, columnwise=True, with_2d_quantization=True
    )(w_deq)
    _assert_columnwise_identical(sut, expected, w.shape)
    assert not torch.equal(
        sut._columnwise_data.view(dtype=torch.uint8),
        default._columnwise_data.view(dtype=torch.uint8),
    )


@pytest.mark.skipif(not recipe_available, reason=reason_for_no_recipe)
@pytest.mark.parametrize("M, N", FUSED_SHAPES[:4] + UNFUSED_SHAPES[:3])
@pytest.mark.parametrize("use_4over6", [False, True], ids=["standard", "row4over6"])
def test_activation_columnwise_rht_from_rowwise(M, N, use_4over6):
    """X: the columnwise RHT copy equals RHT-quantize(dequantize(rowwise copy))."""
    _seed()
    x = torch.randn((M, N), dtype=torch.bfloat16, device=DEVICE)
    e4m3_max = 256 if use_4over6 else 0
    kwargs = dict(
        fp4_dtype=FP4_DTYPE,
        with_rht=True,
        with_post_rht_amax=True,
        nvfp4_use_4over6=use_4over6,
        nvfp4_e4m3_max=e4m3_max,
    )
    sut = NVFP4Quantizer(columnwise_from_rowwise=True, **kwargs)(x)
    default = NVFP4Quantizer(**kwargs)(x)

    torch.testing.assert_close(
        sut._rowwise_data.view(dtype=torch.uint8),
        default._rowwise_data.view(dtype=torch.uint8),
        atol=0.0,
        rtol=0.0,
    )
    torch.testing.assert_close(
        _valid_scale(sut._rowwise_scale_inv, x.shape, False),
        _valid_scale(default._rowwise_scale_inv, x.shape, False),
        atol=0.0,
        rtol=0.0,
    )
    torch.testing.assert_close(sut._amax_rowwise, default._amax_rowwise, atol=0.0, rtol=0.0)

    x_deq = _reference_dequantize_rowwise(sut, x.shape, e4m3_max or 448, torch.bfloat16)
    torch.testing.assert_close(sut.dequantize(dtype=torch.bfloat16), x_deq, atol=0.0, rtol=0.0)

    # Standard columnwise RHT quantization of the dequantized rowwise copy. That path is
    # checked against NVFP4QuantizerRef in test_nvfp4_rht_quantize_exact.py.
    expected = NVFP4Quantizer(
        fp4_dtype=FP4_DTYPE,
        rowwise=False,
        columnwise=True,
        with_rht=True,
        with_post_rht_amax=True,
    )(x_deq)
    _assert_columnwise_identical(sut, expected, x.shape)
    assert not torch.equal(
        sut._columnwise_data.view(dtype=torch.uint8),
        default._columnwise_data.view(dtype=torch.uint8),
    )
    assert sut._nvfp4_columnwise_e4m3_max == (448 if use_4over6 else 0)


@pytest.mark.skipif(not recipe_available, reason=reason_for_no_recipe)
def test_recipe_backward_source_options():
    recipe = NVFP4BlockScaling()
    assert recipe.nvfp4_backward_weight_source == "high_precision"
    assert recipe.nvfp4_backward_activation_source == "high_precision"
    for q in _make_quantizers(recipe, "forward", 3) + _make_quantizers(recipe, "backward", 2):
        assert not q.columnwise_from_rowwise

    recipe = NVFP4BlockScaling(nvfp4_backward_weight_source="forward_quantized")
    inp, weight, _ = _make_quantizers(recipe, "forward", 3)
    assert weight.columnwise_from_rowwise and not inp.columnwise_from_rowwise

    recipe = NVFP4BlockScaling(nvfp4_backward_activation_source="forward_quantized")
    inp, weight, _ = _make_quantizers(recipe, "forward", 3)
    assert inp.columnwise_from_rowwise and not weight.columnwise_from_rowwise
    for q in _make_quantizers(recipe, "backward", 2):
        assert not q.columnwise_from_rowwise

    with pytest.raises((AssertionError, ValueError)):
        NVFP4BlockScaling(nvfp4_backward_weight_source="invalid")
    with pytest.raises(ValueError, match="row-scaled"):
        _make_quantizers(
            NVFP4BlockScaling(
                disable_rht=True,
                row_scaled_activation=True,
                nvfp4_backward_activation_source="forward_quantized",
            ),
            "forward",
            3,
        )


@pytest.mark.skipif(not recipe_available, reason=reason_for_no_recipe)
@pytest.mark.parametrize("nvfp4_4over6", ["none", "all"])
@pytest.mark.parametrize(
    "batch, seq, in_features, out_features",
    [(8, 128, 512, 256), (4, 88, 288, 160)],
    ids=["aligned", "unaligned"],
)
def test_linear_backward_source(nvfp4_4over6, batch, seq, in_features, out_features):
    """te.Linear with 1D weights: the option leaves fprop unchanged and changes dgrad / wgrad
    only through the operand whose source was switched."""
    base_kwargs = dict(disable_2d_quantization=True, nvfp4_4over6=nvfp4_4over6)
    _seed()
    weight = torch.randn((out_features, in_features), dtype=torch.bfloat16, device=DEVICE) * 0.05
    x = torch.randn((batch, seq, in_features), dtype=torch.bfloat16, device=DEVICE)
    dy = torch.randn((batch, seq, out_features), dtype=torch.bfloat16, device=DEVICE)

    def run(**kwargs):
        return _run_linear(NVFP4BlockScaling(**base_kwargs, **kwargs), weight, x, dy, seed=7)

    y_d, dx_d, dw_d = run()
    y_w, dx_w, dw_w = run(nvfp4_backward_weight_source="forward_quantized")
    y_a, dx_a, dw_a = run(nvfp4_backward_activation_source="forward_quantized")
    y_b, dx_b, dw_b = run(
        nvfp4_backward_weight_source="forward_quantized",
        nvfp4_backward_activation_source="forward_quantized",
    )

    # Forward is unchanged.
    for y in (y_w, y_a, y_b):
        torch.testing.assert_close(y, y_d, atol=0.0, rtol=0.0)
    # Weight source: only dgrad (dY @ W) changes.
    assert not torch.equal(dx_w, dx_d)
    torch.testing.assert_close(dw_w, dw_d, atol=0.0, rtol=0.0)
    # Activation source: only wgrad (dY.T @ X) changes.
    assert not torch.equal(dw_a, dw_d)
    torch.testing.assert_close(dx_a, dx_d, atol=0.0, rtol=0.0)
    # Both: each gradient equals the single-option result.
    torch.testing.assert_close(dx_b, dx_w, atol=0.0, rtol=0.0)
    torch.testing.assert_close(dw_b, dw_a, atol=0.0, rtol=0.0)
    for t in (dx_b, dw_b):
        assert torch.isfinite(t.float()).all()

    # dgrad stays close to the default: the weight operands differ by re-quantization only.
    rel = (dx_w.double() - dx_d.double()).norm() / dx_d.double().norm()
    assert rel < 0.2, f"dgrad relative change {rel:.3e}"
    rel = (dw_a.double() - dw_d.double()).norm() / dw_d.double().norm()
    assert rel < 0.2, f"wgrad relative change {rel:.3e}"
