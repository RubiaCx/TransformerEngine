# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""MXFP8 quantization with stochastic rounding of the elements.

The block scales must match the round-to-nearest (RN) path bit for bit. Every
element must be one of the two E4M3 values that enclose the exact scaled
value, chosen with probabilities that make the result unbiased.
"""

import math
from typing import Optional, Tuple

import pytest
import torch

import transformer_engine.pytorch as te
import transformer_engine_torch as tex
from transformer_engine.pytorch import MXFP8Quantizer
from transformer_engine.pytorch.cpp_extensions import general_gemm

recipe_available, reason_for_no_recipe = te.is_mxfp8_available(return_reason=True)
pytestmark = pytest.mark.skipif(not recipe_available, reason=reason_for_no_recipe)

DEVICE = "cuda"
BLOCK = 32
MODES = ["rowwise_only", "columnwise_only", "both_directions"]


def make_quantizer(
    mode: str, *, stochastic_rounding: bool, optimize_for_gemm: bool = False
) -> MXFP8Quantizer:
    quantizer = MXFP8Quantizer(
        fp8_dtype=te.DType.kFloat8E4M3,
        rowwise=mode != "columnwise_only",
        columnwise=mode != "rowwise_only",
        stochastic_rounding=stochastic_rounding,
    )
    quantizer.optimize_for_gemm = optimize_for_gemm
    return quantizer


def quantize(quantizer: MXFP8Quantizer, x: torch.Tensor, seed: Optional[int] = None):
    """Quantize, optionally after reseeding the CUDA generator the RNG state is taken from."""
    if seed is not None:
        torch.cuda.manual_seed(seed)
    return quantizer(x)


def flat_2d(x: torch.Tensor) -> torch.Tensor:
    return x.reshape(-1, x.shape[-1])


def direction_outputs(q, columnwise: bool) -> Tuple[torch.Tensor, torch.Tensor]:
    if columnwise:
        return q._columnwise_data.view(torch.uint8), q._columnwise_scale_inv
    return q._rowwise_data.view(torch.uint8), q._rowwise_scale_inv


def directions(mode: str):
    return [
        d
        for d, used in ((False, mode != "columnwise_only"), (True, mode != "rowwise_only"))
        if used
    ]


def valid_scales(scale: torch.Tensor, rows: int, cols: int, columnwise: bool) -> torch.Tensor:
    """Compact (non-swizzled) scales without the padding of the scale tensor."""
    if columnwise:
        return scale[: math.ceil(rows / BLOCK), :cols]
    return scale[:rows, : math.ceil(cols / BLOCK)]


def scale_per_element(scale: torch.Tensor, rows: int, cols: int, columnwise: bool) -> torch.Tensor:
    """FP64 power-of-two scale of every element, from compact E8M0 scales."""
    s = valid_scales(scale, rows, cols, columnwise).to(torch.float64) - 127.0
    if columnwise:
        s = s.repeat_interleave(BLOCK, dim=0)[:rows]
    else:
        s = s.repeat_interleave(BLOCK, dim=1)[:, :cols]
    return torch.exp2(s)


def e4m3_levels() -> torch.Tensor:
    """The 127 non-negative finite E4M3 values in ascending order (0 ... 448)."""
    codes = torch.arange(127, dtype=torch.uint8, device=DEVICE)
    return codes.view(torch.float8_e4m3fn).float().double()


def decode_e4m3(codes: torch.Tensor) -> torch.Tensor:
    return codes.contiguous().view(torch.float8_e4m3fn).float().double()


def decode(codes: torch.Tensor, scale: torch.Tensor, rows: int, cols: int, columnwise: bool):
    return decode_e4m3(flat_2d(codes)) * scale_per_element(scale, rows, cols, columnwise)


def neighbours(x2d: torch.Tensor, scale: torch.Tensor, columnwise: bool):
    """Exact scaled magnitudes (FP64) and their enclosing E4M3 levels (saturating at 448)."""
    rows, cols = x2d.shape
    levels = e4m3_levels()
    v = x2d.double().abs() / scale_per_element(scale, rows, cols, columnwise)
    v = torch.nan_to_num(v, nan=448.0, posinf=448.0)
    hi = levels[torch.searchsorted(levels, v.contiguous()).clamp(max=126)]
    lo = levels[(torch.searchsorted(levels, v.contiguous(), right=True) - 1).clamp(0, 126)]
    return v, lo, hi


def check_against_oracle(x: torch.Tensor, q_sr, q_rn, mode: str) -> None:
    """Scales equal RN; every SR element is an enclosing E4M3 level with the input sign."""
    x2d = flat_2d(x)
    rows, cols = x2d.shape
    for columnwise in directions(mode):
        codes, scale = direction_outputs(q_sr, columnwise)
        codes_rn, scale_rn = direction_outputs(q_rn, columnwise)
        assert codes.shape == x.shape
        assert scale.shape == scale_rn.shape
        assert torch.equal(
            valid_scales(scale, rows, cols, columnwise),
            valid_scales(scale_rn, rows, cols, columnwise),
        ), "SR block scales differ from RN"

        codes = flat_2d(codes)
        magnitude = codes & 0x7F
        assert int(magnitude.max()) <= 0x7E, "SR produced a non-finite E4M3 code"
        v, lo, hi = neighbours(x2d, scale, columnwise)
        q = decode_e4m3(magnitude)
        assert bool(((q == lo) | (q == hi)).all()), "SR element is not an enclosing E4M3 level"
        on_grid = lo == hi
        assert torch.equal(q[on_grid], v.clamp(max=448.0)[on_grid])
        # Sign is kept (NaN maps to +448).
        finite = ~torch.isnan(x2d)
        assert torch.equal((codes >= 0x80)[finite], torch.signbit(x2d)[finite])
        assert bool((codes[~finite] == 0x7E).all())

        # RN picks the nearest of the same two levels, so wherever SR and RN differ they are the
        # two neighbours. Skipped for non-finite inputs and all-NaN blocks (scale 0xFF).
        q_rn_mag = decode_e4m3(flat_2d(codes_rn) & 0x7F)
        regular = torch.isfinite(x2d) & (
            scale_per_element(scale, rows, cols, columnwise) < 2.0**128
        )
        assert bool(((q_rn_mag == lo) | (q_rn_mag == hi))[regular].all())


def random_input(shape, dtype, seed: int = 0, spread: float = 4.0) -> torch.Tensor:
    """Normal values with a per-element log-uniform magnitude spread, to reach subnormals."""
    gen = torch.Generator(device=DEVICE).manual_seed(seed)
    x = torch.randn(shape, device=DEVICE, generator=gen)
    x = x * torch.exp2(spread * (torch.rand(shape, device=DEVICE, generator=gen) * 2 - 1) * 3)
    return x.to(dtype).contiguous()


SHAPES = [
    (32, 32),
    (128, 128),
    (256, 1024),
    (64, 4096),
    (1024, 256),
    # Partial blocks / kernel chunks: the last dimension only has to be a multiple of 16.
    (1, 32),
    (3, 16),
    (33, 48),
    (100, 160),
    (129, 144),
    (257, 1040),
    (31, 2064),
    # Leading dimensions are flattened.
    (4, 32, 64),
    (3, 5, 80),
]


@pytest.mark.parametrize("shape", SHAPES, ids=str)
@pytest.mark.parametrize("x_dtype", [torch.float32, torch.bfloat16], ids=str)
@pytest.mark.parametrize("mode", MODES)
def test_sr_scales_and_adjacent_levels(shape, x_dtype, mode):
    x = random_input(shape, x_dtype, seed=sum(shape) + len(mode))
    q_rn = quantize(make_quantizer(mode, stochastic_rounding=False), x)
    q_sr = quantize(make_quantizer(mode, stochastic_rounding=True), x, seed=7)
    check_against_oracle(x, q_sr, q_rn, mode)

    # The RN result is not perturbed by the SR option being available.
    q_rn2 = quantize(make_quantizer(mode, stochastic_rounding=False), x)
    for columnwise in directions(mode):
        assert torch.equal(
            direction_outputs(q_rn, columnwise)[0], direction_outputs(q_rn2, columnwise)[0]
        )

    # Off-grid inputs must actually be rounded both ways.
    x2d = flat_2d(x)
    for columnwise in directions(mode):
        codes, scale = direction_outputs(q_sr, columnwise)
        v, lo, hi = neighbours(x2d, scale, columnwise)
        q = decode_e4m3(flat_2d(codes) & 0x7F)
        off_grid = hi > lo
        if int(off_grid.sum()) >= 256:
            ups = float((q == hi)[off_grid].double().mean())
            assert 0.05 < ups < 0.95, f"suspicious round-up fraction {ups}"


@pytest.mark.parametrize(
    "shape", [(128, 128), (256, 384), (100, 160), (33, 48), (512, 2048)], ids=str
)
@pytest.mark.parametrize("x_dtype", [torch.float32, torch.bfloat16], ids=str)
@pytest.mark.parametrize("mode", MODES)
def test_sr_gemm_swizzled_scales(shape, x_dtype, mode):
    """GEMM-swizzled scale layout: scale buffers equal RN (padding included) and the data
    equals the compact-layout SR result for the same RNG state."""
    x = random_input(shape, x_dtype, seed=3)
    q_rn = quantize(make_quantizer(mode, stochastic_rounding=False, optimize_for_gemm=True), x)
    q_sr = quantize(
        make_quantizer(mode, stochastic_rounding=True, optimize_for_gemm=True), x, seed=11
    )
    q_sr_compact = quantize(make_quantizer(mode, stochastic_rounding=True), x, seed=11)
    assert q_sr._with_gemm_swizzled_scales and q_rn._with_gemm_swizzled_scales
    for columnwise in directions(mode):
        codes, scale = direction_outputs(q_sr, columnwise)
        _, scale_rn = direction_outputs(q_rn, columnwise)
        codes_compact, _ = direction_outputs(q_sr_compact, columnwise)
        assert torch.equal(scale, scale_rn), "swizzled SR scales differ from RN"
        assert torch.equal(codes, codes_compact)


@pytest.mark.parametrize("x_dtype", [torch.float32, torch.bfloat16], ids=str)
@pytest.mark.parametrize("mode", MODES)
def test_sr_keeps_representable_values(x_dtype, mode):
    """Inputs that are exactly representable after scaling are returned unchanged."""
    rows, cols = 256, 512
    gen = torch.Generator(device=DEVICE).manual_seed(5)
    levels = e4m3_levels().float()
    idx = torch.randint(0, 127, (rows, cols), device=DEVICE, generator=gen)
    sign = torch.randint(0, 2, (rows, cols), device=DEVICE, generator=gen).float() * 2 - 1
    x = levels[idx] * sign
    # Pin the amax of every 32x32 tile to 448 so that all block scales are 2^k, with a different
    # exponent per tile.
    x[::BLOCK, :] = 448.0
    x[:, ::BLOCK] = -448.0
    k = torch.randint(-20, 21, (rows // BLOCK, cols // BLOCK), device=DEVICE, generator=gen)
    k = k.repeat_interleave(BLOCK, dim=0).repeat_interleave(BLOCK, dim=1)
    x = (x * torch.exp2(k.float())).to(x_dtype)
    # E4M3 values times a power of two fit in BF16 (4 significant bits).
    q_rn = quantize(make_quantizer(mode, stochastic_rounding=False), x)
    for seed in range(8):
        q_sr = quantize(make_quantizer(mode, stochastic_rounding=True), x, seed=seed)
        for columnwise in directions(mode):
            codes, scale = direction_outputs(q_sr, columnwise)
            codes_rn, scale_rn = direction_outputs(q_rn, columnwise)
            assert torch.equal(codes, codes_rn)
            assert torch.equal(
                valid_scales(scale, rows, cols, columnwise),
                valid_scales(scale_rn, rows, cols, columnwise),
            )
            assert torch.equal(decode(codes, scale, rows, cols, columnwise), x.double())


def bernstein_bound(p: torch.Tensor, draws: int, log_term: float) -> torch.Tensor:
    """Bernstein bound on |mean of `draws` Bernoulli(p) - p|, exceeded with probability at
    most 2 * exp(-log_term) per element."""
    return torch.sqrt(2.0 * p * (1.0 - p) * log_term / draws) + log_term / (3.0 * draws)


@pytest.mark.parametrize("x_dtype", [torch.float32, torch.bfloat16], ids=str)
@pytest.mark.parametrize("mode", MODES)
def test_sr_is_unbiased(x_dtype, mode):
    """The mean over independent draws converges to the input.

    Per element the round-up indicator is Bernoulli(p) with p = (v - low) / (high - low), so
    the mean of the decoded values deviates from the input by gap * (mean indicator - p).
    The per-element tolerance is the Bernstein bound with exp(-L) = 1e-11; with 2 directions
    x 32768 elements the test fails by chance with probability below 1e-5. The aggregate
    check sums the standardized deviation of all elements, which is sensitive to a common
    bias of a fraction of a percent of a grid step.
    """
    rows, cols = 64, 512
    draws = 1024
    log_term = -math.log(1e-11)
    x = random_input((rows, cols), x_dtype, seed=21)
    q_rn = quantize(make_quantizer(mode, stochastic_rounding=False), x)
    quantizer = make_quantizer(mode, stochastic_rounding=True)
    torch.cuda.manual_seed(99)
    sums = {
        c: torch.zeros((rows, cols), dtype=torch.float64, device=DEVICE) for c in directions(mode)
    }
    scales = {}
    for _ in range(draws):
        q_sr = quantizer(x)
        for columnwise in directions(mode):
            codes, scale = direction_outputs(q_sr, columnwise)
            sums[columnwise] += decode_e4m3(codes & 0x7F)
            scales[columnwise] = scale

    for columnwise in directions(mode):
        v, lo, hi = neighbours(x, scales[columnwise], columnwise)
        gap = hi - lo
        off_grid = gap > 0
        assert float(off_grid.double().mean()) > 0.5, "test input is mostly on the E4M3 grid"
        p = torch.where(off_grid, (v - lo) / torch.where(off_grid, gap, torch.ones_like(gap)), 0.0)
        mean = sums[columnwise] / draws
        # Per element, in units of the local grid step.
        deviation = torch.where(off_grid, (mean - v) / torch.where(off_grid, gap, 1.0), mean - v)
        bound = bernstein_bound(p, draws, log_term)
        assert bool(
            (deviation.abs() <= bound).all()
        ), f"max excess {float((deviation.abs() - bound).max())} grid steps"
        # Aggregate: sum of (ups - draws * p) over all elements is ~N(0, draws * sum p(1-p)).
        z = float((deviation * draws).sum() / torch.sqrt(draws * (p * (1 - p)).sum()))
        assert abs(z) < 5.0, f"aggregate bias z-score {z}"

        # The SR mean is much closer to the input than RN. The expected ratio is about
        # 1 / sqrt(draws) times a constant near 1.4.
        scale_elt = scale_per_element(scales[columnwise], rows, cols, columnwise)
        x64 = x.double()
        sign = torch.where(torch.signbit(x), -1.0, 1.0)
        sr_err = (mean * scale_elt * sign - x64).abs()
        codes_rn, scale_rn = direction_outputs(q_rn, columnwise)
        rn_err = (decode(codes_rn, scale_rn, rows, cols, columnwise) - x64).abs()
        assert float(sr_err.mean()) < 0.1 * float(rn_err.mean())
        assert float(sr_err.square().mean().sqrt()) < 0.1 * float(rn_err.square().mean().sqrt())


@pytest.mark.parametrize("x_dtype", [torch.float32, torch.bfloat16], ids=str)
@pytest.mark.parametrize("columnwise", [False, True], ids=["rowwise", "columnwise"])
def test_sr_round_up_probability(x_dtype, columnwise):
    """Round-up frequency of fixed scaled values, pooled over positions and draws, and
    independence of neighbouring positions."""
    rows, cols = 128, 256
    draws = 64
    # (scaled value, low, high): normal range, top binade, subnormal range, negative.
    cases = [
        (1.03125, 1.0, 1.125),
        (1.09375, 1.0, 1.125),
        (300.0, 288.0, 320.0),
        (0.001220703125, 0.0, 0.001953125),
        (0.0166015625, 0.015625, 0.017578125),
        (-3.375, 3.25, 3.5),
        (-0.00048828125, 0.0, 0.001953125),
    ]
    mode = "columnwise_only" if columnwise else "rowwise_only"
    quantizer = make_quantizer(mode, stochastic_rounding=True)
    for case, (value, lo, hi) in enumerate(cases):
        p = (abs(value) - lo) / (hi - lo)
        x = torch.full((rows, cols), value, dtype=torch.float32, device=DEVICE)
        # One 448 per block pins the block scale to 1.
        if columnwise:
            x[::BLOCK, :] = 448.0
        else:
            x[:, ::BLOCK] = 448.0
        x = x.to(x_dtype)
        assert torch.equal(x.float(), x.float().to(x_dtype).float())
        probe = x.float() == value
        torch.cuda.manual_seed(1000 + case)
        ups = torch.zeros((rows, cols), dtype=torch.float64, device=DEVICE)
        for _ in range(draws):
            codes, scale = direction_outputs(quantizer(x), columnwise)
            assert int(valid_scales(scale, rows, cols, columnwise).min()) == 127
            assert int(valid_scales(scale, rows, cols, columnwise).max()) == 127
            q = decode_e4m3(codes)
            assert bool(((q.abs() == lo) | (q.abs() == hi))[probe].all())
            assert bool((torch.signbit(q) == (value < 0))[probe].all())
            ups += (q.abs() == hi).double()
        n = int(probe.sum()) * draws
        freq = float(ups[probe].sum()) / n
        sigma = math.sqrt(p * (1 - p) / n)
        assert abs(freq - p) < 6 * sigma, f"value {value}: round-up frequency {freq}, expected {p}"

        # Positions are independent: the per-position counts over the draws are Binomial(draws, p)
        # and neighbouring positions are uncorrelated. The correlation estimate over n_pairs
        # pairs has standard deviation 1 / sqrt(n_pairs).
        centered = (ups - draws * p) / math.sqrt(draws * p * (1 - p))
        pairs = (
            (centered[:, :-1], centered[:, 1:], probe[:, :-1] & probe[:, 1:]),
            (centered[:-1, :], centered[1:, :], probe[:-1, :] & probe[1:, :]),
        )
        for first, second, both in pairs:
            corr = float((first * second)[both].mean())
            assert abs(corr) < 6 / math.sqrt(int(both.sum())), f"neighbour correlation {corr}"
        variance = float(centered[probe].square().mean())
        assert abs(variance - 1.0) < 0.1, f"per-position variance ratio {variance}"


@pytest.mark.parametrize("x_dtype", [torch.float32, torch.bfloat16], ids=str)
@pytest.mark.parametrize("mode", MODES)
def test_sr_determinism(x_dtype, mode):
    x = random_input((256, 512), x_dtype, seed=31)
    quantizer = make_quantizer(mode, stochastic_rounding=True)

    def data(q):
        return [direction_outputs(q, c)[0].clone() for c in directions(mode)]

    a = data(quantize(quantizer, x, seed=123))
    b = data(quantize(quantizer, x, seed=123))
    c = data(quantize(quantizer, x, seed=124))
    d = data(quantizer(x))  # generator advanced by the previous call
    for qa, qb, qc, qd in zip(a, b, c, d):
        assert torch.equal(qa, qb), "same RNG state must give the same output"
        assert not torch.equal(qa, qc), "different seeds must give different output"
        assert not torch.equal(qc, qd), "consecutive calls must draw new random numbers"
        # Different seeds decorrelate: about half of the off-grid elements differ.
        assert float((qa != qc).double().mean()) > 0.1
    if mode == "both_directions":
        # The two directions do not reuse the same random numbers. With 1x32 and 32x1 blocks
        # scaled alike (same amax in every block), equal draws would give equal data.
        y = torch.full((256, 512), 1.0625, dtype=x_dtype, device=DEVICE)
        y[::BLOCK, :] = 448.0
        y[:, ::BLOCK] = 448.0
        q = quantize(quantizer, y, seed=5)
        assert torch.equal(
            valid_scales(q._rowwise_scale_inv, 256, 512, False).unique(),
            valid_scales(q._columnwise_scale_inv, 256, 512, True).unique(),
        )
        differ = float((q._rowwise_data != q._columnwise_data).double().mean())
        assert 0.3 < differ < 0.6, f"rowwise/columnwise differing fraction {differ}"


def special_inputs(x_dtype) -> dict:
    rows, cols = 96, 160
    gen = torch.Generator(device=DEVICE).manual_seed(41)
    base = torch.randn((rows, cols), device=DEVICE, generator=gen)
    huge = 3.0e38 if x_dtype == torch.float32 else 3.0e38
    cases = {
        "zeros": torch.zeros((rows, cols), device=DEVICE),
        "negative_zeros": -torch.zeros((rows, cols), device=DEVICE),
        "tiny": base * 1e-30,
        "subnormal_fp32": base * 1e-40,
        "huge": base.clamp(-1, 1) * huge,
        "large": base * 1e30,
        "mixed_sign_constant": torch.where(base > 0, 1.7, -1.7),
        "one_hot_blocks": torch.zeros((rows, cols), device=DEVICE),
        "mixed_magnitude_blocks": base
        * torch.exp2(
            torch.randint(
                -120, 120, (rows // BLOCK, cols // BLOCK), device=DEVICE, generator=gen
            ).float()
        )
        .repeat_interleave(BLOCK, dim=0)
        .repeat_interleave(BLOCK, dim=1),
        "wide_range_in_block": base
        * torch.exp2(torch.randint(-40, 40, (rows, cols), device=DEVICE, generator=gen).float()),
        "zero_and_nonzero_blocks": base
        * (torch.rand((rows // BLOCK, cols // BLOCK), device=DEVICE, generator=gen) > 0.5)
        .float()
        .repeat_interleave(BLOCK, dim=0)
        .repeat_interleave(BLOCK, dim=1),
    }
    cases["one_hot_blocks"][::7, ::13] = -5.5
    with_inf = base.clone()
    with_inf[3, 5] = float("inf")
    with_inf[40, 100] = float("-inf")
    cases["with_inf"] = with_inf
    with_nan = base.clone()
    with_nan[2, 7] = float("nan")
    with_nan[64:96, 32:64] = float("nan")
    cases["with_nan"] = with_nan
    return {name: value.to(x_dtype).contiguous() for name, value in cases.items()}


@pytest.mark.parametrize(
    "case",
    [
        "zeros",
        "negative_zeros",
        "tiny",
        "subnormal_fp32",
        "huge",
        "large",
        "mixed_sign_constant",
        "one_hot_blocks",
        "mixed_magnitude_blocks",
        "wide_range_in_block",
        "zero_and_nonzero_blocks",
        "with_inf",
        "with_nan",
    ],
)
@pytest.mark.parametrize("x_dtype", [torch.float32, torch.bfloat16], ids=str)
@pytest.mark.parametrize("mode", MODES)
def test_sr_special_values(case, x_dtype, mode):
    x = special_inputs(x_dtype)[case]
    q_rn = quantize(make_quantizer(mode, stochastic_rounding=False), x)
    for seed in range(3):
        q_sr = quantize(make_quantizer(mode, stochastic_rounding=True), x, seed=seed)
        check_against_oracle(x, q_sr, q_rn, mode)
        if case in ("zeros", "negative_zeros"):
            for columnwise in directions(mode):
                codes, scale = direction_outputs(q_sr, columnwise)
                expected = 0x80 if case == "negative_zeros" else 0
                assert bool((codes == expected).all())
                assert int(valid_scales(scale, *x.shape, columnwise).max()) == 0


@pytest.mark.parametrize("x_dtype", [torch.float32, torch.bfloat16], ids=str)
@pytest.mark.parametrize("stochastic_rounding", [False, True], ids=["rn", "sr"])
@pytest.mark.parametrize("shape", [(32, 40), (64, 8), (16, 100)], ids=str)
def test_unsupported_shapes_match_rn(x_dtype, stochastic_rounding, shape):
    """SR has the same shape restriction as RN: the last dimension is a multiple of 16."""
    x = random_input(shape, x_dtype)
    for mode in MODES:
        quantizer = make_quantizer(mode, stochastic_rounding=stochastic_rounding)
        with pytest.raises(RuntimeError, match="Shape not supported"):
            quantizer(x)


@pytest.mark.parametrize("stochastic_rounding", [False, True], ids=["rn", "sr"])
@pytest.mark.parametrize("mode", MODES)
def test_unaligned_input_matches_rn(stochastic_rounding, mode):
    """SR has the same pointer restriction as the generic RN kernel (16-byte alignment)."""
    rows, cols = 64, 48
    storage = torch.randn(rows * cols + 1, dtype=torch.float32, device=DEVICE)
    x = storage[1:].view(rows, cols)
    assert x.is_contiguous() and x.data_ptr() % 16 != 0
    quantizer = make_quantizer(mode, stochastic_rounding=stochastic_rounding)
    with pytest.raises(RuntimeError, match="16B aligned"):
        quantizer(x)


@pytest.mark.parametrize("x_dtype", [torch.float32, torch.bfloat16], ids=str)
@pytest.mark.parametrize("mode", ["rowwise_only", "both_directions"])
@pytest.mark.parametrize("shape", [(128, 256), (100, 160), (4, 32, 64)], ids=str)
def test_sr_dequantize_round_trip(x_dtype, mode, shape):
    x = random_input(shape, x_dtype, seed=51)
    x2d = flat_2d(x)
    rows, cols = x2d.shape
    q_sr = quantize(make_quantizer(mode, stochastic_rounding=True), x, seed=9)
    codes, scale = direction_outputs(q_sr, False)
    expected = decode(codes, scale, rows, cols, False)
    dequantized = q_sr.dequantize(dtype=torch.float32)
    assert dequantized.shape == x.shape
    assert torch.equal(flat_2d(dequantized).double(), expected)
    # The round trip error is below one grid step of the element.
    v, lo, hi = neighbours(x2d, scale, False)
    scale_elt = scale_per_element(scale, rows, cols, False)
    assert bool(((expected - x2d.double()).abs() <= (hi - lo) * scale_elt).all())

    # update_quantized into an existing tensor takes the same path.
    quantizer = make_quantizer(mode, stochastic_rounding=True)
    dst = quantizer.make_empty(x.shape, dtype=x_dtype, device=DEVICE)
    torch.cuda.manual_seed(9)
    quantizer.update_quantized(x, dst)
    assert torch.equal(direction_outputs(dst, False)[0], codes)
    assert torch.equal(
        valid_scales(direction_outputs(dst, False)[1], rows, cols, False),
        valid_scales(scale, rows, cols, False),
    )


def dequantize_usage(q, columnwise: bool) -> torch.Tensor:
    """FP32 values of one usage of a compact-scale MXFP8 tensor."""
    codes, scale = direction_outputs(q, columnwise)
    rows, cols = flat_2d(codes).shape
    return decode(codes, scale, rows, cols, columnwise).float()


@pytest.mark.parametrize("x_dtype", [torch.float32, torch.bfloat16], ids=str)
@pytest.mark.parametrize("layout", ["TN", "NN", "NT"])
@pytest.mark.parametrize("optimize_for_gemm", [False, True], ids=["compact", "swizzled"])
def test_sr_gemm_smoke(x_dtype, layout, optimize_for_gemm):
    """GEMM on SR-quantized operands equals the product of the values they encode.

    Layouts as used by Linear: TN is the forward product x @ w.T, NN the dgrad product
    dy @ w, and NT the wgrad product dy.T @ x.
    """
    batch, in_features, out_features = 256, 512, 384
    gen = torch.Generator(device=DEVICE).manual_seed(61)
    w = torch.randn((out_features, in_features), device=DEVICE, generator=gen).to(x_dtype)
    x = torch.randn((batch, in_features), device=DEVICE, generator=gen).to(x_dtype)
    dy = torch.randn((batch, out_features), device=DEVICE, generator=gen).to(x_dtype)
    if layout == "TN":
        a_hp, b_hp = w, x
    elif layout == "NN":
        a_hp, b_hp = w, dy
    else:
        a_hp, b_hp = x, dy
    transa, transb = layout[0] == "T", layout[1] == "T"
    a_columnwise, b_columnwise = not transa, transb

    def sr_operand(value, columnwise):
        compact = quantize(
            make_quantizer(
                "columnwise_only" if columnwise else "rowwise_only", stochastic_rounding=True
            ),
            value,
            seed=77,
        )
        if not optimize_for_gemm:
            return compact, compact
        swizzled = quantize(
            make_quantizer(
                "columnwise_only" if columnwise else "rowwise_only",
                stochastic_rounding=True,
                optimize_for_gemm=True,
            ),
            value,
            seed=77,
        )
        assert torch.equal(
            direction_outputs(swizzled, columnwise)[0], direction_outputs(compact, columnwise)[0]
        )
        return swizzled, compact

    a_q, a_compact = sr_operand(a_hp, a_columnwise)
    b_q, b_compact = sr_operand(b_hp, b_columnwise)
    a_val = dequantize_usage(a_compact, a_columnwise)
    b_val = dequantize_usage(b_compact, b_columnwise)
    # The SR operands differ from the RN ones, so this exercises SR data in the GEMM.
    a_rn = quantize(
        make_quantizer(
            "columnwise_only" if a_columnwise else "rowwise_only", stochastic_rounding=False
        ),
        a_hp,
    )
    assert not torch.equal(
        direction_outputs(a_rn, a_columnwise)[0], direction_outputs(a_compact, a_columnwise)[0]
    )

    actual = general_gemm(a_q, b_q, out_dtype=torch.float32, layout=layout)[0]
    if layout == "TN":
        expected = b_val @ a_val.T
    elif layout == "NN":
        expected = b_val @ a_val
    else:
        expected = b_val.T @ a_val
    assert actual.shape == expected.shape
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4 * float(expected.abs().max()))
    # And the result is close to the high-precision product, at the FP8 quantization noise level.
    if layout == "TN":
        hp = b_hp.float() @ a_hp.float().T
    elif layout == "NN":
        hp = b_hp.float() @ a_hp.float()
    else:
        hp = b_hp.float().T @ a_hp.float()
    rel = float((actual - hp).norm() / hp.norm())
    assert rel < 0.1, f"relative error vs high precision {rel}"


def test_quantizer_flag_defaults_and_copy():
    quantizer = MXFP8Quantizer(fp8_dtype=te.DType.kFloat8E4M3)
    assert quantizer.stochastic_rounding is False
    assert quantizer.is_requantization_safe()
    sr = MXFP8Quantizer(fp8_dtype=te.DType.kFloat8E4M3, stochastic_rounding=True)
    assert sr.stochastic_rounding is True
    assert sr.copy().stochastic_rounding is True
    assert not sr.is_requantization_safe()
    assert quantizer.copy().stochastic_rounding is False
    assert sr != quantizer


def test_sr_unsupported_configurations():
    """Options and fused paths that only round to nearest reject stochastic rounding."""
    x = random_input((128, 128), torch.bfloat16)

    # E5M2 output
    quantizer = MXFP8Quantizer(fp8_dtype=te.DType.kFloat8E5M2, stochastic_rounding=True)
    with pytest.raises(RuntimeError, match="only supports Float8E4M3"):
        quantizer(x)

    # 2D block scaling
    quantizer = MXFP8Quantizer(
        fp8_dtype=te.DType.kFloat8E4M3, with_2d_quantization=True, stochastic_rounding=True
    )
    with pytest.raises(RuntimeError, match="2D block scaling"):
        quantizer(x)

    # FP16 input
    quantizer = MXFP8Quantizer(fp8_dtype=te.DType.kFloat8E4M3, stochastic_rounding=True)
    with pytest.raises(RuntimeError, match="only supports Float32 and BFloat16"):
        quantizer(x.to(torch.float16))

    # Fused dbias, activation and activation-gradient kernels
    with pytest.raises(RuntimeError, match="does not support MXFP8 stochastic rounding"):
        tex.bgrad_quantize(x, quantizer)
    with pytest.raises(RuntimeError, match="does not support MXFP8 stochastic rounding"):
        tex.gelu(x, quantizer)
    with pytest.raises(RuntimeError, match="does not support MXFP8 stochastic rounding"):
        tex.dgelu(x, x, quantizer)
    with pytest.raises(RuntimeError, match="does not support MXFP8 stochastic rounding"):
        tex.dbias_dgelu(x, x, quantizer)

    # Grouped quantization
    with pytest.raises(RuntimeError, match="does not support MXFP8 stochastic rounding"):
        tex.group_quantize(x, quantizer, 2, None, None, None, None, None)

    # The same calls work with round to nearest.
    rn = MXFP8Quantizer(fp8_dtype=te.DType.kFloat8E4M3)
    tex.bgrad_quantize(x, rn)
    tex.gelu(x, rn)


def test_sr_split_quantize():
    """split_quantize quantizes every split with its own quantizer, SR included."""
    x = random_input((256, 128), torch.bfloat16, seed=71)
    sections = [64, 192]
    sr = [make_quantizer("both_directions", stochastic_rounding=True) for _ in sections]
    rn = [make_quantizer("both_directions", stochastic_rounding=False) for _ in sections]
    torch.cuda.manual_seed(3)
    out_sr = tex.split_quantize(x, sections, sr)
    out_rn = tex.split_quantize(x, sections, rn)
    any_diff = False
    for chunk, q_sr, q_rn in zip(torch.split(x, sections), out_sr, out_rn):
        check_against_oracle(chunk.contiguous(), q_sr, q_rn, "both_directions")
        any_diff |= not torch.equal(q_sr._rowwise_data, q_rn._rowwise_data)
    assert any_diff
