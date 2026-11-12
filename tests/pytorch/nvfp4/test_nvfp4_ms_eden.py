# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""MS-EDEN NVFP4 quantization with tiled, paired, resampled random Hadamard transforms.

Covers the columnwise MS-EDEN kernel against the pure-PyTorch reference of
Quartet II (arXiv:2601.22813) Algorithm 1, unbiasedness, rotation cancellation
in WGRAD, default-option invariance and sign-vector reproducibility.
"""

import pytest
import torch

import transformer_engine.pytorch as te
from transformer_engine.common.recipe import NVFP4BlockScaling
from transformer_engine.pytorch import NVFP4Quantizer
from transformer_engine.pytorch.quantization import RecipeState
from transformer_engine.pytorch.tensor import nvfp4_tensor

import ms_eden_reference as ref

recipe_available, reason_for_no_recipe = te.is_nvfp4_available(return_reason=True)


def _ms_eden_kernel_available() -> bool:
    """The MS-EDEN kernel uses UMMA/TMEM: SM100 family only."""
    if not torch.cuda.is_available():
        return False
    major, minor = torch.cuda.get_device_capability()
    return 100 <= major * 10 + minor <= 110


requires_nvfp4 = pytest.mark.skipif(not recipe_available, reason=reason_for_no_recipe)
requires_ms_eden_kernel = pytest.mark.skipif(
    not (recipe_available and _ms_eden_kernel_available()),
    reason="MS-EDEN quantization requires NVFP4 support on the SM100 family.",
)


@pytest.fixture(autouse=True)
def _reset_rht_step():
    """Every test starts and ends at step 0 of the resampled-RHT schedule."""
    if recipe_available:
        te.nvfp4_rht_set_step(0)
    yield
    if recipe_available:
        te.nvfp4_rht_set_step(0)


def _make_quantizer(rht_dim=128, ms_eden=True, policy="static", seed=0, interval=1, **kwargs):
    quantizer = NVFP4Quantizer(
        rowwise=kwargs.pop("rowwise", False),
        columnwise=True,
        with_amax_reduction=False,
        with_rht=True,
        with_post_rht_amax=True,
        stochastic_rounding=kwargs.pop("stochastic_rounding", False),
        rht_dim=rht_dim,
        ms_eden=ms_eden,
        rht_sign_policy=policy,
        rht_sign_seed=seed,
        rht_resample_interval=interval,
        **kwargs,
    )
    quantizer.optimize_for_gemm = False  # compact scales
    return quantizer


def _columnwise(out, shape):
    """FP4 values [C, R], E4M3 scale values [C, R / 16] and amax of the columnwise copy."""
    rows, cols = shape
    codes = ref.unpack_fp4(out._columnwise_data.view(dtype=torch.uint8).reshape(cols, rows // 2))
    values = ref.fp4_codes_to_values(codes)
    scales = ref.e4m3_decode(out._columnwise_scale_inv.view(dtype=torch.uint8)[:cols, : rows // 16])
    return values, scales, out._amax_columnwise.reshape(()).clone()


def _rowwise(out, shape):
    """FP4 values [R, C], E4M3 scale values [R, C / 16] and amax of the rowwise copy."""
    rows, cols = shape
    codes = ref.unpack_fp4(out._rowwise_data.view(dtype=torch.uint8).reshape(rows, cols // 2))
    values = ref.fp4_codes_to_values(codes)
    scales = ref.e4m3_decode(out._rowwise_scale_inv.view(dtype=torch.uint8)[:rows, : cols // 16])
    return values, scales, out._amax_rowwise.reshape(()).clone()


def _dequantize_and_unrotate(out, shape, sign_bits):
    """Inverse-rotated dequantized columnwise copy, [C, R] float64."""
    values, scales, amax = _columnwise(out, shape)
    return ref.rht_rotate_inverse(ref.dequantize_columnwise(values, scales, amax), sign_bits)


def _grid_input(rows, cols, seed):
    """BF16-exact inputs whose rotation is exact in FP32 before the final scaling."""
    generator = torch.Generator(device="cuda").manual_seed(seed)
    ints = torch.randint(-127, 128, (rows, cols), device="cuda", generator=generator)
    return (ints.to(torch.float32) / 32.0).to(torch.bfloat16)


def _gaussian_input(rows, cols, seed):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    return torch.randn(rows, cols, device="cuda", generator=generator).to(torch.bfloat16)


# ---------------------------------------------------------------------------
# (a) Kernel versus the reference of Algorithm 1
# ---------------------------------------------------------------------------


@requires_ms_eden_kernel
@pytest.mark.parametrize("rht_dim", [16, 128])
@pytest.mark.parametrize("shape", [(128, 128), (256, 384), (1024, 256), (2048, 1024)])
def test_ms_eden_matches_reference_exact_grid(rht_dim, shape):
    """FP4 values and amax are exact; every scale is an E4M3 neighbour of S * scale."""
    x = _grid_input(*shape, seed=1234)
    quantizer = _make_quantizer(rht_dim=rht_dim)
    out = quantizer(x)
    values, scales, amax = _columnwise(out, shape)

    expected = ref.ms_eden_reference(x, quantizer.rht_sign_bits())
    torch.testing.assert_close(amax, expected["amax_stored"], rtol=0, atol=0)
    torch.testing.assert_close(values, expected["fp4_values"], rtol=0, atol=0)

    # The EDEN factor is a ratio of FP32 sums whose rounding depends on FMA
    # contraction, so allow a relative slack on the rounding target.
    target = expected["scale_targets"]
    lo, _ = ref.e4m3_neighbours(target * (1 - 1e-4))
    _, hi = ref.e4m3_neighbours(target * (1 + 1e-4))
    assert torch.all(scales >= lo) and torch.all(scales <= hi)
    # Headroom: the corrected scales must not saturate E4M3.
    assert target.max() < ref.E4M3_MAX


@requires_ms_eden_kernel
@pytest.mark.parametrize("rht_dim", [16, 128])
def test_ms_eden_matches_reference_gaussian(rht_dim):
    """Gaussian inputs: tensor-core accumulation order may differ in the last bit."""
    shape = (1024, 512)
    x = _gaussian_input(*shape, seed=7)
    quantizer = _make_quantizer(rht_dim=rht_dim)
    out = quantizer(x)
    values, scales, amax = _columnwise(out, shape)

    expected = ref.ms_eden_reference(x, quantizer.rht_sign_bits())
    torch.testing.assert_close(amax, expected["amax_stored"], rtol=1e-6, atol=0)
    mismatch = (values != expected["fp4_values"]).float().mean().item()
    assert mismatch < 1e-3, f"FP4 mismatch fraction {mismatch}"
    target = expected["scale_targets"]
    lo, _ = ref.e4m3_neighbours(target * (1 - 1e-3))
    _, hi = ref.e4m3_neighbours(target * (1 + 1e-3))
    in_range = ((scales >= lo) & (scales <= hi)).float().mean().item()
    assert in_range > 0.999, f"fraction of scales adjacent to the target: {in_range}"


@requires_ms_eden_kernel
@pytest.mark.parametrize("rht_dim", [16, 128])
def test_ms_eden_scale_rounding_is_unbiased(rht_dim):
    """The mean over stochastic-rounding draws of each scale matches S * scale."""
    shape = (256, 256)
    x = _grid_input(*shape, seed=99)
    quantizer = _make_quantizer(rht_dim=rht_dim)
    expected = ref.ms_eden_reference(x, quantizer.rht_sign_bits())
    target = expected["scale_targets"].to(torch.float64)

    num_draws = 400
    torch.cuda.manual_seed(2028)
    mean = torch.zeros_like(target)
    first_values = None
    distinct = False
    for _ in range(num_draws):
        values, scales, _ = _columnwise(quantizer(x), shape)
        if first_values is None:
            first_values, first_scales = values, scales
        else:
            # FP4 values are deterministic (round-to-nearest); only scales are random.
            torch.testing.assert_close(values, first_values, rtol=0, atol=0)
            distinct = distinct or not torch.equal(scales, first_scales)
        mean += scales.to(torch.float64)
    mean /= num_draws
    assert distinct, "Scale rounding never changed between draws."
    rel = (mean - target) / target
    # Per-scale standard error is at most (1/16) / 2 / sqrt(num_draws) ~= 0.16 %.
    # (This catches correlated draws across calls, e.g. reduced-round Philox.)
    assert rel.abs().max() < 0.012
    assert rel.mean().abs() < 5e-4


@requires_ms_eden_kernel
def test_ms_eden_zero_and_constant_inputs():
    """All-zero tensors and all-zero chunks must not produce NaN/Inf."""
    shape = (256, 128)
    quantizer = _make_quantizer()
    zeros = torch.zeros(shape, device="cuda", dtype=torch.bfloat16)
    values, scales, amax = _columnwise(quantizer(zeros), shape)
    assert amax.item() == 0.0
    assert torch.count_nonzero(values) == 0
    assert torch.isfinite(scales).all()

    x = _gaussian_input(*shape, seed=3)
    x[:128, :] = 0  # first rotation chunk of every column is zero
    out = quantizer(x)
    deq = _dequantize_and_unrotate(out, shape, quantizer.rht_sign_bits())
    assert torch.isfinite(deq).all()
    assert torch.count_nonzero(deq[:, :128]) == 0


# ---------------------------------------------------------------------------
# (b) Unbiasedness and MSE versus RHT + stochastic rounding
# ---------------------------------------------------------------------------


@requires_ms_eden_kernel
def test_ms_eden_is_unbiased_and_beats_stochastic_rounding():
    shape = (256, 128)
    x = _gaussian_input(*shape, seed=11)
    xt = x.to(torch.float64).t()
    signal = xt.pow(2).mean().item()

    ms_eden = _make_quantizer(rht_dim=128, policy="resampled", seed=5)
    rht_sr = NVFP4Quantizer(
        rowwise=False,
        columnwise=True,
        with_rht=True,
        with_post_rht_amax=True,
        stochastic_rounding=True,
    )
    rht_sr.optimize_for_gemm = False

    num_draws = 600
    torch.cuda.manual_seed(31)
    sums = {"ms_eden": torch.zeros_like(xt), "rht_sr": torch.zeros_like(xt)}
    mse = {"ms_eden": 0.0, "rht_sr": 0.0}
    for draw in range(num_draws):
        te.nvfp4_rht_set_step(draw)  # new omega_RHT; omega_SR advances with the CUDA generator
        for name, quantizer in (("ms_eden", ms_eden), ("rht_sr", rht_sr)):
            estimate = _dequantize_and_unrotate(quantizer(x), shape, quantizer.rht_sign_bits())
            sums[name] += estimate
            mse[name] += (estimate - xt).pow(2).mean().item() / num_draws

    rel_mse = {name: value / signal for name, value in mse.items()}
    # Reference (CPU) values for Gaussian inputs: MS-EDEN ~0.0104, RHT-16 + SR ~0.019.
    assert rel_mse["ms_eden"] < 0.8 * rel_mse["rht_sr"], rel_mse
    assert rel_mse["ms_eden"] < 0.02, rel_mse

    # Unbiased: the error of the mean is pure sampling noise, i.e. ~ mse / num_draws.
    mean = sums["ms_eden"] / num_draws
    bias_mse = (mean - xt).pow(2).mean().item() / signal
    assert bias_mse < 1.5 * rel_mse["ms_eden"] / num_draws, (bias_mse, rel_mse)
    projection = (mean * xt).sum().item() / xt.pow(2).sum().item()
    assert abs(projection - 1.0) < 1e-3, projection


# ---------------------------------------------------------------------------
# (c) Paired rotation: the rotation cancels in WGRAD
# ---------------------------------------------------------------------------


def _recipe_quantizers(**recipe_kwargs):
    """Forward-input and output-gradient quantizers built by the recipe."""
    recipe = NVFP4BlockScaling(**recipe_kwargs)
    forward = RecipeState.create(recipe, mode="forward", num_quantizers=3).make_quantizers()
    backward = RecipeState.create(recipe, mode="backward", num_quantizers=2).make_quantizers()
    return forward[0], backward[0]


@requires_nvfp4
@pytest.mark.parametrize("rht_dim", [16, 128])
@pytest.mark.parametrize("policy", ["static", "resampled"])
def test_wgrad_operands_share_sign_vector(rht_dim, policy):
    input_quantizer, grad_quantizer = _recipe_quantizers(
        grad_bias_correction="ms_eden",
        rht_dim=rht_dim,
        rht_sign_policy=policy,
        rht_sign_seed=17,
        rht_resample_interval=2,
    )
    for step in range(5):
        te.nvfp4_rht_set_step(step)
        input_quantizer.refresh_rht_signs()
        grad_quantizer.refresh_rht_signs()
        assert input_quantizer.rht_sign_bits() == grad_quantizer.rht_sign_bits()
        assert torch.equal(input_quantizer.rht_matrix, grad_quantizer.rht_matrix)
        assert tuple(input_quantizer.rht_matrix.shape) == (rht_dim, 16)
        # The matrix handed to the kernel encodes exactly these signs.
        signs = ref.sign_vector(input_quantizer.rht_sign_bits(), device="cuda")
        blocks = ref.hadamard_matrix(16).to("cuda").repeat(rht_dim // 16, 1) * 0.25
        torch.testing.assert_close(
            input_quantizer.rht_matrix.to(torch.float64), signs.unsqueeze(1) * blocks
        )


@requires_nvfp4
@pytest.mark.parametrize("rht_dim", [16, 128])
def test_paired_rotation_cancels_in_high_precision_wgrad(rht_dim):
    """dW from paired-rotated high-precision operands equals the un-rotated product."""
    tokens, in_features, out_features = 256, 96, 64
    generator = torch.Generator().manual_seed(0)
    x = torch.randn(tokens, in_features, generator=generator, dtype=torch.float64)
    dy = torch.randn(tokens, out_features, generator=generator, dtype=torch.float64)
    expected = dy.t() @ x

    input_quantizer, grad_quantizer = _recipe_quantizers(
        grad_bias_correction="ms_eden", rht_dim=rht_dim, rht_sign_policy="resampled"
    )
    te.nvfp4_rht_set_step(3)
    x_rot = ref.rht_rotate_exact(x.t().contiguous(), input_quantizer.rht_sign_bits())
    dy_rot = ref.rht_rotate_exact(dy.t().contiguous(), grad_quantizer.rht_sign_bits())
    torch.testing.assert_close(dy_rot @ x_rot.t(), expected, rtol=1e-10, atol=1e-10)

    # Negative control: operands rotated with different sign vectors do not cancel.
    te.nvfp4_rht_set_step(4)
    dy_other = ref.rht_rotate_exact(dy.t().contiguous(), grad_quantizer.rht_sign_bits())
    error = (dy_other @ x_rot.t() - expected).norm() / expected.norm()
    assert error > 0.5


@requires_ms_eden_kernel
@pytest.mark.parametrize("rht_dim", [16, 128])
@pytest.mark.parametrize("scope", ["gradients", "all"])
def test_paired_rotation_cancels_in_quantized_wgrad(rht_dim, scope):
    """dW from the quantized, rotated columnwise copies approximates dY^T X."""
    tokens, in_features, out_features = 512, 256, 128
    x = _gaussian_input(tokens, in_features, seed=21)
    dy = _gaussian_input(tokens, out_features, seed=22)
    expected = dy.to(torch.float64).t() @ x.to(torch.float64)

    input_quantizer, grad_quantizer = _recipe_quantizers(
        grad_bias_correction="ms_eden",
        ms_eden_scope=scope,
        rht_dim=rht_dim,
        rht_sign_policy="resampled",
    )
    assert grad_quantizer.ms_eden and input_quantizer.ms_eden == (scope == "all")
    for quantizer in (input_quantizer, grad_quantizer):
        quantizer.set_usage(rowwise=True, columnwise=True)
        quantizer.optimize_for_gemm = False

    def rotated_dequantized(quantizer, tensor):
        values, scales, amax = _columnwise(quantizer(tensor), tuple(tensor.shape))
        return ref.dequantize_columnwise(values, scales, amax)

    te.nvfp4_rht_set_step(1)
    x_q = rotated_dequantized(input_quantizer, x)
    dy_q = rotated_dequantized(grad_quantizer, dy)
    error = (dy_q @ x_q.t() - expected).norm() / expected.norm()
    assert error < 0.25, error

    # Negative control: advancing the schedule between the two operands breaks pairing.
    te.nvfp4_rht_set_step(2)
    dy_unpaired = rotated_dequantized(grad_quantizer, dy)
    error_unpaired = (dy_unpaired @ x_q.t() - expected).norm() / expected.norm()
    assert error_unpaired > 0.7, error_unpaired


@requires_ms_eden_kernel
@pytest.mark.parametrize("rht_dim", [16, 128])
def test_linear_backward_with_ms_eden(rht_dim):
    """End to end: te.Linear weight gradients stay aligned with the BF16 gradients."""
    tokens, in_features, out_features = 512, 256, 128
    torch.manual_seed(0)
    module = te.Linear(in_features, out_features, bias=False, params_dtype=torch.bfloat16).cuda()
    x = _gaussian_input(tokens, in_features, seed=5)
    target = _gaussian_input(tokens, out_features, seed=6)

    def weight_grad(recipe):
        module.zero_grad(set_to_none=True)
        with te.autocast(enabled=recipe is not None, recipe=recipe):
            y = module(x)
        (y.float() * target.float()).sum().backward()
        return module.weight.grad.detach().to(torch.float64).clone()

    reference = weight_grad(None)
    recipe = NVFP4BlockScaling(
        grad_bias_correction="ms_eden", rht_dim=rht_dim, rht_sign_policy="resampled"
    )
    grad = weight_grad(recipe)
    cosine = (grad * reference).sum() / (grad.norm() * reference.norm())
    assert cosine > 0.95, cosine


# ---------------------------------------------------------------------------
# (d) Defaults are unchanged
# ---------------------------------------------------------------------------


@requires_nvfp4
def test_default_recipe_and_quantizer_are_unchanged():
    recipe = NVFP4BlockScaling()
    assert recipe.grad_bias_correction == "sr"
    assert recipe.rht_dim == 16
    assert recipe.rht_sign_policy == "static"
    assert recipe.dgrad_rht is False
    assert recipe.ms_eden_scope == "gradients" and recipe.ms_eden_grid_max == 6.0
    for qparams in (recipe.fp4_quant_fwd_inp, recipe.fp4_quant_bwd_grad):
        assert qparams.rht_dim == 16 and not qparams.ms_eden
        assert "ms_eden" not in repr(qparams) and "rht_dim" not in repr(qparams)
    assert "grad_bias_correction" not in repr(recipe)

    input_quantizer, grad_quantizer = _recipe_quantizers()
    for quantizer in (input_quantizer, grad_quantizer):
        assert quantizer.rht_dim == 16
        assert not quantizer.ms_eden
        assert quantizer.rht_sign_policy == "static"
        assert not quantizer.rht_dynamic_signs
        device = torch.cuda.current_device()
        assert quantizer.rht_matrix is nvfp4_tensor.get_rht_matrix(True, device)
        assert quantizer.rht_matrix_random_sign_mask_t == nvfp4_tensor.get_random_sign_mask_for_rht(
            True, device
        )
        # The schedule does not affect static quantizers.
        te.nvfp4_rht_set_step(123)
        quantizer.refresh_rht_signs()
        assert quantizer.rht_matrix is nvfp4_tensor.get_rht_matrix(True, device)
    assert grad_quantizer.stochastic_rounding and not input_quantizer.stochastic_rounding


@requires_nvfp4
def test_default_quantizer_output_ignores_rht_schedule():
    """Default (dim-16, static, SR-free) quantization is bit-identical across steps and
    equal to a quantizer constructed with the new options left at their defaults."""
    x = _gaussian_input(256, 128, seed=8)

    def snapshot(quantizer):
        quantizer.optimize_for_gemm = False
        out = quantizer(x)
        return [
            t.clone()
            for t in (
                out._rowwise_data,
                out._rowwise_scale_inv,
                out._columnwise_data,
                out._columnwise_scale_inv,
                out._amax_rowwise,
                out._amax_columnwise,
            )
        ]

    base = snapshot(NVFP4Quantizer(with_rht=True, with_post_rht_amax=True))
    te.nvfp4_rht_set_step(7)
    explicit = snapshot(
        NVFP4Quantizer(
            with_rht=True,
            with_post_rht_amax=True,
            rht_dim=16,
            ms_eden=False,
            rht_sign_policy="static",
            rht_sign_seed=99,
        )
    )
    for lhs, rhs in zip(base, explicit):
        assert torch.equal(lhs, rhs)


@requires_nvfp4
def test_static_dim16_seeded_matrix_matches_legacy_construction():
    """The generic matrix builder reproduces the legacy 16x16 RHT matrix."""
    device = torch.cuda.current_device()
    mask = nvfp4_tensor.get_random_sign_mask_for_rht(True, device)
    bits = tuple((mask >> i) & 1 for i in range(16))
    assert torch.equal(
        nvfp4_tensor.get_rht_matrix_from_sign_bits(bits, device),
        nvfp4_tensor.get_rht_matrix(True, device),
    )


@requires_nvfp4
def test_unsupported_configurations_raise():
    with pytest.raises(ValueError):
        NVFP4BlockScaling(grad_bias_correction="ms_eden", disable_rht=True)
    with pytest.raises(ValueError):
        NVFP4BlockScaling(grad_bias_correction="ms_eden", nvfp4_4over6="all")
    with pytest.raises(ValueError):
        NVFP4BlockScaling(rht_dim=128, nvfp4_4over6="activations")
    with pytest.raises(ValueError):
        NVFP4BlockScaling(dgrad_rht=True, disable_rht=True)
    with pytest.raises(ValueError):
        NVFP4Quantizer(with_rht=True, ms_eden=True, ms_eden_grid_max=0.0)
    with pytest.raises(ValueError):
        NVFP4Quantizer(with_rht=True, rht_dim=128, with_2d_quantization=True)
    with pytest.raises(ValueError):
        NVFP4Quantizer(with_rht=True, ms_eden=True, nvfp4_use_4over6=True)
    with pytest.raises(ValueError):
        NVFP4Quantizer(with_rht=False, ms_eden=True)
    with pytest.raises(ValueError):
        NVFP4Quantizer(with_rht=True, rht_sign_policy="per_call")


@requires_ms_eden_kernel
def test_ms_eden_rejects_unaligned_shapes():
    quantizer = _make_quantizer()
    x = torch.randn(192, 128, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(RuntimeError):
        quantizer(x)


@requires_ms_eden_kernel
def test_ms_eden_rowwise_copy_is_plain_nvfp4():
    """The un-rotated rowwise copy equals the one of a quantizer without RHT."""
    x = _gaussian_input(256, 128, seed=13)
    ms_eden = _make_quantizer(rowwise=True)
    plain = NVFP4Quantizer(rowwise=True, columnwise=False)
    plain.optimize_for_gemm = False
    out, expected = ms_eden(x), plain(x)
    assert torch.equal(out._rowwise_data, expected._rowwise_data)
    assert torch.equal(out._rowwise_scale_inv, expected._rowwise_scale_inv)
    assert torch.equal(out._amax_rowwise, expected._amax_rowwise)


# ---------------------------------------------------------------------------
# (e) Sign-vector schedule and RNG reproducibility
# ---------------------------------------------------------------------------


@requires_nvfp4
@pytest.mark.parametrize("rht_dim", [16, 128])
def test_sign_bits_are_reproducible(rht_dim):
    bits = nvfp4_tensor.get_rht_sign_bits(rht_dim, 3, 5)
    assert len(bits) == rht_dim and set(bits) <= {0, 1}
    nvfp4_tensor.get_rht_sign_bits.cache_clear()
    assert nvfp4_tensor.get_rht_sign_bits(rht_dim, 3, 5) == bits
    assert nvfp4_tensor.get_rht_sign_bits(rht_dim, 3, 6) != bits
    assert nvfp4_tensor.get_rht_sign_bits(rht_dim, 4, 5) != bits
    # Roughly balanced signs.
    many = [b for g in range(64) for b in nvfp4_tensor.get_rht_sign_bits(rht_dim, 0, g)]
    assert 0.4 < sum(many) / len(many) < 0.6


@requires_nvfp4
def test_resample_interval_and_step_api():
    quantizer = _make_quantizer(policy="resampled", seed=1, interval=3)
    assert quantizer.rht_dynamic_signs
    seen = []
    for step in range(9):
        assert te.nvfp4_rht_get_step() == step
        seen.append(quantizer.rht_sign_bits())
        te.nvfp4_rht_step()
    assert seen[0] == seen[1] == seen[2]
    assert seen[3] == seen[4] == seen[5]
    assert len({seen[0], seen[3], seen[6]}) == 3
    te.nvfp4_rht_set_step(4)
    assert quantizer.rht_sign_bits() == seen[3]

    static = _make_quantizer(policy="static", seed=1)
    assert not static.rht_dynamic_signs
    te.nvfp4_rht_set_step(0)
    bits = static.rht_sign_bits()
    te.nvfp4_rht_set_step(50)
    assert static.rht_sign_bits() == bits

    # A copy rotates identically.
    copy = quantizer.copy()
    assert copy.rht_sign_bits() == quantizer.rht_sign_bits()
    assert copy.ms_eden and copy.rht_dim == 128 and copy.rht_resample_interval == 3


@requires_nvfp4
def test_auto_step_follows_optimizer():
    parameter = torch.nn.Parameter(torch.zeros(4, device="cuda"))
    optimizer = torch.optim.SGD([parameter], lr=0.1)
    try:
        te.nvfp4_rht_enable_auto_step(True)
        for expected in range(1, 4):
            parameter.grad = torch.ones_like(parameter)
            optimizer.step()
            assert te.nvfp4_rht_get_step() == expected
    finally:
        te.nvfp4_rht_enable_auto_step(False)
    optimizer.step()
    assert te.nvfp4_rht_get_step() == 3


@requires_ms_eden_kernel
@pytest.mark.parametrize("rht_dim", [16, 128])
def test_ms_eden_is_reproducible_for_fixed_seeds(rht_dim):
    """Same (omega_RHT, omega_SR) gives identical tensors; changing either changes them."""
    shape = (256, 256)
    x = _gaussian_input(*shape, seed=17)
    device = torch.cuda.current_device()

    def run(step, sr_seed, seed=9):
        quantizer = _make_quantizer(rht_dim=rht_dim, policy="resampled", seed=seed)
        te.nvfp4_rht_set_step(step)
        with torch.random.fork_rng(devices=[device]):
            torch.cuda.manual_seed(sr_seed)
            out = quantizer(x)
            torch.cuda.synchronize()
        return (
            out._columnwise_data.clone(),
            out._columnwise_scale_inv.clone(),
            out._amax_columnwise.clone(),
        )

    base = run(step=2, sr_seed=100)
    again = run(step=2, sr_seed=100)
    for lhs, rhs in zip(base, again):
        assert torch.equal(lhs, rhs)

    other_sr = run(step=2, sr_seed=101)
    assert torch.equal(other_sr[0], base[0])  # FP4 values do not depend on omega_SR
    assert torch.equal(other_sr[2], base[2])
    assert not torch.equal(other_sr[1], base[1])

    other_rht = run(step=3, sr_seed=100)
    assert not torch.equal(other_rht[0], base[0])
    other_seed = run(step=2, sr_seed=100, seed=10)
    assert not torch.equal(other_seed[0], base[0])


@requires_ms_eden_kernel
def test_ms_eden_rng_allocation_independence():
    """The RNG state and workspace must stay alive until the kernels consumed them."""
    device = torch.cuda.current_device()
    was_enabled = torch.are_deterministic_algorithms_enabled()
    was_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    was_fill = torch.utils.deterministic.fill_uninitialized_memory
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.utils.deterministic.fill_uninitialized_memory = True
        x = _gaussian_input(256, 128, seed=2027)
        quantizer = _make_quantizer(rowwise=True, stochastic_rounding=True)

        def snapshot():
            with torch.random.fork_rng(devices=[device]):
                torch.cuda.manual_seed(77)
                out = quantizer(x)
                torch.cuda.synchronize()
            return [
                t.clone()
                for t in (
                    out._rowwise_data,
                    out._rowwise_scale_inv,
                    out._columnwise_data,
                    out._columnwise_scale_inv,
                    out._amax_columnwise,
                )
            ]

        first = snapshot()
        junk = [torch.empty(2, dtype=torch.int64, device="cuda") for _ in range(64)]
        second = snapshot()
        del junk
        for lhs, rhs in zip(first, second):
            assert torch.equal(lhs, rhs)
    finally:
        torch.utils.deterministic.fill_uninitialized_memory = was_fill
        torch.use_deterministic_algorithms(was_enabled, warn_only=was_warn_only)


# ---------------------------------------------------------------------------
# Range-aligned rotated quantization (round-to-nearest / element-level SR)
# ---------------------------------------------------------------------------


@requires_ms_eden_kernel
@pytest.mark.parametrize("shape", [(128, 128), (256, 384), (2048, 1024)])
def test_aligned_rtn_matches_reference_exact_grid(shape):
    """Two-pass rotated round-to-nearest quantization: values, scales and amax exact.

    Dimension 128 only: columnwise dimension-16 copies without MS-EDEN use the
    pre-existing RHT kernels (covered by their own tests). The tiled dimension-16
    aligned mode is covered by the rowwise tests below.
    """
    rht_dim = 128
    x = _grid_input(*shape, seed=4321)
    quantizer = _make_quantizer(rht_dim=rht_dim, ms_eden=False, seed=3, policy="resampled")
    out = quantizer(x)
    values, scales, amax = _columnwise(out, shape)
    expected = ref.aligned_reference(x, quantizer.rht_sign_bits())
    torch.testing.assert_close(amax, expected["amax"], rtol=0, atol=0)
    torch.testing.assert_close(scales, expected["scales"], rtol=0, atol=0)
    torch.testing.assert_close(values, expected["fp4_values"], rtol=0, atol=0)


@requires_ms_eden_kernel
def test_aligned_stochastic_rounding_is_adjacent_and_unbiased():
    """Dimension-128 RHT with element-level SR."""
    rht_dim = 128
    shape = (256, 256)
    x = _grid_input(*shape, seed=55)
    quantizer = _make_quantizer(
        rht_dim=rht_dim, ms_eden=False, seed=3, policy="resampled", stochastic_rounding=True
    )
    expected = ref.aligned_reference(x, quantizer.rht_sign_bits())
    scaled = expected["scaled"].clamp(-6.0, 6.0).to(torch.float64)
    grid = torch.tensor(
        sorted({-m for m in ref.FP4_MAGNITUDES} | set(ref.FP4_MAGNITUDES)),
        dtype=torch.float64,
        device="cuda",
    )
    hi_idx = torch.searchsorted(grid, scaled.contiguous()).clamp(max=grid.numel() - 1)
    hi = grid[hi_idx]
    lo = torch.where(hi == scaled, hi, grid[(hi_idx - 1).clamp(min=0)])

    num_draws = 300
    torch.cuda.manual_seed(77)
    mean = torch.zeros_like(scaled)
    for _ in range(num_draws):
        values, scales, amax = _columnwise(quantizer(x), shape)
        torch.testing.assert_close(scales, expected["scales"], rtol=0, atol=0)
        torch.testing.assert_close(amax, expected["amax"], rtol=0, atol=0)
        values = values.to(torch.float64)
        assert torch.all((values == lo) | (values == hi))
        mean += values / num_draws
    # Largest FP4 gap is 2, so the standard error of the mean is at most 1 / sqrt(draws).
    assert (mean - scaled).abs().max() < 6.0 / num_draws**0.5
    assert (mean - scaled).mean().abs() < 2e-3


@requires_ms_eden_kernel
def test_rht128_stochastic_rounding_is_unbiased():
    """Mosaic 'Update 2': dimension-128 RHT with element-level stochastic rounding."""
    shape = (256, 128)
    x = _gaussian_input(*shape, seed=12)
    xt = x.to(torch.float64).t()
    signal = xt.pow(2).mean().item()
    quantizer = _make_quantizer(
        rht_dim=128, ms_eden=False, policy="resampled", seed=6, stochastic_rounding=True
    )
    num_draws = 600
    torch.cuda.manual_seed(32)
    total = torch.zeros_like(xt)
    mse = 0.0
    for draw in range(num_draws):
        te.nvfp4_rht_set_step(draw)
        estimate = _dequantize_and_unrotate(quantizer(x), shape, quantizer.rht_sign_bits())
        total += estimate
        mse += (estimate - xt).pow(2).mean().item() / num_draws
    rel_mse = mse / signal
    bias_mse = (total / num_draws - xt).pow(2).mean().item() / signal
    assert rel_mse < 0.03, rel_mse
    assert bias_mse < 1.5 * rel_mse / num_draws, (bias_mse, rel_mse)


# ---------------------------------------------------------------------------
# MS-EDEN grid parameter
# ---------------------------------------------------------------------------


@requires_ms_eden_kernel
@pytest.mark.parametrize("grid_max", [5.5, 6.0, 6.0 / ((17.0 / 16.0) * 0.93), 7.0])
def test_ms_eden_grid_max_matches_reference(grid_max):
    shape = (256, 384)
    x = _grid_input(*shape, seed=808)
    quantizer = _make_quantizer(rht_dim=128, ms_eden_grid_max=grid_max)
    values, scales, amax = _columnwise(quantizer(x), shape)
    expected = ref.ms_eden_reference(x, quantizer.rht_sign_bits(), grid_max=grid_max)
    torch.testing.assert_close(amax, expected["amax_stored"], rtol=0, atol=0)
    torch.testing.assert_close(values, expected["fp4_values"], rtol=0, atol=0)
    target = expected["scale_targets"]
    lo, _ = ref.e4m3_neighbours(target * (1 - 1e-4))
    _, hi = ref.e4m3_neighbours(target * (1 + 1e-4))
    assert torch.all(scales >= lo) and torch.all(scales <= hi)
    assert target.max() < ref.E4M3_MAX


# ---------------------------------------------------------------------------
# MS-EDEN operand scope
# ---------------------------------------------------------------------------


@requires_nvfp4
@pytest.mark.parametrize("scope", ["gradients", "all"])
@pytest.mark.parametrize("rht_dim", [16, 128])
def test_ms_eden_scope_selects_operands(scope, rht_dim):
    recipe = NVFP4BlockScaling(
        grad_bias_correction="ms_eden", ms_eden_scope=scope, rht_dim=rht_dim, dgrad_rht=True
    )
    forward = RecipeState.create(recipe, mode="forward", num_quantizers=3).make_quantizers()
    backward = RecipeState.create(recipe, mode="backward", num_quantizers=2).make_quantizers()
    input_quantizer, weight_quantizer, grad_quantizer = forward[0], forward[1], backward[0]
    assert grad_quantizer.ms_eden and grad_quantizer.dgrad_rht
    assert input_quantizer.ms_eden == (scope == "all")
    assert not input_quantizer.dgrad_rht and not input_quantizer.rht_rowwise
    assert not weight_quantizer.with_rht
    # The rotated DGRAD weight copy follows the scope and uses the DGRAD sign stream.
    grad_quantizer.set_rowwise_rht(True)
    dgrad_weight_quantizer = grad_quantizer.dgrad_weight_quantizer()
    assert dgrad_weight_quantizer.ms_eden == (scope == "all")
    assert dgrad_weight_quantizer.rht_dim == rht_dim
    assert dgrad_weight_quantizer.rht_sign_bits() == grad_quantizer.rht_sign_bits(rowwise=True)
    assert torch.equal(dgrad_weight_quantizer.rht_matrix, grad_quantizer.rht_matrix_rowwise)
    # WGRAD and DGRAD use independent sign vectors.
    assert grad_quantizer.rht_sign_bits(rowwise=True) != nvfp4_tensor.get_rht_sign_bits(
        rht_dim, recipe.rht_sign_seed, 0
    )


# ---------------------------------------------------------------------------
# DGRAD paired rotation
# ---------------------------------------------------------------------------


def _dgrad_quantizers(rht_dim, weight_source="high_precision", **recipe_kwargs):
    recipe = NVFP4BlockScaling(
        rht_dim=rht_dim,
        dgrad_rht=True,
        dgrad_rht_weight_source=weight_source,
        rht_sign_policy="resampled",
        **recipe_kwargs,
    )
    grad_quantizer = RecipeState.create(
        recipe, mode="backward", num_quantizers=2
    ).make_quantizers()[0]
    grad_quantizer.optimize_for_gemm = False
    grad_quantizer.set_usage(rowwise=True, columnwise=True)
    grad_quantizer.set_rowwise_rht(True)
    weight_quantizer = grad_quantizer.dgrad_weight_quantizer()
    weight_quantizer.optimize_for_gemm = False
    return grad_quantizer, weight_quantizer


@requires_ms_eden_kernel
@pytest.mark.parametrize("rht_dim", [16, 128])
@pytest.mark.parametrize("shape", [(128, 128), (384, 256), (1024, 2048)])
def test_rowwise_ms_eden_matches_reference_exact_grid(rht_dim, shape):
    """Rowwise copy rotated along the last dimension (row-major kernel input)."""
    x = _grid_input(*shape, seed=2468)
    grad_quantizer, _ = _dgrad_quantizers(rht_dim, grad_bias_correction="ms_eden")
    out = grad_quantizer(x)
    # The rowwise copy of x equals the columnwise copy of x^T.
    transposed = x.t().contiguous()
    expected = ref.ms_eden_reference(transposed, grad_quantizer.rht_sign_bits(rowwise=True))
    values, scales, amax = _rowwise(out, shape)
    torch.testing.assert_close(amax, expected["amax_stored"], rtol=0, atol=0)
    torch.testing.assert_close(values, expected["fp4_values"], rtol=0, atol=0)
    target = expected["scale_targets"]
    lo, _ = ref.e4m3_neighbours(target * (1 - 1e-4))
    _, hi = ref.e4m3_neighbours(target * (1 + 1e-4))
    assert torch.all(scales >= lo) and torch.all(scales <= hi)
    # The columnwise copy of the same call is still rotated along the first dimension.
    expected_col = ref.ms_eden_reference(x, grad_quantizer.rht_sign_bits())
    values_col, _, amax_col = _columnwise(out, shape)
    torch.testing.assert_close(amax_col, expected_col["amax_stored"], rtol=0, atol=0)
    torch.testing.assert_close(values_col, expected_col["fp4_values"], rtol=0, atol=0)


@requires_ms_eden_kernel
@pytest.mark.parametrize("rht_dim", [16, 128])
def test_rowwise_aligned_matches_reference_exact_grid(rht_dim):
    shape = (384, 256)
    x = _grid_input(*shape, seed=1357)
    grad_quantizer, _ = _dgrad_quantizers(rht_dim, disable_stochastic_rounding=True)
    assert not grad_quantizer.ms_eden and not grad_quantizer.stochastic_rounding
    values, scales, amax = _rowwise(grad_quantizer(x), shape)
    expected = ref.aligned_reference(x.t().contiguous(), grad_quantizer.rht_sign_bits(rowwise=True))
    torch.testing.assert_close(amax, expected["amax"], rtol=0, atol=0)
    torch.testing.assert_close(scales, expected["scales"], rtol=0, atol=0)
    torch.testing.assert_close(values, expected["fp4_values"], rtol=0, atol=0)


@requires_ms_eden_kernel
@pytest.mark.parametrize("rht_dim", [16, 128])
@pytest.mark.parametrize("bias_correction", ["sr", "ms_eden"])
@pytest.mark.parametrize("scope", ["gradients", "all"])
@pytest.mark.parametrize("weight_source", ["high_precision", "dequantized"])
def test_paired_rotation_cancels_in_quantized_dgrad(rht_dim, bias_correction, scope, weight_source):
    """dX from the rotated rowwise dY copy and the rotated columnwise W copy ~= dY W."""
    tokens, in_features, out_features = 512, 256, 384
    dy = _gaussian_input(tokens, out_features, seed=41)
    weight = _gaussian_input(out_features, in_features, seed=42)
    grad_quantizer, _ = _dgrad_quantizers(
        rht_dim, weight_source, grad_bias_correction=bias_correction, ms_eden_scope=scope
    )
    forward_weight_quantizer = NVFP4Quantizer(rowwise=True, columnwise=False)
    forward_weight = forward_weight_quantizer(weight)
    weight_reference = (
        forward_weight.dequantize(dtype=torch.bfloat16)
        if weight_source == "dequantized"
        else weight
    )
    expected = dy.to(torch.float64) @ weight_reference.to(torch.float64)

    te.nvfp4_rht_set_step(1)
    dy_values, dy_scales, dy_amax = _rowwise(grad_quantizer(dy), tuple(dy.shape))
    dy_q = ref.dequantize_columnwise(dy_values, dy_scales, dy_amax)  # [tokens, out], rotated
    rotated_weight = nvfp4_tensor.nvfp4_dgrad_rotated_weight(grad_quantizer, forward_weight, weight)
    w_values, w_scales, w_amax = _columnwise(rotated_weight, tuple(weight.shape))
    w_q = ref.dequantize_columnwise(w_values, w_scales, w_amax)  # [in, out], rotated
    error = (dy_q @ w_q.t() - expected).norm() / expected.norm()
    assert error < 0.25, error

    # Negative control: a weight copy from another sign generation does not cancel.
    te.nvfp4_rht_set_step(2)
    unpaired = nvfp4_tensor.nvfp4_dgrad_rotated_weight(grad_quantizer, forward_weight, weight)
    u_values, u_scales, u_amax = _columnwise(unpaired, tuple(weight.shape))
    u_q = ref.dequantize_columnwise(u_values, u_scales, u_amax)
    error_unpaired = (dy_q @ u_q.t() - expected).norm() / expected.norm()
    assert error_unpaired > 0.7, error_unpaired


# ---------------------------------------------------------------------------
# te.Linear end to end
# ---------------------------------------------------------------------------


def _linear_grads(module, x, target, recipe):
    module.zero_grad(set_to_none=True)
    inp = x.clone().requires_grad_(True)
    with te.autocast(enabled=recipe is not None, recipe=recipe):
        y = module(inp)
    (y.float() * target.float()).sum().backward()
    return (
        module.weight.grad.detach().to(torch.float64).clone(),
        inp.grad.detach().to(torch.float64).clone(),
    )


@requires_nvfp4
@pytest.mark.parametrize(
    "rht_dim,dtype,tolerance",
    [
        # FP32 operands: TE runs pure-FP32 GEMMs with TF32 tensor cores (10-bit operand
        # mantissas), which bounds the agreement of two GEMMs on different operands.
        # (The dimension-16 forward path only accepts BF16 inputs, so FP32 is exercised
        # with dimension 128.) The exact FP64 identity is checked in
        # test_linear_rotation_operands_cancel_in_fp64 below.
        (128, torch.float32, 2e-3),
        # BF16 operands: tolerance of BF16 rounding of the rotated operands and outputs.
        (16, torch.bfloat16, 3e-2),
        (128, torch.bfloat16, 3e-2),
    ],
)
@pytest.mark.parametrize("policy", ["static", "resampled"])
def test_linear_rotation_cancels_with_high_precision_operands(
    rht_dim, dtype, tolerance, policy, monkeypatch
):
    """With rotation on both backward GEMMs and unquantized operands, WGRAD and DGRAD
    equal the un-rotated results up to floating-point error."""
    tokens, in_features, out_features = 256, 256, 128
    torch.manual_seed(0)
    module = te.Linear(in_features, out_features, bias=False, params_dtype=dtype).cuda()
    x = torch.randn(tokens, in_features, device="cuda").to(dtype)
    target = torch.randn(tokens, out_features, device="cuda").to(dtype)
    kwargs = dict(
        backward_override="high_precision",
        rht_dim=rht_dim,
        rht_sign_policy=policy,
        dgrad_rht=True,
    )
    te.nvfp4_rht_set_step(3)
    monkeypatch.setenv("NVTE_NVFP4_RHT_HIGH_PRECISION_DEBUG", "0")
    wgrad_ref, dgrad_ref = _linear_grads(module, x, target, NVFP4BlockScaling(**kwargs))
    monkeypatch.setenv("NVTE_NVFP4_RHT_HIGH_PRECISION_DEBUG", "1")
    wgrad, dgrad = _linear_grads(module, x, target, NVFP4BlockScaling(**kwargs))
    # The rotated path really ran (results differ in the last bits) ...
    assert not torch.equal(wgrad, wgrad_ref)
    assert not torch.equal(dgrad, dgrad_ref)
    # ... and both gradients agree with the un-rotated ones.
    for grad, reference in ((wgrad, wgrad_ref), (dgrad, dgrad_ref)):
        error = (grad - reference).norm() / reference.norm()
        assert error < tolerance, error
        torch.testing.assert_close(
            grad, reference, rtol=0, atol=10 * tolerance * reference.abs().max().item()
        )


@requires_nvfp4
@pytest.mark.parametrize("rht_dim", [16, 128])
@pytest.mark.parametrize("policy", ["static", "resampled"])
def test_linear_rotation_operands_cancel_in_fp64(rht_dim, policy):
    """The rotations te.Linear applies in its debug mode (same helper, same matrices
    as handed to the kernels) cancel exactly: WGRAD and DGRAD in FP64."""
    tokens, in_features, out_features = 256, 384, 128
    generator = torch.Generator(device="cuda").manual_seed(0)
    x = torch.randn(tokens, in_features, device="cuda", generator=generator, dtype=torch.float64)
    dy = torch.randn(tokens, out_features, device="cuda", generator=generator, dtype=torch.float64)
    weight = torch.randn(
        out_features, in_features, device="cuda", generator=generator, dtype=torch.float64
    )
    recipe = NVFP4BlockScaling(rht_dim=rht_dim, rht_sign_policy=policy, dgrad_rht=True)
    forward = RecipeState.create(recipe, mode="forward", num_quantizers=3).make_quantizers()
    backward = RecipeState.create(recipe, mode="backward", num_quantizers=2).make_quantizers()
    input_quantizer, grad_quantizer = forward[0], backward[0]
    te.nvfp4_rht_set_step(5)
    grad_quantizer.set_rowwise_rht(True)
    grad_quantizer.refresh_rht_signs()
    input_quantizer.refresh_rht_signs()
    rotate = nvfp4_tensor.rht_rotate_high_precision

    wgrad = rotate(dy, grad_quantizer.rht_matrix, 0).t() @ rotate(x, input_quantizer.rht_matrix, 0)
    torch.testing.assert_close(wgrad, dy.t() @ x, rtol=1e-10, atol=1e-10)
    dgrad = rotate(dy, grad_quantizer.rht_matrix_rowwise, -1) @ rotate(
        weight, grad_quantizer.rht_matrix_rowwise, 0
    )
    torch.testing.assert_close(dgrad, dy @ weight, rtol=1e-10, atol=1e-10)
    # The helper matches the reference rotation of the kernels' oracle.
    torch.testing.assert_close(
        rotate(x, input_quantizer.rht_matrix, 0),
        ref.rht_rotate_exact(x.t().contiguous(), input_quantizer.rht_sign_bits()).t(),
        rtol=1e-12,
        atol=1e-12,
    )
    torch.testing.assert_close(
        rotate(dy, grad_quantizer.rht_matrix_rowwise, -1),
        ref.rht_rotate_exact(dy, grad_quantizer.rht_sign_bits(rowwise=True)),
        rtol=1e-12,
        atol=1e-12,
    )


@requires_ms_eden_kernel
@pytest.mark.parametrize("rht_dim", [16, 128])
@pytest.mark.parametrize("bias_correction", ["sr", "ms_eden"])
@pytest.mark.parametrize("scope", ["gradients", "all"])
@pytest.mark.parametrize("weight_source", ["high_precision", "dequantized"])
def test_linear_backward_with_rotated_dgrad(rht_dim, bias_correction, scope, weight_source):
    """End to end: quantized WGRAD and rotated DGRAD stay aligned with BF16 gradients."""
    tokens, in_features, out_features = 512, 256, 384
    torch.manual_seed(0)
    module = te.Linear(in_features, out_features, bias=False, params_dtype=torch.bfloat16).cuda()
    x = _gaussian_input(tokens, in_features, seed=5)
    target = _gaussian_input(tokens, out_features, seed=6)
    wgrad_ref, dgrad_ref = _linear_grads(module, x, target, None)
    recipe = NVFP4BlockScaling(
        grad_bias_correction=bias_correction,
        ms_eden_scope=scope,
        rht_dim=rht_dim,
        rht_sign_policy="resampled",
        dgrad_rht=True,
        dgrad_rht_weight_source=weight_source,
    )
    wgrad, dgrad = _linear_grads(module, x, target, recipe)
    for grad, reference in ((wgrad, wgrad_ref), (dgrad, dgrad_ref)):
        cosine = (grad * reference).sum() / (grad.norm() * reference.norm())
        assert cosine > 0.95, cosine


@requires_ms_eden_kernel
def test_linear_detects_sign_step_between_forward_and_backward():
    tokens, in_features, out_features = 256, 128, 128
    module = te.Linear(in_features, out_features, bias=False, params_dtype=torch.bfloat16).cuda()
    x = _gaussian_input(tokens, in_features, seed=5)
    recipe = NVFP4BlockScaling(grad_bias_correction="ms_eden", rht_sign_policy="resampled")
    with te.autocast(enabled=True, recipe=recipe):
        y = module(x)
    te.nvfp4_rht_step()  # too early: the backward pass has not run yet
    with pytest.raises(RuntimeError, match="sign schedule advanced"):
        y.float().sum().backward()

    # Stepping after the backward pass is fine; so is an interval that keeps the generation.
    with te.autocast(enabled=True, recipe=recipe):
        y = module(x)
    y.float().sum().backward()
    te.nvfp4_rht_step()
