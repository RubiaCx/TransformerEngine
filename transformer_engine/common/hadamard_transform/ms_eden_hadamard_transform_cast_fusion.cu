/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <cuda.h>
#include <cudaTypedefs.h>
#include <cuda_bf16.h>
#include <cuda_pipeline.h>
#include <cuda_runtime.h>
#include <cutlass/arch/barrier.h>
#include <transformer_engine/hadamard_transform.h>

#include <cuda/barrier>
#include <cute/tensor.hpp>

#include "common/cast/nvfp4/core_nvfp4.cuh"
#include "common/common.h"
#include "common/util/cuda_runtime.h"
#include "common/util/curanddx.hpp"
#include "common/util/ptx.cuh"
#include "common/utils.cuh"
#include "cutlass/arch/barrier.h"
#include "cutlass/cutlass.h"
#include "cutlass/gemm/collective/builders/sm100_common.inl"
#include "cutlass/numeric_conversion.h"
#include "cutlass/pipeline/pipeline.hpp"
#include "cutlass/util/GPU_Clock.hpp"
#include "cutlass/util/command_line.h"
#include "cutlass/util/print_error.hpp"

// clang-format off

namespace transformer_engine {
namespace detail {
namespace {

using namespace cute;

using cute::Tensor;  // Avoid conflict with transformer_engine::Tensor
using cute::Shape;  // Avoid conflict with transformer_engine::Shape

// Kernel modes of the tiled-RHT cast kernel.
//   kTiledRhtModeMsEden:  rotation + RTN + EDEN factors + amax (post hoc range alignment).
//   kTiledRhtModeAmax:    rotation + amax only (first pass of the aligned path).
//   kTiledRhtModeAligned: rotation + standard NVFP4 cast against a known amax
//                         (round-to-nearest or element-level stochastic rounding).
constexpr int kTiledRhtModeMsEden = 0;
constexpr int kTiledRhtModeAmax = 1;
constexpr int kTiledRhtModeAligned = 2;

// The MS-EDEN epilogue (rotation completion, quantization, EDEN sums) is the
// compute-bound part of that kernel. Rows of a 128 x 128 chunk map to the 128 threads
// of an epilogue group; several groups take consecutive chunks in round-robin order so
// that chunks are processed concurrently. Each chunk occupies two 128 x 64 accumulator
// stages (eight 128 x 16 MMA accumulators, 16 TMEM columns each), and every group owns
// its own pair of stages.
//
// The number of groups is bounded by the SM register file shared by all threads of the
// CTA: each epilogue thread keeps a full 128-element row in registers (see the
// --maxrregcount option of this file in CMakeLists.txt). The amax and aligned-cast
// modes are not epilogue-bound and use a single group.
// Thread layout: warp 0 = MMA, warp 1 = TMA loads, then 4 warps per epilogue group.
constexpr int kTiledRhtMaxEpilogueGroups = 2;
constexpr int tiled_rht_epilogue_groups(int mode) {
  return mode == kTiledRhtModeMsEden ? kTiledRhtMaxEpilogueGroups : 1;
}
constexpr int kTiledRhtAccumulatorCount = 8 * kTiledRhtMaxEpilogueGroups;
static_assert(16 * kTiledRhtAccumulatorCount <= 512, "Accumulators must fit into TMEM.");
constexpr int kTiledRhtFirstEpilogueWarp = 2;
constexpr int tiled_rht_threads_per_block(int mode) {
  return 32 * kTiledRhtFirstEpilogueWarp + 128 * tiled_rht_epilogue_groups(mode);
}

template <class ElementA,
          class ElementB,
          class ASmemLayout,
          class BSmemLayout>
struct SharedStorage {
  static constexpr int AccumulatorPipelineStageCount = kTiledRhtAccumulatorCount;
  using AtomThrShapeMNK = cute::Shape<_1, _1, _1>;

  using AccumulatorPipeline = cutlass::PipelineUmmaAsync<AccumulatorPipelineStageCount / 4, AtomThrShapeMNK>;
  using AccumulatorPipelineStorage = typename AccumulatorPipeline::SharedStorage;

  static constexpr int MainloopPipelineStageCount = size<3>(ASmemLayout{});
  using MainloopPipeline = cutlass::PipelineTmaUmmaAsync<
                             MainloopPipelineStageCount,
                             Shape<_1,_1,_1>,
                             AtomThrShapeMNK>;
  using MainloopPipelineStorage = typename MainloopPipeline::SharedStorage;

  alignas(16) AccumulatorPipelineStorage accumulator;
  alignas(16) MainloopPipelineStorage mainloop;
  alignas(16) cute::uint64_t tma_barrier[1];
  uint32_t tmem_base_ptr;

  struct TensorStorage : cute::aligned_struct<128, _1> {
    // cute::array_aligned<ElementA, cute::cosize_v<ASmemLayout>> smem_A;
    cute::array_aligned<ElementA, cute::cosize_v<ASmemLayout>> smem_A;
    cute::array_aligned<ElementB, cute::cosize_v<BSmemLayout>> smem_B;
  } tensors;

};

CUTLASS_DEVICE
cutlass::Array<cutlass::float_e2m1_t, 8>
StochasticNumericConverterBase(cutlass::Array<float, 8> const &input, cutlass::Array<uint32_t, 2> const &rbits) {
  using result_type = cutlass::Array<cutlass::float_e2m1_t, 8>;
  result_type output;
  constexpr bool has_rs = ARCH_HAS_STOCHASTIC_ROUNDING;
  if constexpr (has_rs) {
    auto output_ptr = reinterpret_cast<uint16_t *>(&output);
    asm volatile( \
        "{\n" \
        "cvt.rs.satfinite.e2m1x4.f32   %0, {%5, %4, %3, %2}, %10;\n" \
        "cvt.rs.satfinite.e2m1x4.f32   %1, {%9, %8, %7, %6}, %11;\n" \
        "}" \
        : "=h"(output_ptr[0]),
        "=h"(output_ptr[1])
        : "f"(input[0]), "f"(input[1]), "f"(input[2]), "f"(input[3]),
        "f"(input[4]), "f"(input[5]), "f"(input[6]), "f"(input[7]),
        "r"(rbits[0]), "r"(rbits[1]));
  } else {
#if FP4_TYPE_SUPPORTED
    // No cvt.rs on this architecture (e.g. SM110): use the software rounder.
    auto output_ptr = reinterpret_cast<uint16_t *>(&output);
    output_ptr[0] =
        ptx::stochastic_round_fp4_e2m1x4(input[0], input[1], input[2], input[3], rbits[0]);
    output_ptr[1] =
        ptx::stochastic_round_fp4_e2m1x4(input[4], input[5], input[6], input[7], rbits[1]);
#else
    NVTE_DEVICE_ERROR("FP4 cvt PTX instructions are architecture-specific. "
        "Try recompiling with sm_XXXa instead of sm_XXX.");
#endif
  }
  return output;
}

CUTLASS_DEVICE
cutlass::Array<cutlass::float_e2m1_t, 16>
StochasticNumericConverter(cutlass::Array<float, 16> const &input, cutlass::Array<uint32_t, 4> const *rbits) {
  using result_type = cutlass::Array<cutlass::float_e2m1_t, 16>;
  result_type output;
  cutlass::Array<cutlass::float_e2m1_t, 8> *result_ptr = reinterpret_cast<cutlass::Array<cutlass::float_e2m1_t, 8> *>(&output);
  cutlass::Array<float, 8> const *source_ptr = reinterpret_cast<cutlass::Array<float, 8> const *>(&input);
  cutlass::Array<uint32_t, 2> const *rbits_ptr = reinterpret_cast<cutlass::Array<uint32_t, 2> const *>(rbits);
  CUTLASS_PRAGMA_UNROLL
  for (int i = 0; i < 2; i++) {
    result_ptr[i] = StochasticNumericConverterBase(source_ptr[i], rbits_ptr[i]);
  }
  return output;
}

// ---------------------------------------------------------------------------
// MS-EDEN helpers (Quartet II, arXiv:2601.22813, Algorithm 1 and Section 7).
// ---------------------------------------------------------------------------

// 1 / sqrt(8): normalization of the 8-point Hadamard butterfly that extends the
// 16-point tensor-core Hadamard (already scaled by 1/4) to 128 points.
constexpr float kMsEdenInvSqrt8 = 0.35355339059327373f;

// Round a non-negative finite FP32 value to 3 mantissa bits (round-to-nearest-even).
// This is the "E8M3" extended-range proxy of an E4M3 micro-scale: same mantissa
// grid as E4M3, FP32 exponent range.
CUTLASS_DEVICE float ms_eden_round_to_e8m3(float x) {
  uint32_t bits = __float_as_uint(x);
  const uint32_t lsb = (bits >> 20) & 1u;
  bits += 0x7FFFFu + lsb;
  bits &= 0xFFF00000u;
  return __uint_as_float(bits);
}

// Value of the E2M1 code that round-to-nearest-even produces for |x| = a (a >= 0).
// E2M1 magnitudes: {0, 0.5, 1, 1.5, 2, 3, 4, 6}.
// The grid step is 0.5 below 2, 1 below 4 and 2 above; adding and subtracting
// 1.5 * 2^23 * step rounds to the nearest multiple of the step (ties to even
// multiples, which are the even E2M1 codes).
CUTLASS_DEVICE float ms_eden_fp4_rtn_abs(float a) {
  const float magic = (a < 2.0f) ? 6291456.0f : ((a < 4.0f) ? 12582912.0f : 25165824.0f);
  return fminf((a + magic) - magic, 6.0f);
}

// Complete the rotation of one 128-element chunk (8 vectors of 16) held in registers:
// for kRhtDim == 128 an 8-point Hadamard butterfly across the vectors and a
// multiplication by 1 / sqrt(8); nothing for kRhtDim == 16.
template <int kRhtDim>
CUTLASS_DEVICE void tiled_rht_finish_rotation(cutlass::Array<float, 16> *vals) {
  static_assert(kRhtDim == 16 || kRhtDim == 128, "Unsupported RHT dimension.");
  if constexpr (kRhtDim == 128) {
    CUTLASS_PRAGMA_UNROLL
    for (int len = 1; len < 8; len <<= 1) {
      CUTLASS_PRAGMA_UNROLL
      for (int base = 0; base < 8; base += 2 * len) {
        CUTLASS_PRAGMA_UNROLL
        for (int j = base; j < base + len; ++j) {
          CUTLASS_PRAGMA_UNROLL
          for (int e = 0; e < 16; ++e) {
            const float lo = vals[j][e];
            const float hi = vals[j + len][e];
            vals[j][e] = lo + hi;
            vals[j + len][e] = lo - hi;
          }
        }
      }
    }
    CUTLASS_PRAGMA_UNROLL
    for (int g = 0; g < 8; ++g) {
      CUTLASS_PRAGMA_UNROLL
      for (int e = 0; e < 16; ++e) {
        vals[g][e] = vals[g][e] * kMsEdenInvSqrt8;
      }
    }
  }
}

// Process one 128-element chunk (8 NVFP4 groups of 16) held in registers:
//   1. (kRhtDim == 128) 8-point Hadamard butterfly across the 8 groups, which
//      completes the 128-point RHT started by the 16-point tensor-core transform.
//   2. Per-group amax, E8M3 pseudo-scale d_g = E8M3(amax_g / s), RTN to E2M1
//      against d_g. s (grid_max_inv = 1 / s) is the grid parameter of the paper's
//      Q_RTN: s = 6 maps the group amax to the largest E2M1 value, s > 6 clips.
//   3. EDEN factor S = <x, x> / <x, x_rtn> per rotation chunk
//      (128 elements for kRhtDim == 128, 16 elements for kRhtDim == 16).
//   4. corrected_scales[g] = S * d_g (FP32, not yet aligned to the global scale).
// thread_amax accumulates the post-rotation amax.
template <int kRhtDim>
CUTLASS_DEVICE void ms_eden_process_chunk(
    cutlass::Array<float, 16> *vals,
    cutlass::Array<cutlass::float_e2m1_t, 16> *codes,
    float *corrected_scales,
    float &thread_amax,
    float grid_max_inv) {
  tiled_rht_finish_rotation<kRhtDim>(vals);

  cutlass::maximum_absolute_value_reduction<cutlass::Array<float, 16>, true> amax_reduction;
  float pseudo_scale[8];
  float sum_xx[8];
  float sum_xq[8];
  CUTLASS_PRAGMA_UNROLL
  for (int g = 0; g < 8; ++g) {
    const float group_amax = amax_reduction(float(0), vals[g]);
    thread_amax = fmaxf(thread_amax, group_amax);
    const float d = ms_eden_round_to_e8m3(group_amax * grid_max_inv);
    // d == 0 only for an all-zero group; the clamp keeps 0 * inv finite (== 0).
    const float d_inv = fminf(1.0f / d, cutlass::platform::numeric_limits<float>::max());
    const cutlass::Array<float, 16> scaled =
        cutlass::multiplies<cutlass::Array<float, 16>>{}(vals[g], d_inv);
    codes[g] = cutlass::NumericArrayConverter<cutlass::float_e2m1_t, float, 16>{}(scaled);
    // Inner products in units of the pseudo-scale: x = scaled * d, x_rtn = q * d.
    float xx = 0.0f;
    float xq = 0.0f;
    CUTLASS_PRAGMA_UNROLL
    for (int e = 0; e < 16; ++e) {
      const float a = fabsf(scaled[e]);
      xx += a * a;
      xq += a * ms_eden_fp4_rtn_abs(a);
    }
    pseudo_scale[g] = d;
    sum_xx[g] = xx;  // <x, x> / d^2 restricted to this group
    sum_xq[g] = xq;  // <x, x_rtn> / d^2 restricted to this group
  }

  if constexpr (kRhtDim == 128) {
    // Combine the groups with weights (d_g / d_max)^2, which keeps the sums in range
    // for any tensor magnitude.
    float d_max = 0.0f;
    CUTLASS_PRAGMA_UNROLL
    for (int g = 0; g < 8; ++g) {
      d_max = fmaxf(d_max, pseudo_scale[g]);
    }
    const float d_max_inv = fminf(1.0f / d_max, cutlass::platform::numeric_limits<float>::max());
    float num = 0.0f;
    float den = 0.0f;
    CUTLASS_PRAGMA_UNROLL
    for (int g = 0; g < 8; ++g) {
      const float w = pseudo_scale[g] * d_max_inv;
      num += sum_xx[g] * (w * w);
      den += sum_xq[g] * (w * w);
    }
    float s = num / den;
    // All-zero chunk (0/0) or non-finite input: leave the scales uncorrected.
    if (!(den > 0.0f) || !(s <= cutlass::platform::numeric_limits<float>::max())) {
      s = 1.0f;
    }
    CUTLASS_PRAGMA_UNROLL
    for (int g = 0; g < 8; ++g) {
      corrected_scales[g] = pseudo_scale[g] * s;
    }
  } else {
    CUTLASS_PRAGMA_UNROLL
    for (int g = 0; g < 8; ++g) {
      float s = sum_xx[g] / sum_xq[g];
      if (!(sum_xq[g] > 0.0f) || !(s <= cutlass::platform::numeric_limits<float>::max())) {
        s = 1.0f;
      }
      corrected_scales[g] = pseudo_scale[g] * s;
    }
  }
}

// Stochastically round a non-negative FP32 value to E4M3 (saturating at 448).
// rbits supplies 24 uniform random bits (the top 24 bits are used).
CUTLASS_DEVICE uint8_t ms_eden_stochastic_round_e4m3(float t, uint32_t rbits) {
  if (!(t > 0.0f)) {
    return 0;  // zero, negative or NaN
  }
  if (t >= 448.0f) {
    return 0x7E;  // E4M3 max (448)
  }
  int e = static_cast<int>((__float_as_uint(t) >> 23) & 0xFFu) - 127;  // floor(log2(t))
  e = (e < -6) ? -6 : e;  // below 2^-6 the grid is the subnormal one (quantum 2^-9)
  // k = t / 2^(e - 3): exact power-of-two scaling, k in [8, 16) for normals, [0, 8) below.
  const float k = t * __uint_as_float(static_cast<uint32_t>(127 + 3 - e) << 23);
  const float k_floor = floorf(k);
  const float frac = k - k_floor;
  const float u = static_cast<float>(rbits >> 8) * (1.0f / 16777216.0f);  // [0, 1)
  int ki = static_cast<int>(k_floor) + ((u < frac) ? 1 : 0);
  if (ki == 16) {
    ki = 8;
    e += 1;
  }
  if (ki < 8) {
    return static_cast<uint8_t>(ki);  // subnormal: ki * 2^-9
  }
  return static_cast<uint8_t>(((e + 7) << 3) | (ki - 8));
}

// Global encode scale for a given amax and scale bound (same formula as
// compute_global_encode_scaling_factor_FP4, with a runtime scale bound).
CUTLASS_DEVICE float tiled_rht_global_encode_scale(float global_amax, float scale_max) {
  float encode_scale = scale_max * 6.0f / global_amax;
  encode_scale = fminf(encode_scale, cutlass::platform::numeric_limits<float>::max());
  if (global_amax == 0.0f || encode_scale == 0.0f) {
    encode_scale = 1.0f;
  }
  return encode_scale;
}

// SplitMix64 output function. The micro-scale rounding uses its own counter-based
// generator instead of Philox: one scale gets a single draw per call, so the result
// must be unbiased across consecutive RNG offsets, which reduced-round Philox builds
// (NVTE_BUILD_NUM_PHILOX_ROUNDS < 10) do not provide.
CUTLASS_DEVICE uint64_t ms_eden_splitmix64(uint64_t z) {
  z += 0x9E3779B97F4A7C15ull;
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
  return z ^ (z >> 31);
}

// Second (light) pass of post hoc range alignment: align the FP32 corrected scales
// S * d_g to the global scale derived from the post-rotation amax and stochastically
// round them to E4M3. Each thread handles 4 consecutive scales (two SplitMix64 draws
// keyed by the RNG seed, the RNG offset and the scale index).
__global__ static void ms_eden_finalize_scales_kernel(const float *corrected_scales,
                                                      uint8_t *scales_out,
                                                      const float *raw_amax,
                                                      float scale_max,
                                                      float headroom,
                                                      size_t num_words,
                                                      const size_t *rng_state) {
  const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= num_words) {
    return;
  }
  // Same formula as compute_global_encode_scaling_factor_FP4, applied to the amax
  // that will be stored in the output tensor (raw amax * headroom).
  const float stored_amax = (*raw_amax) * headroom;
  const float encode_scale = tiled_rht_global_encode_scale(stored_amax, scale_max);

  const size_t rng_seed = rng_state != nullptr ? rng_state[0] : 0;
  const size_t rng_offset = rng_state != nullptr ? rng_state[1] : 0;
  const uint64_t stream = ms_eden_splitmix64(static_cast<uint64_t>(rng_seed) ^
                                             ms_eden_splitmix64(static_cast<uint64_t>(rng_offset)));
  const uint64_t draw0 = ms_eden_splitmix64(stream + 2 * static_cast<uint64_t>(idx));
  const uint64_t draw1 = ms_eden_splitmix64(stream + 2 * static_cast<uint64_t>(idx) + 1);
  const uint32_t rbits[4] = {static_cast<uint32_t>(draw0), static_cast<uint32_t>(draw0 >> 32),
                             static_cast<uint32_t>(draw1), static_cast<uint32_t>(draw1 >> 32)};

  uint32_t packed = 0;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const float t = corrected_scales[idx * 4 + i] * encode_scale;
    packed |= static_cast<uint32_t>(ms_eden_stochastic_round_e4m3(t, rbits[i])) << (8 * i);
  }
  reinterpret_cast<uint32_t *>(scales_out)[idx] = packed;
}

// Replace the raw post-rotation amax by the amax that defines the global scale
// (raw amax * headroom). Runs after ms_eden_finalize_scales_kernel on the same stream.
__global__ static void ms_eden_store_amax_kernel(float *amax, float headroom) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    *amax = (*amax) * headroom;
  }
}

template <class MShape, class NShape, class KShape, class ClusterTileShape,
          class TA, class AStride, class ASmemLayout, class TmaLoadA,
          class TB, class BStride, class BSmemLayout, class TmaLoadB,
          class TC, class CStride, class CSmemLayout,
          class TSFC,
          class TiledMMA,
          int kRhtDim = 128,
          int kMode = kTiledRhtModeMsEden,
          bool kEnableStochasticRounding = false>
__global__ static
void
ms_eden_rht_gemm_device(MShape M, NShape N, KShape K, ClusterTileShape cluster_tile,
            TA const* A, AStride dA, ASmemLayout sAlayout, CUTE_GRID_CONSTANT TmaLoadA const tma_load_a,
            TB const* B, BStride dB, BSmemLayout sBlayout, CUTE_GRID_CONSTANT TmaLoadB const tma_load_b,
            TC      * C, CStride dC, CSmemLayout         ,
            float   * WS,
            TSFC    * SFC,
            TiledMMA mma,
            float   * global_amax,
            float grid_max_inv,
            float scale_max,
            const size_t* rng_state)
{
  using namespace cute;
  constexpr bool is_blackwell_arch = ARCH_BLACKWELL_FAMILY;
  if constexpr (!is_blackwell_arch) {
    NVTE_DEVICE_ERROR("MS-EDEN RHT fusion is only supported on Blackwell.");
    return;
  } else {
  using X = Underscore;
  // static constexpr bool kApplyStochasticRounding = true;
  using ElementAccumulator = float;
  static constexpr int K_PIPE_MAX = size<3>(ASmemLayout{});
  using AtomThrShapeMNK = Shape<decltype(shape<0>(typename TiledMMA::ThrLayoutVMNK{})), _1, _1>;
  static constexpr uint32_t kTmaTransactionBytes =
    cutlass::bits_to_bytes(size(AtomThrShapeMNK{}) * cosize(take<0,3>(ASmemLayout{})) * cute::sizeof_bits_v<TA>);

  static_assert(kRhtDim == 16 || kRhtDim == 128, "Unsupported RHT dimension.");
  // Number of 128-thread epilogue groups of this kernel mode. Every group owns the
  // same accumulator stage pairs in each round.
  static constexpr int kEpilogueGroups = tiled_rht_epilogue_groups(kMode);
  static_assert(kEpilogueGroups == 1 || kTiledRhtAccumulatorCount == 8 * kEpilogueGroups,
                "Stage ownership assumes one stage pair per group (or a single group).");
  // Number of 16x16 (sign-folded) Hadamard blocks: one per 16-element K block of a
  // rotation chunk. Block j holds 0.25 * H16 * diag(signs[16 * j : 16 * j + 16]).
  static constexpr int kNumRhtBlocks = kRhtDim / 16;
  static_assert(size<3>(BSmemLayout{}) >= kNumRhtBlocks,
                "Not enough SMEM stages to hold the RHT matrix blocks.");
  static constexpr int kTmaRhtTensorTransactionBytes =
    cutlass::bits_to_bytes(16 * kRhtDim * cute::sizeof_bits_v<TB>);
  static constexpr int AccumulatorPipelineStageCount = kTiledRhtAccumulatorCount;

  static constexpr int MainloopPipelineStageCount = size<3>(ASmemLayout{});
  using MainloopPipeline = cutlass::PipelineTmaUmmaAsync<
                             MainloopPipelineStageCount,
                             Shape<_1,_1,_1>,
                             AtomThrShapeMNK>;
  using MainloopPipelineState = typename MainloopPipeline::PipelineState;

  using TmemAllocator = cute::TMEM::Allocator1Sm;
  static constexpr int VectorSize = 16;
  // Preconditions
  CUTE_STATIC_ASSERT(is_static<ASmemLayout>::value);
  CUTE_STATIC_ASSERT(is_static<BSmemLayout>::value);
  CUTE_STATIC_ASSERT(is_static<CSmemLayout>::value);

  // Represent the full tensors
  Tensor mA = tma_load_a.get_tma_tensor(make_shape(M,N));
  Tensor mB = tma_load_b.get_tma_tensor(make_shape(16,kRhtDim));
  Tensor mC = make_tensor(cute::subbyte_iterator<TC>(C), make_shape(M,N), dC);      // (M,N)

  auto sfc_shape  = make_shape(
    M,
    make_shape( make_shape(Int<16>{}, _4{}), N / 64 )
  );

  auto sfc_stride = make_stride(
    N / 16,
    make_stride( make_stride(_0{}, _1{}), _4{} )
  );

  auto sfc_layout = make_layout(sfc_shape, sfc_stride);
  // FP32 workspace with the same (M, N/16) layout as the scale-factor tensor: receives
  // the corrected, not yet range-aligned scales S * d_g.
  Tensor mWS = make_tensor(make_gmem_ptr(WS), sfc_layout);
  // E4M3 scale factors (aligned mode only).
  Tensor mSFC = make_tensor(make_gmem_ptr(SFC), sfc_layout);

  auto cluster_shape = Shape<  _1,  _1, _1>{};

  // Get the appropriate blocks for this Cluster
  dim3 cluster_coord_in_grid = cluster_id_in_grid();

  // Total number of k-tiles
  const int K_TILE_MAX  = min(N, K) / 64;
  uint32_t tiles_in_m = (M + size<0>(cluster_tile) - 1) / size<0>(cluster_tile);
  uint32_t tiles_in_n = (N + 64 - 1) / 64;
  uint32_t linear_tile_idx = blockIdx.x;
  uint32_t tile_idx_m = linear_tile_idx % tiles_in_m;
  uint32_t tile_idx_n = (linear_tile_idx / tiles_in_m) * K_TILE_MAX;


  auto mainloop_tiler = Shape<_128,_16,_64>{};
  auto epilogue_tiler = Shape<_128,_64,_64>{};
  Tensor gA_mk = local_tile(mA, mainloop_tiler, make_coord(_,_, _), Step<_1, X,_1>{});
  Tensor gB_nk = local_tile(mB, cluster_tile, make_coord(_,_, _), Step< X,_1,_1>{});  // (BLK_N,BLK_K,k)
  Tensor gC_mn = local_tile(mC, epilogue_tiler, make_coord(_,_, _), Step<_1,_1, X>{});  // (BLK_M,BLK_N)

  Tensor gWS_mn = local_tile(mWS, epilogue_tiler, make_coord(_,_, _), Step<_1,_1, X>{});  // (BLK_M,BLK_N)
  Tensor gSFC_mn = local_tile(mSFC, epilogue_tiler, make_coord(_,_, _), Step<_1,_1, X>{});  // (BLK_M,BLK_N)
  // Allocate SMEM
  extern __shared__ char shared_memory[];
  using SharedStorage = SharedStorage<TA, TB, ASmemLayout, BSmemLayout>;
  SharedStorage& shared_storage = *reinterpret_cast<SharedStorage*>(shared_memory);
  Tensor tCsA = make_tensor(make_smem_ptr(shared_storage.tensors.smem_A.data()), sAlayout);  // (MMA,MMA_M,MMA_N,PIPE)
  Tensor tCsB = make_tensor(make_smem_ptr(shared_storage.tensors.smem_B.data()), sBlayout);  // (MMA,MMA_N,MMA_K,PIPE)


  //
  // MMA: Define C accumulators and A/B partitioning
  //

  int block_rank_in_cluster = cute::block_rank_in_cluster();
  ThrMMA thr_mma = mma.get_slice(block_rank_in_cluster);               // blk idx
  Tensor tCgB = thr_mma.partition_B(gB_nk);                               // (MMA,MMA_N,MMA_K,k)

  auto mma_epilogue = make_tiled_mma(SM100_MMA_F16BF16_SS<TA, TB, ElementAccumulator,
                                               128, 64,
                                               UMMA::Major::MN, UMMA::Major::MN>{},
                            Layout<Shape<_1,_1>>{});
  ThrMMA thr_mma_epilogue = mma_epilogue.get_slice(block_rank_in_cluster);


  using TiledMmaEpilogue = decltype(mma_epilogue);
  Tensor tCgA = thr_mma.partition_A(gA_mk);
  // Allocate "fragments" -- these are actually umma smem descriptors
  Tensor tCrA = thr_mma.make_fragment_A(tCsA);                         // (MMA,MMA_M,MMA_K,PIPE)
  Tensor tCrB = thr_mma.make_fragment_B(tCsB);                         // (MMA,MMA_M,MMA_K,PIPE)

  auto acc_shape_mma = partition_shape_C(TiledMMA{}, take<0,2>(ClusterTileShape{}));
  auto acc_shape_epilogue = partition_shape_C(TiledMmaEpilogue{}, take<0,2>(epilogue_tiler));

  auto bulk_tmem_mma = TiledMMA::make_fragment_C(append(acc_shape_mma,
                                                      Int<AccumulatorPipelineStageCount>{}));

  auto bulk_tmem_epilogue = TiledMmaEpilogue::make_fragment_C(append(acc_shape_epilogue,
                                                      Int<AccumulatorPipelineStageCount / 4>{}));

  TmemAllocator tmem_allocator{};
  cutlass::arch::NamedBarrier tmem_allocation_result_barrier(32 + 128 * kEpilogueGroups, cutlass::arch::ReservedNamedBarriers::TmemAllocBarrier);

  Layout cta_layout_mnk  = make_layout(cluster_shape);
  Layout cta_layout_vmnk = tiled_divide(cta_layout_mnk, make_tile(typename TiledMMA::AtomThrID{}));
  auto cta_coord_vmnk  = cta_layout_vmnk.get_flat_coord(block_rank_in_cluster);

  auto [tAgA, tAsA] = tma_partition(tma_load_a,
    get<2>(cta_coord_vmnk), make_layout(size<2>(cta_layout_vmnk)),
    group_modes<0,3>(tCsA), group_modes<0,3>(tCgA));

  auto [tBgB, tBsB] = tma_partition(tma_load_b,
    get<1>(cta_coord_vmnk), make_layout(size<1>(cta_layout_vmnk)),
    group_modes<0,3>(tCsB), group_modes<0,3>(tCgB));

  uint16_t tma_mcast_mask_a = create_tma_multicast_mask<2>(cta_layout_vmnk, cta_coord_vmnk);
  uint16_t tma_mcast_mask_b = create_tma_multicast_mask<1>(cta_layout_vmnk, cta_coord_vmnk);

  int warp_idx = cutlass::canonical_warp_idx_sync();

  bool is_mma_warp = (warp_idx == 0);
  bool is_dma_warp = (warp_idx == 1);
  bool is_epilogue_warp = (warp_idx >= kTiledRhtFirstEpilogueWarp &&
                           warp_idx < kTiledRhtFirstEpilogueWarp + 4 * kEpilogueGroups);

  typename MainloopPipeline::Params mainloop_pipeline_params;
  if (is_dma_warp) {
    mainloop_pipeline_params.role = MainloopPipeline::ThreadCategory::Producer;
  }
  if (is_mma_warp) {
    mainloop_pipeline_params.role = MainloopPipeline::ThreadCategory::Consumer;
  }
  mainloop_pipeline_params.is_leader = cute::elect_one_sync() && is_dma_warp;
  mainloop_pipeline_params.transaction_bytes = kTmaTransactionBytes;
  mainloop_pipeline_params.initializing_warp = 0;
  MainloopPipeline mainloop_pipeline(shared_storage.mainloop,
                                       mainloop_pipeline_params,
                                       cluster_shape,
                                       cute::true_type{},   // Perform barrier init
                                       cute::true_type{}); // Delay mask calculation

  MainloopPipelineState mainloop_pipe_consumer_state;
  MainloopPipelineState mainloop_pipe_producer_state = cutlass::make_producer_start_state<MainloopPipeline>();



  using AccumulatorPipeline = cutlass::PipelineUmmaAsync<AccumulatorPipelineStageCount / 4, AtomThrShapeMNK>;
  using AccumulatorPipelineState = typename AccumulatorPipeline::PipelineState;

  AccumulatorPipelineState accumulator_pipe_consumer_state;
  AccumulatorPipelineState accumulator_pipe_producer_state = cutlass::make_producer_start_state<AccumulatorPipeline>();

  typename AccumulatorPipeline::Params accumulator_pipeline_params;
  if (is_mma_warp) {
    accumulator_pipeline_params.role = AccumulatorPipeline::ThreadCategory::Producer;
  }
  if (is_epilogue_warp) {
    accumulator_pipeline_params.role = AccumulatorPipeline::ThreadCategory::Consumer;
  }
  // Only one producer thread arrives on this barrier.
  accumulator_pipeline_params.producer_arv_count = 1;
  accumulator_pipeline_params.consumer_arv_count = size(AtomThrShapeMNK{}) * 128;
  accumulator_pipeline_params.initializing_warp = 1;
  AccumulatorPipeline accumulator_pipeline(shared_storage.accumulator,
                                           accumulator_pipeline_params,
                                           cluster_shape,
                                           cute::true_type{},   // Perform barrier init
                                           cute::true_type{}); // Delay mask calculation

  if (is_dma_warp && elect_one_sync()) {
    cute::initialize_barrier(shared_storage.tma_barrier[0], /* num_threads */ 1);
    // Publish this separately initialized barrier before TMA can complete against it.
    cutlass::arch::fence_barrier_init();
  }
  __syncthreads();
  using TMEM_LOAD_NEW = cute::SM100::TMEM::LOAD::SM100_TMEM_LOAD_32dp32b64x;

  if (is_dma_warp) {
    if (elect_one_sync()) {
      cute::set_barrier_transaction_bytes(shared_storage.tma_barrier[0], kTmaRhtTensorTransactionBytes);
      CUTE_UNROLL
      for (int rht_block = 0; rht_block < kNumRhtBlocks; ++rht_block) {
        copy(tma_load_b.with(shared_storage.tma_barrier[0], tma_mcast_mask_b), tBgB(_,0,rht_block), tBsB(_,rht_block));
      }
    }
    cute::wait_barrier(shared_storage.tma_barrier[0], 0 /*tma_phase_bit*/);
    do {
      bool is_first_wave = linear_tile_idx == blockIdx.x;
      uint32_t skip_wait = is_first_wave;
      auto tAgA_mk = tAgA(_,tile_idx_m,_);
      int k_tile = 0;
      auto barrier_token = mainloop_pipeline.producer_try_acquire(mainloop_pipe_producer_state, skip_wait);


      CUTE_NO_UNROLL
      while (k_tile < K_TILE_MAX && k_tile + tile_idx_n < tiles_in_n) {
        int k_tile_idx_n = tile_idx_n + k_tile;
        ++k_tile;
        skip_wait = (is_first_wave && k_tile < MainloopPipelineStageCount);
        mainloop_pipeline.producer_acquire(mainloop_pipe_producer_state, barrier_token);
        using BarrierType = typename MainloopPipeline::ProducerBarrierType;
        BarrierType* tma_barrier = mainloop_pipeline.producer_get_barrier(mainloop_pipe_producer_state);
        int write_stage = mainloop_pipe_producer_state.index();
        ++mainloop_pipe_producer_state;
        barrier_token = mainloop_pipeline.producer_try_acquire(mainloop_pipe_producer_state, skip_wait);
        if (cute::elect_one_sync()) {
          copy(tma_load_a.with(*tma_barrier, tma_mcast_mask_a), tAgA_mk(_,k_tile_idx_n), tAsA(_,write_stage));
        }
      }
      linear_tile_idx += gridDim.x;
      tile_idx_m = linear_tile_idx % tiles_in_m;
      tile_idx_n = (linear_tile_idx / tiles_in_m) * K_TILE_MAX;
    } while (tile_idx_m < tiles_in_m && tile_idx_n < tiles_in_n);
    mainloop_pipeline.producer_tail(mainloop_pipe_producer_state);
  } else if (is_mma_warp) {
    mma.accumulate_ = UMMA::ScaleOut::Zero;

    tmem_allocator.allocate(cute::TMEM::Sm100TmemCapacityColumns, &shared_storage.tmem_base_ptr);
    __syncwarp();
    tmem_allocation_result_barrier.arrive();
    uint32_t tmem_base_ptr = shared_storage.tmem_base_ptr;
    bulk_tmem_mma.data() = tmem_base_ptr;

    do {
      uint32_t skip_wait = K_TILE_MAX <= 0;
      auto barrier_token = mainloop_pipeline.consumer_try_wait(mainloop_pipe_consumer_state, skip_wait);
      CUTE_NO_UNROLL
      for (int k_tile = 0; k_tile < K_TILE_MAX && k_tile + tile_idx_n < tiles_in_n; )
      {
        mainloop_pipeline.consumer_wait(mainloop_pipe_consumer_state, barrier_token);
        int read_stage = mainloop_pipe_consumer_state.index();
        auto tCrA_mk = tCrA(_,_,_,read_stage);
        // First 16-element K block of this 64-wide tile within its rotation chunk.
        const int rht_block_base = ((tile_idx_n + k_tile) * 4) % kNumRhtBlocks;
        CUTE_UNROLL
        for (int k_block = 0; k_block < size<2>(tCrA) / 4; ++k_block)
        {
          accumulator_pipeline.producer_acquire(accumulator_pipe_producer_state);
          CUTE_UNROLL
          for (int i = 0; i < 4; i++) {
            auto accumulators = bulk_tmem_mma(_,_,_,accumulator_pipe_producer_state.index() * 4 + i);
            auto tCrB_nk = tCrB(_,_,0,(rht_block_base + k_block * 4 + i) % kNumRhtBlocks);
            gemm(mma, tCrA_mk(_,_,k_block * 4 + i), tCrB_nk, accumulators);
          }

          accumulator_pipeline.producer_commit(accumulator_pipe_producer_state);
          ++accumulator_pipe_producer_state;
        }
        auto curr_mainloop_pipe_consumer_state = mainloop_pipe_consumer_state;
        ++mainloop_pipe_consumer_state;
        ++k_tile;
        skip_wait = k_tile >= K_TILE_MAX;
        barrier_token = mainloop_pipeline.consumer_try_wait(mainloop_pipe_consumer_state, skip_wait);
        mainloop_pipeline.consumer_release(curr_mainloop_pipe_consumer_state);
      }

      linear_tile_idx += gridDim.x;
      tile_idx_m = linear_tile_idx % tiles_in_m;
      tile_idx_n = (linear_tile_idx / tiles_in_m) * K_TILE_MAX;
    } while (tile_idx_m < tiles_in_m && tile_idx_n < tiles_in_n);
    tmem_allocator.release_allocation_lock();
    accumulator_pipeline.producer_tail(accumulator_pipe_producer_state);
    tmem_allocator.free(tmem_base_ptr, cute::TMEM::Sm100TmemCapacityColumns);
  } else if (is_epilogue_warp) {
    static constexpr int FragmentSize = 256 / sizeof_bits_v<TC>;
    static_assert(FragmentSize == 4 * VectorSize,
                  "The epilogue expects 64 FP4 values (4 NVFP4 groups) per thread and tile.");
    // One rotation / EDEN work unit is 128 elements along N: two 64-wide epilogue tiles.
    static constexpr int kTilesPerChunk = 2;
    static constexpr int kVecsPerTile = FragmentSize / VectorSize;
    static constexpr int kVecsPerChunk = kTilesPerChunk * kVecsPerTile;

    tmem_allocation_result_barrier.arrive_and_wait();
    uint32_t tmem_base_ptr = shared_storage.tmem_base_ptr;
    bulk_tmem_epilogue.data() = tmem_base_ptr;
    // TMEM lanes are tied to the warp position modulo 4, so the row of a thread is its
    // index modulo 128 for any four consecutive warps.
    int thread_idx = threadIdx.x % 128;
    // Epilogue group of this thread and the first accumulator stage pair it owns.
    const int epilogue_group = (warp_idx - kTiledRhtFirstEpilogueWarp) / 4;
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < 2 * epilogue_group; ++i) {
      ++accumulator_pipe_consumer_state;
    }
    // Index of the next chunk produced by the MMA warp of this CTA.
    uint32_t chunk_counter = 0;

    Tensor tCgC = thr_mma_epilogue.partition_C(gC_mn);                             // (MMA,MMA_M,MMA_N)
    auto tiled_t2r = make_tmem_copy(TMEM_LOAD_NEW{}, bulk_tmem_epilogue(_,_,_,_0{}));
    auto tiled_r2g = make_tiled_copy_D(Copy_Atom<SM100_STORE_256bit_CACHE_NOALLOCATION, TC>{}, tiled_t2r);
    auto thr_t2r   = tiled_t2r.get_slice(thread_idx);
    auto thr_r2g = tiled_r2g.get_slice(thread_idx);

    // Post-rotation amax seen by this thread (reduced into global_amax at the end).
    float thread_amax = 0.0f;

    // Aligned mode: global scales from the amax computed by the amax pass, and RNG state.
    float global_encode_scale_multiplier = 1.0f;
    float global_decode_scale = 1.0f;
    // Element-level stochastic rounding uses a SplitMix64 counter generator keyed by the
    // RNG seed and the per-call RNG offset, with one counter per (thread, chunk, vector).
    // Philox with a reduced round count (NVTE_BUILD_NUM_PHILOX_ROUNDS < 10) leaves some
    // output bytes constant across calls (the offset advances by a fixed stride) and
    // identical across threads within a call, which repeats rounding decisions.
    const uint64_t sr_stream = ms_eden_splitmix64(
        static_cast<uint64_t>(rng_state != nullptr ? rng_state[0] : 0) ^
        ms_eden_splitmix64(static_cast<uint64_t>(rng_state != nullptr ? rng_state[1] : 0)));
    if constexpr (kMode == kTiledRhtModeAligned) {
      const float global_encode_scale = tiled_rht_global_encode_scale(*global_amax, scale_max);
      global_decode_scale = 1.0f / global_encode_scale;
      global_encode_scale_multiplier = global_encode_scale * (1.0f / 6.0f);
    }

    do {
      // The host guarantees that K_TILE_MAX and tiles_in_n are even, so 64-wide tiles
      // always come in (even, odd) pairs that form one 128-element chunk.
      for (int k_tile = 0; k_tile < K_TILE_MAX && k_tile + tile_idx_n < tiles_in_n; k_tile += kTilesPerChunk) {
        // Chunks are handed to the epilogue groups in round-robin order.
        if ((chunk_counter++) % kEpilogueGroups != static_cast<uint32_t>(epilogue_group)) {
          continue;
        }
        cutlass::Array<ElementAccumulator, VectorSize> chunk_vals[kVecsPerChunk];
        cutlass::Array<TC, VectorSize> chunk_codes[kVecsPerChunk];
        float chunk_scales[kVecsPerChunk];
        cutlass::Array<TSFC, kVecsPerChunk> chunk_sfs;

        // Phase 1: TMEM -> registers for both halves of the chunk.
        CUTLASS_PRAGMA_UNROLL
        for (int h = 0; h < kTilesPerChunk; ++h) {
          Tensor tCgC_mn = tCgC(_,_,_,tile_idx_m,tile_idx_n+k_tile+h);
          accumulator_pipeline.consumer_wait(accumulator_pipe_consumer_state);

          auto tCtC = bulk_tmem_epilogue(_,_,_,accumulator_pipe_consumer_state.index());
          Tensor tDtC = thr_t2r.partition_S(tCtC);                   // ((TMEM_LOAD,#TMEM_LOAD),MMA_M,MMA_N)
          Tensor tDgC = thr_t2r.partition_D(tCgC_mn);                   // ((TMEM_LOAD,#TMEM_LOAD),MMA_M,MMA_N)

          Tensor tTR_rAcc = make_tensor<ElementAccumulator>(shape(tDgC));                 // ((TMEM_LOAD,#TMEM_LOAD),MMA_M,MMA_N)
          Tensor tTR_rAcc_frag = recast<cutlass::Array<ElementAccumulator, FragmentSize>>(coalesce(tTR_rAcc));

          // TMEM_LOAD
          copy(tiled_t2r, tDtC, tTR_rAcc);
          cutlass::arch::fence_view_async_tmem_load();

          accumulator_pipeline.consumer_release(accumulator_pipe_consumer_state);

          ++accumulator_pipe_consumer_state;

          auto compute_frgs = reinterpret_cast<cutlass::Array< ElementAccumulator, VectorSize> *>(tTR_rAcc_frag.data());
          CUTLASS_PRAGMA_UNROLL
          for (int v = 0; v < kVecsPerTile; v++) {
            chunk_vals[h * kVecsPerTile + v] = compute_frgs[v];
          }
        }

        // Skip the accumulator stages owned by the other epilogue groups.
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < 2 * (kEpilogueGroups - 1); ++i) {
          ++accumulator_pipe_consumer_state;
        }

        // Phase 2: finish the rotation and quantize. The RHT output stays in FP32
        // (no BF16 round trip).
        if constexpr (kMode == kTiledRhtModeMsEden) {
          // RTN to E2M1, EDEN factor, corrected (not range-aligned) scales.
          ms_eden_process_chunk<kRhtDim>(chunk_vals, chunk_codes, chunk_scales, thread_amax,
                                         grid_max_inv);
        } else if constexpr (kMode == kTiledRhtModeAmax) {
          tiled_rht_finish_rotation<kRhtDim>(chunk_vals);
          cutlass::maximum_absolute_value_reduction<cutlass::Array<ElementAccumulator, VectorSize>, true> amax_reduction;
          CUTLASS_PRAGMA_UNROLL
          for (int v = 0; v < kVecsPerChunk; v++) {
            thread_amax = fmaxf(thread_amax, amax_reduction(ElementAccumulator(0), chunk_vals[v]));
          }
        } else {
          // Standard NVFP4 cast against the known global amax.
          tiled_rht_finish_rotation<kRhtDim>(chunk_vals);
          cutlass::maximum_absolute_value_reduction<cutlass::Array<ElementAccumulator, VectorSize>, true> amax_reduction;
          cutlass::Array<ElementAccumulator, kVecsPerChunk> vec_maxs;
          CUTLASS_PRAGMA_UNROLL
          for (int v = 0; v < kVecsPerChunk; v++) {
            vec_maxs[v] = amax_reduction(ElementAccumulator(0), chunk_vals[v]);
          }
          auto pvscales = cutlass::multiplies<cutlass::Array<ElementAccumulator, kVecsPerChunk>>{}(vec_maxs, global_encode_scale_multiplier);
          chunk_sfs = cutlass::NumericArrayConverter<TSFC, ElementAccumulator, kVecsPerChunk>{}(pvscales);
          auto qpvscale_ups = cutlass::NumericArrayConverter<ElementAccumulator, TSFC, kVecsPerChunk>{}(chunk_sfs);
          auto qpvscale_scaled = cutlass::multiplies<cutlass::Array<ElementAccumulator, kVecsPerChunk>>{}(qpvscale_ups, global_decode_scale);
          auto acc_scales = cutlass::divides<cutlass::Array<ElementAccumulator, kVecsPerChunk>>{}(1.0, qpvscale_scaled);

          // 16 counters per (thread, chunk): two 64-bit draws per 16-element vector.
          const uint64_t sr_counter = sr_stream +
              ((static_cast<uint64_t>(thread_idx) + static_cast<uint64_t>(k_tile) * 256 +
                static_cast<uint64_t>(linear_tile_idx) * K_TILE_MAX * 256) << 4);
          uint4 random_uint4 = uint4{0, 0, 0, 0};

          CUTLASS_PRAGMA_UNROLL
          for (int v = 0; v < kVecsPerChunk; v++) {
            auto acc_scale = cutlass::minimum_with_nan_propagation<ElementAccumulator>{}(acc_scales[v], cutlass::platform::numeric_limits<ElementAccumulator>::max());
            if constexpr (kEnableStochasticRounding) {
              const uint64_t draw0 = ms_eden_splitmix64(sr_counter + 2 * v);
              const uint64_t draw1 = ms_eden_splitmix64(sr_counter + 2 * v + 1);
              random_uint4 = uint4{static_cast<uint32_t>(draw0), static_cast<uint32_t>(draw0 >> 32),
                                   static_cast<uint32_t>(draw1), static_cast<uint32_t>(draw1 >> 32)};
              chunk_codes[v] = StochasticNumericConverter(
                cutlass::multiplies<cutlass::Array<ElementAccumulator, VectorSize>>{}(
                  chunk_vals[v],
                  acc_scale
                ),
                reinterpret_cast<cutlass::Array<uint32_t, 4>*>(&random_uint4));
            } else {
              chunk_codes[v] = cutlass::NumericArrayConverter<TC, ElementAccumulator, VectorSize>{}(cutlass::multiplies<cutlass::Array<ElementAccumulator, VectorSize>>{}(chunk_vals[v], acc_scale));
            }
          }
        }

        // Phase 3: store FP4 codes and scales for both halves.
        if constexpr (kMode != kTiledRhtModeAmax) {
        CUTLASS_PRAGMA_UNROLL
        for (int h = 0; h < kTilesPerChunk; ++h) {
          Tensor tCgC_mn = tCgC(_,_,_,tile_idx_m,tile_idx_n+k_tile+h);
          Tensor tDgC = thr_t2r.partition_D(tCgC_mn);                   // ((TMEM_LOAD,#TMEM_LOAD),MMA_M,MMA_N)
          Tensor tDrC = make_tensor<TC>(shape(tDgC));
          Tensor tDrC_frag = recast<cutlass::Array<TC, FragmentSize>>(coalesce(tDrC));

          Tensor src = thr_r2g.retile_S(tDrC);
          Tensor dst = thr_r2g.retile_D(tDgC);

          auto output_frgs = reinterpret_cast<cutlass::Array< TC, VectorSize> *>(tDrC_frag.data());
          CUTLASS_PRAGMA_UNROLL
          for (int v = 0; v < kVecsPerTile; v++) {
            output_frgs[v] = chunk_codes[h * kVecsPerTile + v];
          }
          copy(tiled_r2g, src, dst);

          if constexpr (kMode == kTiledRhtModeMsEden) {
            // FP32 corrected scales (range-aligned and rounded by the second kernel).
            Tensor tCgWS_mn = gWS_mn(_,_,tile_idx_m,tile_idx_n+k_tile+h);
            Tensor tCgWS = make_tensor(tCgWS_mn.data(), make_layout(
                                       make_shape(shape(tCgWS_mn), Int<1>{}, Int<1>{}),
                                       make_stride(stride(tCgWS_mn), Int<0>{}, Int<0>{})
                                      ));
            Tensor tDgWS = filter(thr_t2r.partition_D(tCgWS));
            CUTLASS_PRAGMA_UNROLL
            for (int v = 0; v < kVecsPerTile; v++) {
              tDgWS(v) = chunk_scales[h * kVecsPerTile + v];
            }
          } else {
            // Final E4M3 scales.
            Tensor tCgSFC_mn = gSFC_mn(_,_,tile_idx_m,tile_idx_n+k_tile+h);
            Tensor tCgSFC = make_tensor(tCgSFC_mn.data(), make_layout(
                                        make_shape(shape(tCgSFC_mn), Int<1>{}, Int<1>{}),
                                        make_stride(stride(tCgSFC_mn), Int<0>{}, Int<0>{})
                                       ));
            Tensor tDgSFC = filter(thr_t2r.partition_D(tCgSFC));
            CUTLASS_PRAGMA_UNROLL
            for (int v = 0; v < kVecsPerTile; v++) {
              tDgSFC(v) = chunk_sfs[h * kVecsPerTile + v];
            }
          }
        }
        }
      }
      linear_tile_idx += gridDim.x;
      tile_idx_m = linear_tile_idx % tiles_in_m;
      tile_idx_n = (linear_tile_idx / tiles_in_m) * K_TILE_MAX;
    } while (tile_idx_m < tiles_in_m && tile_idx_n < tiles_in_n);

    // Non-negative floats order like their bit patterns, so an integer atomic max
    // implements the FP32 amax reduction. The host zeroes *global_amax beforehand.
    if constexpr (kMode != kTiledRhtModeAligned) {
      if (thread_amax > 0.0f) {
        atomicMax(reinterpret_cast<int *>(global_amax), __float_as_int(thread_amax));
      }
    }
  }
  }
}

// Stride of the (M, N) input view: column-major (M contiguous) or row-major (N contiguous).
template <bool kRowMajorA>
auto tiled_rht_input_stride(int m, int n) {
  using namespace cute;
  if constexpr (kRowMajorA) {
    return make_stride(n, Int<1>{});
  } else {
    return make_stride(Int<1>{}, m);
  }
}

// Launch the tiled-RHT cast kernel for
// A: m x n (BF16): col-major (kRowMajorA == false) or row-major (kRowMajorA == true)
// B: 16 x kRhtDim: col-major (BF16), block j = 0.25 * H16 * diag(signs of block j)
// C: m x n: row-major (FP4)
// WS: m x (n/16): row-major (FP32 corrected scales, not range-aligned; MS-EDEN mode)
// SFC: m x (n/16): row-major (E4M3 scales; aligned mode)
// The rotation and the 16-element NVFP4 groups both run along n.
template <typename TA, typename TB, typename TC, typename TSFC, int kRhtDim, int kMode,
          bool kEnableStochasticRounding, bool kRowMajorA>
void
ms_eden_rht_gemm_ntt(int m, int n,
        TA const* A,
        TB const* B,
        TC      * C,
        float   * WS,
        TSFC    * SFC,
        float   * global_amax,
        float grid_max_inv,
        float scale_max,
        const size_t* rng_state,
        uint32_t sm_count,
        cudaStream_t stream,
        int k_tile_size = 2048)
{
  using namespace cute;

  // Define shapes (dynamic)
  auto M = static_cast<int>(m);
  auto N = static_cast<int>(n);

  // Define strides (mixed)
  auto dA = tiled_rht_input_stride<kRowMajorA>(m, n);  // (dM,dK)
  constexpr UMMA::Major kMajorA = kRowMajorA ? UMMA::Major::K : UMMA::Major::MN;
  auto dB = make_stride(Int<1>{}, 16);  // (dN,dK)
  auto dC = make_stride(n, Int<1>{});  // (dM,dN)

  auto cga_shape      = Shape<  _1,  _1, _1>{};
  auto cga_tile_shape = Shape<_128,_16,_16>{};
  auto cluster_tile_mainloop = Shape<_128,_16,_64>{};

  // Construct the MMA
  auto mma = make_tiled_mma(SM100_MMA_F16BF16_SS<TA, TB, float,
                                               128, 16,
                                               kMajorA, UMMA::Major::MN>{},
                            Layout<Shape<_1,_1>>{});

  // Assert that the TiledMMA uses all CTAs in the CGA.
  CUTE_STATIC_ASSERT_V(size(cga_shape) == size(mma));
  CUTE_STATIC_ASSERT_V(evenly_divides(cga_tile_shape, tile_shape(mma)));

  // Determine the A and B shapes
  auto mma_shape_B = partition_shape_B(mma, make_shape(size<1>(cga_tile_shape), size<2>(cga_tile_shape)));

  using TiledMma = decltype(mma);
  using AtomThrID = typename TiledMma::AtomThrID;

  using SmemShape_M = decltype(shape_div(shape<0>(cga_tile_shape), shape_div(shape<0>(cga_tile_shape), size<0>(cga_tile_shape) / size(AtomThrID{}))));
  using SmemShape_N = decltype(shape_div(shape<1>(cga_tile_shape), shape_div(shape<1>(cga_tile_shape), size<1>(cga_tile_shape) / size(AtomThrID{}))));
  using SmemShape_K = decltype(cute::get<2>(cga_tile_shape));

  using SmemLayoutAtomB = decltype(cutlass::gemm::collective::detail::sm100_smem_selector<
      cute::UMMA::Major::MN, TB, SmemShape_N, SmemShape_K>());

  auto mma_shape_A = partition_shape_A(mma, make_shape(size<0>(cluster_tile_mainloop), size<2>(cluster_tile_mainloop)));
  using SmemShape_M_A = decltype(shape_div(shape<0>(cluster_tile_mainloop), shape_div(shape<0>(cluster_tile_mainloop), size<0>(cluster_tile_mainloop) / size(AtomThrID{}))));
  using SmemShape_K_A = decltype(cute::get<2>(cluster_tile_mainloop));
  using SmemLayoutAtomA = decltype(cutlass::gemm::collective::detail::sm100_smem_selector<
      kMajorA, TA, SmemShape_M_A, SmemShape_K_A>());

  // Define the smem layouts (static)
  // Calculate max pipeline stages based on Blackwell SM100's 232KB shared memory
  constexpr int kBlackwellSmemSize = 232448; // 232KB in bytes
  constexpr int kBytesPerStage = cute::size(mma_shape_A) * sizeof(TA) + cute::size(mma_shape_B) * sizeof(TB);
  constexpr int kReservedBytes = 256; // Reserve for barriers and other uses
  constexpr int kMaxStages = (kBlackwellSmemSize - kReservedBytes) / kBytesPerStage;
  // Every pipeline stage reserves one 16x16 B block; the kRhtDim / 16 RHT blocks
  // are stored in the first stages of that (otherwise unused) B storage.
  static_assert(kMaxStages >= kRhtDim / 16, "Not enough SMEM stages for the RHT matrix blocks.");
  auto sP = Int<kMaxStages>{};      // SMEM pipelines
  auto sA = UMMA::tile_to_mma_shape(SmemLayoutAtomA{}, append(mma_shape_A, sP)); // (MMA,MMA_M,MMA_K,PIPE)
  auto sB = UMMA::tile_to_mma_shape(SmemLayoutAtomB{}, append(mma_shape_B, sP)); // (MMA,MMA_N,MMA_K,PIPE)
  auto sC = Layout<_1>{};  // XXX Dummy

  // Create GMEM tensors
  Tensor tensorA = make_tensor(A, make_layout(make_shape(M,N), dA));      // (M,N)
  Tensor tensorB = make_tensor(B, make_layout(make_shape(16,kRhtDim), dB));      // (16,kRhtDim)

  // Create the TiledCopy

  auto tma_load_a = make_tma_copy_A_sm100(
        SM90_TMA_LOAD{},
        tensorA,
        sA(_,_,_,0),
        cluster_tile_mainloop,
        mma);
  auto tma_load_b =  make_tma_copy_B_sm100(
        SM90_TMA_LOAD{},
        tensorB,
        sB(_,_,_,0),
        cga_tile_shape,
        mma);

  // Assert checks on tile sizes -- no predication
  NVTE_CHECK(M % size<0>(cga_tile_shape) == 0,
             "Outer dimension must be divisible by ", static_cast<size_t>(size<0>(cga_tile_shape)), " but got ", M, ".");
  NVTE_CHECK(N % 128 == 0,
             "Inner (rotation) dimension must be divisible by 128 but got ", N, ".");
  NVTE_CHECK(k_tile_size % 128 == 0,
             "k_tile_size must be divisible by 128 but got ", k_tile_size, ".");

  uint32_t tiles = size(ceil_div(M, get<0>(cga_tile_shape))) * size(ceil_div(N, k_tile_size));

  tiles = (tiles < sm_count) ? tiles : sm_count;

  dim3 dimBlock(tiled_rht_threads_per_block(kMode));
  dim3 dimCluster(size<0>(cga_shape), size<1>(cga_shape), size<2>(cga_shape));
  dim3 dimGrid(tiles, 1, 1);

  int smem_size = sizeof(SharedStorage<TA, TB, decltype(sA), decltype(sB)>);
  auto* kernel_ptr = &ms_eden_rht_gemm_device<
                                  decltype(M), decltype(N), decltype(k_tile_size), decltype(cga_tile_shape),
                                  TA, decltype(dA), decltype(sA), decltype(tma_load_a),
                                  TB, decltype(dB), decltype(sB), decltype(tma_load_b),
                                  TC, decltype(dC), decltype(sC),
                                  TSFC,
                                  decltype(mma),
                                  kRhtDim, kMode, kEnableStochasticRounding>;

  NVTE_CHECK_CUDA(
      cudaFuncSetAttribute(*kernel_ptr,
                           cudaFuncAttributeMaxDynamicSharedMemorySize,
                           smem_size)
  );

  (*kernel_ptr)
      <<< dimGrid, dimBlock, smem_size, stream >>>
      (M,  N,  k_tile_size, cga_tile_shape,
       A, dA, sA, tma_load_a,
       B, dB, sB, tma_load_b,
       C, dC, sC,
       WS, SFC,
       mma, global_amax, grid_max_inv, scale_max, rng_state);
  NVTE_CHECK_CUDA(cudaGetLastError());
}

// All passes for one (RHT dimension, input layout) combination.
template <int kRhtDim, bool kRowMajorA>
void
tiled_rht_quantize(int m, int n, const void* input, const void* hadamard_matrix, void* data,
                   void* scales, float* amax, float* workspace, bool ms_eden, bool stochastic,
                   float grid_max, float scale_max, const size_t* rng_state, uint32_t sm_count,
                   cudaStream_t stream, int k_tile_size) {
  using TA = cute::bfloat16_t;
  using TB = cute::bfloat16_t;
  using TC = cutlass::float_e2m1_t;
  using TSFC = fp8e4m3;
  const auto *A = reinterpret_cast<TA const *>(input);
  const auto *B = reinterpret_cast<TB const *>(hadamard_matrix);
  auto *C = reinterpret_cast<TC *>(data);
  auto *SFC = reinterpret_cast<TSFC *>(scales);
  const size_t num_scales = static_cast<size_t>(m) * (n / 16);

  NVTE_CHECK_CUDA(cudaMemsetAsync(amax, 0, sizeof(float), stream));
  if (ms_eden) {
    // Micro-scales are generated against scale_max / headroom so that the EDEN
    // correction has room to scale them up; the stored amax is inflated by the same
    // factor so that standard NVFP4 consumers decode the tensor unchanged. With the
    // grid parameter s, the per-group decode scale is group_amax / s, which adds the
    // factor 6 / s.
    const float headroom = (448.0f / 256.0f) * (6.0f / grid_max);
    // Pass 1 (heavy): rotation + RTN + EDEN factors + post-rotation amax.
    ms_eden_rht_gemm_ntt<TA, TB, TC, TSFC, kRhtDim, kTiledRhtModeMsEden, false, kRowMajorA>(
        m, n, A, B, C, workspace, SFC, amax, 1.0f / grid_max, scale_max, nullptr, sm_count,
        stream, k_tile_size);
    // Pass 2 (light, scales only): range alignment + stochastic rounding to E4M3.
    const size_t num_words = num_scales / 4;
    constexpr unsigned int kFinalizeThreads = 256;
    const unsigned int num_blocks =
        static_cast<unsigned int>((num_words + kFinalizeThreads - 1) / kFinalizeThreads);
    ms_eden_finalize_scales_kernel<<<num_blocks, kFinalizeThreads, 0, stream>>>(
        workspace, reinterpret_cast<uint8_t *>(scales), amax, scale_max, headroom, num_words,
        rng_state);
    NVTE_CHECK_CUDA(cudaGetLastError());
    // Store the amax that defines the global scale.
    ms_eden_store_amax_kernel<<<1, 1, 0, stream>>>(amax, headroom);
    NVTE_CHECK_CUDA(cudaGetLastError());
  } else {
    // Pass 1: rotation + post-rotation amax.
    ms_eden_rht_gemm_ntt<TA, TB, TC, TSFC, kRhtDim, kTiledRhtModeAmax, false, kRowMajorA>(
        m, n, A, B, C, workspace, SFC, amax, 1.0f / 6.0f, scale_max, nullptr, sm_count, stream,
        k_tile_size);
    // Pass 2: rotation + range-aligned NVFP4 cast.
    if (stochastic) {
      ms_eden_rht_gemm_ntt<TA, TB, TC, TSFC, kRhtDim, kTiledRhtModeAligned, true, kRowMajorA>(
          m, n, A, B, C, workspace, SFC, amax, 1.0f / 6.0f, scale_max, rng_state, sm_count,
          stream, k_tile_size);
    } else {
      ms_eden_rht_gemm_ntt<TA, TB, TC, TSFC, kRhtDim, kTiledRhtModeAligned, false, kRowMajorA>(
          m, n, A, B, C, workspace, SFC, amax, 1.0f / 6.0f, scale_max, rng_state, sm_count,
          stream, k_tile_size);
    }
  }
}

}  // namespace
}  // namespace detail

// clang-format on

void quantize_tiled_rht(const Tensor &input_, Tensor &output_, const Tensor &hadamard_matrix_,
                        Tensor *workspace_, QuantizationConfig quant_config, bool rowwise,
                        bool ms_eden, float ms_eden_grid_max, cudaStream_t stream) {
  NVTE_API_CALL(quantize_tiled_rht);

  // Check input and output tensors
  NVTE_CHECK(input_.scaling_mode == NVTE_DELAYED_TENSOR_SCALING,
             "Input tensor must be BF16 tensor, but scaling mode is ",
             to_string(input_.scaling_mode), ".");
  NVTE_CHECK(input_.dtype() == transformer_engine::DType::kBFloat16,
             "Input tensor must be BF16 tensor, but dtype is ", to_string(input_.dtype()), ".");
  NVTE_CHECK(input_.dim() >= 2, "Input must be a 2D tensor.");
  NVTE_CHECK(output_.scaling_mode == NVTE_NVFP4_1D_SCALING,
             "Output tensor must be an NVFP4 tensor, but scaling mode is ",
             to_string(output_.scaling_mode), ".");
  NVTE_CHECK(!output_.with_gemm_swizzled_scales,
             "Tiled-RHT NVFP4 quantization emits scales in compact format.");
  NVTE_CHECK(!output_.row_scaled_nvfp4,
             "Tiled-RHT NVFP4 quantization does not support row-scaled NVFP4.");
  const SimpleTensor &input = input_.data;
  SimpleTensor &global_amax = rowwise ? output_.amax : output_.columnwise_amax;
  SimpleTensor &output_t = rowwise ? output_.data : output_.columnwise_data;
  SimpleTensor &scale_inv_t = rowwise ? output_.scale_inv : output_.columnwise_scale_inv;
  NVTE_CHECK(output_t.dptr != nullptr, "Tiled-RHT NVFP4 quantization: missing output data.");
  NVTE_CHECK(scale_inv_t.dptr != nullptr, "Tiled-RHT NVFP4 quantization: missing output scales.");
  NVTE_CHECK(global_amax.dptr != nullptr, "Tiled-RHT NVFP4 quantization: missing amax buffer.");
  NVTE_CHECK(scale_inv_t.dtype == DType::kFloat8E4M3,
             "Tiled-RHT NVFP4 quantization only supports E4M3 scales, but got ",
             to_string(scale_inv_t.dtype), ".");
  NVTE_CHECK(ms_eden_grid_max > 0.0f, "The MS-EDEN grid parameter must be positive.");

  // Stochastic rounding state (micro-scales for MS-EDEN, elements otherwise)
  const size_t *rng_state = nullptr;
  if (quant_config.rng_state != nullptr) {
    Tensor &rng_state_tensor = *convertNVTETensor(quant_config.rng_state);
    NVTE_CHECK(rng_state_tensor.dtype() == DType::kInt64,
               "RNG state should contain 2 64-bit values.");
    NVTE_CHECK(rng_state_tensor.data.shape == Shape{2},
               "Shape of the RNG state should be [2], but got ", rng_state_tensor.data.shape);
    rng_state = reinterpret_cast<const size_t *>(rng_state_tensor.data.dptr);
  }
  const bool stochastic = !ms_eden && quant_config.stochastic_rounding;

  checkCuDriverContext(stream);

  const int sm_arch = transformer_engine::cuda::sm_arch(transformer_engine::cuda::current_device());
  NVTE_CHECK(sm_arch >= 100 && sm_arch <= 110,
             "Tiled-RHT NVFP4 quantization requires the SM100 family (UMMA/TMEM), but got SM",
             sm_arch, ".");

  // Check Hadamard matrix: rht_dim / 16 sign-folded 16x16 blocks, shape [rht_dim, 16].
  NVTE_CHECK(hadamard_matrix_.scaling_mode == NVTE_DELAYED_TENSOR_SCALING,
             "Hadamard matrix must be BF16 tensor, but scaling mode is ",
             to_string(hadamard_matrix_.scaling_mode), ".");
  NVTE_CHECK(hadamard_matrix_.dtype() == transformer_engine::DType::kBFloat16,
             "Hadamard matrix must be BF16 tensor, but dtype is ",
             to_string(hadamard_matrix_.dtype()), ".");
  const SimpleTensor &hadamard_matrix = hadamard_matrix_.data;
  const bool is_rht_16 = (hadamard_matrix_.shape() == Shape{16, 16});
  const bool is_rht_128 = (hadamard_matrix_.shape() == Shape{128, 16});
  NVTE_CHECK(is_rht_16 || is_rht_128,
             "Tiled-RHT Hadamard matrix must have shape [16, 16] or [128, 16], but got shape=",
             hadamard_matrix_.shape(), ".");

  const size_t ndim = input.shape.size();
  const size_t n = input.shape[ndim - 1];
  size_t m = 1;
  for (size_t i = 0; i < ndim - 1; ++i) {
    m *= input.shape[i];
  }

  // The kernel works on 128 x 128 element tiles in both directions.
  NVTE_CHECK(m % 128 == 0,
             "Tiled-RHT NVFP4 quantization requires the product of all dimensions except the "
             "last to be divisible by 128, but got ",
             m, ".");
  NVTE_CHECK(n % 128 == 0,
             "Tiled-RHT NVFP4 quantization requires the last dimension to be divisible by 128, "
             "but got ",
             n, ".");

  // Kernel view: (M, N) with rotation and NVFP4 groups along N.
  //   columnwise: the [m, n] row-major input is an [n, m] column-major matrix, which makes
  //               the kernel emit the transposed (columnwise) layout rotated along m.
  //   rowwise:    the input is read as an [m, n] row-major matrix and rotated along n.
  const size_t kernel_m = rowwise ? m : n;
  const size_t kernel_n = rowwise ? n : m;
  const size_t num_scales = kernel_m * (kernel_n / 16);
  NVTE_CHECK(scale_inv_t.numel() == num_scales, "Unexpected number of scales (got ",
             scale_inv_t.numel(), ", expected ", num_scales, ").");
  float *workspace_ptr = nullptr;
  if (ms_eden) {
    NVTE_CHECK(workspace_ != nullptr && workspace_->data.dptr != nullptr &&
                   workspace_->data.dtype == DType::kFloat32 &&
                   workspace_->data.numel() >= num_scales,
               "MS-EDEN quantization requires an FP32 workspace with at least ", num_scales,
               " elements.");
    workspace_ptr = reinterpret_cast<float *>(workspace_->data.dptr);
  }

  auto sm_count = transformer_engine::cuda::sm_count();

  // Same tile-size heuristics as the RHT cast-fusion kernel (all multiples of 128),
  // expressed in terms of the kernel view (kernel_n is the rotated dimension).
  int k_tile_size = 1024;
  if (kernel_n == 8192 && kernel_m == 5120) {
    k_tile_size = 512;
  } else if (kernel_n == 8192 && kernel_m == 2560) {
    k_tile_size = 1280;
  } else if (kernel_n == 8192 && kernel_m == 512) {
    k_tile_size = 256;
  } else if (kernel_n == 8192 && kernel_m == 3584) {
    k_tile_size = 512;
  } else if (kernel_n == 5120 && kernel_m == 8192) {
    k_tile_size = 512;
  } else if (kernel_n == 2560 && kernel_m == 8192) {
    k_tile_size = 1280;
  } else if (kernel_n == 512 && kernel_m == 8192) {
    k_tile_size = 256;
  } else if (kernel_n == 3584 && kernel_m == 8192) {
    k_tile_size = 512;
  } else if (kernel_m < 1024 || kernel_n < 1024) {
    k_tile_size = 512;
  }

  // Global scale bound of the output tensor (448 for E4M3 unless overridden).
  const float scale_max = static_cast<float>(output_.get_nvfp4_scale_max());
  auto *amax_ptr = reinterpret_cast<float *>(global_amax.dptr);
  const int km = static_cast<int>(kernel_m);
  const int kn = static_cast<int>(kernel_n);

  if (is_rht_128) {
    if (rowwise) {
      detail::tiled_rht_quantize<128, true>(km, kn, input.dptr, hadamard_matrix.dptr,
                                            output_t.dptr, scale_inv_t.dptr, amax_ptr,
                                            workspace_ptr, ms_eden, stochastic, ms_eden_grid_max,
                                            scale_max, rng_state, sm_count, stream, k_tile_size);
    } else {
      detail::tiled_rht_quantize<128, false>(km, kn, input.dptr, hadamard_matrix.dptr,
                                             output_t.dptr, scale_inv_t.dptr, amax_ptr,
                                             workspace_ptr, ms_eden, stochastic, ms_eden_grid_max,
                                             scale_max, rng_state, sm_count, stream, k_tile_size);
    }
  } else {
    if (rowwise) {
      detail::tiled_rht_quantize<16, true>(km, kn, input.dptr, hadamard_matrix.dptr,
                                           output_t.dptr, scale_inv_t.dptr, amax_ptr,
                                           workspace_ptr, ms_eden, stochastic, ms_eden_grid_max,
                                           scale_max, rng_state, sm_count, stream, k_tile_size);
    } else {
      detail::tiled_rht_quantize<16, false>(km, kn, input.dptr, hadamard_matrix.dptr,
                                            output_t.dptr, scale_inv_t.dptr, amax_ptr,
                                            workspace_ptr, ms_eden, stochastic, ms_eden_grid_max,
                                            scale_max, rng_state, sm_count, stream, k_tile_size);
    }
  }
}

}  // namespace transformer_engine

void nvte_quantize_ms_eden_columnwise(const NVTETensor input, NVTETensor output,
                                      const NVTETensor hadamard_matrix, NVTETensor workspace,
                                      const NVTEQuantizationConfig quant_config,
                                      cudaStream_t stream) {
  NVTE_API_CALL(nvte_quantize_ms_eden_columnwise);
  using namespace transformer_engine;
  QuantizationConfig quant_config_cpp;
  if (quant_config != nullptr) {
    quant_config_cpp = *reinterpret_cast<QuantizationConfig *>(quant_config);
  }
  quantize_tiled_rht(*convertNVTETensorCheck(input), *convertNVTETensorCheck(output),
                     *convertNVTETensorCheck(hadamard_matrix), convertNVTETensorCheck(workspace),
                     quant_config_cpp, /*rowwise=*/false, /*ms_eden=*/true,
                     /*ms_eden_grid_max=*/6.0f, stream);
}

void nvte_quantize_tiled_rht(const NVTETensor input, NVTETensor output,
                             const NVTETensor hadamard_matrix, NVTETensor workspace,
                             const NVTEQuantizationConfig quant_config, int rowwise, int ms_eden,
                             float ms_eden_grid_max, cudaStream_t stream) {
  NVTE_API_CALL(nvte_quantize_tiled_rht);
  using namespace transformer_engine;
  QuantizationConfig quant_config_cpp;
  if (quant_config != nullptr) {
    quant_config_cpp = *reinterpret_cast<QuantizationConfig *>(quant_config);
  }
  quantize_tiled_rht(*convertNVTETensorCheck(input), *convertNVTETensorCheck(output),
                     *convertNVTETensorCheck(hadamard_matrix),
                     workspace != nullptr ? convertNVTETensorCheck(workspace) : nullptr,
                     quant_config_cpp, rowwise != 0, ms_eden != 0, ms_eden_grid_max, stream);
}
