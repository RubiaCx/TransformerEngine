/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

/*! \file quantize_sr_mxfp8.cuh
 *  \brief CUDA kernels to quantize to MXFP8 with stochastic rounding of the elements.
 *
 *  The E8M0 block scales are computed exactly as in the round-to-nearest kernels
 *  (quantize_mxfp8.cuh): float_to_e8m0(amax * max_norm_rcp) per 1x32 (rowwise) or
 *  32x1 (columnwise) block, written in the same compact or GEMM-swizzled layout.
 *  Only the element rounding differs: every scaled element x * 2^-e is rounded
 *  to one of its two neighbouring E4M3 values by ptx::stochastic_round_fp8_e4m3.
 *
 *  Random numbers come from Philox4x32 used as a counter-based generator, with
 *  max(NVTE_BUILD_NUM_PHILOX_ROUNDS, 7) rounds. One
 *  generate4() call yields four 32-bit words that serve four elements, of which
 *  the top 24 bits each are consumed. The key is the seed in rng_state[0], the
 *  low counter half is the offset in rng_state[1], and the high counter half is
 *  (group_index << 1) | direction, where group_index identifies the four
 *  elements by their position in the tensor. Elements therefore have
 *  independent streams, the rowwise and columnwise outputs of one call draw
 *  different numbers, and the output only depends on rng_state, the input and
 *  the tensor shape (not on the launch configuration).
 *
 *  The kernels read and write global memory directly: adjacent threads of a
 *  warp touch adjacent addresses, and nothing is staged in shared memory. The
 *  rowwise and columnwise outputs are produced by two separate launches.
 *
 *  Shapes and alignment follow the RN path: the last dimension must be a
 *  multiple of 16 and the data pointers 16-byte aligned (the TMA constraints of
 *  the generic RN kernel), so all vector loads and stores below are aligned.
 *  Either dimension may be a non-multiple of 32; partial blocks are scaled by
 *  the amax of their in-bounds elements.
 */

#ifndef TRANSFORMER_ENGINE_QUANTIZE_SR_MXFP8_CUH_
#define TRANSFORMER_ENGINE_QUANTIZE_SR_MXFP8_CUH_

#include <cuda.h>
#include <cudaTypedefs.h>
#include <cuda_runtime.h>
#include <transformer_engine/transformer_engine.h>

#include "../../common.h"
#include "../../util/curanddx.hpp"
#include "../../util/cuda_runtime.h"
#include "../../util/math.h"
#include "../../util/ptx_arch_spec.cuh"
#include "../../utils.cuh"
#include "../core/common.cuh"
#include "quantize_mxfp8.cuh"
#include "swizzle.cuh"

namespace transformer_engine {
namespace dispatch {
namespace mxfp8 {
namespace quantize_sr_kernel {

constexpr size_t SR_BLOCK_DIM = 32;  // MXFP8 block length
constexpr size_t SR_THREADS = 256;
constexpr size_t SR_WARPS = SR_THREADS / THREADS_PER_WARP;

// Rowwise: a thread owns 4 consecutive elements of a row, so a warp covers 128 columns
// (four 1x32 blocks) and 8 neighbouring lanes share a block.
constexpr size_t SR_ROW_ELTS_PER_THREAD = 4;
constexpr size_t SR_ROW_TILE_DIM_X = SR_ROW_ELTS_PER_THREAD * THREADS_PER_WARP;  // 128
constexpr size_t SR_ROW_LANES_PER_BLOCK = SR_BLOCK_DIM / SR_ROW_ELTS_PER_THREAD;  // 8
constexpr size_t SR_ROW_TILES_PER_WARP = 4;

// Columnwise: a thread owns 8 bytes of input per row (2 FP32 or 4 BF16 columns) over the
// 32 rows of a block and keeps them in registers between the amax and the rounding pass.
constexpr size_t SR_COL_BYTES_PER_THREAD = 8;

// Philox is used as a counter-based generator here: every group of four elements hashes
// its own counter once, and counters of neighbouring groups differ in a few low bits only.
// That needs the full diffusion of the cipher, unlike the NVFP4 kernels, which advance one
// stream per thread. Philox4x32-7 is the smallest variant that passes BigCrush, so the
// build-time round count is honoured only when it is at least 7.
constexpr unsigned int SR_MIN_PHILOX_ROUNDS = 7;
constexpr unsigned int SR_PHILOX_ROUNDS = (NVTE_BUILD_NUM_PHILOX_ROUNDS > SR_MIN_PHILOX_ROUNDS)
                                              ? NVTE_BUILD_NUM_PHILOX_ROUNDS
                                              : SR_MIN_PHILOX_ROUNDS;

// Direction tags mixed into the Philox counter.
constexpr uint64_t SR_STREAM_ROWWISE = 0;
constexpr uint64_t SR_STREAM_COLWISE = 1;

// 128 random bits for the group of four elements `group_index` of stream `direction`.
__device__ __forceinline__ uint4 sr_random_bits(const uint64_t rng_seed, const uint64_t rng_offset,
                                                const uint64_t group_index,
                                                const uint64_t direction) {
  transformer_engine::curanddx::detail::philox4x32_native_state<SR_PHILOX_ROUNDS> rng;
  rng.init(rng_seed, /*subsequence=*/(group_index << 1) | direction, /*offset=*/rng_offset);
  return rng.generate4();
}

// Raw input bits to FP32. BF16 is the upper half of the FP32 encoding.
template <typename IType>
__device__ __forceinline__ float sr_bits_to_float(const uint32_t bits) {
  if constexpr (std::is_same_v<IType, float>) {
    return __uint_as_float(bits);
  } else {
    return __uint_as_float(bits << 16);
  }
}

template <typename IType, bool WITH_GEMM_SWIZZLED_SCALES>
__global__ void __launch_bounds__(SR_THREADS)
    quantize_sr_rowwise_kernel(const IType *const input, uint8_t *const output,
                               e8m0_t *const scales, const float *const noop, const size_t rows,
                               const size_t cols, const size_t scale_stride,
                               const size_t *const rng_state) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  using transformer_engine::dispatch::mxfp8::swizzle::gemm_swizzled_scale_idx;
  constexpr size_t ELTS = SR_ROW_ELTS_PER_THREAD;

  if (noop != nullptr && noop[0] == 1.0f) {
    return;
  }
  const uint64_t rng_seed = rng_state != nullptr ? rng_state[0] : 0;
  const uint64_t rng_offset = rng_state != nullptr ? rng_state[1] : 0;

  const size_t tiles_per_row = DIVUP(cols, static_cast<size_t>(SR_ROW_TILE_DIM_X));
  const size_t groups_per_row = DIVUP(cols, ELTS);
  const size_t num_tiles = rows * tiles_per_row;
  const size_t lane = threadIdx.x % THREADS_PER_WARP;
  const size_t warp = threadIdx.x / THREADS_PER_WARP;

  size_t tile = (blockIdx.x * SR_WARPS + warp) * SR_ROW_TILES_PER_WARP;
  if (tile >= num_tiles) {
    return;
  }
  size_t row = tile / tiles_per_row;
  size_t tile_x = tile % tiles_per_row;

  for (size_t iter = 0; iter < SR_ROW_TILES_PER_WARP && row < rows; ++iter) {
    const size_t col = tile_x * SR_ROW_TILE_DIM_X + lane * ELTS;
    const size_t offset = row * cols + col;

    // 1. Load four elements. cols is a multiple of 4, so they are all inside or all
    //    outside the row. Out-of-bounds elements count as zero, as in the RN kernels.
    const bool in_bounds = col < cols;
    float x[ELTS] = {0.0f, 0.0f, 0.0f, 0.0f};
    if (in_bounds) {
      if constexpr (std::is_same_v<IType, float>) {
        const float4 v = *reinterpret_cast<const float4 *>(input + offset);
        x[0] = v.x;
        x[1] = v.y;
        x[2] = v.z;
        x[3] = v.w;
      } else {
        const uint64_t v = *reinterpret_cast<const uint64_t *>(input + offset);
#pragma unroll
        for (int e = 0; e < ELTS; ++e) {
          x[e] = sr_bits_to_float<IType>(static_cast<uint32_t>(v >> (16 * e)) & 0xFFFFu);
        }
      }
    }

    // 2. Block amax: fmaxf drops NaNs unless the whole block is NaN, as in the RN kernels.
    float amax = fmaxf(fmaxf(fabsf(x[0]), fabsf(x[1])), fmaxf(fabsf(x[2]), fabsf(x[3])));
#pragma unroll
    for (int step = 1; step < SR_ROW_LANES_PER_BLOCK; step *= 2) {
      amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFFu, amax, step));
    }

    // 3. E8M0 scale, identical to the RN path.
    const e8m0_t biased_exponent =
        ptx::float_to_e8m0(amax * Quantized_Limits<fp8e4m3>::max_norm_rcp);
    if ((lane % SR_ROW_LANES_PER_BLOCK == 0) && in_bounds) {
      const size_t scale_x = col / SR_BLOCK_DIM;
      size_t scale_idx;
      if constexpr (WITH_GEMM_SWIZZLED_SCALES) {
        scale_idx =
            gemm_swizzled_scale_idx(row, scale_x, DIVUP(cols, static_cast<size_t>(128)));
      } else {
        scale_idx = row * scale_stride + scale_x;
      }
      scales[scale_idx] = biased_exponent;
    }
    const float block_scale_inverse = ptx::exp2f_rcp<float>(biased_exponent);

    // 4. Scale and round stochastically.
    const uint4 rbits = sr_random_bits(rng_seed, rng_offset, row * groups_per_row + col / ELTS,
                                       SR_STREAM_ROWWISE);
    const uint32_t q0 = ptx::stochastic_round_fp8_e4m3(x[0] * block_scale_inverse, rbits.x);
    const uint32_t q1 = ptx::stochastic_round_fp8_e4m3(x[1] * block_scale_inverse, rbits.y);
    const uint32_t q2 = ptx::stochastic_round_fp8_e4m3(x[2] * block_scale_inverse, rbits.z);
    const uint32_t q3 = ptx::stochastic_round_fp8_e4m3(x[3] * block_scale_inverse, rbits.w);

    // 5. Store.
    if (in_bounds) {
      *reinterpret_cast<uint32_t *>(output + offset) = q0 | (q1 << 8) | (q2 << 16) | (q3 << 24);
    }

    if (++tile_x == tiles_per_row) {
      tile_x = 0;
      ++row;
    }
  }
#endif  // #if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
}

template <typename IType, bool WITH_GEMM_SWIZZLED_SCALES>
__global__ void __launch_bounds__(SR_THREADS)
    quantize_sr_colwise_kernel(const IType *const input, uint8_t *const output,
                               e8m0_t *const scales, const float *const noop, const size_t rows,
                               const size_t cols, const size_t scale_stride,
                               const size_t *const rng_state) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  using transformer_engine::dispatch::mxfp8::swizzle::gemm_swizzled_scale_idx;
  // Columns per thread, and rows that share one Philox call (4 elements per call).
  constexpr int COLS_PT = SR_COL_BYTES_PER_THREAD / sizeof(IType);
  constexpr int ROWS_PER_CALL = 4 / COLS_PT;
  constexpr int ELT_BITS = 8 * sizeof(IType);
  constexpr uint64_t ELT_MASK = (uint64_t{1} << ELT_BITS) - 1;
  constexpr size_t TILE_DIM_X = COLS_PT * THREADS_PER_WARP;
  static_assert(COLS_PT == 2 || COLS_PT == 4);

  if (noop != nullptr && noop[0] == 1.0f) {
    return;
  }
  const uint64_t rng_seed = rng_state != nullptr ? rng_state[0] : 0;
  const uint64_t rng_offset = rng_state != nullptr ? rng_state[1] : 0;

  const size_t tiles_per_row = DIVUP(cols, TILE_DIM_X);
  const size_t groups_per_row = DIVUP(cols, static_cast<size_t>(COLS_PT));
  const size_t num_tiles = DIVUP(rows, static_cast<size_t>(SR_BLOCK_DIM)) * tiles_per_row;
  const size_t lane = threadIdx.x % THREADS_PER_WARP;
  const size_t warp = threadIdx.x / THREADS_PER_WARP;

  const size_t tile = blockIdx.x * SR_WARPS + warp;
  if (tile >= num_tiles) {
    return;
  }
  const size_t tile_y = tile / tiles_per_row;
  const size_t tile_x = tile % tiles_per_row;
  const size_t row0 = tile_y * SR_BLOCK_DIM;
  const size_t col = tile_x * TILE_DIM_X + lane * COLS_PT;
  if (col >= cols) {
    return;
  }

  // 1. Load the 32 x COLS_PT elements and find the amax of each column. cols is a multiple
  //    of COLS_PT, so the columns of this thread are all in bounds. Out-of-bounds rows count
  //    as zero, as in the RN kernels.
  uint64_t raw[SR_BLOCK_DIM];
  float amax[COLS_PT];
#pragma unroll
  for (int c = 0; c < COLS_PT; ++c) {
    amax[c] = __int_as_float(0x7FC00000);
  }
#pragma unroll
  for (int i = 0; i < SR_BLOCK_DIM; ++i) {
    const size_t row = row0 + i;
    const size_t offset = row * cols + col;
    uint64_t v = 0;
    if (row < rows) {
      v = *reinterpret_cast<const uint64_t *>(input + offset);
    }
    raw[i] = v;
#pragma unroll
    for (int c = 0; c < COLS_PT; ++c) {
      const float x =
          sr_bits_to_float<IType>(static_cast<uint32_t>((v >> (ELT_BITS * c)) & ELT_MASK));
      amax[c] = fmaxf(amax[c], fabsf(x));
    }
  }

  // 2. E8M0 scales, identical to the RN path.
  float block_scale_inverse[COLS_PT];
#pragma unroll
  for (int c = 0; c < COLS_PT; ++c) {
    const e8m0_t biased_exponent =
        ptx::float_to_e8m0(amax[c] * Quantized_Limits<fp8e4m3>::max_norm_rcp);
    size_t scale_idx;
    if constexpr (WITH_GEMM_SWIZZLED_SCALES) {
      scale_idx = gemm_swizzled_scale_idx(col + c, tile_y, DIVUP(rows, static_cast<size_t>(128)));
    } else {
      scale_idx = tile_y * scale_stride + col + c;
    }
    scales[scale_idx] = biased_exponent;
    block_scale_inverse[c] = ptx::exp2f_rcp<float>(biased_exponent);
  }

  // 3. Scale, round stochastically and store.
#pragma unroll
  for (int i0 = 0; i0 < SR_BLOCK_DIM; i0 += ROWS_PER_CALL) {
    const uint4 rbits4 =
        sr_random_bits(rng_seed, rng_offset,
                       ((row0 + i0) / ROWS_PER_CALL) * groups_per_row + col / COLS_PT,
                       SR_STREAM_COLWISE);
    const uint32_t rbits[4] = {rbits4.x, rbits4.y, rbits4.z, rbits4.w};
#pragma unroll
    for (int k = 0; k < ROWS_PER_CALL; ++k) {
      const int i = i0 + k;
      const size_t row = row0 + i;
      const size_t offset = row * cols + col;
      uint32_t q[COLS_PT];
#pragma unroll
      for (int c = 0; c < COLS_PT; ++c) {
        const float x =
            sr_bits_to_float<IType>(static_cast<uint32_t>((raw[i] >> (ELT_BITS * c)) & ELT_MASK));
        q[c] = ptx::stochastic_round_fp8_e4m3(x * block_scale_inverse[c], rbits[k * COLS_PT + c]);
      }
      if (row < rows) {
        if constexpr (COLS_PT == 2) {
          *reinterpret_cast<uint16_t *>(output + offset) =
              static_cast<uint16_t>(q[0] | (q[1] << 8));
        } else {
          *reinterpret_cast<uint32_t *>(output + offset) =
              q[0] | (q[1] << 8) | (q[2] << 16) | (q[3] << 24);
        }
      }
    }
  }
#endif  // #if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
}

}  // namespace quantize_sr_kernel

/*! \brief Plain cast to MXFP8 (1D block scaling, E4M3) with stochastic rounding. */
inline void quantize_sr(const Tensor &input, const Tensor *noop, Tensor *output,
                        const QuantizationConfig *quant_config, cudaStream_t stream) {
  using namespace quantize_sr_kernel;

  const bool use_rowwise_scaling = output->has_data();
  const bool use_colwise_scaling = output->has_columnwise_data();
  NVTE_CHECK(transformer_engine::cuda::sm_arch() >= 100,
             "MXFP8 quantization with stochastic rounding requires SM 10.0+.");
  NVTE_CHECK(input.has_data(), "Cannot quantize tensor without rowwise data.");
  NVTE_CHECK(output->dtype() == DType::kFloat8E4M3,
             "MXFP8 quantization with stochastic rounding only supports Float8E4M3 output, got ",
             to_string(output->dtype()), ".");
  NVTE_CHECK(input.dtype() == DType::kFloat32 || input.dtype() == DType::kBFloat16,
             "MXFP8 quantization with stochastic rounding only supports Float32 and BFloat16 "
             "input, got ",
             to_string(input.dtype()), ".");
  NVTE_CHECK(!quant_config->mxfp8_2d_quantization,
             "MXFP8 quantization with stochastic rounding does not support 2D block scaling.");
  NVTE_CHECK(output->amax.dptr == nullptr,
             "MXFP8 quantization with stochastic rounding does not compute amax.");
  if (use_rowwise_scaling) {
    NVTE_CHECK(output->scale_inv.dptr != nullptr, "Scaling tensor must be allocated");
  }
  if (use_colwise_scaling) {
    NVTE_CHECK(output->columnwise_scale_inv.dptr != nullptr,
               "Columnwise scaling tensor must be allocated");
  }
  CheckNoopTensor(*noop, "cast_noop");

  // RNG state: [seed, offset], read on the device.
  const size_t *rng_state = nullptr;
  if (quant_config->rng_state != nullptr) {
    const Tensor &rng_state_tensor = *convertNVTETensorCheck(quant_config->rng_state);
    NVTE_CHECK(rng_state_tensor.dtype() == DType::kInt64,
               "RNG state should contain 2 64-bit values.");
    NVTE_CHECK(rng_state_tensor.data.shape == Shape{2},
               "Shape of the RNG state should be [2], but got ", rng_state_tensor.data.shape);
    rng_state = reinterpret_cast<const size_t *>(rng_state_tensor.data.dptr);
  }

  const auto [rows, cols] = input.flat_2d_dims();
  if (rows == 0 || cols == 0) {
    return;
  }

  // Same restrictions as the TMA tensor maps of the generic RN kernel.
  constexpr size_t cols_multiple = TMA_GMEM_ALIGNMENT;  // 16 one-byte output elements
  NVTE_CHECK(cols % cols_multiple == 0, "Shape not supported. Expected multiple of ",
             cols_multiple, ", got ", cols);
  NVTE_CHECK(is_aligned_ptr(input.data.dptr, TMA_GMEM_ALIGNMENT) &&
                 (!use_rowwise_scaling || is_aligned_ptr(output->data.dptr, TMA_GMEM_ALIGNMENT)) &&
                 (!use_colwise_scaling ||
                  is_aligned_ptr(output->columnwise_data.dptr, TMA_GMEM_ALIGNMENT)),
             "Tensor data pointer must be 16B aligned");

  const bool with_gemm_swizzled_scales = output->with_gemm_swizzled_scales;
  const float *const noop_ptr = reinterpret_cast<const float *>(noop->data.dptr);

  // Clear the padding of GEMM-swizzled scales, as the RN path does.
  if (with_gemm_swizzled_scales && (cols % 128 != 0 || rows % 128 != 0)) {
    constexpr size_t zero_threads = 256;
    for (const auto *scale : {use_rowwise_scaling ? &output->scale_inv : nullptr,
                              use_colwise_scaling ? &output->columnwise_scale_inv : nullptr}) {
      if (scale == nullptr) continue;
      const size_t size_bytes = scale->buffer_size_bytes();
      if (size_bytes > 0) {
        quantize_kernel::zero_scales_kernel<<<DIVUP(size_bytes, zero_threads), zero_threads, 0,
                                              stream>>>(reinterpret_cast<uint8_t *>(scale->dptr),
                                                        size_bytes, noop_ptr);
        NVTE_CHECK_CUDA(cudaGetLastError());
      }
    }
  }

  TRANSFORMER_ENGINE_SWITCH_CONDITION(
      input.dtype() == DType::kFloat32, IS_FP32,
      TRANSFORMER_ENGINE_SWITCH_CONDITION(
          with_gemm_swizzled_scales, WITH_GEMM_SWIZZLED_SCALES,

          using IType = std::conditional_t<IS_FP32, float, bf16>;
          const IType *const input_ptr = reinterpret_cast<const IType *>(input.data.dptr);

          if (use_rowwise_scaling) {
            const size_t num_tiles = rows * DIVUP(cols, SR_ROW_TILE_DIM_X);
            const size_t grid = DIVUP(num_tiles, SR_WARPS * SR_ROW_TILES_PER_WARP);
            NVTE_CHECK(grid <= 0x7FFFFFFFul, "Tensor is too large for MXFP8 quantization.");
            quantize_sr_rowwise_kernel<IType, WITH_GEMM_SWIZZLED_SCALES>
                <<<grid, SR_THREADS, 0, stream>>>(
                    input_ptr, reinterpret_cast<uint8_t *>(output->data.dptr),
                    reinterpret_cast<e8m0_t *>(output->scale_inv.dptr), noop_ptr, rows, cols,
                    output->scale_inv.shape[1], rng_state);
            NVTE_CHECK_CUDA(cudaGetLastError());
          }

          if (use_colwise_scaling) {
            constexpr size_t cols_per_thread = SR_COL_BYTES_PER_THREAD / sizeof(IType);
            const size_t num_tiles =
                DIVUP(rows, SR_BLOCK_DIM) * DIVUP(cols, cols_per_thread * THREADS_PER_WARP);
            const size_t grid = DIVUP(num_tiles, SR_WARPS);
            NVTE_CHECK(grid <= 0x7FFFFFFFul, "Tensor is too large for MXFP8 quantization.");
            quantize_sr_colwise_kernel<IType, WITH_GEMM_SWIZZLED_SCALES>
                <<<grid, SR_THREADS, 0, stream>>>(
                    input_ptr, reinterpret_cast<uint8_t *>(output->columnwise_data.dptr),
                    reinterpret_cast<e8m0_t *>(output->columnwise_scale_inv.dptr), noop_ptr, rows,
                    cols, output->columnwise_scale_inv.shape[1], rng_state);
            NVTE_CHECK_CUDA(cudaGetLastError());
          }););  // NOLINT(*)
}

}  // namespace mxfp8
}  // namespace dispatch
}  // namespace transformer_engine

#endif  // TRANSFORMER_ENGINE_QUANTIZE_SR_MXFP8_CUH_
