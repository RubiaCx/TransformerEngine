/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

/*! \file rtc_dispatch.h
 *  \brief Host-side NVRTC dispatch for the NVFP4 4over6 quantize kernel.
 */

#ifndef TRANSFORMER_ENGINE_CAST_NVFP4_RTC_DISPATCH_H_
#define TRANSFORMER_ENGINE_CAST_NVFP4_RTC_DISPATCH_H_

#include <cuda.h>

#include <string>

#include "../../util/rtc.h"
// do not include util/string.h here — it pulls in <regex>, which is heavy.
#include "core_nvfp4.cuh"

namespace transformer_engine {
namespace dispatch {
namespace nvfp4 {
namespace rtc_nvfp4 {

#if FP4_TYPE_SUPPORTED

using nvfp4_scale_t = fp8e4m3;

void compile_quantize_4over6_rtc(const std::string &kernel_label, const std::string &itype_name,
                                 bool use_2d, bool return_identity, bool return_transpose,
                                 bool row_scaled, const std::string &mode_name, bool err_fast_math,
                                 int e4m3_max, bool col_standard, bool from_rowwise = false,
                                 bool amax_only = false, int col_e4m3_max = 448);

template <typename T>
inline const char *rtc_type_name() {
  return detail::type_name<T>();
}
template <>
inline const char *rtc_type_name<fp16>() {
  return "fp16";
}
template <>
inline const char *rtc_type_name<bf16>() {
  return "bf16";
}

// Enum literal spelling for the __MODE__ substitution / cache key.
inline const char *rtc_mode_name(NVTENVFP44Over6Mode mode) {
  switch (mode) {
    case kNVTENVFP44Over6MinMSE:
      return "kNVTENVFP44Over6MinMSE";
    case kNVTENVFP44Over6MinMAE:
      return "kNVTENVFP44Over6MinMAE";
    default:
      NVTE_ERROR("Unsupported NVFP4 4over6 mode.");
  }
}

template <bool USE_2D_QUANTIZATION, typename Cfg, int E4M3_MAX, typename IType>
inline void launch_quantize_4over6_rtc(const IType *input, fp4e2m1x2 *output, fp4e2m1x2 *output_t,
                                       nvfp4_scale_t *scales, nvfp4_scale_t *scales_t,
                                       const float *amax_rowwise, const float *amax_colwise,
                                       size_t rows, size_t cols, size_t scale_stride,
                                       size_t scale_stride_t, const float *noop,
                                       bool return_identity, bool return_transpose,
                                       bool row_scaled_nvfp4, dim3 grid, dim3 block, size_t shmem,
                                       cudaStream_t stream, bool col_standard = false) {
  const char *itype_name = rtc_type_name<IType>();
  const char *mode_name = rtc_mode_name(Cfg::mode);

  // Cache key encodes everything that varies the compiled kernel.
  std::string kernel_label;
  kernel_label.reserve(160);
  kernel_label.append("quantize_4over6,itype=").append(itype_name);
  kernel_label.append(",use2d=").push_back(USE_2D_QUANTIZATION ? '1' : '0');
  kernel_label.append(",id=").push_back(return_identity ? '1' : '0');
  kernel_label.append(",t=").push_back(return_transpose ? '1' : '0');
  kernel_label.append(",rowscaled=").push_back(row_scaled_nvfp4 ? '1' : '0');
  kernel_label.append(",mode=").append(mode_name);
  kernel_label.append(",fastmath=").push_back(Cfg::err_use_fast_math ? '1' : '0');
  kernel_label.append(",e4m3max=").append(std::to_string(E4M3_MAX));
  kernel_label.append(",colstd=").push_back(col_standard ? '1' : '0');

  auto &mgr = rtc::KernelManager::instance();
  if (!mgr.is_compiled(kernel_label)) {
    compile_quantize_4over6_rtc(kernel_label, itype_name, USE_2D_QUANTIZATION, return_identity,
                                return_transpose, row_scaled_nvfp4, mode_name,
                                Cfg::err_use_fast_math, E4M3_MAX, col_standard);
  }

  if (shmem > 0) {
    mgr.set_function_attribute(kernel_label, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                               static_cast<int>(shmem));
  }

  mgr.launch(kernel_label, grid, block, static_cast<unsigned int>(shmem), stream, input, output,
             output_t, scales, scales_t, amax_rowwise, amax_colwise, rows, cols, scale_stride,
             scale_stride_t, noop);
}

// Columnwise copy quantized from the dequantized rowwise copy (or, with amax_only, the amax of
// the dequantized rowwise copy).
template <bool USE_2D_QUANTIZATION, typename Cfg, int ROW_E4M3_MAX, int COL_E4M3_MAX,
          typename IType>
inline void launch_quantize_4over6_from_rowwise_rtc(
    const fp4e2m1x2 *codes, const nvfp4_scale_t *row_scales, const float *amax_rowwise,
    fp4e2m1x2 *output_t, nvfp4_scale_t *scales_t, float *amax_colwise, size_t rows, size_t cols,
    size_t scale_stride, size_t scale_stride_t, bool col_standard, bool amax_only, dim3 grid,
    dim3 block, size_t shmem, cudaStream_t stream) {
  const char *itype_name = rtc_type_name<IType>();
  const char *mode_name = rtc_mode_name(Cfg::mode);

  std::string kernel_label;
  kernel_label.reserve(192);
  kernel_label.append("quantize_4over6_from_rowwise,itype=").append(itype_name);
  kernel_label.append(",use2d=").push_back(USE_2D_QUANTIZATION ? '1' : '0');
  kernel_label.append(",mode=").append(mode_name);
  kernel_label.append(",fastmath=").push_back(Cfg::err_use_fast_math ? '1' : '0');
  kernel_label.append(",rowmax=").append(std::to_string(ROW_E4M3_MAX));
  kernel_label.append(",colmax=").append(std::to_string(COL_E4M3_MAX));
  kernel_label.append(",colstd=").push_back(col_standard ? '1' : '0');
  kernel_label.append(",amaxonly=").push_back(amax_only ? '1' : '0');

  auto &mgr = rtc::KernelManager::instance();
  if (!mgr.is_compiled(kernel_label)) {
    compile_quantize_4over6_rtc(kernel_label, itype_name, USE_2D_QUANTIZATION,
                                /*return_identity=*/false, /*return_transpose=*/true,
                                /*row_scaled=*/false, mode_name, Cfg::err_use_fast_math,
                                ROW_E4M3_MAX, col_standard, /*from_rowwise=*/true, amax_only,
                                COL_E4M3_MAX);
  }
  mgr.set_function_attribute(kernel_label, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                             static_cast<int>(shmem));
  mgr.launch(kernel_label, grid, block, static_cast<unsigned int>(shmem), stream, codes,
             row_scales, amax_rowwise, output_t, scales_t, amax_colwise, rows, cols, scale_stride,
             scale_stride_t);
}

#endif  // FP4_TYPE_SUPPORTED

}  // namespace rtc_nvfp4
}  // namespace nvfp4
}  // namespace dispatch
}  // namespace transformer_engine

#endif  // TRANSFORMER_ENGINE_CAST_NVFP4_RTC_DISPATCH_H_
