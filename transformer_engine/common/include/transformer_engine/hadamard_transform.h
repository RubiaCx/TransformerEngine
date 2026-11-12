/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

/*! \file hadamard_transform.h
 *  \brief Functions for Hadamard transforms.
 */

#ifndef TRANSFORMER_ENGINE_HADAMARD_TRANSFORM_H_
#define TRANSFORMER_ENGINE_HADAMARD_TRANSFORM_H_

#include "transformer_engine.h"

#ifdef __cplusplus
extern "C" {
#endif

/*! \brief Perform a randomized Hadamard transform on the input tensor.
 *
 *  This function is experimental and the API is not stable.
 *
 *  \param[in]      input              Input tensor to apply Hadamard transform.
 *  \param[in,out]  output             Output tensor.
 *  \param[in]      random_sign_mask   16-bit sign mask.
 *  \param[in]      random_sign_mask_t 16-bit sign mask.
 *  \param[in]      stream             CUDA stream used for the operation.
 */
void nvte_hadamard_transform(const NVTETensor input, NVTETensor output, int random_sign_mask,
                             int random_sign_mask_t, cudaStream_t stream);

/*! \brief Perform the absolute maximum reduction on the input tensor with/without
 *         randomized hadamard transform. The rowwise result is the absolute maximum
 *         of the input tensor. The columnwise result is the absolute maximum of the
 *         input tensor transposed and applied randomized hadamard transformation.
 *
 *  This function is experimental and the API is not stable.
 *
 *  \param[in]      input              Input tensor to apply Hadamard transform.
 *  \param[in,out]  output             Output tensor.
 *  \param[in]      random_sign_mask   16-bit sign mask.
 *  \param[in]      random_sign_mask_t 16-bit sign mask.
 *  \param[in]      stream             CUDA stream used for the operation.
 */
void nvte_hadamard_transform_amax(const NVTETensor input, NVTETensor output, int random_sign_mask,
                                  int random_sign_mask_t, cudaStream_t stream);

/*! \brief Perform the columnwise hadamard transform cast fusion.
 *
 *  \deprecated This function has been deprecated in favor of nvte_quantize_with_hadamard_transform.
 *
 *  \param[in]      input           Input tensor to apply Hadamard transform.
 *  \param[in,out]  output          Output tensor.
 *  \param[in]      hadamard_matrix Hadamard matrix.
 *  \param[in]      quant_config    Quantization configuration.
 *  \param[in]      stream          CUDA stream used for the operation.
 */
void nvte_hadamard_transform_cast_fusion_columnwise(const NVTETensor input, NVTETensor output,
                                                    const NVTETensor hadamard_matrix,
                                                    const NVTEQuantizationConfig quant_config,
                                                    cudaStream_t stream);

/*! \brief Perform the regular rowwise cast and columnwise hadamard transform cast fusion.
 *
 *  This function is experimental and the API is not stable.
 *
 *  \param[in]      input           Input tensor to apply Hadamard transform.
 *  \param[in,out]  output          Output tensor.
 *  \param[in]      hadamard_matrix Hadamard matrix.
 *  \param[in]      quant_config    Quantization configuration.
 *  \param[in]      stream          CUDA stream used for the operation.
 */
void nvte_quantize_with_hadamard_transform(const NVTETensor input, NVTETensor output,
                                           const NVTETensor hadamard_matrix,
                                           const NVTEQuantizationConfig quant_config,
                                           cudaStream_t stream);

/*! \brief Columnwise MS-EDEN NVFP4 quantization with a tiled random Hadamard transform.
 *
 *  This function is experimental and the API is not stable.
 *
 *  Implements MS-EDEN (Quartet II, arXiv:2601.22813, Algorithm 1) with post hoc
 *  range alignment. Only the columnwise output (data, scales and amax) is written.
 *  For a 2D input ``x`` of shape [m, n] the columnwise output holds ``x^T``; each
 *  row of ``x^T`` (length m) is split into chunks of ``rht_dim`` elements which are
 *  rotated with the randomized Hadamard transform, rounded to E2M1 with
 *  round-to-nearest against one micro-scale per 16 elements, and the micro-scales
 *  are multiplied by the EDEN factor ``<x_rht, x_rht> / <x_rht, x_rtn>`` of their
 *  chunk and stochastically rounded to E4M3. No element-level stochastic rounding
 *  is applied.
 *
 *  Pass structure: one fused kernel reads the input once (rotation, RTN, EDEN
 *  factors, post-rotation amax) and a second kernel touches only the scales
 *  (range alignment against the amax, stochastic rounding).
 *
 *  The micro-scales are generated against a global scale bound of 256 instead of
 *  448 so that the EDEN correction cannot overflow E4M3. To keep the output a
 *  standard NVFP4 tensor, the stored columnwise amax is the post-rotation amax
 *  multiplied by 448 / 256.
 *
 *  Requirements: SM100 family, BF16 input, m and n divisible by 128, E4M3 scales
 *  in compact (non-swizzled) layout.
 *
 *  \param[in]      input           Input tensor (BF16).
 *  \param[in,out]  output          NVFP4 output tensor. Only columnwise fields are written.
 *  \param[in]      hadamard_matrix BF16 tensor of shape [rht_dim, 16] with rht_dim in
 *                                  {16, 128}. Row k holds
 *                                  ``sign[k] * H16[k % 16, :] / 4``.
 *  \param[in,out]  workspace       FP32 workspace with at least ``m * n / 16`` elements.
 *  \param[in]      quant_config    Quantization configuration. ``rng_state`` seeds the
 *                                  stochastic rounding of the micro-scales.
 *  \param[in]      stream          CUDA stream used for the operation.
 */
void nvte_quantize_ms_eden_columnwise(const NVTETensor input, NVTETensor output,
                                      const NVTETensor hadamard_matrix, NVTETensor workspace,
                                      const NVTEQuantizationConfig quant_config,
                                      cudaStream_t stream);

/*! \brief NVFP4 quantization of one copy (rowwise or columnwise) with a tiled random
 *         Hadamard transform of dimension 16 or 128.
 *
 *  This function is experimental and the API is not stable.
 *
 *  Generalization of nvte_quantize_ms_eden_columnwise.
 *
 *  - ``rowwise == 0``: writes the columnwise copy (``x^T``), rotated along the first
 *    (flattened) input dimension. This is the inner dimension of WGRAD.
 *  - ``rowwise != 0``: writes the rowwise copy, rotated along the last input
 *    dimension. This is the inner dimension of DGRAD for the output gradient.
 *
 *  - ``ms_eden != 0``: MS-EDEN with post hoc range alignment (one pass over the
 *    data plus a pass over the scales). ``ms_eden_grid_max`` is the grid parameter
 *    ``s`` of the paper's round-to-nearest quantizer: the micro-scale of a group is
 *    ``group_amax / s`` before rounding, so ``s = 6`` maps the group amax to the
 *    largest E2M1 value, ``s > 6`` clips values above ``6 / s`` of the group amax
 *    and ``s < 6`` leaves part of the E2M1 range unused. The stored amax is the
 *    post-rotation amax times ``(448 / 256) * (6 / s)``.
 *  - ``ms_eden == 0``: standard, range-aligned NVFP4 quantization of the rotated
 *    tensor in two passes over the data (rotation + amax, then rotation + cast).
 *    Elements are rounded to nearest, or stochastically if
 *    ``quant_config.stochastic_rounding`` is set. ``workspace`` may be NULL.
 *
 *  Requirements: SM100 family, BF16 input, both flattened dimensions divisible by
 *  128, E4M3 scales in compact (non-swizzled) layout.
 *
 *  \param[in]      input            Input tensor (BF16).
 *  \param[in,out]  output           NVFP4 output tensor. Only the selected copy is written.
 *  \param[in]      hadamard_matrix  BF16 tensor of shape [rht_dim, 16]; row k holds
 *                                   ``sign[k] * H16[k % 16, :] / 4``.
 *  \param[in,out]  workspace        FP32 workspace with at least ``numel / 16`` elements
 *                                   (MS-EDEN only).
 *  \param[in]      quant_config     Quantization configuration (RNG state, stochastic
 *                                   rounding).
 *  \param[in]      rowwise          Which copy to write (see above).
 *  \param[in]      ms_eden          Whether to use MS-EDEN.
 *  \param[in]      ms_eden_grid_max Grid parameter ``s`` (MS-EDEN only).
 *  \param[in]      stream           CUDA stream used for the operation.
 */
void nvte_quantize_tiled_rht(const NVTETensor input, NVTETensor output,
                             const NVTETensor hadamard_matrix, NVTETensor workspace,
                             const NVTEQuantizationConfig quant_config, int rowwise, int ms_eden,
                             float ms_eden_grid_max, cudaStream_t stream);

/*! \brief Split a tensor along dimension 0 and compute RHT amaxes for each split.
 *
 *  This function is experimental and the API is not stable.
 *
 *  This is intended for quantizing to NVFP4 with random Hadamard
 *  transforms (RHT). For each tensor split, compute the maximum
 *  absolute value (amax) and populate the row-wise amax of the
 *  corresponding output tensor. Also, compute the amax after a
 *  transposed RHT and populate the column-wise amax of the
 *  corresponding output tensor.
 *
 *  \param[in]      input               Input tensor.
 *  \param[in,out]  outputs             Array of NVFP4 output tensors. Only the row-wise and
 *                                      column-wise amaxes are updated.
 *  \param[in]      split_sections      Size of each tensor split along dimension 0.
 *  \param[in]      num_tensors         Number of tensor splits.
 *  \param[in]      random_sign_mask    16-bit sign mask for RHT.
 *  \param[in]      random_sign_mask_t  16-bit sign mask for transposed RHT.
 *  \param[in]      stream              CUDA stream used for the operation.
 */
void nvte_group_hadamard_transform_amax(const NVTETensor input, NVTETensor* outputs,
                                        const size_t* split_sections, size_t num_tensors,
                                        int random_sign_mask, int random_sign_mask_t,
                                        cudaStream_t stream);

/*! \brief Grouped-tensor amax with Hadamard transform (graph safe, device-managed grouping).
 *
 *  This function is experimental and the API is not stable.
 *
 *  This API assumes that the split info (grouping of tensors) is on device and unknown to the host;
 *  therefore, this is a graph safe API and the grouped-tensor argument is passed as a single device structure.
 *
 *  \param[in]      input               NVTEGroupedTensor representing grouped input tensors.
 *  \param[in,out]  output              NVTEGroupedTensor for output amax (row/col). Only the row-wise and
 *                                      column-wise amaxes are updated.
 *  \param[in]      random_sign_mask    16-bit sign mask for RHT.
 *  \param[in]      random_sign_mask_t  16-bit sign mask for transposed RHT.
 *  \param[in]      stream              CUDA stream used for the operation.
 */
void nvte_group_hadamard_transform_amax_graph_safe(const NVTEGroupedTensor input,
                                                   NVTEGroupedTensor output, int random_sign_mask,
                                                   int random_sign_mask_t, cudaStream_t stream);

/*!
 * \brief Perform the grouped-tensor columnwise Hadamard transform cast fusion operation.
 *
 *  This function is experimental and the API is not stable. Group_ prefix means contiguous input concatenated
 *
 *  \param[in]      input             Input tensor to apply Hadamard transform.
 *  \param[in,out]  outputs           Array of output tensors.
 *  \param[in]      hadamard_matrix   Hadamard matrix to use for transformation.
 *  \param[in]      split_sections    Array specifying splits in dimension 0 for each output tensor.
 *  \param[in]      num_tensors       Number of output tensors, must be > 0.
 *  \param[in]      quant_config      Quantization configuration.
 *  \param[in]      stream            CUDA stream used for the operation.
 */
void nvte_group_hadamard_transform_cast_fusion_columnwise(
    const NVTETensor input, NVTETensor* outputs, const NVTETensor hadamard_matrix,
    const size_t* split_sections, size_t num_tensors, const NVTEQuantizationConfig quant_config,
    cudaStream_t stream);

/*!
 * \brief Perform the grouped-tensor row quantize (without Hadamard) and columnwise Hadamard transform cast fusion operation.
 *
 *  This function is experimental and the API is not stable. Group_ prefix means contiguous input concatenated
 *
 *  \param[in]      input             Input tensor to apply Hadamard transform.
 *  \param[in,out]  outputs           Array of output tensors.
 *  \param[in]      hadamard_matrix   Hadamard matrix to use for transformation.
 *  \param[in]      split_sections    Array specifying splits in dimension 0 for each output tensor.
 *  \param[in]      num_tensors       Number of output tensors, must be > 0.
 *  \param[in]      quant_config      Quantization configuration.
 *  \param[in]      quant_workspace   Workspace buffer. Must be at least 4 bytes.
 *  \param[in]      stream            CUDA stream used for the operation.
 */
void nvte_group_hadamard_transform_cast_fusion(const NVTETensor input, NVTETensor* outputs,
                                               const NVTETensor hadamard_matrix,
                                               const size_t* split_sections, size_t num_tensors,
                                               const NVTEQuantizationConfig quant_config,
                                               NVTETensor quant_workspace, cudaStream_t stream);

/*!
 * \brief Perform the grouped-tensor Hadamard transform cast fusion operation in graph-safe mode.
 *
 *  This function is experimental and the API is not stable. Group_ prefix means contiguous input concatenated.
 *
 *  \param[in]      input             NVTEGroupedTensor representing grouped input tensors.
 *  \param[in,out]  output            NVTEGroupedTensor for output (row/column-wise quantized results).
 *  \param[in]      hadamard_matrix   Hadamard matrix to use for transformation.
 *  \param[in]      quant_config      Quantization configuration.
 *  \param[in]      quant_workspace   Workspace buffer. Must be at least 4 bytes.
 *  \param[in]      stream            CUDA stream used for the operation.
 */
void nvte_group_hadamard_transform_cast_fusion_graph_safe(
    const NVTEGroupedTensor input, NVTEGroupedTensor output, const NVTETensor hadamard_matrix,
    const NVTEQuantizationConfig quant_config, NVTETensor quant_workspace, cudaStream_t stream);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // TRANSFORMER_ENGINE_HADAMARD_TRANSFORM_H_
