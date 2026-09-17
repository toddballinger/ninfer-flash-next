#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops {

/**
 * Qwen4Exp HyperConnection reference primitives.
 *
 * Logical storage follows NInfer tensor convention:
 *
 *   hidden          [H,T]
 *   hyper           [C*H,T]
 *   low-rank        [R,T]
 *   inject logits   [C,T]
 *
 * Flash-Next canonical geometry is C=4, H=2560, R=320, but these
 * primitives validate their supplied geometry rather than hard-coding H/R.
 */

/**
 * Repeat one hidden vector into C residual streams:
 *
 *   out[c*H + h, t] = input[h,t]
 */
void hyper_connection_repeat(const Tensor& input,
                             std::int32_t hc_count,
                             Tensor& out,
                             cudaStream_t stream);

/**
 * Independently RMS-normalize C hidden-width streams.
 *
 * For each token t and stream c:
 *
 *   inverse = 1 / sqrt(mean_h(x[c,h,t]^2) + eps)
 *   out[c,h,t] = x[c,h,t] * inverse * weight[c,h]
 *
 * The norm weight therefore has logical width C*H.
 */
void hyper_connection_group_rmsnorm(const Tensor& input,
                                    const Tensor& weight,
                                    std::int32_t hidden_size,
                                    std::int32_t hc_count,
                                    float eps,
                                    Tensor& out,
                                    cudaStream_t stream);

/**
 * Reference low-rank activation:
 *
 *   out = SiLU(input / divisor)
 */
void hyper_connection_silu_divide(const Tensor& input,
                                  float divisor,
                                  Tensor& out,
                                  cudaStream_t stream);

/**
 * Collapse C normalized streams using sigmoid mixing logits:
 *
 *   out[h,t] =
 *       mean_c(sigmoid(logits[c,h,t]) * normalized[c,h,t])
 */
void hyper_connection_read_mix(const Tensor& normalized,
                               const Tensor& logits,
                               std::int32_t hidden_size,
                               std::int32_t hc_count,
                               Tensor& out,
                               cudaStream_t stream);

/**
 * Per-block HyperConnection injection:
 *
 *   alpha[c,t] = 2 * sigmoid(inject_logits[c,t] / hc_count)
 *
 *   hyper[c,h,t] += block_output[h,t] * alpha[c,t]
 *
 * `hyper` is updated in place.
 */
void hyper_connection_inject_update(const Tensor& block_output,
                                    const Tensor& inject_logits,
                                    std::int32_t hidden_size,
                                    std::int32_t hc_count,
                                    Tensor& hyper,
                                    cudaStream_t stream);

} // namespace ninfer::ops
