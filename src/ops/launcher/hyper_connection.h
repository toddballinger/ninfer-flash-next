#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops::detail {

void hyper_connection_repeat_launch(const Tensor& input,
                                    std::int32_t hc_count,
                                    Tensor& out,
                                    cudaStream_t stream);

void hyper_connection_group_rmsnorm_launch(const Tensor& input,
                                           const Tensor& weight,
                                           std::int32_t hidden_size,
                                           std::int32_t hc_count,
                                           float eps,
                                           Tensor& out,
                                           cudaStream_t stream);

void hyper_connection_silu_divide_launch(const Tensor& input,
                                         float divisor,
                                         Tensor& out,
                                         cudaStream_t stream);

void hyper_connection_read_mix_launch(const Tensor& normalized,
                                      const Tensor& logits,
                                      std::int32_t hidden_size,
                                      std::int32_t hc_count,
                                      Tensor& out,
                                      cudaStream_t stream);

void hyper_connection_inject_update_launch(const Tensor& block_output,
                                           const Tensor& inject_logits,
                                           std::int32_t hidden_size,
                                           std::int32_t hc_count,
                                           Tensor& hyper,
                                           cudaStream_t stream);

} // namespace ninfer::ops::detail
