#include "ops/launcher/hyper_connection.h"

#include "core/device.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace ninfer::ops::detail {
namespace {

__device__ __forceinline__ float bf16_load(const __nv_bfloat16* p) {
    return __bfloat162float(*p);
}

__device__ __forceinline__ __nv_bfloat16 bf16_store(float value) {
    return __float2bfloat16_rn(value);
}

__device__ __forceinline__ float sigmoidf_stable(float x) {
    return 1.0F / (1.0F + expf(-x));
}

__device__ __forceinline__ float siluf(float x) {
    return x * sigmoidf_stable(x);
}

__global__ void repeat_kernel(const __nv_bfloat16* input,
                              __nv_bfloat16* out,
                              std::int32_t hidden,
                              std::int32_t hc_count,
                              std::int32_t tokens) {
    const std::int64_t expanded =
        static_cast<std::int64_t>(hidden) * hc_count;

    const std::int64_t total = expanded * tokens;

    for (std::int64_t index =
             static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < total;
         index += static_cast<std::int64_t>(blockDim.x) * gridDim.x) {

        const std::int32_t feature =
            static_cast<std::int32_t>(index % expanded);

        const std::int32_t token =
            static_cast<std::int32_t>(index / expanded);

        const std::int32_t hidden_index = feature % hidden;

        out[index] =
            input[static_cast<std::int64_t>(token) * hidden + hidden_index];
    }
}

template <int Block>
__global__ void group_rmsnorm_kernel(const __nv_bfloat16* input,
                                     const __nv_bfloat16* weight,
                                     __nv_bfloat16* out,
                                     std::int32_t hidden,
                                     std::int32_t hc_count,
                                     float eps) {
    const std::int32_t logical_row = blockIdx.x;

    const std::int32_t stream =
        logical_row % hc_count;

    const std::int32_t token =
        logical_row / hc_count;

    const std::int64_t expanded =
        static_cast<std::int64_t>(hidden) * hc_count;

    const std::int64_t base =
        static_cast<std::int64_t>(token) * expanded +
        static_cast<std::int64_t>(stream) * hidden;

    float local_sum = 0.0F;

    for (std::int32_t h = threadIdx.x; h < hidden; h += Block) {
        const float value = bf16_load(input + base + h);
        local_sum += value * value;
    }

    __shared__ float warp_sums[32];

    for (int offset = 16; offset > 0; offset >>= 1) {
        local_sum += __shfl_down_sync(0xffffffffU, local_sum, offset);
    }

    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;

    if (lane == 0) {
        warp_sums[warp] = local_sum;
    }

    __syncthreads();

    float sum = 0.0F;

    if (warp == 0) {
        const int warp_count = (Block + 31) / 32;
        sum = lane < warp_count ? warp_sums[lane] : 0.0F;

        for (int offset = 16; offset > 0; offset >>= 1) {
            sum += __shfl_down_sync(0xffffffffU, sum, offset);
        }

        if (lane == 0) {
            warp_sums[0] =
                rsqrtf(sum / static_cast<float>(hidden) + eps);
        }
    }

    __syncthreads();

    const float inverse = warp_sums[0];

    for (std::int32_t h = threadIdx.x; h < hidden; h += Block) {
        const float value = bf16_load(input + base + h);
        const float gain =
            bf16_load(weight + static_cast<std::int64_t>(stream) * hidden + h);

        out[base + h] = bf16_store(value * inverse * gain);
    }
}

__global__ void silu_divide_kernel(const __nv_bfloat16* input,
                                   __nv_bfloat16* out,
                                   std::int64_t count,
                                   float divisor) {
    for (std::int64_t index =
             static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < count;
         index += static_cast<std::int64_t>(blockDim.x) * gridDim.x) {

        const float value = bf16_load(input + index) / divisor;
        out[index] = bf16_store(siluf(value));
    }
}

__global__ void read_mix_kernel(const __nv_bfloat16* normalized,
                                const __nv_bfloat16* logits,
                                __nv_bfloat16* out,
                                std::int32_t hidden,
                                std::int32_t hc_count,
                                std::int32_t tokens) {
    const std::int64_t output_count =
        static_cast<std::int64_t>(hidden) * tokens;

    const std::int64_t expanded =
        static_cast<std::int64_t>(hidden) * hc_count;

    for (std::int64_t index =
             static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < output_count;
         index += static_cast<std::int64_t>(blockDim.x) * gridDim.x) {

        const std::int32_t h =
            static_cast<std::int32_t>(index % hidden);

        const std::int32_t token =
            static_cast<std::int32_t>(index / hidden);

        const std::int64_t token_base =
            static_cast<std::int64_t>(token) * expanded;

        float sum = 0.0F;

        for (std::int32_t stream = 0; stream < hc_count; ++stream) {
            const std::int64_t offset =
                token_base +
                static_cast<std::int64_t>(stream) * hidden +
                h;

            const float normalized_value =
                bf16_load(normalized + offset);

            const float logit =
                bf16_load(logits + offset);

            sum += sigmoidf_stable(logit) * normalized_value;
        }

        out[index] =
            bf16_store(sum / static_cast<float>(hc_count));
    }
}

__global__ void inject_update_kernel(const __nv_bfloat16* block_output,
                                     const __nv_bfloat16* inject_logits,
                                     __nv_bfloat16* hyper,
                                     std::int32_t hidden,
                                     std::int32_t hc_count,
                                     std::int32_t tokens) {
    const std::int64_t expanded =
        static_cast<std::int64_t>(hidden) * hc_count;

    const std::int64_t total =
        expanded * tokens;

    for (std::int64_t index =
             static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < total;
         index += static_cast<std::int64_t>(blockDim.x) * gridDim.x) {

        const std::int32_t feature =
            static_cast<std::int32_t>(index % expanded);

        const std::int32_t token =
            static_cast<std::int32_t>(index / expanded);

        const std::int32_t stream =
            feature / hidden;

        const std::int32_t h =
            feature % hidden;

        const float injection_logit =
            bf16_load(
                inject_logits +
                static_cast<std::int64_t>(token) * hc_count +
                stream);

        const float alpha =
            2.0F *
            sigmoidf_stable(
                injection_logit /
                static_cast<float>(hc_count));

        const float block =
            bf16_load(
                block_output +
                static_cast<std::int64_t>(token) * hidden +
                h);

        const float previous =
            bf16_load(hyper + index);

        hyper[index] =
            bf16_store(previous + block * alpha);
    }
}

int grid_for(std::int64_t count) {
    constexpr int block = 256;
    constexpr int max_grid = 4096;

    const auto needed =
        (count + block - 1) / block;

    return static_cast<int>(
        std::max<std::int64_t>(
            1,
            std::min<std::int64_t>(max_grid, needed)));
}

} // namespace

void hyper_connection_repeat_launch(const Tensor& input,
                                    std::int32_t hc_count,
                                    Tensor& out,
                                    cudaStream_t stream) {
    constexpr int block = 256;

    const std::int64_t total = out.numel();

    repeat_kernel<<<grid_for(total), block, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(input.data),
        static_cast<__nv_bfloat16*>(out.data),
        input.ne[0],
        hc_count,
        input.ne[1]);

    CUDA_CHECK(cudaGetLastError());
}

void hyper_connection_group_rmsnorm_launch(const Tensor& input,
                                           const Tensor& weight,
                                           std::int32_t hidden_size,
                                           std::int32_t hc_count,
                                           float eps,
                                           Tensor& out,
                                           cudaStream_t stream) {
    constexpr int block = 256;

    const int rows = input.ne[1] * hc_count;

    group_rmsnorm_kernel<block><<<rows, block, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(input.data),
        static_cast<const __nv_bfloat16*>(weight.data),
        static_cast<__nv_bfloat16*>(out.data),
        hidden_size,
        hc_count,
        eps);

    CUDA_CHECK(cudaGetLastError());
}

void hyper_connection_silu_divide_launch(const Tensor& input,
                                         float divisor,
                                         Tensor& out,
                                         cudaStream_t stream) {
    constexpr int block = 256;

    const auto count = input.numel();

    silu_divide_kernel<<<grid_for(count), block, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(input.data),
        static_cast<__nv_bfloat16*>(out.data),
        count,
        divisor);

    CUDA_CHECK(cudaGetLastError());
}

void hyper_connection_read_mix_launch(const Tensor& normalized,
                                      const Tensor& logits,
                                      std::int32_t hidden_size,
                                      std::int32_t hc_count,
                                      Tensor& out,
                                      cudaStream_t stream) {
    constexpr int block = 256;

    const auto count = out.numel();

    read_mix_kernel<<<grid_for(count), block, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(normalized.data),
        static_cast<const __nv_bfloat16*>(logits.data),
        static_cast<__nv_bfloat16*>(out.data),
        hidden_size,
        hc_count,
        out.ne[1]);

    CUDA_CHECK(cudaGetLastError());
}

void hyper_connection_inject_update_launch(const Tensor& block_output,
                                           const Tensor& inject_logits,
                                           std::int32_t hidden_size,
                                           std::int32_t hc_count,
                                           Tensor& hyper,
                                           cudaStream_t stream) {
    constexpr int block = 256;

    const auto count = hyper.numel();

    inject_update_kernel<<<grid_for(count), block, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(block_output.data),
        static_cast<const __nv_bfloat16*>(inject_logits.data),
        static_cast<__nv_bfloat16*>(hyper.data),
        hidden_size,
        hc_count,
        hyper.ne[1]);

    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
