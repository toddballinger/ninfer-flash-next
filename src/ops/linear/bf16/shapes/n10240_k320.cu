#include "ops/linear/bf16/bf16_shapes.h"
#include "core/device.h"
#include "ops/common/math.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {

// BF16 [10240,320] — HyperConnection low-rank up projection (LinearUp).
//
// K=320 is divisible by no phase width (128/256/512) that the phase-based gemv/SIMT
// ladders require, so this shape has no feasible phase schedule at any T. It is
// therefore served exclusively by a narrow generic/reference GEMV: one thread per
// output row, a serial K=320 loop with scalar BF16 loads and exact per-element FP32
// FMA accumulation, then the standard BF16 store epilogue (__float2bfloat16_rn).
// The reference covers the whole serving interval at every T without GPU validation,
// matching the existing linear epilogue/BF16-conversion semantics. The existing
// specialized [320,10240] and [4,10240] paths are left unchanged.
namespace {

struct RefContiguousOutput {
    __nv_bfloat16* data;
    std::int32_t rows;

    __device__ __forceinline__ void store(std::int32_t row, std::int32_t token, float value) const {
        data[static_cast<std::int64_t>(token) * rows + row] = __float2bfloat16_rn(value);
    }
};

__global__ void bf16_ref_gemv_kernel(const __nv_bfloat16* __restrict__ x,
                                     const __nv_bfloat16* __restrict__ weight,
                                     RefContiguousOutput output, int tokens, std::int32_t rows) {
    const std::int32_t row = static_cast<std::int32_t>(blockIdx.x * static_cast<unsigned>(blockDim.x)) +
                             static_cast<std::int32_t>(threadIdx.x);
    if (row >= rows) return;
    const __nv_bfloat16* wrow = weight + static_cast<std::int64_t>(row) * 320;
    for (std::int32_t token = 0; token < tokens; ++token) {
        float acc = 0.0F;
        const __nv_bfloat16* xrow = x + static_cast<std::int64_t>(token) * 320;
        for (std::int32_t k = 0; k < 320; ++k) {
            acc = fmaf(__bfloat162float(wrow[k]), __bfloat162float(xrow[k]), acc);
        }
        output.store(row, token, acc);
    }
}

void bf16_ref_gemv_launch(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    const RefContiguousOutput output{static_cast<__nv_bfloat16*>(out.data), 10240};
    const int tokens = x.ne[1];
    bf16_ref_gemv_kernel<<<div_up(10240, 256), 256, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), static_cast<const __nv_bfloat16*>(weight.qdata),
        output, tokens, 10240);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

Bf16Launch select_bf16_n10240_k320_reference(std::int32_t tokens) {
    if (tokens <= 0) throw std::invalid_argument("bf16 linear: T must be positive");
    return &bf16_ref_gemv_launch;
}

} // namespace ninfer::ops::detail