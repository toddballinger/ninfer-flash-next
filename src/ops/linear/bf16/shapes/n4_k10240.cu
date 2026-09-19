#include "core/device.h"
#include "ops/common/math.h"
#include "ops/linear/bf16/bf16_shapes.h"
#include "ops/linear/bf16/bf16_launch.cuh"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace ninfer::ops::detail {
namespace {
// BF16 [4,10240] - HyperConnection per-block injection weights (BlockInject).
// Four output rows (one per stream) over a 10240-wide input; gemv at T=1, then the SIMT
// ladder for T=2..4. N=4 cannot meet the MMA geometry asserts (kBlockRows % kWarpRows and the
// 1024-thread bound), so T>4 prefill is served by a narrow generic/reference GEMV: one thread
// per output row, a serial K=10240 loop with scalar BF16 loads and per-element FP32 FMA
// accumulation, then the standard BF16 store epilogue (__float2bfloat16_rn). The reference
// covers the whole serving interval at every T, matching the existing linear
// epilogue/BF16-conversion semantics; this mirrors the [10240,320] reference.
using Geometry = Bf16Geometry<4, 10240>;
using Gemv     = Bf16GemvSchedule<4, 1, 1, 8, 4, Bf16ActivationAccess::Direct, Bf16WeightCache::Default,
                              Bf16PhaseOrder::Sequential, 1, 1, 1, 2>;
using C2      = Bf16SimtSchedule<1, 1, 4, 4, 1, 2, Bf16SimtActivationAccess::WarpPacked,
                               Bf16WeightCache::Default, Bf16PhaseOrder::Sequential, 1, 1, 1, 2>;
using C4      = Bf16SimtSchedule<1, 1, 2, 4, 1, 4, Bf16SimtActivationAccess::WarpPacked,
                               Bf16WeightCache::Default, Bf16PhaseOrder::Sequential, 1, 2, 1, 2>;

} // namespace

namespace {

struct RefContiguousOutput {
    __nv_bfloat16* data;
    std::int32_t rows;

    __device__ __forceinline__ void store(std::int32_t row, std::int32_t token, float value) const {
        data[static_cast<std::int64_t>(token) * rows + row] = __float2bfloat16_rn(value);
    }
};

__global__ void bf16_ref_gemv_n4_k10240_kernel(const __nv_bfloat16* __restrict__ x,
                                                const __nv_bfloat16* __restrict__ weight,
                                                RefContiguousOutput output, int tokens, std::int32_t rows) {
    const std::int32_t row = static_cast<std::int32_t>(blockIdx.x * static_cast<unsigned>(blockDim.x)) +
                             static_cast<std::int32_t>(threadIdx.x);
    if (row >= rows) return;
    const __nv_bfloat16* wrow = weight + static_cast<std::int64_t>(row) * 10240;
    for (std::int32_t token = 0; token < tokens; ++token) {
        float acc = 0.0F;
        const __nv_bfloat16* xrow = x + static_cast<std::int64_t>(token) * 10240;
        for (std::int32_t k = 0; k < 10240; ++k) {
            acc = fmaf(__bfloat162float(wrow[k]), __bfloat162float(xrow[k]), acc);
        }
        output.store(row, token, acc);
    }
}

void bf16_ref_gemv_n4_k10240_launch(const Tensor& x, const Weight& weight, Tensor& out,
                                    cudaStream_t stream) {
    const RefContiguousOutput output{static_cast<__nv_bfloat16*>(out.data), 4};
    const int tokens = x.ne[1];
    bf16_ref_gemv_n4_k10240_kernel<<<div_up(4, 256), 256, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), static_cast<const __nv_bfloat16*>(weight.qdata),
        output, tokens, 4);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

Bf16Launch select_bf16_n4_k10240(std::int32_t tokens) {
    if (tokens == 1) return launch_bf16_gemv<Geometry, Gemv>;
    if (tokens <= 2) return launch_bf16_simt<Geometry, 2, C2>;
    if (tokens <= 4) return launch_bf16_simt<Geometry, 4, C4>;
    return &bf16_ref_gemv_n4_k10240_launch;
}

} // namespace ninfer::ops::detail