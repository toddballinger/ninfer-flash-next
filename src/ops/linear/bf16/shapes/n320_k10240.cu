#include "ops/linear/bf16/bf16_shapes.h"
#include "ops/linear/bf16/bf16_launch.cuh"

namespace ninfer::ops::detail {
namespace {
// BF16 [320,10240] — HyperConnection low-rank down projection (LinearDown).
// T=1 decode uses gemv; prefill uses the SIMT ladder, then the MMA path for the largest T.
using Geometry = Bf16Geometry<320, 10240>;
using Gemv     = Bf16GemvSchedule<4, 1, 8, 8, 4, Bf16ActivationAccess::Direct, Bf16WeightCache::Default,
                              Bf16PhaseOrder::Sequential, 1, 1, 1, 2>;
using C2      = Bf16SimtSchedule<4, 1, 2, 16, 1, 2, Bf16SimtActivationAccess::WarpPacked,
                               Bf16WeightCache::Default, Bf16PhaseOrder::Sequential, 1, 1, 1, 2>;
using C4      = Bf16SimtSchedule<4, 1, 2, 8, 1, 4, Bf16SimtActivationAccess::WarpPacked,
                               Bf16WeightCache::Default, Bf16PhaseOrder::Sequential, 1, 2, 1, 2>;
using C8      = Bf16SimtSchedule<4, 1, 1, 8, 1, 4, Bf16SimtActivationAccess::WarpPacked,
                               Bf16WeightCache::Default, Bf16PhaseOrder::Sequential, 1, 2, 1, 2>;
using Mma     = Bf16MmaSchedule<32, 128, 64, 16, 32, 2, 2, Cache::cg, Cache::cg,
                               Bf16MmaFragmentPipeline::PingPong, Bf16MmaRaster::TokenFast>;

} // namespace

Bf16Launch select_bf16_n320_k10240(std::int32_t tokens) {
    if (tokens == 1) return launch_bf16_gemv<Geometry, Gemv>;
    if (tokens <= 2) return launch_bf16_simt<Geometry, 2, C2>;
    if (tokens <= 4) return launch_bf16_simt<Geometry, 4, C4>;
    if (tokens <= 8) return launch_bf16_simt<Geometry, 8, C8>;
    return launch_bf16_mma<Geometry, Mma>;
}

} // namespace ninfer::ops::detail