#include "ops/launcher/hyper_connection_executor.h"

#include "core/device.h"

#include "ops/launcher/hyper_connection.h"
#include "ninfer/ops/linear.h"

#include "core/arena.h"

#include <cuda_runtime.h>
#include <cuda_bf16.h>

#include <algorithm>
#include <cstddef>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

/// Bridge a module `Weight` payload (a `const void*`) into a non-const
/// `Tensor` view so it can be passed to the existing 3C1 grouped-RMSNorm
/// launcher, which takes `const Tensor&`. `Weight` and `Tensor` are both
/// non-owning views over the same device payload, so no copy or layout
/// change occurs (contract s.8: no transposition / re-layout).
Tensor weight_as_tensor(const Weight& w) {
    const std::int32_t rows =
        w.n != 0 ? w.n : (w.shape[0] != 0 ? w.shape[0] : 1);
    return Tensor(const_cast<void*>(w.payload), DType::BF16, {rows});
}

/// Device-to-device identity copy of one BF16 tensor into another.
void copy_bf16(const Tensor& src, Tensor& dst, cudaStream_t stream) {
    if (src.data == nullptr || dst.data == nullptr) {
        throw std::invalid_argument(
            "hyper_connection_read_launch: copy source/destination "
            "payload is null");
    }
    const std::size_t bytes =
        static_cast<std::size_t>(src.ne[0]) * src.ne[1] * 2u;
    const cudaError_t err = cudaMemcpyAsync(
        dst.data, src.data, bytes, cudaMemcpyDeviceToDevice, stream);
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string(
            "hyper_connection_read_launch: prior-snapshot copy failed: ") +
            cudaGetErrorString(err));
    }
}

/// Device-side BF16 load/store helpers (translation-unit private).
__device__ __forceinline__ float bf16_load(const __nv_bfloat16* p) {
    return __bfloat162float(*p);
}

__device__ __forceinline__ __nv_bfloat16 bf16_store(float value) {
    return __float2bfloat16_rn(value);
}

__device__ __forceinline__ float sigmoidf_stable(float x) {
    return 1.0F / (1.0F + expf(-x));
}

/// One-shot in-place four-stream inject update:
///
///   hyper[c, h, t] = prior_read[c, h, t] + block[h, t] * 2 * sigmoid(raw[c, t] / C)
///
/// `prior_read` is read-only; `hyper` is the in-place destination.
/// `raw` are the block-inject logits (one value per stream per token);
/// `block` is the caller block output [H, T].
__global__ void inject_from_prior_kernel(
    const __nv_bfloat16* prior_read,
    const __nv_bfloat16* block_output,
    const __nv_bfloat16* raw_logits,
    __nv_bfloat16* hyper,
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
        const std::int32_t stream_idx = feature / hidden;
        const std::int32_t h = feature % hidden;

        const float raw_logit =
            bf16_load(
                raw_logits +
                static_cast<std::int64_t>(token) * hc_count +
                stream_idx);
        const float alpha =
            2.0F *
            sigmoidf_stable(
                raw_logit / static_cast<float>(hc_count));
        const float block_val =
            bf16_load(
                block_output +
                static_cast<std::int64_t>(token) * hidden +
                h);
        const float prior_val =
            bf16_load(prior_read + index);

        hyper[index] =
            bf16_store(prior_val + block_val * alpha);
    }
}

/// Grid-sizing helper (same pattern as the 3C1 launcher).
int grid_for(std::int64_t count) {
    constexpr int kBlock = 256;
    constexpr int kMaxGrid = 4096;
    const auto needed = (count + kBlock - 1) / kBlock;
    return static_cast<int>(
        std::max<std::int64_t>(1, std::min<std::int64_t>(kMaxGrid, needed)));
}

} // namespace

void hyper_connection_read_launch(
    const Tensor& hyper, std::int32_t T, std::int32_t hidden,
    std::int32_t hc_count, const Weight& norm, const Weight& mix_down,
    const Weight& mix_up, float eps, Tensor& mixed, Tensor& prior,
    WorkspaceArena& workspace, cudaStream_t stream) {
    if (hidden <= 0 || hc_count <= 0 || T <= 0) {
        throw std::invalid_argument(
            "hyper_connection_read_launch: H, C, and T must be positive (H=" +
            std::to_string(hidden) + ", C=" + std::to_string(hc_count) +
            ", T=" + std::to_string(T) + ")");
    }

    const std::int64_t CH =
        static_cast<std::int64_t>(hidden) * hc_count;
    const std::int32_t ch32 = static_cast<std::int32_t>(CH);
    const std::int32_t R = mix_down.n;  // low-rank width

    if (hyper.data == nullptr) {
        throw std::invalid_argument(
            "hyper_connection_read_launch: hyper payload is null");
    }
    if (hyper.ne[0] != ch32 || hyper.ne[1] != T) {
        throw std::invalid_argument(
            "hyper_connection_read_launch: hyper must be [C*H, T]");
    }
    if (mixed.ne[0] != hidden || mixed.ne[1] != T) {
        throw std::invalid_argument(
            "hyper_connection_read_launch: mixed must be [H, T]");
    }
    if (prior.ne[0] != ch32 || prior.ne[1] != T) {
        throw std::invalid_argument(
            "hyper_connection_read_launch: prior must be [C*H, T]");
    }

    // Three caller-provisioned transient buffers (all BF16, 2 bytes/elem):
    //   hn   : normalized [CH, T]   -> 2*CH*T  (= 2*C*H per token)
    //   down : low-rank     [R, T]  -> 2*R*T   (single buffer; SiLU in place)
    //   up   : up-projected [CH, T] -> 2*CH*T  (= 2*C*H per token)
    // Total = 2*T*(2*C*H + R), matching the frozen capacity term.
    const DeviceSpan hn_span =
        workspace.alloc_bytes(2 * static_cast<std::size_t>(CH) * T);
    const DeviceSpan down_span =
        workspace.alloc_bytes(2 * static_cast<std::size_t>(R) * T);
    const DeviceSpan up_span =
        workspace.alloc_bytes(2 * static_cast<std::size_t>(ch32) * T);

    Tensor hn =
        Tensor(hn_span.data, DType::BF16, {ch32, T});
    Tensor down =
        Tensor(down_span.data, DType::BF16, {R, T});
    Tensor up =
        Tensor(up_span.data, DType::BF16, {ch32, T});

    const Tensor norm_tensor = weight_as_tensor(norm);

    // (1) Hn = GroupRMSNorm(hyper, module norm_weight, eps)  [CH, T]
    hyper_connection_group_rmsnorm_launch(
        hyper, norm_tensor, hidden, hc_count, eps, hn, stream);

    // (2) down = linear(Hn, mix_down [R, CH])  [R, T]   (A16, no workspace)
    linear(hn, mix_down, down, stream);

    // (3) down = SiLU(down / C), in place -> one low-rank buffer only
    hyper_connection_silu_divide_launch(down,
                                        static_cast<float>(hc_count), down,
                                        stream);

    // (4) up = linear(down, mix_up [CH, R])  [CH, T]   (A16, no workspace)
    linear(down, mix_up, up, stream);

    // (5) mixed = read_mix(Hn, up) -> mean_c(sigmoid(up) * Hn)  [H, T]
    hyper_connection_read_mix_launch(hn, up, hidden, hc_count, mixed,
                                     stream);

    // (6) prior = preserved prior_read snapshot: exact copy of live hyper
    copy_bf16(hyper, prior, stream);
}

void hyper_connection_final_mixer_launch(
    const Tensor& hyper, std::int32_t T, std::int32_t hidden,
    std::int32_t hc_count, const Weight& norm, const Weight& mix_down,
    const Weight& mix_up, float eps, Tensor& final_mix,
    WorkspaceArena& workspace, cudaStream_t stream) {
    if (hidden <= 0 || hc_count <= 0 || T <= 0) {
        throw std::invalid_argument(
            "hyper_connection_final_mixer_launch: H, C, and T must be positive (H=" +
            std::to_string(hidden) + ", C=" + std::to_string(hc_count) +
            ", T=" + std::to_string(T) + ")");
    }

    const std::int64_t CH =
        static_cast<std::int64_t>(hidden) * hc_count;
    const std::int32_t ch32 = static_cast<std::int32_t>(CH);
    const std::int32_t R = mix_down.n;  // low-rank width

    if (hyper.data == nullptr) {
        throw std::invalid_argument(
            "hyper_connection_final_mixer_launch: hyper payload is null");
    }
    if (hyper.ne[0] != ch32 || hyper.ne[1] != T) {
        throw std::invalid_argument(
            "hyper_connection_final_mixer_launch: hyper must be [C*H, T]");
    }
    if (final_mix.ne[0] != hidden || final_mix.ne[1] != T) {
        throw std::invalid_argument(
            "hyper_connection_final_mixer_launch: final_mix must be [H, T]");
    }

    // Three caller-provisioned transient buffers (all BF16, 2 bytes/elem):
    //   hn   : normalized  [CH, T] -> 2*CH*T  (= 2*C*H per token)
    //   down : low-rank    [R, T]  -> 2*R*T   (single buffer; SiLU in place)
    //   up   : up-project  [CH, T] -> 2*CH*T  (= 2*C*H per token)
    // Total = 2*T*(2*C*H + R), matching the frozen capacity term.
    const DeviceSpan hn_span =
        workspace.alloc_bytes(2 * static_cast<std::size_t>(CH) * T);
    const DeviceSpan down_span =
        workspace.alloc_bytes(2 * static_cast<std::size_t>(R) * T);
    const DeviceSpan up_span =
        workspace.alloc_bytes(2 * static_cast<std::size_t>(ch32) * T);

    Tensor hn =
        Tensor(hn_span.data, DType::BF16, {ch32, T});
    Tensor down =
        Tensor(down_span.data, DType::BF16, {R, T});
    Tensor up =
        Tensor(up_span.data, DType::BF16, {ch32, T});

    const Tensor norm_tensor = weight_as_tensor(norm);

    // (1) Hn = GroupRMSNorm(hyper, final module norm_weight, eps)  [CH, T]
    hyper_connection_group_rmsnorm_launch(
        hyper, norm_tensor, hidden, hc_count, eps, hn, stream);

    // (2) down = linear(Hn, mix_down [R, CH])  [R, T]   (A16, no workspace)
    linear(hn, mix_down, down, stream);

    // (3) down = SiLU(down / C), in place -> one low-rank buffer only
    hyper_connection_silu_divide_launch(down,
                                        static_cast<float>(hc_count), down,
                                        stream);

    // (4) up = linear(down, mix_up [CH, R])  [CH, T]   (A16, no workspace)
    linear(down, mix_up, up, stream);

    // (5) final_mix = read_mix(Hn, up) -> mean_c(sigmoid(up) * Hn)  [H, T]
    hyper_connection_read_mix_launch(hn, up, hidden, hc_count, final_mix,
                                     stream);
}

void hyper_connection_inject_launch(
    const Tensor& block, Tensor& hyper, std::int32_t T,
    std::int32_t hidden, std::int32_t hc_count,
    const Weight& norm, const Weight& inject_weight,
    float eps,
    const Tensor& prior_read,
    WorkspaceArena& workspace, cudaStream_t stream) {
    if (hidden <= 0 || hc_count <= 0 || T <= 0) {
        throw std::invalid_argument(
            "hyper_connection_inject_launch: H, C, and T must be positive (H=" +
            std::to_string(hidden) + ", C=" + std::to_string(hc_count) +
            ", T=" + std::to_string(T) + ")");
    }

    const std::int64_t CH =
        static_cast<std::int64_t>(hidden) * hc_count;
    const std::int32_t ch32 = static_cast<std::int32_t>(CH);

    if (hyper.data == nullptr) {
        throw std::invalid_argument(
            "hyper_connection_inject_launch: hyper payload is null");
    }
    if (block.data == nullptr) {
        throw std::invalid_argument(
            "hyper_connection_inject_launch: block payload is null");
    }
    if (prior_read.data == nullptr) {
        throw std::invalid_argument(
            "hyper_connection_inject_launch: prior_read payload is null");
    }
    if (hyper.ne[0] != ch32 || hyper.ne[1] != T) {
        throw std::invalid_argument(
            "hyper_connection_inject_launch: hyper must be [C*H, T]");
    }
    if (block.ne[0] != hidden || block.ne[1] != T) {
        throw std::invalid_argument(
            "hyper_connection_inject_launch: block must be [H, T]");
    }
    if (prior_read.ne[0] != ch32 || prior_read.ne[1] != T) {
        throw std::invalid_argument(
            "hyper_connection_inject_launch: prior_read must be [C*H, T]");
    }
    if (inject_weight.n != hc_count) {
        throw std::invalid_argument(
            "hyper_connection_inject_launch: inject_weight.n must equal C (" +
            std::to_string(hc_count) + "), got " +
            std::to_string(inject_weight.n));
    }

    // Two caller-provisioned transient buffers (all BF16, 2 bytes/elem):
    //   hn   : normalized    [CH, T] -> 2*CH*T bytes
    //   raw  : inject logits [C,  T] -> 2*C*T  bytes
    // Total = 2*T*(C*H + C), matching the frozen inject capacity term
    // (contract doc s.9.5).
    const DeviceSpan hn_span =
        workspace.alloc_bytes(2 * static_cast<std::size_t>(CH) * T);
    const DeviceSpan raw_span =
        workspace.alloc_bytes(2 * static_cast<std::size_t>(hc_count) * T);

    Tensor hn =
        Tensor(hn_span.data, DType::BF16, {ch32, T});
    Tensor raw_logits =
        Tensor(raw_span.data, DType::BF16, {hc_count, T});

    const Tensor norm_tensor = weight_as_tensor(norm);

    // (1) Hn = GroupRMSNorm(live hyper, norm_weight, eps)  [CH, T]
    //     alpha depends on the LIVE (pre-inject) normalized hyper.
    hyper_connection_group_rmsnorm_launch(
        hyper, norm_tensor, hidden, hc_count, eps, hn, stream);

    // (2) raw = linear(Hn, inject [C, CH])  [C, T]
    //     raw block-inject logits (pre-sigmoid); A16 path, no workspace.
    linear(hn, inject_weight, raw_logits, stream);

    // (3) hyper[c,h,t] = prior_read[c,h,t] + block[h,t] * 2*sigmoid(raw[c,t]/C)
    //     prior_read is the sole immutable additive base; hyper is the
    //     in-place destination.
    {
        constexpr int kBlock = 256;
        const std::int64_t count = static_cast<std::int64_t>(ch32) * T;
        inject_from_prior_kernel<<<grid_for(count), kBlock, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(prior_read.data),
            static_cast<const __nv_bfloat16*>(block.data),
            static_cast<const __nv_bfloat16*>(raw_logits.data),
            static_cast<__nv_bfloat16*>(hyper.data),
            hidden,
            hc_count,
            T);
        CUDA_CHECK(cudaGetLastError());
    }
}

} // namespace ninfer::ops::detail

namespace ninfer::ops {

std::size_t hyper_connection_executor_workspace_capacity_bytes(
    std::int32_t streams, std::int32_t hidden, std::int32_t lowrank,
    std::int32_t tokens) {
    if (streams <= 0 || hidden <= 0 || lowrank <= 0 || tokens <= 0) {
        return 0;
    }

    // Capacity mirrors the DeviceArena `alloc_bytes` placement of the
    // executor's transient buffers: each allocation's start is aligned up
    // to the arena's 256-byte allocation alignment (arena.h
    // alloc_bytes/alloc default `align = 256`) and `off_` advances to
    // the allocation end (arena.cu alloc_bytes), so the sequence's true
    // footprint is the final arena offset, not the sum of raw element
    // bytes.
    const std::size_t align = 256;  // DeviceArena default allocation alignment
    const std::size_t kMax = std::numeric_limits<std::size_t>::max();
    const auto align_up = [align, kMax](std::size_t v) {
        const std::size_t mask = align - 1u;
        if (v > kMax - mask) {
            throw std::overflow_error(
                "hyper_connection_executor_workspace_capacity_bytes: "
                "aligned offset addition overflow");
        }
        return (v + mask) & ~mask;
    };
    const auto checked_multiply = [kMax](std::size_t lhs,
                                     std::size_t rhs) -> std::size_t {
        if (lhs != 0 && rhs > kMax / lhs) {
            throw std::overflow_error(
                "hyper_connection_executor_workspace_capacity_bytes: "
                "size_t multiplication overflow");
        }
        return lhs * rhs;
    };
    const auto checked_add = [kMax](std::size_t lhs,
                                    std::size_t rhs) -> std::size_t {
        if (rhs > kMax - lhs) {
            throw std::overflow_error(
                "hyper_connection_executor_workspace_capacity_bytes: "
                "size_t addition overflow");
        }
        return lhs + rhs;
    };

    const std::size_t c = static_cast<std::size_t>(streams);
    const std::size_t h = static_cast<std::size_t>(hidden);
    const std::size_t r = static_cast<std::size_t>(lowrank);
    const std::size_t t = static_cast<std::size_t>(tokens);
    const std::size_t ch = checked_multiply(c, h);
    // Check the 2x element-width doublings before use so a near-max
    // width cannot wrap size_t before the checked helper sees it.
    const std::size_t doubled_ch = checked_multiply(2u, ch);
    const std::size_t doubled_r = checked_multiply(2u, r);
    const std::size_t doubled_c = checked_multiply(2u, c);

    // read / final-mixer pass: hn(2*CH*T), down(2*R*T), up(2*CH*T).
    std::size_t read_off = 0;
    read_off = checked_add(align_up(read_off), checked_multiply(doubled_ch, t));
    read_off = checked_add(align_up(read_off), checked_multiply(doubled_r, t));
    read_off = checked_add(align_up(read_off), checked_multiply(doubled_ch, t));

    // inject pass: hn(2*CH*T), raw(2*C*T).
    std::size_t inject_off = 0;
    inject_off = checked_add(align_up(inject_off), checked_multiply(doubled_ch, t));
    inject_off = checked_add(align_up(inject_off), checked_multiply(doubled_c, t));

    return std::max(read_off, inject_off);
}


} // namespace ninfer::ops
