#pragma once

#include "models/qwen4_exp/config.h"
#include "models/qwen4_exp/weights.h"

namespace ninfer::models::qwen4_exp {

// Binds the normalized NInfer artifact namespace. Source-model prefixes are a converter concern.
// The returned graph is logical; Binding parts may point into fused serialized objects.
[[nodiscard]] TextWeights bind_text_weights(artifact::Binder& binder,
                                            const TextConfig& config);

} // namespace ninfer::models::qwen4_exp
