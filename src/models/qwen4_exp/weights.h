#pragma once

#include "artifact/binder.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <variant>
#include <vector>

namespace ninfer::models::qwen4_exp {

using Weight = artifact::ParameterReference;

struct HyperConnectionWeights {
    Weight norm;
    Weight down;
    Weight up;
    std::optional<Weight> block_inject;
};

struct DenseWeights {
    Weight gate;
    Weight up;
    Weight down;
};

// Each logical expert remains addressable even when several bindings slice one serialized object.
struct ExpertBank {
    std::vector<DenseWeights> experts;

    [[nodiscard]] const DenseWeights& at(std::size_t expert_id) const {
        return experts.at(expert_id);
    }
};

struct MoeWeights {
    Weight router;
    Weight shared_gate;
    DenseWeights shared_expert;
    ExpertBank routed_experts;
};

struct GdnWeights {
    Weight a_log;
    Weight dt_bias;
    Weight convolution;
    Weight in_proj_a;
    Weight in_proj_b;
    Weight qkv;
    Weight z;
    Weight norm;
    Weight output;
};

struct QsaIndexerWeights {
    Weight query_key;
    Weight query_norm;
    Weight key_norm;
};

struct QsaWeights {
    Weight query;
    Weight key;
    Weight value;
    Weight output;
    Weight query_norm;
    Weight key_norm;
    QsaIndexerWeights indexer;
};

struct PleWeights {
    Weight convolution;
    Weight key;
    Weight value;
    Weight key_norm;
    Weight query_norm;
    Weight conv_norm;
    std::vector<Weight> embedding_shards;
    std::vector<std::int64_t> layer_multipliers;
    std::vector<std::int64_t> ngram_heads_offsets;
    std::vector<std::int64_t> ngram_heads_vocab_sizes;
};

using MixerWeights = std::variant<GdnWeights, QsaWeights>;

struct BlockWeights {
    HyperConnectionWeights attention_hyper_connection;
    HyperConnectionWeights mlp_hyper_connection;
    MixerWeights mixer;
    std::optional<PleWeights> ple;
    MoeWeights moe;
};

struct TextWeights {
    Weight token_embedding;
    Weight output_head;
    HyperConnectionWeights hyper_connection;
    std::vector<BlockWeights> layers;
};

} // namespace ninfer::models::qwen4_exp
