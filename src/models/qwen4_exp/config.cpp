#include "models/qwen4_exp/config.h"

#include "artifact/schema.h"

#include <cmath>
#include <limits>
#include <set>
#include <string>

namespace ninfer::models::qwen4_exp {
namespace {

using artifact::ArtifactError;
using artifact::Json;
using artifact::require_members;

constexpr std::uint32_t kDefaultPleSeed = 1234;

std::uint32_t integer(const Json& value, std::string_view label, bool positive = true) {
    const auto n = artifact::require_u64(value, label, positive);
    if (n > std::numeric_limits<std::uint32_t>::max()) {
        throw ArtifactError(std::string(label) + ": config dimension exceeds u32");
    }
    return static_cast<std::uint32_t>(n);
}

std::uint32_t dimension(const Json& config, const char* name) {
    return integer(config.at(name), name);
}

float real_value(const Json& config, const char* name, bool positive) {
    const auto& value = config.at(name);
    if (!value.is_number()) { throw ArtifactError(std::string(name) + " must be a real value"); }
    const float result = value.get<float>();
    if (!std::isfinite(result) || (positive ? result <= 0 : result < 0)) {
        throw ArtifactError(std::string(name) +
                            (positive ? " must be positive finite FP32"
                                      : " must be nonnegative finite FP32"));
    }
    return result;
}

std::string architecture(const Json& config) {
    const auto& names = config.at("architectures");
    if (!names.is_array() || names.size() != 1) {
        throw ArtifactError("architectures must name one implementation");
    }
    return artifact::require_id(names[0], "architecture");
}

bool boolean(const Json& config, const char* name) {
    const auto& value = config.at(name);
    if (!value.is_boolean()) { throw ArtifactError(std::string(name) + " must be boolean"); }
    return value.get<bool>();
}

void require_canonical(bool condition, const char* message) {
    if (!condition) { throw ArtifactError(message); }
}

AttentionConfig attention(const Json& value) {
    AttentionConfig out;
    out.num_attention_heads   = dimension(value, "num_attention_heads");
    out.num_key_value_heads   = dimension(value, "num_key_value_heads");
    out.head_dim              = dimension(value, "head_dim");
    out.partial_rotary_factor = real_value(value, "partial_rotary_factor", true);
    out.attention_bias        = boolean(value, "attention_bias");
    out.attention_dropout     = real_value(value, "attention_dropout", false);
    if (out.num_attention_heads % out.num_key_value_heads) {
        throw ArtifactError("attention query heads must be divisible by KV heads");
    }
    if (out.partial_rotary_factor > 1) {
        throw ArtifactError("partial_rotary_factor exceeds one");
    }
    const auto rotary_dim =
        static_cast<std::uint32_t>(double(out.head_dim) * out.partial_rotary_factor);
    if (!rotary_dim || rotary_dim % 2) {
        throw ArtifactError("rotary dimension must be positive and even");
    }
    const auto& rope = value.at("rope_parameters");
    require_members(rope, {"rope_theta", "partial_rotary_factor"}, {}, "text RoPE");
    out.rope_theta = real_value(rope, "rope_theta", true);
    if (real_value(rope, "partial_rotary_factor", true) != out.partial_rotary_factor) {
        throw ArtifactError("text and RoPE partial_rotary_factor differ");
    }
    return out;
}

GdnConfig gdn(const Json& value) {
    GdnConfig out;
    out.linear_num_key_heads   = dimension(value, "linear_num_key_heads");
    out.linear_num_value_heads = dimension(value, "linear_num_value_heads");
    out.linear_key_head_dim    = dimension(value, "linear_key_head_dim");
    out.linear_value_head_dim  = dimension(value, "linear_value_head_dim");
    out.linear_conv_kernel_dim = dimension(value, "linear_conv_kernel_dim");
    out.output_gate_type =
        artifact::require_id(value.at("output_gate_type"), "output_gate_type");
    if (out.linear_num_value_heads % out.linear_num_key_heads) {
        throw ArtifactError("GDN value heads must be divisible by key heads");
    }
    return out;
}

HyperConnectionConfig hyper_connection(const Json& value) {
    return {dimension(value, "hc_count"), dimension(value, "hc_lowrank")};
}

MoeConfig moe(const Json& value) {
    MoeConfig out;
    out.num_experts                     = dimension(value, "num_experts");
    out.num_experts_per_tok             = dimension(value, "num_experts_per_tok");
    out.moe_intermediate_size           = dimension(value, "moe_intermediate_size");
    out.shared_expert_intermediate_size =
        dimension(value, "shared_expert_intermediate_size");
    out.norm_topk_prob =
        value.contains("norm_topk_prob") ? boolean(value, "norm_topk_prob") : true;
    if (out.num_experts_per_tok > out.num_experts) {
        throw ArtifactError("selected experts exceed expert count");
    }
    return out;
}

PleConfig ple(const Json& value, std::uint32_t num_hidden_layers) {
    PleConfig out;
    out.ngram_size = dimension(value, "ngram_size");
    out.ngram_vocab_size_base =
        artifact::require_u64(value.at("ngram_vocab_size_base"), "ngram_vocab_size_base", true);
    out.make_ngram_vocab_size_divisible_by =
        dimension(value, "make_ngram_vocab_size_divisible_by");
    out.heads_per_ngram      = dimension(value, "heads_per_ngram");
    out.split_ngram_parts    = dimension(value, "split_ngram_parts");
    out.ple_embed_dim        = dimension(value, "ple_embed_dim");
    out.ple_conv_kernel_size = dimension(value, "ple_conv_kernel_size");
    out.seed = value.contains("seed") ? integer(value.at("seed"), "seed", false) : kDefaultPleSeed;
    if (out.ngram_size < 2) { throw ArtifactError("ngram_size must be at least two"); }
    const auto total_heads = out.total_heads();
    if (!total_heads || out.ple_embed_dim % total_heads) {
        throw ArtifactError("PLE embedding width must be divisible by total ngram heads");
    }
    if (out.ngram_vocab_size_base % out.make_ngram_vocab_size_divisible_by) {
        throw ArtifactError("ngram vocabulary base is not divisible by its configured divisor");
    }
    const auto& ids = value.at("ple_layer_ids");
    if (!ids.is_array() || ids.empty()) {
        throw ArtifactError("ple_layer_ids must be a nonempty array");
    }
    std::set<std::uint32_t> unique;
    for (const auto& id : ids) {
        const auto one_based = integer(id, "PLE layer ID", false);
        if (one_based == 0 || one_based > num_hidden_layers || !unique.insert(one_based).second) {
            throw ArtifactError("PLE layer IDs must be unique and in one-based layer range");
        }
        out.ple_layer_ids.push_back(one_based);
    }
    return out;
}

IndexerConfig indexer(const Json& value) {
    IndexerConfig out{dimension(value, "indexer_budget"),
                      dimension(value, "indexer_compress_ratio"),
                      dimension(value, "indexer_head_dim"),
                      dimension(value, "indexer_kv_heads"),
                      dimension(value, "indexer_n_heads")};
    if (out.indexer_budget % out.indexer_compress_ratio) {
        throw ArtifactError("indexer budget must be divisible by compression ratio");
    }
    return out;
}

std::optional<MtpConfig> mtp(const Json& value) {
    const bool has_layers = value.contains("mtp_num_hidden_layers");
    const bool has_embed  = value.contains("mtp_use_dedicated_embeddings");
    if (has_layers != has_embed) {
        throw ArtifactError("MTP metadata must provide layers and dedicated-embedding flag");
    }
    if (!has_layers) { return std::nullopt; }
    return MtpConfig{dimension(value, "mtp_num_hidden_layers"),
                     boolean(value, "mtp_use_dedicated_embeddings")};
}

void validate_canonical(const TextConfig& out) {
    require_canonical(out.hidden_size == 2560, "Flash-Next hidden_size must be 2560");
    require_canonical(out.vocab_size == 248320, "Flash-Next vocab_size must be 248320");
    require_canonical(out.num_hidden_layers == 48,
                      "Flash-Next num_hidden_layers must be 48");
    require_canonical(out.max_position_embeddings == 262144,
                      "Flash-Next max_position_embeddings must be 262144");
    require_canonical(out.full_attention_interval == 4,
                      "Flash-Next full_attention_interval must be 4");
    require_canonical(!out.tie_word_embeddings,
                      "Flash-Next tie_word_embeddings must be false");
    require_canonical(out.rms_norm_eps == 1e-6F, "Flash-Next rms_norm_eps must be 1e-6");

    const auto& a = out.attention;
    require_canonical(a.num_attention_heads == 24 && a.num_key_value_heads == 2 &&
                          a.head_dim == 256 && a.partial_rotary_factor == 0.25F &&
                          a.rope_theta == 10000000.0F && !a.attention_bias &&
                          a.attention_dropout == 0,
                      "Flash-Next full-attention geometry differs from the frozen target");

    const auto& g = out.gdn;
    require_canonical(g.linear_num_key_heads == 16 && g.linear_num_value_heads == 48 &&
                          g.linear_key_head_dim == 128 && g.linear_value_head_dim == 128 &&
                          g.linear_conv_kernel_dim == 4 && g.output_gate_type == "sigmoid",
                      "Flash-Next GDN geometry differs from the frozen target");

    require_canonical(out.hyper_connection.hc_count == 4 &&
                          out.hyper_connection.hc_lowrank == 320 &&
                          out.hyper_connection.width(out.hidden_size) == 10240,
                      "Flash-Next HyperConnection geometry differs from the frozen target");

    const auto& m = out.moe;
    require_canonical(m.num_experts == 512 && m.num_experts_per_tok == 10 &&
                          m.moe_intermediate_size == 640 &&
                          m.shared_expert_intermediate_size == 640 && m.norm_topk_prob,
                      "Flash-Next MoE geometry differs from the frozen target");

    const auto& p = out.ple;
    require_canonical(p.ngram_size == 3 && p.ngram_vocab_size_base == 20000000 &&
                          p.make_ngram_vocab_size_divisible_by == 128 &&
                          p.heads_per_ngram == 8 && p.split_ngram_parts == 128 &&
                          p.ple_embed_dim == 2560 && p.ple_conv_kernel_size == 4 &&
                          p.seed == kDefaultPleSeed &&
                          p.ple_layer_ids == std::vector<std::uint32_t>{2},
                      "Flash-Next PLE geometry differs from the frozen target");

    const auto& i = out.indexer;
    require_canonical(i.indexer_budget == 2048 && i.indexer_compress_ratio == 4 &&
                          i.indexer_head_dim == 128 && i.indexer_kv_heads == 1 &&
                          i.indexer_n_heads == 4 && i.block_budget() == 512,
                      "Flash-Next QSA indexer geometry differs from the frozen target");

    if (out.mtp) {
        require_canonical(out.mtp->mtp_num_hidden_layers == 1 &&
                              !out.mtp->mtp_use_dedicated_embeddings,
                          "Flash-Next MTP metadata differs from the frozen target");
    }
}

TextConfig text(const Json& value) {
    require_members(
        value,
        {"architectures", "model_type", "hidden_size", "vocab_size", "num_hidden_layers",
         "max_position_embeddings", "tie_word_embeddings", "rms_norm_eps",
         "full_attention_interval", "layer_types", "num_attention_heads",
         "num_key_value_heads", "head_dim", "partial_rotary_factor", "attention_bias",
         "attention_dropout", "rope_parameters", "linear_num_key_heads",
         "linear_num_value_heads", "linear_key_head_dim", "linear_value_head_dim",
         "linear_conv_kernel_dim", "output_gate_type", "hc_count", "hc_lowrank",
         "num_experts", "num_experts_per_tok", "moe_intermediate_size",
         "shared_expert_intermediate_size", "ngram_size", "ngram_vocab_size_base",
         "make_ngram_vocab_size_divisible_by", "heads_per_ngram", "split_ngram_parts",
         "ple_embed_dim", "ple_conv_kernel_size", "ple_layer_ids", "indexer_budget",
         "indexer_compress_ratio", "indexer_head_dim", "indexer_kv_heads", "indexer_n_heads"},
        {"norm_topk_prob", "seed", "mtp_num_hidden_layers",
         "mtp_use_dedicated_embeddings"},
        "Qwen4Exp text config");

    TextConfig out;
    out.architecture = resolve_architecture(
        architecture(value), artifact::require_id(value.at("model_type"), "model_type"));
    out.hidden_size             = dimension(value, "hidden_size");
    out.vocab_size              = dimension(value, "vocab_size");
    out.num_hidden_layers       = dimension(value, "num_hidden_layers");
    out.max_position_embeddings = dimension(value, "max_position_embeddings");
    out.full_attention_interval = dimension(value, "full_attention_interval");
    out.tie_word_embeddings     = boolean(value, "tie_word_embeddings");
    out.rms_norm_eps            = real_value(value, "rms_norm_eps", true);

    const auto& layers = value.at("layer_types");
    if (!layers.is_array() || layers.size() != out.num_hidden_layers) {
        throw ArtifactError("layer_types must cover every Qwen4Exp text block");
    }
    for (std::size_t layer = 0; layer < layers.size(); ++layer) {
        LayerKind kind;
        if (layers[layer] == "full_attention") {
            kind = LayerKind::FullAttention;
        } else if (layers[layer] == "linear_attention") {
            kind = LayerKind::LinearAttention;
        } else {
            throw ArtifactError("unknown Qwen4Exp text layer_type");
        }
        const auto expected = (layer + 1) % out.full_attention_interval == 0
                                  ? LayerKind::FullAttention
                                  : LayerKind::LinearAttention;
        if (kind != expected) {
            throw ArtifactError("layer_types disagree with full_attention_interval");
        }
        out.layer_types.push_back(kind);
    }

    out.attention        = attention(value);
    out.gdn              = gdn(value);
    out.hyper_connection = hyper_connection(value);
    out.moe              = moe(value);
    out.ple              = ple(value, out.num_hidden_layers);
    out.indexer          = indexer(value);
    out.mtp              = mtp(value);
    validate_canonical(out);
    return out;
}

} // namespace

Config parse_config(const artifact::Directory& directory) {
    try {
        return {text(directory.component("text").config)};
    } catch (const std::exception& error) {
        throw ArtifactError(std::string("Qwen4Exp config: ") + error.what());
    }
}

} // namespace ninfer::models::qwen4_exp
