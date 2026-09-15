#include "artifact/fixture.h"
#include "artifact/schema.h"
#include "models/qwen4_exp/config.h"

#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

namespace {

using namespace ninfer;
using namespace ninfer::models;
namespace qwen = ninfer::models::qwen4_exp;
using ninfer::test::artifact_fixture::Json;
using ninfer::test::artifact_fixture::rejects;
using ninfer::test::artifact_fixture::require;

Json canonical_config() {
    Json layers = Json::array();
    for (std::uint32_t layer = 0; layer < 48; ++layer) {
        layers.push_back((layer + 1) % 4 == 0 ? "full_attention" : "linear_attention");
    }
    return {
        {"architectures", {"Qwen4ExpForConditionalGeneration"}},
        {"model_type", "qwen4_exp_text"},
        {"hidden_size", 2560},
        {"vocab_size", 248320},
        {"num_hidden_layers", 48},
        {"max_position_embeddings", 262144},
        {"tie_word_embeddings", false},
        {"rms_norm_eps", 1e-6},
        {"full_attention_interval", 4},
        {"layer_types", std::move(layers)},
        {"num_attention_heads", 24},
        {"num_key_value_heads", 2},
        {"head_dim", 256},
        {"partial_rotary_factor", 0.25},
        {"attention_bias", false},
        {"attention_dropout", 0.0},
        {"rope_parameters", {{"rope_theta", 10000000}, {"partial_rotary_factor", 0.25}}},
        {"linear_num_key_heads", 16},
        {"linear_num_value_heads", 48},
        {"linear_key_head_dim", 128},
        {"linear_value_head_dim", 128},
        {"linear_conv_kernel_dim", 4},
        {"output_gate_type", "sigmoid"},
        {"hc_count", 4},
        {"hc_lowrank", 320},
        {"num_experts", 512},
        {"num_experts_per_tok", 10},
        {"moe_intermediate_size", 640},
        {"shared_expert_intermediate_size", 640},
        {"norm_topk_prob", true},
        {"ngram_size", 3},
        {"ngram_vocab_size_base", 20000000},
        {"make_ngram_vocab_size_divisible_by", 128},
        {"heads_per_ngram", 8},
        {"split_ngram_parts", 128},
        {"ple_embed_dim", 2560},
        {"ple_conv_kernel_size", 4},
        {"ple_layer_ids", {2}},
        {"indexer_budget", 2048},
        {"indexer_compress_ratio", 4},
        {"indexer_head_dim", 128},
        {"indexer_kv_heads", 1},
        {"indexer_n_heads", 4},
        {"mtp_num_hidden_layers", 1},
        {"mtp_use_dedicated_embeddings", false},
    };
}

qwen::Config parse(Json config) {
    artifact::Directory directory;
    directory.components.emplace("text", artifact::Component{.config = std::move(config)});
    return qwen::parse_config(directory);
}

void canonical_geometry() {
    const auto config = parse(canonical_config());
    const auto& text  = config.text;
    require(text.architecture == Architecture::Qwen4Exp, "Qwen4Exp architecture was not retained");
    require(text.hidden_size == 2560 && text.vocab_size == 248320 &&
                text.num_hidden_layers == 48 && text.full_attention_interval == 4,
            "authoritative Flash-Next backbone geometry was not retained");
    require(text.layer_types.size() == 48 &&
                text.layer_types[3] == qwen::LayerKind::FullAttention &&
                text.layer_types[47] == qwen::LayerKind::FullAttention,
            "the 48 authoritative layer types were not parsed");
    require(text.attention.num_attention_heads == 24 &&
                text.attention.num_key_value_heads == 2 && text.attention.head_dim == 256 &&
                text.attention.partial_rotary_factor == 0.25F && !text.attention.attention_bias &&
                text.attention.attention_dropout == 0,
            "full-attention source geometry was not retained");
    require(text.gdn.linear_num_key_heads == 16 && text.gdn.linear_num_value_heads == 48 &&
                text.gdn.linear_key_head_dim == 128 && text.gdn.linear_value_head_dim == 128 &&
                text.gdn.linear_conv_kernel_dim == 4 && text.gdn.output_gate_type == "sigmoid",
            "GDN source geometry was not retained");
    require(text.hyper_connection.hc_count == 4 && text.hyper_connection.hc_lowrank == 320 &&
                text.hyper_connection.width(text.hidden_size) == 10240,
            "HyperConnection geometry was not retained");
    require(text.moe.num_experts == 512 && text.moe.num_experts_per_tok == 10 &&
                text.moe.moe_intermediate_size == 640 &&
                text.moe.shared_expert_intermediate_size == 640 && text.moe.norm_topk_prob,
            "sparse MoE geometry was not retained");
    require(text.ple.ngram_size == 3 && text.ple.ngram_vocab_size_base == 20000000 &&
                text.ple.make_ngram_vocab_size_divisible_by == 128 &&
                text.ple.heads_per_ngram == 8 && text.ple.total_heads() == 16 &&
                text.ple.split_ngram_parts == 128 && text.ple.ple_embed_dim == 2560 &&
                text.ple.ple_conv_kernel_size == 4 && text.ple.seed == 1234,
            "PLE source geometry and semantic seed were not retained");
    require(text.ple.has_ple_at_layer(1), "physical layer 1 was not recognized as the PLE layer");
    require(!text.ple.has_ple_at_layer(2),
            "one-based PLE layer ID was incorrectly treated as zero-based");
    require(text.indexer.indexer_budget == 2048 && text.indexer.indexer_compress_ratio == 4 &&
                text.indexer.indexer_head_dim == 128 && text.indexer.indexer_kv_heads == 1 &&
                text.indexer.indexer_n_heads == 4 && text.indexer.block_budget() == 512,
            "QSA geometry or derived block budget was not retained");
    require(text.mtp && text.mtp->mtp_num_hidden_layers == 1 &&
                !text.mtp->mtp_use_dedicated_embeddings,
            "optional MTP metadata was not retained");
}

void identity() {
    require(resolve_architecture("Qwen4ExpForConditionalGeneration", "qwen4_exp_text") ==
                Architecture::Qwen4Exp,
            "authoritative Qwen4Exp identity did not resolve");
    require(architecture_name(Architecture::Qwen4Exp) == "Qwen4ExpForConditionalGeneration",
            "Qwen4Exp architecture name was not canonical");
    require(resolve_architecture("Qwen3_5ForCausalLM", "qwen3_5_text") ==
                Architecture::Qwen3_5 &&
                resolve_architecture("Qwen3_5MoeForCausalLM", "qwen3_5_moe_text") ==
                    Architecture::Qwen3_5Moe,
            "existing Qwen3.5 identities changed");
    rejects<std::invalid_argument>(
        [] {
            (void)resolve_architecture("Qwen4ExpForConditionalGeneration", "qwen3_5_text");
        },
        "Qwen4Exp architecture accepted a Qwen3.5 model type");
}

void invalid_config() {
    const auto bad = [](auto mutate, const char* message) {
        auto config = canonical_config();
        mutate(config);
        rejects([&] { (void)parse(std::move(config)); }, message);
    };
    bad([](Json& value) { value["ple_layer_ids"] = {0}; },
        "zero PLE semantic layer ID was accepted");
    bad([](Json& value) { value["ple_layer_ids"] = {49}; },
        "PLE semantic layer beyond the decoder was accepted");
    bad([](Json& value) { value["num_experts_per_tok"] = 513; },
        "expert TopK exceeding the expert count was accepted");
    bad([](Json& value) { value["moe_intermediate_size"] = 0; },
        "zero routed expert width was accepted");
    bad([](Json& value) { value["linear_key_head_dim"] = -1; },
        "negative GDN head width was accepted");
    bad([](Json& value) { value["layer_types"].erase(value["layer_types"].end() - 1); },
        "short layer_types array was accepted");
    bad([](Json& value) { value["layer_types"][0] = "future_attention"; },
        "unknown layer type was accepted");
    bad([](Json& value) { value["indexer_budget"] = 2049; },
        "non-divisible QSA indexer budget was accepted");
    bad([](Json& value) { value["hidden_size"] = 2561; },
        "noncanonical positive Flash-Next geometry was accepted");
}

} // namespace

int main() {
    try {
        canonical_geometry();
        identity();
        invalid_config();
        std::cout << "Qwen4Exp architecture and Flash-Next config checks passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
