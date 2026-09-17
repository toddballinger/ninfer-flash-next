"""Qwen4Exp Flash-Next BF16/ModelOpt text config and logical checkpoint mapping.

This adapter stops at the frozen text-backbone mathematical model.  Routed
experts may be supplied either as the Batch 3A fused BF16 tensors or, in a
ModelOpt NVFP4 checkpoint, as split per-expert modules that resolve through the
``modelopt_matrix_source`` adapter.  Non-routed parameters always use the
decoded (direct) path.  Runtime execution and recipe support belong to later
batches.
"""

from __future__ import annotations

from math import isfinite

from .model import Model, Parameter
from .sources.modelopt import modelopt_matrix_source
from .sources.safetensors import SafetensorsSource, tensor_source


_ARCHITECTURE = "Qwen4ExpForConditionalGeneration"
_MODEL_TYPE = "qwen4_exp_text"


def _source_value(source: dict, raw: dict, key: str, default=None, *, required=False):
    if key in raw:
        return raw[key]
    if key in source:
        return source[key]
    if required:
        raise ValueError(f"text.{key}: missing required canonical field")
    return default


def _fixed_int(source: dict, raw: dict, key: str, expected: int, *, required=True):
    value = _source_value(source, raw, key, expected, required=required)
    if type(value) is not int or value != expected:
        raise ValueError(f"text.{key}: expected {expected!r}, got {value!r}")
    return value


def _fixed_real(source: dict, raw: dict, key: str, expected: float, *, required=False):
    value = _source_value(source, raw, key, expected, required=required)
    if type(value) not in (int, float) or not isfinite(float(value)):
        raise ValueError(f"text.{key}: expected finite real {expected!r}")
    if float(value) != expected:
        raise ValueError(f"text.{key}: expected {expected!r}, got {value!r}")
    return expected


def _fixed_bool(source: dict, raw: dict, key: str, expected: bool):
    value = _source_value(source, raw, key, expected)
    if type(value) is not bool or value is not expected:
        raise ValueError(f"text.{key}: expected {expected!r}, got {value!r}")
    return value


def text_config(source: dict) -> dict:
    """Normalize an official Qwen4Exp config to the frozen NInfer text config."""

    if not isinstance(source, dict):
        raise ValueError("Qwen4Exp config must be an object")
    raw = source.get("text_config", source)
    if not isinstance(raw, dict):
        raise ValueError("text_config must be an object")

    architectures = source.get("architectures", raw.get("architectures"))
    model_type = raw.get("model_type")
    if architectures is not None and architectures != [_ARCHITECTURE]:
        raise ValueError(f"unsupported Qwen4Exp architecture {architectures!r}")
    # A root multimodal model_type may describe the wrapper rather than its text
    # component. A nested text_config identity, however, is authoritative.
    if raw is not source and model_type is not None and model_type != _MODEL_TYPE:
        raise ValueError(f"unsupported Qwen4Exp model_type {model_type!r}")
    if architectures is None and model_type != _MODEL_TYPE:
        raise ValueError("Qwen4Exp config requires architecture or text model identity")

    result = {"architectures": [_ARCHITECTURE], "model_type": _MODEL_TYPE}
    for key, expected in (
        ("hidden_size", 2560),
        ("vocab_size", 248320),
        ("num_hidden_layers", 48),
        ("max_position_embeddings", 262144),
        ("num_attention_heads", 24),
        ("num_key_value_heads", 2),
        ("head_dim", 256),
        ("linear_num_key_heads", 16),
        ("linear_num_value_heads", 48),
        ("linear_key_head_dim", 128),
        ("linear_value_head_dim", 128),
        ("linear_conv_kernel_dim", 4),
        ("hc_count", 4),
        ("hc_lowrank", 320),
        ("num_experts", 512),
        ("num_experts_per_tok", 10),
        ("moe_intermediate_size", 640),
        ("shared_expert_intermediate_size", 640),
        ("ngram_size", 3),
        ("ngram_vocab_size_base", 20000000),
        ("make_ngram_vocab_size_divisible_by", 128),
        ("heads_per_ngram", 8),
        ("split_ngram_parts", 128),
        ("ple_embed_dim", 2560),
        ("ple_conv_kernel_size", 4),
        ("indexer_budget", 2048),
        ("indexer_compress_ratio", 4),
        ("indexer_head_dim", 128),
        ("indexer_kv_heads", 1),
        ("indexer_n_heads", 4),
    ):
        result[key] = _fixed_int(source, raw, key, expected)

    result["full_attention_interval"] = _fixed_int(
        source, raw, "full_attention_interval", 4, required=False
    )
    result["tie_word_embeddings"] = _fixed_bool(
        source, raw, "tie_word_embeddings", False
    )
    result["rms_norm_eps"] = _fixed_real(source, raw, "rms_norm_eps", 1e-6)
    result["partial_rotary_factor"] = _fixed_real(
        source, raw, "partial_rotary_factor", 0.25
    )
    result["attention_bias"] = _fixed_bool(source, raw, "attention_bias", False)
    result["attention_dropout"] = _fixed_real(
        source, raw, "attention_dropout", 0.0
    )
    result["norm_topk_prob"] = _fixed_bool(source, raw, "norm_topk_prob", True)

    output_gate = _source_value(source, raw, "output_gate_type", "sigmoid")
    if output_gate != "sigmoid":
        raise ValueError(
            f"text.output_gate_type: expected 'sigmoid', got {output_gate!r}"
        )
    result["output_gate_type"] = output_gate

    rope = raw.get("rope_parameters", source.get("rope_parameters", {}))
    if not isinstance(rope, dict):
        raise ValueError("text.rope_parameters must be an object")
    rope_theta = rope.get(
        "rope_theta", _source_value(source, raw, "rope_theta", 10000000)
    )
    rope_factor = rope.get("partial_rotary_factor", result["partial_rotary_factor"])
    if (
        type(rope_theta) not in (int, float)
        or type(rope_factor) not in (int, float)
        or not isfinite(float(rope_theta))
        or not isfinite(float(rope_factor))
        or float(rope_theta) != 10000000.0
        or float(rope_factor) != 0.25
    ):
        raise ValueError("text.rope_parameters differs from frozen Flash-Next RoPE")
    result["rope_parameters"] = {
        "rope_theta": 10000000.0,
        "partial_rotary_factor": 0.25,
    }

    layers = raw.get("layer_types")
    if layers is None:
        layers = [
            "full_attention" if (i + 1) % 4 == 0 else "linear_attention"
            for i in range(48)
        ]
    expected_layers = [
        "full_attention" if (i + 1) % 4 == 0 else "linear_attention"
        for i in range(48)
    ]
    if not isinstance(layers, list) or layers != expected_layers:
        raise ValueError(
            "text.layer_types must be the frozen 36-GDN/12-QSA interval-4 topology"
        )
    result["layer_types"] = list(layers)

    ple_ids = _source_value(source, raw, "ple_layer_ids", None, required=True)
    if ple_ids != [2]:
        raise ValueError("text.ple_layer_ids must preserve one-based placement [2]")
    result["ple_layer_ids"] = list(ple_ids)
    seed = _source_value(source, raw, "seed", 1234)
    if type(seed) is not int or seed != 1234:
        raise ValueError(f"text.seed: expected 1234, got {seed!r}")
    result["seed"] = seed
    return result


class _Builder:
    def __init__(self, model: Model, store: SafetensorsSource):
        self.model = model
        self.store = store

    @staticmethod
    def _check(selected, source_name, source_shape, source_dtype=None):
        info = selected.describe(source_name)
        if info.shape != source_shape:
            raise ValueError(
                f"{source_name}: expected source shape {source_shape}, got {info.shape}"
            )
        if source_dtype is not None and info.dtype != source_dtype:
            raise ValueError(
                f"{source_name}: expected source dtype {source_dtype}, got {info.dtype}"
            )

    def add(
        self,
        name,
        source_name,
        shape,
        *,
        source_shape=None,
        offset=0,
        inputs=(),
        direct="bf16",
        residency="text",
        source_dtype=None,
    ):
        shape = tuple(shape)
        original = shape if source_shape is None else tuple(source_shape)
        self._check(self.store, source_name, original, source_dtype)

        def factory(selected, format=None):
            if format is not None:
                raise ValueError(
                    f"{name}: Batch 3A supports decoded BF16 source values only"
                )
            self._check(selected, source_name, original, source_dtype)
            return tensor_source(
                selected, source_name, shape, offset=offset, source_shape=original
            )

        self.model.add(
            Parameter(
                name,
                shape,
                factory(self.store),
                factory,
                tuple(inputs),
                direct,
                residency,
            )
        )

    def _modelopt_experts(self, source: str) -> bool:
        """True when this layer's routed experts use the ModelOpt NVFP4 split layout."""
        probe = f"{source}.experts.0.gate_proj.weight"
        if not self.store.has(probe):
            return False
        prefix = probe.removesuffix(".weight")
        return self.store.has(prefix + ".weight_scale_2") and self.store.has(
            prefix + ".weight_scale"
        )

    def _add_modelopt(self, name, source_name, shape, *, inputs=(), direct="bf16"):
        shape = tuple(shape)
        self.model.add(
            Parameter(
                name,
                shape,
                modelopt_matrix_source(self.store, source_name, shape),
                lambda selected, format=None: modelopt_matrix_source(
                    selected, source_name, shape
                ),
                tuple(inputs),
                direct,
                "routed_experts",
            )
        )

    def _fused_routed_experts(self, prefix, source):
        """Batch 3A fused-tensor routed experts (frozen and unchanged)."""
        gate_up_name = source + ".experts.gate_up_proj"
        down_name = source + ".experts.down_proj"
        gate_up_shape = (512, 1280, 2560)
        down_shape = (512, 2560, 640)
        self._check(self.store, gate_up_name, gate_up_shape, "BF16")
        self._check(self.store, down_name, down_shape, "BF16")
        gate_elements = 640 * 2560
        down_elements = 2560 * 640
        for expert in range(512):
            target = f"{prefix}/moe/experts/{expert}"
            for role, half in (("gate", 0), ("up", 1)):
                self.add(
                    f"{target}/{role}",
                    gate_up_name,
                    (640, 2560),
                    source_shape=gate_up_shape,
                    offset=(expert * 2 + half) * gate_elements,
                    residency="routed_experts",
                    source_dtype="BF16",
                )
            self.add(
                f"{target}/down",
                down_name,
                (2560, 640),
                source_shape=down_shape,
                offset=expert * down_elements,
                residency="routed_experts",
                source_dtype="BF16",
            )

    def _modelopt_routed_experts(self, prefix, source):
        """ModelOpt split NVFP4 routed experts (per-expert gate/up/down)."""
        for expert in range(512):
            target = f"{prefix}/moe/experts/{expert}"
            module = f"{source}.experts.{expert}"
            self._add_modelopt(f"{target}/gate", module + ".gate_proj.weight", (640, 2560))
            self._add_modelopt(f"{target}/up", module + ".up_proj.weight", (640, 2560))
            self._add_modelopt(f"{target}/down", module + ".down_proj.weight", (2560, 640))

    def hyper_connection(self, canonical, source):
        for role, field, shape in (
            ("norm", "hc_norm.weight", (10240,)),
            ("down", "input_mix_weight_down.weight", (320, 10240)),
            ("up", "input_mix_weight_up.weight", (10240, 320)),
        ):
            self.add(f"{canonical}/{role}", f"{source}.{field}", shape)

    def layer_hyper_connection(self, canonical, source):
        self.hyper_connection(canonical, source)
        self.add(
            canonical + "/block_inject",
            source + ".block_inject_weight.weight",
            (4, 10240),
        )

    def gdn(self, prefix, source):
        source += ".linear_attn"
        for role, field in (("a_log", "A_log"), ("dt_bias", "dt_bias")):
            self.add(
                f"{prefix}/gdn/{role}",
                f"{source}.{field}",
                (48,),
                direct="fp32",
                source_dtype="F32",
            )
        for role, field, shape in (
            ("convolution", "conv1d.weight", (10240, 1, 4)),
            ("in_proj_a", "in_proj_a.weight", (48, 2560)),
            ("in_proj_b", "in_proj_b.weight", (48, 2560)),
            ("qkv", "in_proj_qkv.weight", (10240, 2560)),
            ("z", "in_proj_z.weight", (6144, 2560)),
            ("norm", "norm.weight", (128,)),
            ("output", "out_proj.weight", (2560, 6144)),
        ):
            self.add(f"{prefix}/gdn/{role}", f"{source}.{field}", shape)

    def qsa(self, prefix, source):
        source += ".self_attn"
        for role, field, shape in (
            ("query", "q_proj.weight", (12288, 2560)),
            ("key", "k_proj.weight", (512, 2560)),
            ("value", "v_proj.weight", (512, 2560)),
            ("output", "o_proj.weight", (2560, 6144)),
            ("query_norm", "q_norm.weight", (256,)),
            ("key_norm", "k_norm.weight", (256,)),
            ("indexer/query_key", "indexer.index_qk_proj.weight", (640, 2560)),
            ("indexer/query_norm", "indexer.q_layernorm.weight", (128,)),
            ("indexer/key_norm", "indexer.k_layernorm.weight", (128,)),
        ):
            self.add(f"{prefix}/attention/{role}", f"{source}.{field}", shape)

    def moe(self, prefix, source):
        source += ".mlp"
        for role, field, shape in (
            ("router", "gate.weight", (512, 2560)),
            ("shared_gate", "shared_expert_gate.weight", (1, 2560)),
            ("shared/gate", "shared_expert.gate_proj.weight", (640, 2560)),
            ("shared/up", "shared_expert.up_proj.weight", (640, 2560)),
            ("shared/down", "shared_expert.down_proj.weight", (2560, 640)),
        ):
            self.add(f"{prefix}/moe/{role}", f"{source}.{field}", shape)

        if self._modelopt_experts(source):
            self._modelopt_routed_experts(prefix, source)
        else:
            self._fused_routed_experts(prefix, source)

    def ple(self, prefix, source):
        source += ".ple"
        for role, field, shape in (
            ("convolution", "conv1d.weight", (10240, 1, 4)),
            ("key", "key_proj.weight", (10240, 2560)),
            ("value", "value_proj.weight", (2560, 2560)),
            ("key_norm", "norm_key.weight", (10240,)),
            ("query_norm", "norm_query.weight", (10240,)),
            ("conv_norm", "norm_conv.weight", (10240,)),
        ):
            self.add(f"{prefix}/ple/{role}", f"{source}.{field}", shape)

        semantic = source + ".ple_embedding"
        for role, size in (
            ("layer_multipliers", 3),
            ("ngram_heads_offsets", 16),
            ("ngram_heads_vocab_sizes", 16),
        ):
            self.add(
                f"{prefix}/ple/{role}",
                f"{semantic}.{role}",
                (size,),
                direct="int64",
                residency="values",
                source_dtype="I64",
            )

        shards = semantic + ".ngram_embedding"
        for shard in range(128):
            source_name = f"{shards}.shard_{shard}.weight"
            info = self.store.describe(source_name)
            if len(info.shape) != 2 or info.shape[0] <= 0 or info.shape[1] != 160:
                raise ValueError(
                    f"{source_name}: expected a non-empty matrix with width 160, "
                    f"got {info.shape}"
                )
            self.add(
                f"{prefix}/ple/embedding/shards/{shard}",
                source_name,
                info.shape,
                residency="ple_shards",
                source_dtype="BF16",
            )


def build_model(
    base: SafetensorsSource, *, components: tuple[str, ...] = ("text",)
) -> Model:
    """Build the frozen Qwen4Exp text-only logical model (BF16 or ModelOpt experts)."""

    if tuple(components) != ("text",):
        raise ValueError("Qwen4Exp Batch 3A supports only components=('text',)")
    config = text_config(base.config)
    model = Model({"text": {"config": config}})
    builder = _Builder(model, base)
    source_root = "model.language_model"

    builder.add(
        "text/token_embedding",
        source_root + ".embed_tokens.weight",
        (248320, 2560),
    )
    builder.add("text/output_head", "lm_head.weight", (248320, 2560))
    builder.hyper_connection(
        "text/hyper_connection", source_root + ".hyper_connection_mixer"
    )

    for layer, kind in enumerate(config["layer_types"]):
        prefix = f"text/layers/{layer}"
        source = f"{source_root}.layers.{layer}"
        builder.layer_hyper_connection(
            prefix + "/attention_hyper_connection",
            source + ".attn_hyper_connection",
        )
        builder.layer_hyper_connection(
            prefix + "/mlp_hyper_connection", source + ".mlp_hyper_connection"
        )
        if kind == "linear_attention":
            builder.gdn(prefix, source)
        else:
            builder.qsa(prefix, source)
        builder.moe(prefix, source)
        if layer == 1:
            builder.ple(prefix, source)

    return model


__all__ = ["build_model", "text_config"]