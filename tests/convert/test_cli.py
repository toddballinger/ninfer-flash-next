from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

from tools.artifact.reader import Artifact
from tools.artifact.schema import binding_parts


def test_cli_custom_sources_method_template_and_shards(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    config = {
        "architectures": ["Qwen3_5ForCausalLM"],
        "hidden_size": 128,
        "vocab_size": 8,
        "num_hidden_layers": 1,
        "max_position_embeddings": 128,
        "layer_types": ["full_attention"],
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 8,
        "intermediate_size": 24,
        "rope_parameters": {"partial_rotary_factor": 0.5, "mrope_section": [1, 1, 0]},
    }
    (source / "config.json").write_text(json.dumps(config))
    for name, value in {
        "tokenizer.json": {"model": {"vocab": {str(i): i for i in range(6)}}},
        "tokenizer_config.json": {},
        "generation_config.json": {},
    }.items():
        (source / name).write_text(json.dumps(value))
    template = tmp_path / "template.jinja"
    template.write_text("custom {{ messages }}")
    recipe = tmp_path / "recipe.py"
    recipe.write_text("""import torch
from tools.convert.sources.logical import array_source

def custom(request):
    return request.job(produce=lambda output: output.write_values(0, request.source.values().to(torch.bfloat16)))

def configure(model, recipe, sources):
    for name, parameter in model.parameters.items():
        value = torch.ones(parameter.shape, dtype=torch.bfloat16)
        recipe.assign(name, source=array_source(value, "my-importer"))
    recipe.assign(("text/token_embedding", "text/layers/0/mlp/gate", "text/layers/0/mlp/up"), method=custom)
    for role, format in zip(("query", "key", "gate", "value"), ("q4_g64_fp16", "q5_g64_fp16", "q6_g64_fp16", "q8_g32_fp16")):
        recipe.assign("text/layers/0/attention/" + role, format=format, method="grouped_absmax")
""")
    output = tmp_path / "custom.ninfer"
    command = [
        sys.executable,
        "-B",
        "-m",
        "tools.convert",
        "--model",
        str(source),
        "--recipe",
        str(recipe),
        "--out",
        str(output),
        "--name",
        "my-training-run",
        "--resource",
        f"chat_template.jinja={template}",
        "--source",
        f"unused={tmp_path / 'missing'}",
        "--device",
        "cpu",
        "--max-file-bytes",
        "16384",
        "--rows-per-chunk",
        "3",
    ]
    root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        command,
        cwd=root,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )
    assert result.returncode == 0, result.stderr
    with Artifact(output) as artifact:
        assert set(artifact.directory.components) == {"text"}
        assert artifact.directory.metadata == {"name": "my-training-run"}
        assert len(artifact.directory.files) > 1
        resource = artifact.directory.components["text"]["resources"][
            "chat_template.jinja"
        ]
        assert artifact.read_object(resource) == template.read_bytes()
        embedding = binding_parts(
            artifact.directory.bindings["text/token_embedding"], artifact.by_id
        )[0][0]
        assert artifact.read_object(embedding) == b"\x80\x3f" * (8 * 128)
        parents = []
        for role in ("query", "key", "gate", "value"):
            name = "text/layers/0/attention/" + role
            parents.append(
                binding_parts(artifact.directory.bindings[name], artifact.by_id)[0][0]
            )
        assert len(set(parents)) == 4
        assert [artifact.object(parent).format for parent in parents] == [
            "q4_g64_fp16",
            "q5_g64_fp16",
            "q6_g64_fp16",
            "q8_g32_fp16",
        ]
        gate = binding_parts(
            artifact.directory.bindings["text/layers/0/mlp/gate"], artifact.by_id
        )[0][0]
        up = binding_parts(
            artifact.directory.bindings["text/layers/0/mlp/up"], artifact.by_id
        )[0][0]
        assert gate != up


def test_qwen4_exp_builder_dispatch(monkeypatch):
    import tools.convert.__main__ as cli

    qwen4_sentinel = object()
    qwen35_sentinel = object()
    calls = []

    def qwen4_builder(base, *, components):
        calls.append(("qwen4", base, components))
        return qwen4_sentinel

    def qwen35_builder(
        base,
        *,
        components,
        companions,
        resource_overrides,
    ):
        calls.append(
            (
                "qwen3_5",
                base,
                components,
                companions,
                resource_overrides,
            )
        )
        return qwen35_sentinel

    monkeypatch.setattr(cli, "build_qwen4_exp_model", qwen4_builder)
    monkeypatch.setattr(cli, "build_qwen3_5_model", qwen35_builder)

    class Base:
        pass

    qwen4 = Base()
    qwen4.config = {
        "architectures": ["Qwen4ExpForConditionalGeneration"],
        "text_config": {"model_type": "qwen4_exp_text"},
    }

    result = cli._build_model(
        qwen4,
        components=("text",),
        companions={},
        resource_overrides={},
    )

    assert result is qwen4_sentinel
    assert calls == [("qwen4", qwen4, ("text",))]


def test_qwen35_builder_dispatch_regression(monkeypatch):
    import tools.convert.__main__ as cli

    qwen35_sentinel = object()
    calls = []

    def qwen4_builder(base, *, components):
        raise AssertionError("Qwen4Exp builder must not receive Qwen3.5")

    def qwen35_builder(
        base,
        *,
        components,
        companions,
        resource_overrides,
    ):
        calls.append(
            (
                base,
                components,
                companions,
                resource_overrides,
            )
        )
        return qwen35_sentinel

    monkeypatch.setattr(cli, "build_qwen4_exp_model", qwen4_builder)
    monkeypatch.setattr(cli, "build_qwen3_5_model", qwen35_builder)

    class Base:
        pass

    qwen35 = Base()
    qwen35.config = {
        "architectures": ["Qwen3_5ForCausalLM"],
        "model_type": "qwen3_5_text",
    }

    companions = {"dflash": object()}
    overrides = {"chat_template.jinja": "custom.jinja"}

    result = cli._build_model(
        qwen35,
        components=("text", "dflash"),
        companions=companions,
        resource_overrides=overrides,
    )

    assert result is qwen35_sentinel
    assert calls == [
        (
            qwen35,
            ("text", "dflash"),
            companions,
            overrides,
        )
    ]


def test_qwen4_exp_builder_rejects_qwen35_only_cli_features():
    import tools.convert.__main__ as cli

    class Base:
        pass

    base = Base()
    base.config = {
        "architectures": ["Qwen4ExpForConditionalGeneration"],
        "text_config": {"model_type": "qwen4_exp_text"},
    }

    import pytest

    with pytest.raises(ValueError, match="companion components"):
        cli._build_model(
            base,
            components=("text",),
            companions={"dflash": object()},
            resource_overrides={},
        )

    with pytest.raises(ValueError, match="resource overrides"):
        cli._build_model(
            base,
            components=("text",),
            companions={},
            resource_overrides={"chat_template.jinja": "x"},
        )
