# Batch 3B2 — Qwen4Exp recipe and CLI

- Base: `7d31e56796a9e8d85ec3a4f0b46174841cfd1c0b`
- Target branch: `flash-next/batch3b2-recipe-cli`

## Scope

Converter-only; no C++/CUDA/runtime/GPU/service.

## Allowed files

- `tools/convert/official_recipes.py`
- `tools/convert/qwen4_exp.py`
- `tools/convert/__main__.py`
- `tests/convert/test_qwen4_exp.py`
- `tests/convert/test_cli.py`

## Forbidden files

- `tools/convert/methods.py`
- `tools/convert/recipe.py`
- `tools/convert/sources/modelopt.py`

## Decisions

- routed ModelOpt `input_divisor` present => AllowA4
- `input_divisor` absent => A16Only
- never invent a divisor
- preserve `None` in lazy Qwen4Exp source
- routed Parameters expose mathematical input uses

## Recipe

- token embedding: `q8_g32_fp16` / `grouped_absmax`
- output head: `q6_g64_fp16` / `grouped_absmax`
- routed experts: `nvfp4` / `import_encoded`
- GDN `in_proj_a` / `in_proj_b`: `BF16` / `direct`
- router / `shared_gate`: `BF16` / `direct`
- other linear projections: `q8_g32_fp16` / `grouped_absmax`
- norms / convolutions / scalars / PLE: `direct`
- non-routed: `A16Only`

## Acceptance

- exact assignments
- mixed `input_scale`
- byte/divisor preservation
- activation aux only `AllowA4`
- no-divisor `A16Only`
- forced `AllowA4` failure
- routed artifact uses
- Qwen4Exp CLI
- Qwen3.5 CLI regression
- prior 3A/3B1 tests

## Policy

- Luna orchestration
- local-worker routine implementation/tests/git
- Sol only explicit review
- GPU agents sequential
- no GPU tests

## Deferred

- Qwen4Exp runtime/program
- HyperConnection/PLE/GDN/QSA execution
- sparse-MoE runtime/kernels
- CUDA shape/kernel
- serving/benchmarking/GPU validation

## Implementation notes

- converter CLI entrypoint is `tools/convert/__main__.py`
- routed expert `gate` / `up` inputs are `text/layers/L/ffn_input`
- routed expert `down` input is `text/layers/L/moe/experts/E/product`
- Qwen4Exp source factories preserve genuine `input_divisor=None`
- official recipe name is `qwen4_exp_nvfp4`

## Validation

Batch 3B2 implementation validation:
- `git diff --check`: PASS
- Batch 3B2 acceptance tests: 8 passed
- focused converter suite: 53 passed, 1 existing PyTorch buffer warning
- recipe coverage audit: PASS
- complete `tests/convert` CPU-only regression suite: PASS
- CUDA-specific quantization branches intentionally excluded from Batch 3B2 validation because this milestone is converter-only and the RTX 5080 remains allocated to `ninfer-local-model.service`
- no GPU/service changes were required
- `ninfer-local-model.service` remained running and untouched
- implementation scope remained converter-only
