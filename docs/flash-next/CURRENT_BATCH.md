# Batch 3C1 — Qwen4Exp HyperConnection primitives

- Base: `97a5a9b7acb26aa3a852ed8504d77ad5c9a71bea`
- Branch: `flash-next/batch3c-hyperconnection-reference`

## Scope

Implement only the reusable mathematical primitives required by the
Qwen4Exp HyperConnection reference equations.

No Qwen4Exp execution/program integration in 3C1.

## Semantics

- initial repeat `[H,T] -> [4H,T]`
- four independent RMSNorm groups of width `H`
- `SiLU(x / hc_count)`
- read/mix:
  `mean_streams(sigmoid(logits) * normalized)`
- injection:
  `hyper += block_output * (2 * sigmoid(inject_logits / hc_count))`

Canonical Flash-Next geometry:

- `hc_count = 4`
- `hidden_size = 2560`
- `expanded_width = 10240`
- `hc_lowrank = 320`

## Files

- `include/ninfer/ops/hyper_connection.h`
- `src/ops/launcher/hyper_connection.h`
- `src/ops/launcher/hyper_connection.cu`
- `src/ops/wrapper/hyper_connection.cpp`
- `src/ops/basic_sources.cmake`
- `tests/ops/test_hyper_connection.cpp`
- `tests/ops/tests.cmake`

## Explicitly deferred to 3C2

- bound HyperConnection parameter preparation
- `linear()` composition for down/up/block-inject
- Qwen4Exp layer execution
- final model-level mixer integration
- PLE/GDN/QSA/MoE execution
- full decoder/program ownership

## GPU policy

Compile while `ninfer-local-model.service` remains running.

Before CUDA execution tests:
1. stop `ninfer-local-model.service`
2. run bounded HyperConnection GPU tests
3. restart service
4. verify service/model health

## Validation

- source hygiene (`git diff --check`): PASS
- HyperConnection primitive CUDA oracle tests: PASS
- canonical Flash-Next geometry (`C=4`, `H=2560`, `R=320`): PASS
- CUDA Graph capture/instantiate/replay: PASS
- GPU test performed with live NInfer service stopped
- `ninfer-local-model.service` restarted successfully
- `/v1/models` health check after restoration: PASS
- ccache enabled for C, C++, and CUDA compilation

## Publication state

- Implementation status: COMPLETE
- Validation status: COMPLETE
- Commit: NOT DONE before publication
- Push: NOT DONE before publication
- Next milestone: Batch 3C2 — Qwen4Exp HyperConnection executor (deferred; not started)

## Publication result

- Batch 3C1 implementation commit: c67f31f71b13d4a0fba199fe86348378e8743fbe
- GitHub publication: PASS
- Published branch: lash-next/batch3c-hyperconnection-reference
- Next batch: Batch 3C2 — Qwen4Exp HyperConnection executor
- Next batch status: deferred; not started
