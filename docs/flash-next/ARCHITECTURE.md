# Flash-Next Architecture

## 1. Purpose

This document defines the NInfer implementation architecture for the Qwen4Exp model family used by Qwen3.8-Flash-Next.

The canonical mathematical and state semantics are frozen in `docs/flash-next/REFERENCE_FORMULAS.md`. That document is authoritative.

Priority order:

1. `REFERENCE_FORMULAS.md`
2. `ARCHITECTURE.md`
3. `RTX5080_RUNTIME_STRATEGY.md`
4. `IMPLEMENTATION_PLAN.md`
5. `BASELINE.md`

If there is any conflict, the earlier document wins.

## 2. Primary objective

Add exact Qwen4Exp / Flash-Next inference to NInfer while preserving the existing architecture and existing model families.

The implementation must optimise for:

1. correctness;
2. exact persistent-state semantics;
3. deterministic sequence replay/fork behaviour;
4. RTX 5080 memory efficiency;
5. sustained decode performance;
6. long context;
7. agent workloads.

The intended production workload is long-running local inference for OpenClaw, programming agents, and related interactive workloads.

## 3. Frozen model geometry

The implementation must obtain dimensions from artifact/config metadata wherever available. The frozen reference establishes the canonical initial target:

```text
decoder layers              48
hidden size                 2560
HyperConnection streams     4
HyperConnection width       10240
HyperConnection low-rank    320
MoE routed experts          512
experts selected/token      10
QSA indexer query heads     4
QSA indexer KV heads        1
QSA indexer head dimension  128
QSA compression ratio       4
QSA token budget            2048
QSA block TopK              512
PLE ngram_size              3
PLE heads/ngram             8
PLE embedding width         2560
PLE conv kernel             4
PLE conv dilation           3
PLE conv history            9
```

Where artifact metadata contains the corresponding value, NInfer should validate rather than silently override it. Unexpected incompatible geometry must fail explicitly.

## 4. Major architecture components

Flash-Next is not simply another ordinary Transformer + MoE model.

```text
token embedding
      |
      v
initial 4-stream HyperConnection state
      |
      v
48 decoder layers
      |
      +--> optional PLE injection
      +--> attention-side HyperConnection read
      +--> GDN or QSA block
      +--> HyperConnection injection
      +--> MLP-side HyperConnection read
      +--> sparse routed MoE + shared expert
      +--> HyperConnection injection
      |
      v
global HyperConnection mixer
      |
      v
final model head
```

The four-stream state and its update semantics are fundamental model semantics and must not be represented as an ordinary residual sum.

## 5. HyperConnection

Immediately after token embedding, the 2560-wide embedding is repeated across four streams, producing a 10240-wide `hyper_input`. No learned transform occurs before this initial four-stream replication.

The HyperConnection read/mix operation performs grouped RMSNorm independently over each stream, applies the low-rank down projection, divides by `hc_count`, applies SiLU, applies the up projection and sigmoid, reshapes into four stream weights, then computes the weighted mean of the normalized streams. The resulting block input is width 2560.

Per-block HyperConnection modules also generate four injection weights using the frozen formula:

```text
2 * sigmoid(BlockInject(Hn) / hc_count)
```

The block output is broadcast into the four residual streams according to those weights and added to the previous hyper state.

After decoder layer 47, the global HyperConnection module collapses the four-stream width back to 2560. The final mixer has no block injection weight.

## 6. PLE

PLE is a stateful architecture-level operation, not a stateless embedding lookup.

The frozen reference uses one-based configuration indices. For example, `ple_layer_ids = [2]` means physical zero-based `layer_idx = 1`. This conversion must be explicit.

For `ngram_size = 3`, PLE retains two prior token IDs. Missing lexical history is padded with EOS rather than zero, and EOS forms a lexical-history boundary. Sequence snapshot, restore, fork, prefix reuse, and replay must preserve this lexical history.

PLE constructs hashes for n=2 and n=3 with eight heads per n-gram class, yielding sixteen hash heads. The sixteen embedding fragments concatenate to width 2560. NInfer must preserve the frozen shifting, multiplier, XOR, prime-vocabulary, local-ID, and head-offset semantics exactly.

PLE projects its 2560-wide embedding into key/value terms interacting with the current four-stream state. The gate path includes grouped RMSNorm, per-stream dot product, division by `sqrt(2560)`, signed square-root transform, and sigmoid gating. The resulting four-stream value is flattened to width 10240.

PLE then applies a depthwise Conv1D over 10240 channels with kernel 4, dilation 3, no bias, SiLU activation, and persistent convolution history of 9 positions. PLE output is added to `hyper_input` before the attention/GDN HyperConnection read. This ordering is mandatory.

## 7. GDN

GDN is a recurrent stateful block. Every GDN layer requires at least a causal-convolution state and a recurrent matrix state.

Decode with T=1 uses the recurrent gated-delta path and updates persistent state in place. Prefill/chunk execution uses the chunk gated-delta path. Both paths must produce compatible continuation state.

The frozen discretisation includes:

```text
a = in_proj_a(x)
b = in_proj_b(x)
beta = sigmoid(b)
g = -exp(A_log) * softplus(a + dt_bias)
```

Required FP32 portions must remain FP32. Query/key heads are repeated where required to match value-head count.

## 8. QSA indexer

QSA uses a separate indexing path before main sparse attention.

Indexer geometry:

```text
query heads = 4
KV heads    = 1
head dim    = 128
```

Indexer query flow includes RMSNorm and RoPE at the active position. Raw token indexer keys are persisted separately from ordinary attention KV, so QSA has at least two distinct cache concepts: ordinary attention KV and raw indexer-key history.

## 9. QSA block compression

Visible historical indexer keys are grouped into complete four-token blocks. For each complete block, the pooled key is the FP32 mean of the four raw token keys. The result is cast as required, RMS-normalized, then RoPE-positioned using the FIRST token position in the group.

Incomplete visible tail tokens are not compressed and remain explicitly visible.

## 10. QSA scoring and selection

For each compressed block, each query head produces a dot product against the block key. The frozen block score is:

```text
sum(ReLU(per_head_score)) / sqrt(128)
```

The reference scoring path requires FP32 semantics.

With token budget 2048 and compression ratio 4, block TopK is 512. NInfer selects up to 512 complete blocks, expands each selected block back to its four token IDs, then appends all incomplete visible tail IDs.

Production NInfer should perform scoring and TopK entirely on GPU and should not construct the dense boolean mask used by the generic Transformers reference. Sparse KV gather/direct addressing is preferred if numerically equivalent.

## 11. QSA main attention

After index selection, ordinary Q/K/V projections execute, Q includes its learned gate, Q/K receive RMSNorm and RoPE, attention executes only over selected visible tokens, attention softmax follows FP32 semantics, and output projection follows normally.

## 12. Sparse routed MoE

The router consumes the 2560-wide mixed block input. Reference router logits have shape `[tokens, 512]`.

Routing softmax is explicitly FP32. TopK selects 10 experts. Selected probabilities are renormalized only after TopK, and route weights are cast back only after this normalization.

For selected expert `e`:

```text
[gate, up] = gate_up_e(x)
activation = SiLU(gate) * up
y_e = down_e(activation)
routed_output += route_weight_e * y_e
```

Only experts selected by at least one active token need execute. This enables exact selected-expert streaming on constrained VRAM.

## 13. Shared expert

The shared expert executes independently of routed expert probability:

```text
shared = shared_expert(x)
shared_gate = sigmoid(shared_expert_gate(x))
shared_output = shared_gate * shared
output = routed_output + shared_output
```

No routed-expert probability multiplies the shared branch. Because it executes on every MoE invocation, the shared expert should receive stronger GPU-residency priority than arbitrary routed experts.

## 14. Exact decoder-layer order

Each decoder layer follows:

```text
1. optional PLE update
2. attention HyperConnection read
3. execute GDN or QSA
4. inject block result into four streams
5. MLP HyperConnection read
6. execute sparse routed MoE + shared expert
7. inject MoE result into four streams
```

After all 48 layers, the global HyperConnection mixer produces the final 2560-wide hidden state.

## 15. Persistent sequence-state planes

Flash-Next requires more than ordinary Transformer KV.

GDN layers require causal-convolution state and recurrent matrix state.

The PLE layer requires dilated-convolution state and prior-token lexical state.

QSA layers require ordinary attention KV and raw indexer-key cache.

Global generation state must retain the full position history required by indexer/RoPE semantics.

All required state must participate in reset, snapshot, restore, fork, prefix, and replay semantics.

## 16. FlashNextStateImage

The implementation should have one explicit logical representation of all state required to reproduce the next token exactly. Conceptually call this `FlashNextStateImage`; the actual C++ type may follow existing NInfer naming conventions.

The architectural requirement is completeness. Capturing ordinary KV while omitting GDN, PLE, or QSA indexer state is invalid.

## 17. Artifact/config boundary

Raw artifact metadata should be normalized into a validated Flash-Next config before graph construction. Downstream code should not repeatedly reinterpret raw metadata.

Validation should cover architecture identity, decoder layer count, hidden size, HyperConnection geometry, layer-type schedule, PLE layer IDs and geometry, QSA geometry, MoE expert count, routed TopK, shared-expert presence, and context/position parameters.

Unsupported values should fail with precise diagnostics.

## 18. Tensor binding

Tensor-name decoding belongs in the artifact/binder layer. Graph construction should receive resolved tensor handles.

The binder must inventory and validate embeddings, HyperConnection tensors, PLE tensors, GDN tensors, QSA/indexer tensors, Q/K/V/output tensors, router, routed experts, shared expert, shared-expert gate, normalization, final mixer, and output head.

Do not infer a missing tensor by analogy with another Qwen family.

## 19. Runtime placement separation

Mathematical graph code must not encode RTX 5080 assumptions. The same logical expert operation must work whether its backing tensor is device resident, host resident, mapped, staged through a device cache, or eventually resident on another GPU.

## 20. Stable allocation

After model/session creation, decode should reuse stable allocations. Avoid per-token `cudaMalloc`, `cudaFree`, host malloc/free, and large tensor repacking.

Reusable runtime buffers should cover routing output, route IDs, route weights, expert dispatch, expert staging/cache, expert accumulation, QSA scoring, QSA TopK, sparse-KV gather, recurrent-state scratch, and attention scratch.

## 21. CUDA Graph boundary

CUDA Graphs are an optimization, not a correctness dependency. Graph capture should only be attempted after stable addresses, stable decode topology, correct state ownership, and understood expert-cache behavior exist. A correct non-graph path must always exist during implementation.

## 22. Blackwell execution boundary

RTX 5080 is SM120. Existing NInfer Blackwell low-precision kernels should be reused where their mathematics and layouts match exactly.

New kernels require an independently validated reference path, an isolated benchmark, exact formula validation, and comparison against the previous correct path.

## 23. Correctness validation layers

Validation should proceed through structural checks, formula/operator checks, persistent-state checks, then end-to-end deterministic generation. Plausible generated English is not sufficient proof of correctness.

## 24. Architectural completion

Architecture support is established when NInfer can positively identify Qwen4Exp, validate config, bind the canonical tensor set, initialize four-stream state correctly, execute exact HyperConnection, PLE, GDN, QSA, routed MoE, and shared-expert semantics, maintain all persistent state planes, perform deterministic end-to-end inference, and preserve existing model behavior.

Only then should aggressive RTX 5080 optimization become the main task.
