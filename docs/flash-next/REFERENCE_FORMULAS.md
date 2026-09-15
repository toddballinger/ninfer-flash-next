# Qwen4Exp Authoritative Reference Formulas

Reference source:
Hugging Face Transformers Qwen4Exp implementation corresponding to
Qwen3.8-Flash-Next.

This document freezes the semantic equations that NInfer must reproduce
before optimization.

---

# 1. HyperConnection

Input layout:

    hyper_input[..., hc_count * hidden_size]

For Flash-Next:

    hc_count = 4
    hidden_size = 2560
    hc_hidden_size = 10240
    hc_lowrank = 320

## Read / mix

Let:

    H = hyper_input

First grouped RMSNorm:

    Hn = GroupRMSNorm(H)

where each 2560-wide residual stream is independently normalized.

Then:

    D = LinearDown(Hn) / hc_count

    D = SiLU(D)

    W = sigmoid(LinearUp(D))

Reshape:

    W -> [..., 4, 2560]
    Hn -> [..., 4, 2560]

The block input is:

    mixed_input =
        mean_over_streams(W * Hn)

Shape:

    [..., 2560]

This is NOT an ordinary residual-stream sum.

## Injection weights

For per-block HyperConnection modules:

    injection_weights =
        2 * sigmoid(BlockInject(Hn) / hc_count)

Shape:

    [..., 4]

The block executes on mixed_input and produces:

    block_output[..., 2560]

Injection:

    injection =
        block_output[..., None, :] *
        injection_weights[..., :, None]

Then flatten the four streams and update:

    next_hyper_input =
        previous_hyper_input +
        flatten(injection)

For the final model-level HyperConnection mixer:

    block_inject_weight is absent

and the module returns only mixed_input, collapsing four streams to the
final 2560-wide hidden state.

---

# 2. Initial HyperConnection state

After normal token embedding:

    embedding shape = [..., 2560]

The reference model initializes the four-stream representation by
repeating embedding features four times:

    hyper_input =
        repeat(embedding, hc_count)

Shape:

    [..., 10240]

No learned initial stream transform occurs before layer 0.

---

# 3. PLE layer indexing

Config stores PLE layers using one-based layer numbers.

Reference test:

    if layer_idx + 1 in ple_layer_ids

Therefore:

    ple_layer_ids = [2]

means:

    physical zero-based layer_idx = 1

NInfer must preserve this explicitly.

Do NOT interpret ple_layer_ids directly as zero-based indices.

---

# 4. PLE n-gram state

For ngram_size = 3:

    context_len = 2

Persistent token-history state uses:

    conv_state state_idx = 2

If no previous state exists:

    previous_context = [EOS, EOS]

The current input IDs are cached for the next call.

When an initial chunk is shorter than context_len, EOS is used for left
padding rather than zero.

EOS resets lexical history.

The reference shift operation never crosses an EOS boundary.

---

# 5. PLE hash construction

For ngram sizes:

    n = 2
    n = 3

and heads_per_ngram = 8:

    total ngram heads = 16

For each n-gram:

    mixed_ids =
        shifted_token_0 * layer_multiplier_0

then for each additional position:

    mixed_ids =
        mixed_ids XOR
        (shifted_token_position * layer_multiplier_position)

For each hash head:

    local_id =
        mixed_ids mod head_vocab_size

    global_id =
        local_id + head_offset

The 16 resulting embedding rows are concatenated.

For ple_embed_dim = 2560:

    dimension/head = 2560 / 16 = 160

The concatenated PLE embedding width is therefore 2560.

The head vocabulary sizes are distinct primes derived from
ngram_vocab_size_base and ple_layer_index.

---

# 6. PLE injection formula

Inputs:

    hyper_input [..., 4*2560]
    PLE embedding [..., 2560]

Compute:

    K = key_proj(PLE)
      shape [..., 4*2560]

    V = value_proj(PLE)
      shape [..., 2560]

Grouped-normalize:

    Kn = GroupRMSNorm(K)
    Qn = GroupRMSNorm(hyper_input)

Reshape both:

    [..., 4, 2560]

Per-stream scalar gate:

    g =
        sum(Kn * Qn, hidden_dim) /
        sqrt(2560)

Then signed square-root transform:

    g =
        sqrt(max(abs(g), 1e-6)) *
        sign(g)

Then:

    gated_value =
        sigmoid(g)[..., :, None] *
        V[..., None, :]

Shape:

    [..., 4, 2560]

Flatten to 10240.

Normalize:

    gated_value_normed =
        GroupRMSNorm(gated_value_flat)

PLE output:

    PLE_output =
        gated_value_flat +
        DilatedDepthwiseConv(
            gated_value_normed
        )

The decoder adds this directly to the four-stream state BEFORE the
attention/GDN HyperConnection read:

    hyper_input += PLE_output

---

# 7. PLE convolution

PLE conv parameters:

    kernel_size = 4
    dilation = ngram_size = 3

Persistent history required:

    short_conv_state_len =
        (kernel_size - 1) * dilation
        = 9

Persistent PLE convolution state uses:

    state_idx = 1

The conv is:

    depthwise Conv1D
    channels = 10240
    kernel = 4
    dilation = 3
    no bias

Activation:

    SiLU(conv_output)

State must participate in snapshot/restore/prefix semantics.

---

# 8. GDN persistent state

GDN uses:

    state_idx = 0

for causal-convolution state.

For cached T=1 decode:

    causal_conv1d_update

updates this state in place.

GDN also stores a recurrent matrix state separately.

Therefore each GDN layer requires at least:

    conv state
    recurrent state

Both must participate in FlashNextStateImage.

---

# 9. GDN discretization

Given:

    a = in_proj_a(x)
    b = in_proj_b(x)

Then:

    beta = sigmoid(b)

and in FP32:

    g =
        -exp(A_log) *
        softplus(a + dt_bias)

Query/key heads are repeated as needed to match value-head count.

Decode T=1 uses the recurrent gated-delta path.

Prefill/chunk uses the chunk gated-delta path.

The final recurrent state is persisted.

---

# 10. QSA indexer

Indexer geometry:

    query heads = 4
    KV heads = 1
    head dim = 128

Joint projection produces:

    4*128 query features
    1*128 token key features

Query path:

    q = RMSNorm(q)
    q = RoPE(q at current positions)

Raw token indexer keys are persisted in the cache.

---

# 11. QSA block compression

compress_ratio = 4.

For each query, identify all causally visible token positions.

Complete groups of four visible tokens become blocks.

For each block:

    pooled_key =
        FP32 mean(raw token keys across 4 tokens)

Then cast back to source dtype.

Then:

    pooled_key =
        RMSNorm(pooled_key)

RoPE is applied using the position of the FIRST token in each group.

Incomplete visible tail tokens are not compressed and are always
retained.

---

# 12. QSA block scoring

For one query:

    q shape = [4, 128]

Each compressed block key:

    k shape = [128]

Compute one dot product per query head:

    score_head =
        dot(q_head, k)

Then:

    block_score =
        sum_over_heads(ReLU(score_head)) /
        sqrt(128)

All arithmetic used by the reference scoring path is promoted to FP32
for the matrix multiply.

---

# 13. QSA selection budget

Config:

    token_budget = 2048
    compress_ratio = 4

Therefore:

    block_topk =
        2048 / 4
        = 512 blocks

Select:

    TopK(block_score, min(512, available_complete_blocks))

Then expand selected block IDs back into their four constituent token
IDs.

Append all incomplete tail token IDs.

The resulting token set defines the sparse attention visibility.

Production NInfer must perform scoring and TopK entirely on GPU.

It need not materialize the dense boolean mask used by the generic
Transformers implementation.

---

# 14. QSA main attention

After QSA token selection:

- ordinary Q/K/V projections are applied
- Q has an additional learned gate packed into q_proj
- Q and K receive RMSNorm
- Q/K receive RoPE
- sparse attention is evaluated only over selected visible token IDs
- attention probabilities use FP32 softmax semantics
- output is projected normally

The optimized NInfer implementation may gather selected KV directly
rather than constructing a dense attention mask, provided the numerical
result is equivalent.

---

# 15. MoE router

Router:

    logits =
        x @ router_weight^T

Shape:

    [tokens, 512]

Routing softmax is explicitly FP32:

    probs =
        softmax(logits, dtype=float32)

Select:

    top_values, top_ids =
        TopK(probs, 10)

Because:

    norm_topk_prob = true

renormalize only selected routes:

    route_weight_i =
        top_value_i /
        sum(top_values)

Convert route weights back to router-logit dtype after normalization.

This ordering matters for exact reference parity.

---

# 16. Routed expert formula

For each selected routed expert e:

    [gate, up] =
        gate_up_e(x)

    activation =
        SiLU(gate) * up

    y_e =
        down_e(activation)

Then:

    routed_output +=
        route_weight_e * y_e

Only experts hit by at least one token need to execute.

This property enables selected-expert streaming.

---

# 17. Shared expert

Independently:

    shared =
        shared_expert(x)

    shared_gate =
        sigmoid(shared_expert_gate(x))

    shared_output =
        shared_gate * shared

Final MoE output:

    output =
        routed_output +
        shared_output

There is no route probability applied to the shared expert.

---

# 18. Decoder layer ordering

For every layer:

1. If PLE is attached to this layer:

       hyper_input += PLE(
           hyper_input,
           token_ids,
           state
       )

2. Attention/GDN HyperConnection read:

       mixed_input,
       prior_hyper,
       injection_weights =
           attn_hyper_connection(hyper_input)

3. Execute either:

       GDN(mixed_input)

   or:

       QSA(mixed_input)

4. Inject block result:

       hyper_input =
           prior_hyper +
           block_output * injection_weights

5. MLP HyperConnection read.

6. Sparse MoE on mixed input.

7. Inject MoE result back into the four streams.

After all 48 layers:

    final_hidden =
        global_hyper_connection_mixer(hyper_input)

then final model head processing.

---

# 19. Required persistent state planes

The reference source now confirms these state classes:

## Every GDN layer

    GDN conv state
    GDN recurrent state

## PLE physical layer 1

    PLE dilated-conv state
    PLE prior token-ID context state

## Every QSA layer

    ordinary attention KV
    indexer raw-key cache

## Global generation metadata

    full position history required by indexer/RoPE semantics

These must be incorporated into NInfer sequence snapshot/fork/replay
semantics before those features are declared supported.

---

# 20. 5080 execution implications

## Expert streaming

Router completes before expert execution.

Therefore the runtime can:

    route
    -> identify exact ten experts
    -> resolve hot/cold residency
    -> transfer cold misses only
    -> execute

without approximating model semantics.

## QSA

Reference creates a dense mask for convenience.

NInfer should not.

Use:

    compressed key state
    -> GPU scores
    -> GPU TopK block IDs
    -> token-ID expansion
    -> direct selected KV gather
    -> sparse attention

## PLE

Hash row IDs depend only on token history.

Begin PLE host-row gather before the consuming layer executes.

## HyperConnection

The four-stream state width is 10240, but only the mixed 2560-wide
activation enters GDN/QSA/MoE.

This is favorable for compute: heavy sublayers still operate at hidden
size 2560 rather than 10240.

---

# 21. First correctness fixtures

Generate authoritative reference fixtures for:

- HC norm output
- HC down projection
- HC up/sigmoid weights
- HC mixed input
- HC block injection weights

- PLE shifted token history
- PLE hash IDs
- PLE gathered embedding
- PLE gate values
- PLE dilated conv output
- PLE final injection

- QSA pooled block keys
- QSA RoPE block keys
- QSA score vector
- QSA selected block IDs
- QSA final selected token IDs

- router logits
- router FP32 probabilities
- Top-10 IDs
- normalized Top-10 weights
- routed expert output
- shared expert output

- GDN conv transition
- GDN beta
- GDN g
- recurrent state
- final GDN output

These fixtures become the numerical oracles for later CUDA kernels.
