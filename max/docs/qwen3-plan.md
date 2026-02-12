# Qwen3-Coder-Next Bring-Up Plan

## Goal

Implement [Qwen/Qwen3-Coder-Next](https://huggingface.co/Qwen/Qwen3-Coder-Next)
(80B-parameter hybrid attention + MoE model) in the MAX inference platform.
The HF architecture class is `Qwen3NextForCausalLM`.

The model must:

1. Load bfloat16 weights from HuggingFace safetensors
2. Build and compile a MAX computation graph
3. Generate text via the `pipelines generate` CLI
4. Pass logits verification against HF reference
5. Achieve comparable GSM8K accuracy

## Model Architecture Overview

Qwen3-Coder-Next is a **hybrid** model that mixes two kinds of decoder layers:

| Layer type | Count | Mechanism | State |
|---|---|---|---|
| **Full attention** | 12 of 48 | Standard multi-head attention with KV cache | Paged KV cache |
| **Linear attention** | 36 of 48 | GatedDeltaNet (conv + recurrent) | `conv_state` + `recurrent_state` per layer |

Every 4th layer (0, 4, 8, ..., 44) is full-attention; the rest are linear.
The config field `layer_types` lists them explicitly.

### Key HF Config Values

| Parameter | Value | Notes |
|---|---|---|
| `hidden_size` | 8192 | |
| `num_hidden_layers` | 48 | Total (12 full + 36 linear) |
| `num_attention_heads` | 64 | For full-attention layers |
| `num_key_value_heads` | 8 | GQA for full-attention |
| `head_dim` | 256 | Large head dim (not the typical 128) |
| `partial_rotary_factor` | 0.25 | Only 64 of 256 head dims get RoPE |
| `linear_num_value_heads` | 32 | For GatedDeltaNet layers |
| `linear_key_head_dim` | 128 | |
| `linear_value_head_dim` | 128 | |
| `linear_conv_kernel_dim` | 4 | Short conv for GatedDeltaNet |
| `num_experts` | 128 | Sparse MoE |
| `num_experts_per_tok` | 8 | Top-k routing |
| `shared_expert_intermediate_size` | 8192 | Shared expert alongside routed experts |
| `vocab_size` | 152064 | |
| Model size | ~80B params, ~148 GiB bfloat16 | Requires 2+ B200 GPUs |

### Differences from Standard Qwen3 (the parent architecture)

1. **Hybrid layers**: Qwen3 is all full-attention; Qwen3-Next alternates
   full-attention and linear-attention (GatedDeltaNet) blocks.
2. **Gated Attention**: Full-attention layers have an extra sigmoid gate applied
   to the attention output before `o_proj`. HF stores a combined `q_proj` of
   shape `[num_heads * head_dim * 2, hidden]` (query + gate interleaved). Our
   weight adapter splits this into `q_proj` and `attn_gate`.
3. **GatedDeltaNet**: Linear-attention layers use a gated delta-net with:
   - Short 1-D convolution (kernel size 4) on the combined Q/K/V projection
   - Data-dependent gating via `softplus`
   - Recurrent state update: `S_t = g * S + beta * (k^T . v)`
   - `RMSNormGated` output normalization (RMSNorm + SiLU gate)
4. **Shared Experts in MoE**: Both full-attention and linear-attention layers
   can use MoE with a shared expert (gated) alongside routed experts.
5. **Partial Rotary Embeddings**: `partial_rotary_factor = 0.25` means only the
   first 64 of 256 head dimensions receive RoPE; the rest pass through unchanged.
6. **State Management**: Linear-attention layers require `conv_state` and
   `recurrent_state` tensors to be passed between inference steps (similar to
   Mamba-style models). These are NOT stored in the KV cache.

## MAX Implementation — File Inventory

All files under `max/python/max/pipelines/architectures/qwen3_next/`:

| File | Purpose |
|---|---|
| `arch.py` | Registers `Qwen3NextForCausalLM_Legacy` in the pipeline registry |
| `model_config.py` | `Qwen3NextConfig` dataclass extending `Qwen3Config` with hybrid/linear params |
| `model.py` | `Qwen3NextModel` pipeline model — manages state I/O, overrides `execute()` and `_build_graph()` |
| `qwen3_next.py` | `Qwen3Next` (the graph-level nn Module) — hybrid decoder with full + linear layers |
| `weight_adapters.py` | Weight loading: q_proj split, shared expert rename, layer re-indexing |
| `layers/gated_attention.py` | `Qwen3NextGatedAttention` — wraps `Qwen3Attention` with sigmoid gate |
| `layers/linear_attention.py` | `GatedDeltaNet` and `RMSNormGated` — full linear attention implementation |
| `layers/linear_block.py` | `Qwen3NextLinearBlock` — combines `GatedDeltaNet` + MLP/MoE |

### Inheritance Hierarchy

```
LlamaModelBase (model.py)
  └── Qwen3Model (qwen3/model.py)
        └── Qwen3NextModel (qwen3_next/model.py)

Qwen3 (qwen3/qwen3.py)  — nn Module (graph builder)
  └── Qwen3Next (qwen3_next/qwen3_next.py)

Qwen3Config (qwen3/model_config.py)
  └── Qwen3NextConfig (qwen3_next/model_config.py)

Qwen3Attention (qwen3/layers/attention.py)
  └── Qwen3NextGatedAttention (qwen3_next/layers/gated_attention.py)
```

### How State I/O Works (conv_state / recurrent_state)

Unlike KV cache (managed by the engine), the GatedDeltaNet conv/recurrent states
are managed explicitly by `Qwen3NextModel`:

1. **Graph inputs**: `Qwen3Next.input_types()` appends `TensorType` entries for
   `num_linear_layers` conv states + `num_linear_layers` recurrent states after
   the standard KV cache inputs. All use concrete `batch_dim=1`.
2. **Graph outputs**: `Qwen3Next.__call__` returns
   `(logits_tuple, conv_states, recurrent_states)`.
3. **execute()**: `Qwen3NextModel.execute()` passes stored `Buffer` objects as
   extra arguments, then captures new states from the output.
4. **Reset**: `prepare_initial_token_inputs()` calls `_create_zero_states()` to
   zero out states at the start of each new generation.

State shapes per linear layer:
- `conv_state`: `[1, conv_dim, conv_kernel_size - 1]` = `[1, 8192, 3]`
  where `conv_dim = key_dim*2 + value_dim = 128*16*2 + 128*32 = 8192`
- `recurrent_state`: `[1, num_v_heads, head_k_dim, head_v_dim]` = `[1, 32, 128, 128]`

## Completed Work (Steps 1–3)

### Step 1: Branch Management (DONE)

- FP8 work preserved on branch `Qwen3-Coder-Next-FP8`
- `qwen3-coder-next` branch reset to `d890181616` (pre-FP8 baseline)

### Step 2: Architecture Bug Fixes (DONE)

These bugs were discovered during FP8 work but exist in the bfloat16 code too.
All fixed in commits `f4ada34b7e` through `092f920be5`.

| # | Bug | Fix | File |
|---|---|---|---|
| 1 | Conv1D double-permutation | Remove extra permutations; Conv1D with `permute=True` handles it | `linear_attention.py` |
| 2 | `ops.neg` doesn't exist | Use `-expr` (calls `ops.negate`) | `linear_attention.py` |
| 3 | `ops.sum` keeps reduced dim | Add `ops.squeeze` after each `ops.sum(..., axis=...)` | `linear_attention.py` |
| 4 | `repeat_interleave` unsupported on GPU | Use unsqueeze + concat + reshape pattern | `linear_attention.py` |
| 5 | Recurrent step only handles 1 token | Parallel zero-state approximation for all tokens | `linear_attention.py` |
| 6 | batch_dim mismatch for states | Use concrete `batch_dim=1` (ragged input is unsqueezed to `[1, seq, h]`) | `qwen3_next.py` |
| 7 | Weight path doubling | Remove manual `name=` prefixes; Module auto-prefixes by attribute name | `linear_attention.py` |
| 8 | 2D ragged input not handled | Detect 2D input, project in 2D, lift to 3D for conv/recurrent, flatten back | `linear_attention.py` |
| 9 | Device mismatch in RMSNormGated | Move `eps` constant and `weight` to input tensor's device | `linear_attention.py` |
| 10 | `ops.mean` keepdim + spurious unsqueeze | Remove `ops.unsqueeze` after `ops.mean` (which has implicit keepdim) | `linear_attention.py` |

### Step 3: Gated Attention & Shared Experts (DONE)

- **Gated Attention**: Created `Qwen3NextGatedAttention` wrapping `Qwen3Attention`
  with a `sigmoid(gate(x))` applied before `o_proj`
- **Shared Experts**: Added `has_shared_experts` / `shared_experts_dim` to
  `Qwen3VLMoE`; added `shared_expert_gate` weight
- **Weight Adapter**: `q_proj` split logic (query + gate halves), `.shared_expert.`
  → `.shared_experts.` rename

## Step 4: Validate bfloat16 Model (IN PROGRESS)

### Hardware Requirements

The model is ~148 GiB in bfloat16, so it needs at least 2x B200 GPUs
(183 GiB each). Current test system has 4x B200 + 1.5 TiB system RAM.

### Run Command

```bash
./bazelw run //max/python/max/entrypoints:pipelines -- generate \
  --model-path Qwen/Qwen3-Coder-Next \
  --trust-remote-code --max-length 32 --max-batch-size 1 \
  --devices "gpu:0,1" --prompt "Hello"
```

For debug compiler output:
```bash
MODULAR_MAX_DEBUG=True ./bazelw run ...
```

### Compilation Timeline

Graph building + compilation for this 80B model takes **20–25 minutes** and
uses **~200 GiB system RAM** (CPU-side graph optimization) plus ~4 GiB GPU RAM
during the build phase. This is expected given the model's size and complexity
(48 layers, 128 MoE experts, 36 linear attention layers with state).

### Current Blocker: Partial RoPE Incompatibility

**Status**: The model graph builds successfully but **fails at `session.load()`**
(the Mojo compiler step) due to a constraint in the fused QK RoPE kernel.

**Error** (from `max/kernels/src/nn/fused_qk_rope.mojo:336`):
```
constraint failed: Partial RoPE operation only supported for interleaved pattern
```

**Root Cause Analysis**:

Qwen3-Coder-Next has `partial_rotary_factor = 0.25`, meaning only the first 64
of 256 head dimensions receive Rotary Position Embeddings. When `freqs_cis` has
dimension 64 but the KV cache head size is 256, the fused kernel triggers its
"partial RoPE" path. This path has two constraints that conflict with Qwen3-Next:

1. **Wrong end of head dim**: The kernel applies RoPE to the **last** `rope_dim`
   elements of each head (designed for DeepSeek MLA where no-PE dims come first).
   Qwen3-Next needs RoPE on the **first** 64 elements.

2. **Interleaved-only**: The kernel requires `interleaved=True` for partial RoPE,
   but Qwen3-Next uses non-interleaved (half) rotation pattern (HF's
   `rotate_half`), and the safetensors weights encode this as
   `interleaved_rope_weights=False`.

**How DeepSeek handles this** (for reference):
DeepSeek MLA in `multi_latent_attention.py` splits Q into `(xq_nope, xq_rope)`,
passes only `xq_rope` to `fused_qk_ragged_rope` with `interleaved=True`.
Its K cache stores `[nope_dims, rope_dims]` where rope dims come **last**,
matching the kernel's expectation.

**Current Workaround** (commit `092f920be5`):
The partial RoPE override is disabled — full-dim RoPE is used for all 256 head
dimensions. This allows the model to compile but will produce **incorrect
attention patterns** (3/4 of head dims get spurious rotation). A warning is
logged at startup.

**Proper Fix (TODO)**:
Rearrange Q/K projection weight columns in the weight adapter so that:
1. The 192 non-rotary dims come first (positions 0–191)
2. The 64 rotary dims come last (positions 192–255)
3. Within the rotary dims, convert from half layout `[real₀..real₃₁, imag₀..imag₃₁]`
   to interleaved layout `[real₀, imag₀, real₁, imag₁, ..., real₃₁, imag₃₁]`

Then use `interleaved=True` with the fused kernel. The kernel will leave the first
192 dims unchanged and apply interleaved RoPE to the last 64 dims. The `o_proj`
weight rows do NOT need rearranging because V is in the original layout and the
attention output has V's layout.

Weight adapter changes needed:
- Permute columns of `q_proj`, `k_proj` (and `attn_gate` for gated attention layers)
  to put rotary dims last in interleaved order
- No change needed to `v_proj`, `o_proj`, or any MLP/MoE weights
- The `freqs_cis` should remain at `head_dim=64` with `interleaved=True`

## Step 5: Post-Compilation Validation (PENDING)

Once the model compiles and generates output:

1. **Logits verification**: Register in `max/tests/tests/graph/ops/` and compare
   against HF reference model outputs
2. **GSM8K accuracy**: Run `max/tests/tests/accuracy/` against reference scores
3. **Commit**: Final atomic commits following `[Kernels]` style

## MAX Graph API Pitfalls Discovered

These are important lessons for anyone working on MAX graph construction:

| Pitfall | Details |
|---|---|
| `ops.sum` / `ops.mean` keepdim | Both implicitly keep the reduced dimension (like PyTorch `keepdim=True`). You must `ops.squeeze` if you need it removed. |
| `ops.neg` doesn't exist | Use unary `-` operator (calls `ops.negate` internally). |
| `ops.repeat_interleave` unsupported on GPU | Use the `ops.unsqueeze` + `ops.concat` + `ops.reshape` pattern. |
| Device placement for constants | `ops.constant(...)` creates on CPU by default. You MUST `.to(device)` before using with GPU tensors or you get `ValueError: Input values must be on the same device`. |
| Symbolic vs concrete dims | Ragged inputs are 2D `[total_seq, hidden]`. After unsqueezing to `[1, total_seq, hidden]`, the batch dim is concrete `1`. State tensors passed as graph inputs must also use concrete `1`, not symbolic `"batch_size"`. |
| Module weight auto-prefixing | `Module.__init__` auto-prefixes child modules based on attribute names. Do NOT pass manual `name=f"{prefix}.xxx"` to constructors or you get double-prefixed weight paths. |
| Conv1D `permute=True` | When `permute=True`, the Conv1D handles channels-first ↔ channels-last internally. Don't add extra `ops.permute` calls around it. |
| `ops.pad` dimension ordering | `ops.pad` padding list applies to dims in **reverse** order (last dim first), following the PyTorch convention. |

## Commit History (on `qwen3-coder-next` branch)

```
092f920be5 [Kernels] Qwen3-Next: fix RMSNormGated shape and work around partial RoPE
50ec91fdf0 [Kernels] Qwen3-Next: fix device mismatch in RMSNormGated
5adaad1212 [Kernels] Qwen3-Next: state I/O, partial rotary, and batch dim fixes
f4ada34b7e [Kernels] Qwen3-Next architecture fixes: gated attention, shared experts, GatedDeltaNet bugs
d890181616 Fix Qwen3-Next pipeline load: Weight import and model MRO
7f39280ceb Add Qwen3-Next to logits verification and update contributing-models doc
a39044fd84 [Kernels] Qwen3-Next linear parity tests (Phase 6)
d33d8d15d4 [Kernels] Qwen3-Next graph I/O for conv/recurrent state (Phase 5)
b5b1881dbb [Kernels] Qwen3-Next hybrid decoder: 48 layers with full + linear blocks (Phase 4)
0899607e78 [Kernels] Qwen3-Next weight adapter: load all layers and linear_attn (Phase 3)
2a4ca2ed68 [Kernels] Qwen3-Next config: total_num_layers and linear state shapes (Phase 2)
045fc44c1c [Kernels] Qwen3-Next: add Gated DeltaNet linear attention (Phase 1)
66e908a4f4 [Kernels] Register Qwen3-Next architecture in pipeline registry
41985c1aa0 [Kernels] Add Qwen3-Next (Qwen3NextForCausalLM) architecture
413c4ef155 docs: align contributing-models.md with actual architecture patterns
```

## Branches

| Branch | Purpose |
|---|---|
| `qwen3-coder-next` | Main feature branch (bfloat16). HEAD at `092f920be5`. 2 commits ahead of `origin/qwen3-coder-next`. |
| `Qwen3-Coder-Next-FP8` | WIP FP8 debugging work (to be resumed after bfloat16 is validated) |

## Next Steps (in priority order)

1. **Fix partial RoPE** — Implement the weight column rearrangement in
   `weight_adapters.py` as described above (or explore applying RoPE manually
   in graph ops and writing K to cache separately)
2. **Validate compilation** — Run `generate` with 2+ B200 GPUs; expect 20–25 min compile
3. **Check output quality** — Even with full-dim RoPE workaround, verify the model
   produces coherent tokens (sanity check that everything else is correct)
4. **Logits verification** — Register in test infrastructure, compare against HF
5. **GSM8K accuracy** — Run accuracy evaluation
6. **Push and PR** — Push branch to origin, create PR
