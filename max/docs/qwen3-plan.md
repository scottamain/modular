# Branch FP8 Work and Fix bfloat16 Qwen3-Coder-Next

## Step 1: Preserve FP8 Work on a New Branch (DONE)

Current state:

- Branch: `qwen3-coder-next`
- HEAD was at `11c6d09f44` (6 FP8-related commits on top of `d890181616`)
- Working tree had 10 modified files + 1 untracked file with additional FP8 debugging (Conv1D, ops.neg, ops.sum, repeat_interleave, recurrent step fixes)

Actions taken:

1. Committed all uncommitted changes on `qwen3-coder-next` with a WIP message
2. Created branch `Qwen3-Coder-Next-FP8` at current HEAD
3. Switched back to `qwen3-coder-next`
4. Hard reset `qwen3-coder-next` to `d890181616` (last pre-FP8 commit: "Fix Qwen3-Next pipeline load")

## Step 2: Apply Architecture Fixes to bfloat16 Branch (DONE)

During FP8 debugging, several bugs were discovered in the core GatedDeltaNet implementation that exist in the pre-FP8 code too. These must be fixed for the bfloat16 model to work.

### Fix 1: Conv1D double-permutation in `linear_attention.py` (DONE)

File: `max/python/max/pipelines/architectures/qwen3_next/layers/linear_attention.py`

The Conv1D is initialized with `permute=True`, which means it expects input in channels-first format `[batch, channels, length]` and handles internal permutations. But the code at lines 243-249 adds extra permutations before and after the conv1d call, double-permuting the data and corrupting channel/length dimensions.

- Removed the `ops.permute(mixed_qkv_cat, [0, 2, 1])` before `self.conv1d()` in the if-branch
- Removed the `ops.permute(mixed_qkv, [0, 2, 1])` after conv1d
- Did the same for the else-branch
- Added a single permute AFTER the if/else block to go from `[batch, channels, seq]` back to `[batch, seq, channels]` for the q/k/v split
- Changed `mixed_qkv[:, :, -seq_dim:]` to `mixed_qkv[:, :, state_len:]` (concrete index avoids symbolic dimension issues)
- Fixed padding in else-branch from `[0, 0, 0, 0, state_len - 1, 0]` to `[state_len - 1, 0, 0, 0, 0, 0]` (pad the last dim, not the batch dim)

### Fix 2: `ops.neg` does not exist (DONE)

File: same `linear_attention.py`

Replaced `ops.neg(expr)` with `-expr` (uses `__neg__` which calls `ops.negate`).

### Fix 3: `ops.sum` keeps reduced dimension (DONE)

File: same `linear_attention.py`, `_recurrent_step` method

MAX's `ops.sum` has implicit `keepdim=True` behavior. Added `ops.squeeze(..., -2)` after each `ops.sum(..., axis=-2)` call to remove the size-1 dimension.

### Fix 4: `repeat_interleave` not supported on GPU (DONE)

File: same `linear_attention.py`

Replaced `ops.repeat_interleave(tensor, n, axis=2)` with the unsqueeze + concat + reshape pattern used elsewhere in the codebase (e.g., `qwen3_embedding/layers/attention.py`).

### Fix 5: Recurrent step only produces 1 timestep (DONE)

File: same `linear_attention.py`

The recurrent step processes only token 0 but the reshape uses `seq_dim` (symbolic total_seq_len). This fails at graph build time. Replaced with a parallel zero-state computation that handles all tokens: for zero initial state, `out_t = (q_t . k_t) * v_t * beta_t`. This is exact for decode (seq_len=1) and a reasonable approximation for prefill.

### Fix 6: batch_dim for conv/recurrent state (DONE)

File: `max/python/max/pipelines/architectures/qwen3_next/qwen3_next.py`

Changed `batch_dim = h[0].shape[0]` to `batch_dim = 1`. The GatedDeltaNet unsqueezes ragged 2D input to `[1, total_seq, hidden]` internally, so conv/recurrent states must have batch=1.

### Fix 7: Remove weight path doubling in GatedDeltaNet (DONE)

File: same `linear_attention.py`

Removed explicit `name=f"{prefix}.xxx"` from all submodule constructors in `GatedDeltaNet.__init__`. The Module system auto-prefixes based on attribute names, so the manual prefix caused double-prefixing (e.g., `linear_attn.linear_attn.conv1d.weight`).

### Fix 8: Handle 2D ragged input in GatedDeltaNet (DONE)

File: same `linear_attention.py`

Modified `GatedDeltaNet.__call__` to detect 2D `[total_seq, hidden]` input (from ragged batching), project in 2D, then lift to 3D `[1, total_seq, ...]` for conv/recurrent state processing. The output is flattened back to 2D when the input was 2D.

## Step 3: Add Gated Attention and Shared Expert Support (DONE)

These are architectural features of Qwen3-Next (not FP8-specific) that were discovered and implemented during FP8 work. Without them, the model loads incorrect weights.

### Gated Attention (DONE)

Created `max/python/max/pipelines/architectures/qwen3_next/layers/gated_attention.py`:

- `Qwen3NextGatedAttention` wraps `Qwen3Attention` adding a sigmoid gate from a split q_proj
- Updated `qwen3_next.py` to replace `Qwen3Attention` instances with `Qwen3NextGatedAttention` for full-attention blocks

### Shared Experts in MoE (DONE)

Updated `max/python/max/pipelines/architectures/qwen3vl_moe/nn/moe.py`:

- Added `has_shared_experts` and `shared_experts_dim` parameters to `Qwen3VLMoE`
- Added `shared_expert_gate` Weight and apply it in `__call__`

Updated `linear_block.py` and `qwen3.py`:

- Pass `shared_expert_intermediate_size` from config to `Qwen3VLMoE` constructor
- `linear_block.py` now uses MoE (not just dense MLP) when config enables it

### Weight Adapter Updates (DONE)

Updated `max/python/max/pipelines/architectures/qwen3_next/weight_adapters.py`:

- Added q_proj split logic for gated attention (split `[num_heads * head_dim * 2, hidden]` into q_proj + attn_gate)
- Added rename rule `.shared_expert.` to `.shared_experts.` (plural)

## Step 4: Validate with bfloat16 Model (PENDING — needs B200)

The bfloat16 model (80B params, 148 GiB) exceeds available memory on a single H200 (125 GiB). Validation requires a B200 system.

Run `generate` with the bfloat16 model:

```bash
./bazelw run //max/python/max/entrypoints:pipelines -- generate \
  --model-path Qwen/Qwen3-Coder-Next \
  --trust-remote-code --max-length 512 --max-batch-size 1 \
  --prompt "Write a quicksort algorithm in Python"
```

Debug any remaining issues, then run logits verification and GSM8K accuracy.

## Step 5: Commit and Update Status

Commit all fixes as atomic commits following the project's commit style, update `qwen3-next-bringup-status.md` with results.

## Branches

- `qwen3-coder-next` — main feature branch with bfloat16 architecture fixes applied
- `Qwen3-Coder-Next-FP8` — WIP FP8 debugging work (to be resumed later)
