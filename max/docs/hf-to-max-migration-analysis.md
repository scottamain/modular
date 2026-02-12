# HuggingFace to MAX Python API: Migration Analysis and Technical Guide

This document provides a detailed technical comparison of four model implementations that were ported from HuggingFace/PyTorch to the MAX Python API. It covers architecture-level differences, layer-by-layer comparisons, and a comprehensive guide for future developers migrating new models.

## Table of Contents

1. [Model 1: GPT-OSS (gpt_oss_legacy)](#1-gpt-oss-gpt_oss_legacy)
2. [Model 2: InternVL (internvl)](#2-internvl-internvl)
3. [Model 3: Gemma3 Multimodal (gemma3multimodal)](#3-gemma3-multimodal-gemma3multimodal)
4. [Model 4: Mistral3 (mistral3)](#4-mistral3-mistral3)
5. [Cross-Model Pattern Analysis](#5-cross-model-pattern-analysis)
6. [Technical Migration Guide](#6-technical-migration-guide)
7. [Known API Gaps and Workarounds](#7-known-api-gaps-and-workarounds)

---

## 1. GPT-OSS (gpt_oss_legacy)

**HuggingFace Reference:** [`openai/gpt-oss-20b`](https://huggingface.co/openai/gpt-oss-20b) + [OpenAI reference implementation](https://github.com/openai/gpt-oss/blob/main/gpt_oss/torch/model.py) + HuggingFace Transformers `GptOssForCausalLM`

**MAX Implementation:** `max/python/max/pipelines/architectures/gpt_oss_legacy/`

### 1.1 Architecture Overview

GPT-OSS is a decoder-only Transformer with Mixture-of-Experts (MoE) feed-forward layers, attention sinks, mixed sliding/full attention, and YARN RoPE scaling. The MAX port preserves all of these architectural features while replacing PyTorch's eager execution with MAX's graph-compiled approach.

| Component | HuggingFace/OpenAI | MAX |
|---|---|---|
| Execution model | Eager (PyTorch) | Graph-compiled (`max.graph.Graph`) |
| Attention | SDPA / Flash Attention 2 | `flash_attention_ragged` kernel |
| MoE routing | Per-expert loop with `torch.einsum` | Batched `grouped_matmul_ragged` kernel |
| KV cache | `DynamicCache` (dense) | `PagedCacheValues` (paged) |
| Sequence handling | Padded with attention mask | Ragged with `input_row_offsets` |
| Tensor parallelism | Manual `dist.all_reduce` in MoE | `Allreduce` + `ShardingStrategy` |
| Normalization | `torch.nn.Module` RMSNorm | `RMSNorm` from `max.nn.legacy` |

### 1.2 Layer-by-Layer Comparison

#### 1.2.1 Attention Layer

**HuggingFace** (`GptOssAttention` in `modeling_gpt_oss.py`):
- Separate Q, K, V projections via `nn.Linear`
- RoPE via `apply_rotary_pos_emb` (interleaved half-rotation)
- Attention sinks: concatenates learnable sink logits `self.sinks` as an extra column to the attention weight matrix, computes softmax over `[attn_weights, sinks]`, then drops the sink column before multiplying with values
- Sliding window: per-layer `layer_types` config selects `create_sliding_window_causal_mask` or `create_causal_mask`
- Uses HuggingFace `ALL_ATTENTION_FUNCTIONS` dispatch (Flash Attention, SDPA, or eager)

**OpenAI Reference** (`AttentionBlock` in `model.py`):
- Fused QKV projection via single `nn.Linear`
- Separate Q/K/V split from the concatenated output
- Custom `sdpa` function with explicit sink handling
- Sinks appended to Q*K scores before softmax

**MAX** (`GptOssAttention` in `layers/attention.py`):
- Separate Q, K, V, O projections via `Linear` from `max.nn.legacy.linear`
- QKV concatenated at runtime via `ops.concat([wq, wk, wv])` and projected in one `fused_qkv_ragged_matmul` call
- RoPE via `fused_qk_ragged_rope` (fused with ragged sequence handling)
- Attention sinks: passed as `sink_weights` parameter directly to `flash_attention_ragged` kernel, which handles the sink logit injection internally
- Sliding window: `MHAMaskVariant.SLIDING_WINDOW_CAUSAL_MASK` vs `MHAMaskVariant.CAUSAL_MASK` passed to flash attention
- `local_window_size` parameter controls sliding window size

**Migration challenges:**
- The sink attention mechanism had to be integrated into the flash attention kernel rather than implemented as pre/post-processing around standard attention. The HuggingFace eager implementation explicitly concatenates sink logits and drops them after softmax, but the MAX kernel handles this internally for better performance.
- The fused QKV + ragged rope kernels eliminated the need for separate reshape/permute operations that exist in HuggingFace.
- RoPE uses an interleaved format (split into first/second half, not even/odd) matching the OpenAI reference.

#### 1.2.2 Mixture of Experts (MoE)

**HuggingFace** (`GptOssExperts` + `GptOssTopKRouter` in `modeling_gpt_oss.py`):
- Router: `nn.Linear` with bias, top-k selection, softmax on top-k scores
- Expert execution: iterates over hit experts with `torch.where(expert_mask)`, processes each expert's tokens independently via standard matmul
- Weights stored as `nn.Parameter` tensors: `gate_up_proj [num_experts, hidden, 2*intermediate]`, `down_proj [num_experts, intermediate, hidden]`
- Activation: interleaved gate/up split (`[..., ::2]` / `[..., 1::2]`), clamp, `gate * sigmoid(gate * alpha) * (up + 1)`

**OpenAI Reference** (`MLPBlock` in `model.py`):
- Same router structure with bias
- Expert execution: `torch.einsum("beck,bk->bec", weight[indices], x)` -- gathers weights by expert indices, then batched matmul
- Same interleaved SwiGLU activation

**MAX** (`GptOssMoE` + `GptOssMoEGate` in `layers/moe.py`):
- Router: `GptOssMoEGate` extends `MoEGate` with bias via `Linear(has_bias=True)`, top-k via `ops.top_k`, softmax via `ops.softmax`
- Expert execution: `moe_create_indices` kernel creates token-to-expert ordering, then `grouped_matmul_ragged` performs batched expert matmul in a single kernel call
- Weights: `Weight` objects with same shapes, transposed at runtime for `grouped_matmul_ragged`
- Bias: gathered per-token based on expert assignment via `ops.gather`
- Activation: identical logic -- `ops.min` for gate clamping, `clamp` for up, `ops.sigmoid` for gating
- Routing weights applied after expert computation via matrix multiply

**Migration challenges:**
- The per-expert loop in HuggingFace was replaced with the `moe_create_indices` + `grouped_matmul_ragged` pattern, which requires careful token reordering. The `moe_create_indices` kernel returns `token_expert_order`, `expert_start_indices`, `restore_token_order`, `expert_ids`, and `expert_usage_stats` that must be used consistently.
- Bias handling in MoE required gathering biases per-token based on expert assignment (`ops.gather(bias, expert_assignments)`), which doesn't have a direct PyTorch equivalent.
- The interleaved gate/up split (`[..., ::2]` / `[..., 1::2]`) was preserved exactly.
- Tensor parallel sharding of MoE weights required splitting `gate_up_proj` on axis 2 and `down_proj` on axis 1 (columnwise), with biases sharded to match.

#### 1.2.3 Transformer Block

**HuggingFace** (`GptOssDecoderLayer`):
```
residual = x
x = input_layernorm(x)
x = self_attn(x, attention_mask, position_embeddings, past_key_values)
x = residual + x
residual = x
x = post_attention_layernorm(x)
x, router_scores = mlp(x)
x = residual + x
```

**MAX** (`GptOssTransformerBlock` in `layers/transformer_block.py`):
```
x = input_layernorm(x)
x = self_attn(x, kv_collection, input_row_offsets)
x = allreduce(x)  # for tensor parallelism
x = residual + x
x = post_attention_layernorm(x)
x = mlp(x)
x = allreduce(x)  # for tensor parallelism
x = residual + x
```

**Key differences:**
- MAX adds explicit `Allreduce` after attention and MLP for tensor parallel communication
- MAX uses `forward_sharded_layers` for distributed execution across devices
- Router scores are not returned in MAX (inference-only, no auxiliary loss computation)

#### 1.2.4 YARN RoPE

**HuggingFace** (`GptOssRotaryEmbedding`):
- Delegates to `ROPE_INIT_FUNCTIONS["yarn"]` for NTK-by-parts frequency computation
- Returns cos/sin as `[batch, seq_len, head_dim]` tensors

**OpenAI Reference** (`RotaryEmbedding`):
- Direct YARN implementation with concentration scaling
- NTK-by-parts: computes `low`/`high` frequency boundaries, interpolation mask, blends interpolation and extrapolation

**MAX** (`YarnRotaryEmbedding` from `max.nn.legacy.rotary_embedding`):
- Wraps `YarnScalingParams` with `beta_fast`, `beta_slow`, `factor`, `original_max_position_embeddings`
- Pre-computes `freqs_cis` as a weight tensor used by `fused_qk_ragged_rope`
- Applied during the fused kernel call, not as a separate step

#### 1.2.5 Weight Mapping

```python
GPT_OSS_SAFETENSOR_MAP = {
    "model.embed_tokens.":  "language_model.embed_tokens.",
    "model.norm.":          "language_model.norm.",
    "lm_head.":             "language_model.lm_head.",
    "model.layers.":        "language_model.layers.",
    ".mlp.router":          ".mlp.gate.gate_score",
}
```

The MAX model wraps the text model in a `GptOss` class with a `language_model` attribute, requiring the `language_model.` prefix. The MoE router maps from `.mlp.router` (HuggingFace) to `.mlp.gate.gate_score` (MAX's `MoEGate` subclass naming).

### 1.3 Summary of Migration Challenges

1. **Attention sinks** required kernel-level integration rather than pre/post-processing
2. **MoE batched execution** replaced per-expert loops with `moe_create_indices` + `grouped_matmul_ragged`
3. **Ragged sequences** replaced padding + attention mask with `input_row_offsets`
4. **Paged KV cache** replaced `DynamicCache` with page-table-based memory management
5. **Graph compilation** required converting all dynamic control flow to static graph operations
6. **Tensor parallelism** required explicit `Allreduce` and `ShardingStrategy` for each weight

---

## 2. InternVL (internvl)

**HuggingFace Reference:** [`OpenGVLab/InternVL3-8B-Instruct`](https://huggingface.co/OpenGVLab/InternVL3-8B-Instruct) (custom `modeling_internvl_chat.py` + `modeling_intern_vit.py`)

**MAX Implementation:** `max/python/max/pipelines/architectures/internvl/`

### 2.1 Architecture Overview

InternVL is a multimodal model combining InternViT (vision encoder) with Qwen2/Qwen3 (language model) connected by an MLP projector. The MAX port implements all three components natively using MAX graph operations.

| Component | HuggingFace | MAX |
|---|---|---|
| Vision encoder | `InternVisionModel` (custom code in repo) | `InternVLVisionModel` (native MAX) |
| Language model | `Qwen2ForCausalLM` (transformers library) | `InternVLLanguageModel` (native MAX) |
| Projector | `nn.Sequential(LayerNorm, Linear, GELU, Linear)` | `InternVLMLP1` (named submodules) |
| Multimodal merge | Boolean indexing: `embeds[selected] = vit_embeds` | `scatter_nd_skip_oob_indices` kernel |
| Image preprocessing | None (user-implemented) | Custom `InternVLProcessor` + `InternVLTokenizer` |
| Pipeline | Single model forward pass | Two separate compiled models (vision + language) |

### 2.2 Layer-by-Layer Comparison

#### 2.2.1 Vision Embeddings

**HuggingFace** (`InternVisionEmbeddings`):
- Patch embedding: `nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)`
- Class embedding: `nn.Parameter(torch.randn(1, 1, embed_dim))`
- Position embedding: `nn.Parameter(torch.randn(1, num_positions, embed_dim))`
- Position interpolation: `F.interpolate(pos_embed, size=(H, W), mode='bicubic')`
- Forward: Conv2d patches -> flatten -> concat class token -> add position embeddings

**MAX** (`InternVisionEmbeddings`):
- Patch embedding: `Linear(3 * patch_size * patch_size, embed_dim)` -- Conv2d weights reshaped to 2D in weight adapter
- Class embedding: `Weight("class_embedding", shape=[1, 1, embed_dim])`
- Position embedding: `Weight("position_embedding", shape=[1, num_positions, embed_dim])`
- Position interpolation: `ops.resize(pos_embed, ..., method="bicubic")`
- Forward: Reshape input to `[batch, num_patches, 3*patch_size*patch_size]` -> Linear -> concat class token -> add position embeddings

**Migration challenge:** The Conv2d patch embedding was converted to a Linear layer. This required reshaping the Conv2d weight from `[out_channels, in_channels, kernel_h, kernel_w]` to `[out_channels, in_channels * kernel_h * kernel_w]` in the weight adapter (`convert_internvl_vision_model_state_dict`). The input images must also be reshaped from `[batch, channels, height, width]` to `[batch, num_patches, channels * patch_size * patch_size]` before the linear layer.

#### 2.2.2 Vision Attention

**HuggingFace** (`InternAttention`):
- Fused QKV: `nn.Linear(embed_dim, 3 * embed_dim, bias=qkv_bias)`
- QK normalization: `InternRMSNorm` applied to Q and K separately (flatten heads, norm, reshape back)
- Flash attention: `flash_attn_varlen_qkvpacked_func` from `flash_attn` package
- Naive fallback: standard `Q @ K.T / sqrt(d)` -> softmax -> `@ V`
- Output: `nn.Linear(embed_dim, embed_dim)` (called `proj`)

**MAX** (`InternVLMultiheadAttention` in `layers/attention.py`):
- Stacked QKV: `Weight("qkv_proj", shape=[3 * embed_dim, embed_dim])` or separate Q/K/V
- QK normalization: `RMSNorm` on Q and K, with `allgather` for multi-device support
- Flash attention: `flash_attention_gpu` kernel with `MHAMaskVariant.NULL_MASK` (no causal masking for vision)
- Output: `Linear` (called `o_proj`)

**Weight mapping:**
```python
".attn.qkv.":      ".attn.qkv_proj.",
".attn.qkv_bias.": ".attn.qkv_proj_bias.",
".attn.proj.":      ".attn.o_proj.",
```

**Migration challenge:** QK normalization in the multi-device case required `allgather` to reconstruct full Q/K tensors before normalization, since normalization is applied across all heads. The HuggingFace implementation doesn't handle this since it doesn't support tensor parallel inference for the vision encoder.

#### 2.2.3 Vision MLP and Encoder Layer

**HuggingFace** (`InternMLP` + `InternVisionEncoderLayer`):
- MLP: `fc1(x) -> GELU -> fc2(x)`
- Encoder layer: `x + drop_path(attn(norm1(x)) * ls1)` then `x + drop_path(mlp(norm2(x)) * ls2)`
- Layer scaling: `ls1`, `ls2` as learnable parameters (initialized from `config.initializer_factor`)
- Stochastic depth: `DropPath` from `timm.layers`
- Normalization: configurable `layer_norm` or `rms_norm` via `NORM2FN` dict

**MAX** (`InternVLVisionMLP` + `InternVisionEncoderLayer`):
- MLP: Same architecture with `fc1 -> GELU -> fc2`, supports tensor parallelism
- Encoder layer: `x + attn(norm1(x)) * ls1` then `x + mlp(norm2(x)) * ls2`
- Layer scaling: `Weight("ls1")` and `Weight("ls2")`
- No stochastic depth (inference only, drop_path_rate = 0)
- Normalization: `RMSNorm` or `LayerNorm` from `max.nn.legacy`

**Migration note:** `DropPath` from `timm` was removed since it's only active during training. The layer scaling weights are loaded directly.

#### 2.2.4 Pixel Shuffle

**HuggingFace**:
```python
def pixel_shuffle(self, x, scale_factor=0.5):
    n, w, h, c = x.size()
    x = x.view(n, w, int(h * scale_factor), int(c / scale_factor))
    x = x.permute(0, 2, 1, 3).contiguous()
    x = x.view(n, int(h * scale_factor), int(w * scale_factor), int(c / (scale_factor ** 2)))
    x = x.permute(0, 2, 1, 3).contiguous()  # ps_version v2
    return x
```

**MAX**: Implements the same reshape-permute sequence using `ops.reshape` and `ops.transpose`, applied after removing the CLS token from encoder output.

#### 2.2.5 MLP1 Projector (Vision-to-Language)

**HuggingFace**:
```python
self.mlp1 = nn.Sequential(
    nn.LayerNorm(vit_hidden_size * int(1 / downsample_ratio) ** 2),
    nn.Linear(vit_hidden_size * int(1 / downsample_ratio) ** 2, llm_hidden_size),
    nn.GELU(),
    nn.Linear(llm_hidden_size, llm_hidden_size)
)
```

**MAX** (`InternVLMLP1`):
```python
self.layer_norm = LayerNorm(...)   # Maps to mlp1.0
self.fc1 = Linear(...)            # Maps to mlp1.1
# GELU activation (inline)
self.fc2 = Linear(...)            # Maps to mlp1.3  (note: index 3, skipping GELU at index 2)
```

**Weight mapping:**
```python
"mlp1.0.": "mlp1.layer_norm.",
"mlp1.1.": "mlp1.fc1.",
"mlp1.3.": "mlp1.fc2.",
```

#### 2.2.6 Language Model

**HuggingFace**: Delegates entirely to `Qwen2ForCausalLM` from the `transformers` library. No custom code for the language model.

**MAX** (`InternVLLanguageModel`): Re-implements a Qwen2/Qwen3-style decoder from scratch using MAX primitives:
- `DynamicRotaryEmbedding` for RoPE
- `InternVLDecoderLayer` with `TensorParallelAttentionWithRope` for attention
- `MLP` with tensor parallelism for feed-forward
- `RMSNorm` for normalization
- `VocabParallelEmbedding` for token embeddings
- `ColumnParallelLinear` for LM head
- Full KV cache support via `PagedCacheValues`

This is the most significant difference -- the entire Qwen2 decoder had to be re-implemented natively in MAX rather than using the transformers library.

#### 2.2.7 Multimodal Embedding Merge

**HuggingFace**:
```python
input_ids = input_ids.reshape(B * N)
selected = (input_ids == self.img_context_token_id)
input_embeds[selected] = input_embeds[selected] * 0.0 + vit_embeds.reshape(-1, C)
```

**MAX** (`merge_multimodal_embeddings` in `embedding_utils.py`):
```python
scatter_nd_skip_oob_indices(inputs_embeds, multimodal_embeddings, image_token_indices)
```

The MAX version uses a custom kernel that scatters vision embeddings into text embeddings at pre-computed index positions, handling out-of-bounds indices gracefully.

#### 2.2.8 Tokenization and Image Preprocessing

**HuggingFace**: No standard processor. The model repo provides usage examples, but image preprocessing is left to users.

**MAX** (`InternVLTokenizer` + `InternVLProcessor`):
- Custom `InternVLProcessor` that handles chat template formatting and image token insertion
- Dynamic image patching with configurable `max_dynamic_patch` (1-12 patches + thumbnail)
- ImageNet normalization (IMAGENET_MEAN, IMAGENET_STD)
- Patches cropped at aspect-ratio-aware positions
- Images converted to bfloat16 (stored as uint16)

#### 2.2.9 Pipeline Architecture

**HuggingFace**: Single model with `forward()` that calls vision encoder, then language model.

**MAX**: Two separately compiled models:
1. **Vision model**: `pixel_values -> InternVLVisionModel -> image_embeddings`
2. **Language model**: `(input_ids, image_embeddings, image_token_indices) -> InternVLLanguageModel -> logits`

The `InternVLModel.execute()` method orchestrates: run vision model if images present, then run language model with merged embeddings.

### 2.3 Summary of Migration Challenges

1. **Full Qwen2 re-implementation** was required since transformers library cannot be used at runtime
2. **Conv2d-to-Linear conversion** for patch embeddings required weight reshaping in the adapter
3. **Custom image preprocessing** had to be implemented from scratch (no AutoProcessor available)
4. **Two-model compilation** required splitting the forward pass and managing separate weight loading
5. **QK normalization multi-device** required allgather before normalization
6. **Position embedding interpolation** used `ops.resize` instead of `F.interpolate`

---

## 3. Gemma3 Multimodal (gemma3multimodal)

**HuggingFace Reference:** [`google/gemma-3-12b-it`](https://huggingface.co/google/gemma-3-12b-it) (HuggingFace Transformers `Gemma3ForConditionalGeneration`)

**MAX Implementation:** `max/python/max/pipelines/architectures/gemma3multimodal/`

### 3.1 Architecture Overview

Gemma3 is a multimodal model combining a SigLIP vision encoder with a Gemma3 text decoder. The text decoder features alternating global/sliding-window attention, 4-normalization layers per block, dual RoPE, QK normalization, and scaled word embeddings.

| Component | HuggingFace | MAX |
|---|---|---|
| Vision encoder | `AutoModel.from_config(vision_config)` (SigLIP) | Custom `Gemma3VisionModel` (native MAX) |
| Text model | `Gemma3TextModel` | `Gemma3LanguageModel` |
| Projector | `Gemma3MultiModalProjector` (AvgPool2d + RMSNorm + Linear) | `Gemma3MultiModalProjector` (avg_pool2d + Gemma3RMSNorm + Weight matmul) |
| RMSNorm | `(1.0 + weight) * norm(x)` | `RMSNorm(weight_offset=1)` |
| Embeddings | `Gemma3TextScaledWordEmbedding` scales by `hidden_size**0.5` | `ScaledWordEmbedding` scales by `hidden_size**0.5` |
| Multimodal merge | `inputs_embeds.masked_scatter(mask, image_features)` | `scatter_nd_skip_oob_indices` kernel |
| Float8 | Not supported | Supported (`float8_e4m3fn`) |

### 3.2 Layer-by-Layer Comparison

#### 3.2.1 Vision Embeddings

**HuggingFace**: Uses SigLIP vision model loaded via `AutoModel.from_config`. Patch embedding is a Conv2d, position embeddings are learned.

**MAX** (`Gemma3VisionEmbeddings`):
- Patch embedding: `Conv2d` from `max.nn.legacy.conv`
- Position embedding: `Embedding` from `max.nn.legacy.embedding`
- Unlike InternVL, the Conv2d is preserved (not converted to Linear)

#### 3.2.2 Vision Attention

**HuggingFace**: Standard SigLIP multi-head attention with Q, K, V, out projections.

**MAX** (`Gemma3VisionAttention`):
- Q, K, V, out projections via `Linear`
- Uses `flash_attention_gpu` with `NULL_MASK` (bidirectional, no causal masking)
- Input format: `[batch, n_patches, n_heads, head_dim]`

#### 3.2.3 Multimodal Projector

**HuggingFace** (`Gemma3MultiModalProjector`):
```python
self.avg_pool = nn.AvgPool2d(kernel_size=kernel_size, stride=kernel_size)
self.mm_soft_emb_norm = Gemma3RMSNorm(vision_hidden_size, eps=layer_norm_eps)
self.mm_input_projection_weight = nn.Parameter(torch.zeros(vision_hidden, text_hidden))
# Forward: transpose -> reshape -> avg_pool -> flatten -> transpose -> norm -> matmul
```

**MAX** (`Gemma3MultiModalProjector`):
```python
self.mm_input_projection_weight = Weight(...)
self.mm_soft_emb_norm = Gemma3RMSNorm(...)
# Forward: transpose -> reshape -> ops.avg_pool2d -> flatten -> transpose -> norm -> matmul
```

The implementations are nearly identical. The key difference is `nn.AvgPool2d` vs `ops.avg_pool2d` from `max.graph.ops`.

#### 3.2.4 Text Decoder (4-Norm Pattern)

**HuggingFace** (`Gemma3DecoderLayer`):
```python
# Pre-attention norm
hidden_states = self.input_layernorm(hidden_states)
# Attention
hidden_states = self.self_attn(hidden_states, ...)
# Post-attention norm
hidden_states = self.post_attention_layernorm(hidden_states)
hidden_states = residual + hidden_states
# Pre-FFN norm
hidden_states = self.pre_feedforward_layernorm(hidden_states)
# FFN
hidden_states = self.mlp(hidden_states)
# Post-FFN norm
hidden_states = self.post_feedforward_layernorm(hidden_states)
hidden_states = residual + hidden_states
```

**MAX** (`Gemma3TransformerBlock` in `gemma3/layers/transformer_block.py`):
The same 4-norm pattern is preserved exactly, with the addition of `Allreduce` for tensor parallelism after attention and MLP.

#### 3.2.5 RMSNorm (Gemma3 variant)

**HuggingFace**:
```python
class Gemma3RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        self.weight = nn.Parameter(torch.zeros(dim))  # Note: initialized to zeros
    def forward(self, x):
        output = self._norm(x.float())
        return (output * (1.0 + self.weight.float())).type_as(x)
```

**MAX** (`Gemma3RMSNorm` in `gemma3/layers/rms_norm.py`):
- Extends base `RMSNorm` with `weight_offset=1`
- This makes the scale factor `(1.0 + weight)` instead of just `weight`
- Weights initialized to zeros, so initial scale is 1.0

#### 3.2.6 Dual RoPE

**HuggingFace** (`Gemma3RotaryEmbedding`):
- Maintains separate `inv_freq` buffers for each layer type (`full_attention`, `sliding_attention`)
- Global attention uses scaled RoPE (with rope_scaling config)
- Sliding attention uses base RoPE (local base frequency)
- `forward()` takes `layer_type` parameter to select correct frequencies

**MAX** (`Gemma3LanguageModel.__init__`):
- Creates separate `RotaryEmbedding` instances: `global_rope` and `local_rope`
- Global rope uses `RoPEScalingParams` from config
- Local rope uses `rope_local_base_freq` without scaling
- Each transformer block receives the appropriate rope instance based on its attention type

#### 3.2.7 QK Normalization

**HuggingFace** (`Gemma3Attention`):
```python
self.q_norm = Gemma3RMSNorm(dim=config.head_dim, eps=config.rms_norm_eps)
self.k_norm = Gemma3RMSNorm(dim=config.head_dim, eps=config.rms_norm_eps)
# Applied after projection, before RoPE
query_states = self.q_norm(query_states)
key_states = self.k_norm(key_states)
```

**MAX** (`Gemma3Attention` in `gemma3/layers/attention.py`):
- Same pattern: `Gemma3RMSNorm` on Q and K after projection
- For KV cache: uses `rms_norm_key_cache` kernel to normalize keys in the cache

#### 3.2.8 Image Token Handling

**HuggingFace**:
- Uses `token_type_ids` and `image_group_ids` for bidirectional attention within image regions
- `masked_scatter` to replace image placeholder tokens with vision embeddings
- Flex attention support with custom mask functions for image tokens

**MAX**:
- Uses `scatter_nd_skip_oob_indices` to merge vision embeddings at image token positions
- No bidirectional image attention (tokens are treated with standard causal masking)
- `image_token_indices` pre-computed in tokenizer

**Migration note:** The bidirectional image attention from HuggingFace (where image tokens can attend to each other regardless of position) is not implemented in MAX. This is a known functional difference.

### 3.3 Summary of Migration Challenges

1. **SigLIP vision encoder** had to be re-implemented from scratch using MAX primitives
2. **4-norm pattern** required careful replication of the Gemma3-specific normalization placement
3. **Dual RoPE** required creating separate embedding instances rather than a single parameterized one
4. **Gemma3 RMSNorm** with `(1 + weight)` scaling required a custom subclass
5. **Bidirectional image attention** from HuggingFace is not replicated in MAX
6. **Float8 quantization** was added as an enhancement not present in HuggingFace

---

## 4. Mistral3 (mistral3)

**HuggingFace Reference:** [`mistralai/Mistral-Small-3.1-24B-Instruct-2503`](https://huggingface.co/mistralai/Mistral-Small-3.1-24B-Instruct-2503) (HuggingFace Transformers `Mistral3ForConditionalGeneration`)

**MAX Implementation:** `max/python/max/pipelines/architectures/mistral3/`

### 4.1 Architecture Overview

Mistral3 is a multimodal model with a Pixtral vision tower, a multimodal projector with patch merging, and a Mistral text decoder. **The MAX implementation only ports the text decoder**, omitting the vision components entirely.

| Component | HuggingFace | MAX |
|---|---|---|
| Vision tower | `AutoModel.from_config(vision_config)` (Pixtral) | **Not implemented** |
| Patch merger | `Mistral3PatchMerger` (unfold + linear) | **Not implemented** |
| Projector | `Mistral3MultiModalProjector` (RMSNorm + PatchMerger + Linear + GELU + Linear) | **Not implemented** |
| Text decoder | `MistralModel` (via `AutoModel`) | Inherited from base `MistralModel` |
| Tokenizer | `AutoProcessor` | Custom `Mistral3Tokenizer` (loads `chat_template.json`) |
| Context type | `TextAndVisionContext` (supports images) | `TextContext` (text only) |

### 4.2 What Was Ported

#### 4.2.1 Text Decoder (Inherited)

The MAX `Mistral3Model` extends the base `MistralModel` with minimal overrides:

```python
class Mistral3Model(MistralModel):
    def __init__(self, ...):
        text_huggingface_config = huggingface_config.get("text_config", huggingface_config)
        super().__init__(..., huggingface_config=text_huggingface_config)

    @classmethod
    def get_kv_params(cls, ...):
        text_config = huggingface_config.get("text_config", huggingface_config)
        return MistralModel.get_kv_params(..., huggingface_config=text_config)
```

The key pattern is **config extraction**: the multimodal HuggingFace config has `text_config` and `vision_config` nested inside. Mistral3 extracts `text_config` and passes it to the base Mistral implementation.

#### 4.2.2 Configuration

```python
class Mistral3Config(MistralConfig):
    @classmethod
    def initialize(cls, pipeline_config):
        huggingface_config = pipeline_config.huggingface_config
        text_config = huggingface_config.get("text_config", huggingface_config)
        return cls.initialize_from_config(pipeline_config, text_config)
```

#### 4.2.3 Custom Tokenizer

**HuggingFace**: Uses `AutoProcessor` which handles both text and image inputs.

**MAX** (`Mistral3Tokenizer`):
- Extends `TextTokenizer`
- Loads `chat_template.json` from HuggingFace repo (standard Mistral3 tokenizer doesn't include the chat template in `tokenizer_config.json`)
- Falls back to cached version if download fails

#### 4.2.4 Text Encoder (Bonus Component)

MAX includes a `text_encoder/` submodule for use in diffusion pipelines:
- `Mistral3TextEncoderTransformer`: Encoder-only transformer without KV cache
- `EncoderAttention`: Attention without KV cache, using `flash_attention_gpu`
- Returns hidden states from all layers
- Uses `max.nn.Module` (new API, not legacy) with `max.tensor.Tensor`

This component does not exist in the HuggingFace Mistral3 implementation.

#### 4.2.5 Weight Mapping

```python
MISTRAL_SAFETENSOR_MAP = {
    "language_model.model.": "",
    "language_model.": "",
}
```

Simple prefix stripping. The two entries handle both `language_model.model.layers.*` and `language_model.lm_head.*`.

### 4.3 What Was NOT Ported

1. **Pixtral Vision Tower**: Variable-resolution ViT with patch-level RoPE
2. **Patch Merger** (`Mistral3PatchMerger`): Spatial downsampling via unfold + learned linear projection
3. **Multimodal Projector**: RMSNorm + PatchMerger + two-layer MLP with GELU
4. **Image Feature Extraction**: `get_image_features()` method
5. **Multimodal Embedding Merge**: `masked_scatter` for image token replacement
6. **Image Sizes Handling**: Variable-resolution image processing

### 4.4 Summary of Migration Challenges

1. **Minimal effort port** due to heavy reuse of base Mistral implementation
2. **Config extraction** was the primary challenge: unwrapping `text_config` from multimodal config
3. **Chat template loading** required custom tokenizer to handle Mistral's non-standard template distribution
4. **Vision components omitted** -- the model is text-only in MAX, limiting multimodal capabilities
5. **Text encoder for diffusion** was an additional feature not present in HuggingFace

---

## 5. Cross-Model Pattern Analysis

### 5.1 Common Patterns Across All 4 Models

#### Weight Adapter Pattern
Every model implements a weight adapter with:
1. A mapping dictionary (`dict[str, str]`) for name prefix replacement
2. A conversion function that iterates through the state dict, applies mappings, and returns `dict[str, WeightData]`
3. Filtering logic to separate weights for different model components (vision vs language)

#### Architecture Registration
Each model defines a `SupportedArchitecture` instance in `arch.py` with:
- Architecture name matching HuggingFace's `architectures` field in config.json (with `_Legacy` suffix)
- Default and supported encodings
- Pipeline model class, tokenizer class, config class
- Optional weight adapter functions and required arguments

#### Config Pattern
All configs inherit from `ArchConfigWithKVCache` and implement:
- `initialize()`: Creates config from `PipelineConfig`
- `initialize_from_config()`: Parses HuggingFace config dict
- `finalize()`: Sets state-dict-dependent parameters
- `construct_kv_params()`: Builds `KVCacheParams`

#### Pipeline Model Pattern
All models inherit from `PipelineModel` and implement:
- `load_model()`: Builds graph, compiles, loads weights
- `execute()`: Runs inference with input buffers
- `prepare_initial_token_inputs()`: Creates first-step inputs
- `prepare_next_token_inputs()`: Creates subsequent-step inputs

### 5.2 Divergence Points

| Aspect | Text-only (GPT-OSS, Mistral3) | Multimodal (InternVL, Gemma3) |
|---|---|---|
| Compiled models | 1 (language) | 2 (vision + language) |
| Context type | `TextContext` | `TextAndVisionContext` |
| Tokenizer | `TextTokenizer` | `TextAndVisionTokenizer` or custom |
| Weight adapters | 1 adapter | 2 adapters (vision + language) |
| Embedding merge | N/A | `scatter_nd_skip_oob_indices` |
| Memory estimation | Token-based only | Token + image-based |

### 5.3 Reuse Strategies

When porting a new HuggingFace model, a developer must choose one of three strategies based on how similar the new model is to an existing MAX implementation. The following subsections describe each strategy, when to prefer it, and what is concretely required.

#### 5.3.1 Decision Framework

Use this flowchart to choose a strategy:

```
Is the new model's text decoder architecturally identical
to an existing MAX model (same attention, MLP, norms)?
  |
  ├── YES ─── Does the new model add a vision tower
  |           or other modality on top?
  |             |
  |             ├── YES ─── Strategy B: Composition
  |             |           (import layers, write new pipeline)
  |             |
  |             └── NO  ─── Strategy A: Inheritance
  |                         (extend base model, override config)
  |
  └── NO  ─── Does the new model share *some* layers
              (e.g., same norm, same attention pattern)
              with an existing MAX model?
                |
                ├── YES ─── Strategy B: Composition
                |           (import shared layers, write custom layers)
                |
                └── NO  ─── Strategy C: Full Re-implementation
                            (write all layers from scratch)
```

Key factors to consider:

| Factor | Favors Inheritance | Favors Composition | Favors Full Re-impl |
|---|---|---|---|
| Decoder architecture match | Identical to existing model | Shares some layer types | Unique architecture |
| Adds vision/multimodal | No | Yes | Yes (novel vision encoder) |
| Custom attention pattern | No (same as base) | Partially shared | Sinks, novel MoE, etc. |
| Custom activation functions | No | No | Yes |
| Config structure | Wraps existing config | Different config + shared layers | Entirely new config |
| Development effort | Very low (days) | Medium (1-2 weeks) | High (2-4 weeks) |
| Maintenance burden | Lowest (inherits fixes) | Medium (shared layers get fixes) | Highest (independent) |

#### 5.3.2 Strategy A: Inheritance

**Used by:** Mistral3

**When to use:** The new model's text decoder is architecturally identical to an existing MAX model, but the HuggingFace config wraps it differently (e.g., a multimodal config that nests `text_config`) or the model needs minor behavioral adjustments (custom tokenizer, different default parameters).

**What you inherit:** The entire pipeline model implementation -- graph building, compilation, execution, KV cache management, input preparation, tensor parallelism, and all layer implementations. You write no new neural network code.

**Concrete requirements:**

1. **Config class** -- Extend the base config and override `initialize()` to extract the relevant sub-config:

```python
# mistral3/model_config.py (52 lines total)
@dataclass(kw_only=True)
class Mistral3Config(MistralConfig):
    @override
    @classmethod
    def initialize(cls, pipeline_config: PipelineConfig) -> Self:
        huggingface_config = pipeline_config.model.huggingface_config
        # Extract the text_config from the multimodal config
        return cls.initialize_from_config(
            pipeline_config, huggingface_config.text_config
        )
```

2. **Pipeline model class** -- Extend the base model and override methods that receive the raw HuggingFace config to extract the text sub-config first:

```python
# mistral3/model.py (84 lines total)
class Mistral3Model(MistralModel):
    def __init__(self, ..., huggingface_config, ...):
        super().__init__(
            ...,
            text_huggingface_config=huggingface_config.text_config,
        )

    @classmethod
    def get_kv_params(cls, huggingface_config, ...):
        return super().get_kv_params(huggingface_config.text_config, ...)

    @classmethod
    def calculate_max_seq_len(cls, pipeline_config, huggingface_config):
        huggingface_config = getattr(
            huggingface_config, "text_config", huggingface_config
        )
        return super().calculate_max_seq_len(pipeline_config, huggingface_config)
```

3. **Weight adapter** -- Strip multimodal prefixes so the weights match what the base model expects:

```python
# mistral3/weight_adapters.py (36 lines total)
MISTRAL_SAFETENSOR_MAP = {
    "language_model.model.": "",
    "language_model.": "",
}
```

4. **Tokenizer** (optional) -- Subclass if the model has non-standard tokenizer behavior:

```python
# mistral3/tokenizer.py
class Mistral3Tokenizer(TextTokenizer):
    def __init__(self, pipeline_config):
        super().__init__(pipeline_config)
        self._load_and_set_chat_template()  # Custom chat template loading
```

5. **Architecture registration** -- Reference the new config, model, and tokenizer classes:

```python
# mistral3/arch.py
mistral3_arch = SupportedArchitecture(
    name="Mistral3ForConditionalGeneration_Legacy",
    pipeline_model=Mistral3Model,
    config=Mistral3Config,
    tokenizer=Mistral3Tokenizer,
    weight_adapters={WeightsFormat.safetensors: convert_safetensor_state_dict},
    ...
)
```

**Total code:** Mistral3 required approximately **220 lines** across 5 files (not counting the text encoder, which is an unrelated bonus component). No new neural network layers were written.

**Advantages:** Automatic benefit from any improvements or bug fixes to the base model. Minimal code to maintain. Fast to implement.

**Limitations:** Cannot modify the decoder architecture. If the new model has even one architectural difference in the text decoder (different norm placement, different attention pattern), inheritance breaks and you must fall back to composition or full re-implementation.

#### 5.3.3 Strategy B: Composition

**Used by:** Gemma3 Multimodal

**When to use:** The new model shares significant layer-level components with an existing MAX model, but the overall pipeline structure is different. Common scenario: adding a vision encoder to an existing text model, or building a model that uses the same attention/norm/MLP layers but assembles them differently.

**What you reuse:** Individual layer classes (`Attention`, `TransformerBlock`, `RMSNorm`, `ScaledWordEmbedding`, etc.) imported from the base model's `layers/` directory. You write a new pipeline model, new graph building, and any new layer types (vision encoder, multimodal projector).

**Concrete requirements:**

1. **Import shared layers** -- Import layer classes directly from the base model's layers package:

```python
# gemma3multimodal/vision_model/gemma3multimodal.py
from max.pipelines.architectures.gemma3.layers.attention import Gemma3Attention
from max.pipelines.architectures.gemma3.layers.rms_norm import Gemma3RMSNorm
from max.pipelines.architectures.gemma3.layers.scaled_word_embedding import ScaledWordEmbedding
from max.pipelines.architectures.gemma3.layers.transformer_block import Gemma3TransformerBlock
```

2. **Build the language model using imported layers** -- Construct the same layer stack as the base model but in your own `Module` class, which gives you control over the forward pass:

```python
class Gemma3LanguageModel(Module):
    def __init__(self, config):
        # Reuse all Gemma3 layer types
        self.embed_tokens = ScaledWordEmbedding(...)
        self.norm = Gemma3RMSNorm(...)
        self.lm_head = ColumnParallelLinear(...)
        self.layers = LayerList([
            Gemma3TransformerBlock(
                attention=Gemma3Attention(...),
                mlp=MLP(...),
                input_layernorm=Gemma3RMSNorm(...),
                post_attention_layernorm=Gemma3RMSNorm(...),
                pre_feedforward_layernorm=Gemma3RMSNorm(...),
                post_feedforward_layernorm=Gemma3RMSNorm(...),
                devices=config.devices,
            )
            for i in range(num_layers)
        ])

    def __call__(self, tokens, image_embeddings, image_token_indices, ...):
        h = self.embed_tokens(tokens)
        # NEW: merge vision embeddings (not in base Gemma3)
        h = merge_multimodal_embeddings(h, image_embeddings, image_token_indices)
        # REUSED: standard transformer forward pass
        for layer in self.layers:
            h = layer(h, kv_collection, ...)
        return self.lm_head(self.norm(h))
```

3. **Implement new components** -- Write vision encoder, projector, and any model-specific layers from scratch:

```
vision_model/
├── attention.py         # Gemma3VisionAttention (new, for SigLIP encoder)
├── embedding.py         # Gemma3VisionEmbeddings (new, Conv2d patches)
├── encoding.py          # Gemma3VisionEncoder + EncoderLayer (new)
├── gemma3multimodal.py  # Gemma3LanguageModel (reuses layers) + Gemma3VisionModel (new)
└── projection.py        # Gemma3MultiModalProjector (new, avg pool + norm + linear)
```

4. **Write a new pipeline model** -- Since the execution flow differs from the base model (two compiled models, vision input handling, image batching), you need a full `PipelineModel` subclass:

```python
class Gemma3_MultiModalModel(AlwaysSignalBuffersMixin, PipelineModel, KVCacheMixin):
    def load_model(self):
        # Build and compile TWO separate graphs
        self.vision_model = self._build_and_compile_vision_graph(...)
        self.language_model = self._build_and_compile_language_graph(...)

    def execute(self, inputs):
        if inputs.has_vision_inputs:
            image_embeddings = self.vision_model.execute(pixel_values=...)
        logits = self.language_model.execute(
            tokens=..., image_embeddings=..., ...
        )
        return ModelOutputs(next_token_logits=logits)
```

5. **Separate weight adapters** -- One for vision weights, one for language weights:

```python
def convert_safetensor_language_state_dict(state_dict):
    # Filter: only "language_model.*" weights, strip prefix
    ...

def convert_safetensor_vision_state_dict(state_dict):
    # Filter: only "vision_tower.*" and "multi_modal_*" weights, strip prefixes
    ...
```

6. **Config class** -- Typically a new dataclass that contains both text and vision sub-configs:

```python
@dataclass
class Gemma3ForConditionalGenerationConfig(ArchConfigWithKVCache):
    text_config: Gemma3TextConfig  # Reuses or mirrors base Gemma3 config
    vision_config: Gemma3VisionConfig  # New, for vision encoder
    mm_tokens_per_image: int = 256
    boi_token_index: int = 0
    eoi_token_index: int = 0
    image_token_index: int = 0
```

**Total code:** Gemma3 Multimodal required approximately **2,500 lines** across 10 files. The language model layers are imported (not re-implemented), but the vision encoder, projector, pipeline model, config, and weight adapters are all new.

**Advantages:** Shared layers benefit from upstream fixes. Reduces the amount of neural network code to write and test. The language model's attention, normalization, and MLP are known-correct from the base model.

**Limitations:** Tight coupling to the base model's layer API. If the base model refactors its layer interfaces, the composed model may break. You also cannot modify the shared layers without affecting the base model.

**Cross-architecture composition** is also possible. Gemma3 Multimodal imports `merge_multimodal_embeddings` from InternVL's `embedding_utils.py`, demonstrating that utility functions can be shared across model families:

```python
from max.pipelines.architectures.internvl.embedding_utils import merge_multimodal_embeddings
```

#### 5.3.4 Strategy C: Full Re-implementation

**Used by:** GPT-OSS, InternVL

**When to use:** The model has unique architectural features that don't exist in any current MAX implementation (novel attention mechanisms, custom MoE patterns, unique activation functions), or the model's HuggingFace reference uses custom code in the model repo rather than standard transformers classes.

**What you reuse:** Only the base infrastructure from `max.nn.legacy` (Module, Linear, RMSNorm, etc.) and `max.pipelines.lib` (PipelineModel, KVCacheMixin, etc.). All model-specific layers are written from scratch.

**Concrete requirements:**

1. **Custom layer implementations** -- Write every model-specific layer as a `Module` subclass:

For GPT-OSS (attention with sinks, MoE with custom activation):
```
layers/
├── attention.py          # GptOssAttention: sinks, sliding window, YARN RoPE
├── moe.py               # GptOssMoE + GptOssMoEGate: custom SwiGLU, biases
└── transformer_block.py  # GptOssTransformerBlock: attention + MoE + allreduce
```

For InternVL (vision encoder, Qwen2 decoder, multimodal projector):
```
layers/
└── attention.py          # InternVLMultiheadAttention: QK norm, stacked QKV
internvl.py               # InternVLDecoderLayer, InternVLLanguageModel,
                          # InternVisionEmbeddings, InternVisionEncoderLayer,
                          # InternVLVisionModel, InternVLMLP1
```

2. **Each custom layer must implement:**

   - `__init__()`: Create all `Weight` objects and sub-modules with correct shapes, dtypes, and devices
   - `__call__()`: Forward pass using `max.graph.ops` and custom kernels
   - `Shardable` protocol (if tensor parallelism is needed):
     - `sharding_strategy` property (getter/setter)
     - `shard(devices)` method that returns a list of sharded copies

   Example skeleton:
   ```python
   class MyCustomAttention(Module, Shardable):
       def __init__(self, *, num_heads, hidden_size, kv_params, dtype, devices, ...):
           self.q_proj = Linear(hidden_size, num_heads * head_dim, dtype, devices[0])
           self.k_proj = Linear(hidden_size, kv_heads * head_dim, dtype, devices[0])
           self.v_proj = Linear(hidden_size, kv_heads * head_dim, dtype, devices[0])
           self.o_proj = Linear(num_heads * head_dim, hidden_size, dtype, devices[0])
           # Any model-specific weights
           self.custom_weight = Weight("custom", shape=[...], dtype=dtype, device=devices[0])

       def __call__(self, x, kv_collection, **kwargs):
           wqkv = ops.concat([self.q_proj.weight, self.k_proj.weight, self.v_proj.weight])
           xq = fused_qkv_ragged_matmul(self.kv_params, x, wqkv, ...)
           xq = fused_qk_ragged_rope(self.kv_params, xq, ...)
           attn_out = flash_attention_ragged(self.kv_params, xq, ...,
               # Pass model-specific params to kernel
               sink_weights=self.custom_weight,
           )
           return self.o_proj(attn_out.reshape([total_seq_len, -1]))

       @property
       def sharding_strategy(self):
           return self._sharding_strategy

       @sharding_strategy.setter
       def sharding_strategy(self, strategy):
           self.q_proj.sharding_strategy = ShardingStrategy.rowwise(strategy.num_devices)
           self.k_proj.sharding_strategy = ShardingStrategy.rowwise(strategy.num_devices)
           self.v_proj.sharding_strategy = ShardingStrategy.rowwise(strategy.num_devices)
           self.o_proj.sharding_strategy = ShardingStrategy.head_aware_columnwise(...)

       def shard(self, devices):
           # Create per-device copies with sharded weights
           ...
   ```

3. **Full pipeline model** -- Write `load_model()`, `_build_graph()`, `execute()`, `prepare_initial_token_inputs()`, and `prepare_next_token_inputs()` from scratch:

```python
class GptOssModel(AlwaysSignalBuffersMixin, PipelineModel, KVCacheMixin):
    def load_model(self):
        graph = Graph("gpt_oss", ...)
        # Define all input types
        tokens_type = TensorType(DType.int64, ["total_seq_len"], device=DeviceRef.GPU())
        offsets_type = TensorType(DType.uint32, ["batch_plus_1"], device=DeviceRef.GPU())
        # ... kv cache inputs, signal buffers ...

        # Instantiate and call the model architecture
        model = GptOss(config)
        model.load_state_dict(state_dict)
        logits = model(tokens, offsets, kv_collections, signal_buffers, ...)
        graph.output(logits)

        self._model = session.load(graph)
```

4. **Config, weight adapters, and arch registration** -- Same as other strategies, but the config may have many model-specific fields (e.g., GPT-OSS has `num_local_experts`, `num_experts_per_tok`, `swiglu_limit`, `sliding_window`, `layer_types`, `rope_scaling` with YARN-specific params).

5. **For multimodal full re-implementations** (InternVL), also implement:
   - Custom tokenizer and image processor
   - Vision encoder with all layers
   - Multimodal projector
   - Two-model compilation pipeline
   - Image batching and memory estimation

**Total code:** GPT-OSS required approximately **2,200 lines** across 10 files. InternVL required approximately **4,000 lines** across 12 files (including the full Qwen2 decoder, vision encoder, custom tokenizer, and image preprocessing).

**Advantages:** Complete control over every aspect of the model. No coupling to other model implementations. Can implement any architectural novelty.

**Limitations:** Highest development and maintenance cost. No automatic benefit from improvements to other models. Every layer must be independently tested for correctness. Weight loading, sharding, and KV cache integration must all be handled manually.

#### 5.3.5 Strategy Comparison Summary

| Dimension | A: Inheritance | B: Composition | C: Full Re-impl |
|---|---|---|---|
| Lines of code | ~200 | ~2,500 | ~2,000-4,000 |
| Files to create | 5 | 8-12 | 8-12 |
| New layer classes | 0 | Vision layers only | All layers |
| Decoder code | Inherited | Layer classes imported | Written from scratch |
| Pipeline model | Inherited | New (two-model for multimodal) | New |
| Config class | Extends base | New (with sub-configs) | New |
| Weight adapters | Prefix stripping only | Prefix stripping + filtering | Prefix stripping + transforms |
| Tensor parallelism | Inherited | Inherited for shared layers | Must implement per-layer |
| KV cache | Inherited | Inherited (language layers) | Must integrate with kernels |
| Maintenance | Base model fixes propagate | Shared layer fixes propagate | Independent |
| Example models | Mistral3 | Gemma3 Multimodal | GPT-OSS, InternVL |

---

## 6. Technical Migration Guide

This section provides a step-by-step guide for converting a HuggingFace model to the MAX Python API.

### 6.1 Prerequisites

Before starting, gather:
1. The HuggingFace model's `config.json`
2. The model's `modeling_*.py` source code (from transformers or the model repo)
3. The model's architecture class name (from `config.json` -> `architectures`)
4. Sample weights in safetensors format

### 6.2 Step 1: Create the Architecture Directory

```
max/python/max/pipelines/architectures/my_model/
├── __init__.py
├── arch.py
├── model.py
├── model_config.py
└── weight_adapters.py
```

For multimodal models, add vision-specific files:
```
├── vision_model/
│   ├── attention.py
│   ├── embedding.py
│   ├── encoder.py
│   └── my_model_vision.py
├── embedding_utils.py    # Multimodal merge
└── tokenizer.py          # Custom preprocessing
```

### 6.3 Step 2: Configuration Migration

Map HuggingFace config fields to a MAX dataclass:

```python
@dataclass
class MyModelConfig(ArchConfigWithKVCache):
    # Map HuggingFace fields directly
    hidden_size: int = 0
    num_hidden_layers: int = 0
    num_attention_heads: int = 0
    num_key_value_heads: int = 0
    head_dim: int = 0
    vocab_size: int = 0
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0

    # MAX-specific fields
    dtype: DType = DType.bfloat16
    devices: list[DeviceRef] = field(default_factory=list)
    kv_params: KVCacheParams | None = None
    return_logits: ReturnLogits = ReturnLogits.LAST_TOKEN

    @classmethod
    def initialize_from_config(cls, pipeline_config, huggingface_config):
        return cls(
            hidden_size=huggingface_config["hidden_size"],
            num_hidden_layers=huggingface_config["num_hidden_layers"],
            # ... map all fields
        )
```

**Key gotchas:**
- For multimodal models, HuggingFace nests `text_config` and `vision_config`. Extract the relevant sub-config.
- Some models have non-standard field names (e.g., `intermediate_size` vs `feed_forward_length`).
- RoPE scaling parameters vary widely between models. Check `rope_scaling` in config carefully.

### 6.4 Step 3: Layer Migration Reference

#### 6.4.1 Linear Layers

| PyTorch | MAX |
|---|---|
| `nn.Linear(in, out, bias=True)` | `Linear(in_dim=in, out_dim=out, dtype=dtype, device=device, has_bias=True)` |
| `nn.Linear(in, out, bias=False)` | `Linear(in_dim=in, out_dim=out, dtype=dtype, device=device)` |
| Column parallel | `ColumnParallelLinear(in_dim, out_dim, dtype, devices)` |

#### 6.4.2 Normalization

| PyTorch | MAX |
|---|---|
| `nn.LayerNorm(dim, eps)` | `LayerNorm(dim, eps)` from `max.nn.legacy.norm` |
| `nn.RMSNorm(dim, eps)` or custom | `RMSNorm(dim, eps)` from `max.nn.legacy.norm.rms_norm` |
| Gemma3-style `(1+w)*norm(x)` | `RMSNorm(dim, eps, weight_offset=1)` |

#### 6.4.3 Embeddings

| PyTorch | MAX |
|---|---|
| `nn.Embedding(vocab, dim)` | `Embedding(vocab, dim, dtype, device)` from `max.nn.legacy.embedding` |
| Parallel embedding | `VocabParallelEmbedding(vocab, dim, dtype, devices)` |
| Scaled embedding | `ScaledWordEmbedding(vocab, dim, dtype, devices, embed_scale)` |

#### 6.4.4 Attention

| PyTorch Pattern | MAX Equivalent |
|---|---|
| Separate Q, K, V `nn.Linear` | Separate `Linear` + `ops.concat` for fused QKV |
| `F.scaled_dot_product_attention` | `flash_attention_gpu` (dense) or `flash_attention_ragged` (with KV cache) |
| Attention mask (2D/4D tensor) | `MHAMaskVariant` enum (CAUSAL, SLIDING_WINDOW, NULL) |
| `DynamicCache` | `PagedCacheValues` |
| Batched: `[B, heads, seq, dim]` | Ragged: `[total_tokens, heads, dim]` with `input_row_offsets` |

#### 6.4.5 RoPE

| HuggingFace | MAX |
|---|---|
| `ROPE_INIT_FUNCTIONS["default"]` | `RotaryEmbedding` |
| `ROPE_INIT_FUNCTIONS["llama3"]` | `Llama3RotaryEmbedding` |
| `ROPE_INIT_FUNCTIONS["yarn"]` | `YarnRotaryEmbedding` with `YarnScalingParams` |
| `ROPE_INIT_FUNCTIONS["dynamic"]` | `DynamicRotaryEmbedding` |
| `ROPE_INIT_FUNCTIONS["longrope"]` | `LongRoPERotaryEmbedding` |

RoPE is typically applied via `fused_qk_ragged_rope` (fused with ragged sequence handling) rather than as a separate step.

#### 6.4.6 Activation Functions

| PyTorch | MAX |
|---|---|
| `F.gelu(x)` | `ops.gelu(x)` |
| `F.silu(x)` | `ops.silu(x)` |
| `F.sigmoid(x)` | `ops.sigmoid(x)` |
| `F.relu(x)` | `ops.relu(x)` |
| `torch.tanh(x)` | `ops.tanh(x)` |
| `torch.clamp(x, min, max)` | `clamp(x, min, max)` from `max.nn.legacy.clamp` |

### 6.5 Step 4: Weight Adapter

Create a mapping from HuggingFace weight names to MAX weight names:

```python
MY_MODEL_SAFETENSOR_MAP = {
    "model.layers.":        "language_model.layers.",
    "model.embed_tokens.":  "language_model.embed_tokens.",
    "model.norm.":          "language_model.norm.",
    "lm_head.":             "language_model.lm_head.",
}

def convert_safetensor_state_dict(
    state_dict: dict[str, Weights], **kwargs
) -> dict[str, WeightData]:
    new_state_dict: dict[str, WeightData] = {}
    for weight_name, value in state_dict.items():
        max_name = weight_name
        for before, after in MY_MODEL_SAFETENSOR_MAP.items():
            max_name = max_name.replace(before, after)
        new_state_dict[max_name] = value.data()
    return new_state_dict
```

**Common weight transformations:**
- **Prefix stripping**: Remove `model.`, `language_model.model.`, `vision_tower.vision_model.` prefixes
- **Name mapping**: Rename layers (e.g., `.mlp.router` -> `.mlp.gate.gate_score`)
- **Shape conversion**: Conv2d to Linear (reshape 4D weights to 2D)
- **Multimodal splitting**: Filter weights by prefix for vision vs language models

### 6.6 Step 5: Attention with KV Cache

Replace HuggingFace's attention + cache pattern with MAX's fused kernels:

**HuggingFace pattern:**
```python
query = self.q_proj(x).view(B, S, n_heads, head_dim).transpose(1, 2)
key = self.k_proj(x).view(B, S, n_kv_heads, head_dim).transpose(1, 2)
value = self.v_proj(x).view(B, S, n_kv_heads, head_dim).transpose(1, 2)
query, key = apply_rotary_pos_emb(query, key, cos, sin)
key, value = past_key_values.update(key, value, layer_idx)
attn_output = F.scaled_dot_product_attention(query, key, value, attn_mask)
```

**MAX pattern:**
```python
wqkv = ops.concat([self.q_proj.weight, self.k_proj.weight, self.v_proj.weight])
xq = fused_qkv_ragged_matmul(kv_params, x, wqkv, bias, input_row_offsets, kv_collection, layer_idx, n_heads)
xq = xq.reshape((-1, n_heads, head_dim))
xq = fused_qk_ragged_rope(kv_params, xq, input_row_offsets, kv_collection, freqs_cis, layer_idx)
attn_out = flash_attention_ragged(kv_params, xq, kv_collection, layer_idx, input_row_offsets, mask_variant)
```

Key differences:
1. QKV projection, KV cache write, and RoPE are fused into two kernel calls
2. The KV cache (`kv_collection`) is written to during `fused_qkv_ragged_matmul` (K and V are stored in the cache)
3. Only Q is returned from the fused matmul; K and V are read from cache during attention
4. No explicit `cache.update()` call -- it's built into the kernel

### 6.7 Step 6: Pipeline Model Implementation

```python
class MyModel(PipelineModel, KVCacheMixin):
    def load_model(self):
        # 1. Build the graph
        graph = Graph("my_model", ...)
        # 2. Define input types
        tokens_type = TensorType(DType.int64, ["total_seq_len"], device=DeviceRef.GPU())
        offsets_type = TensorType(DType.uint32, ["batch_size_plus_1"], device=DeviceRef.GPU())
        # 3. Create model and call forward
        model = MyModelArchitecture(config)
        model.load_state_dict(state_dict)
        logits = model(tokens, offsets, kv_cache_inputs, ...)
        graph.output(logits)
        # 4. Compile
        self._model = session.load(graph)

    def execute(self, inputs):
        outputs = self._model.execute(**inputs.to_dict())
        return ModelOutputs(next_token_logits=outputs[0])
```

### 6.8 Step 7: Ragged Sequence Handling

Replace padding + attention mask with ragged tensors:

**HuggingFace:**
```python
# Padded: all sequences same length, mask indicates valid tokens
input_ids = [[1, 2, 3, 0, 0],    # seq 1 (len 3, padded to 5)
             [4, 5, 6, 7, 0]]    # seq 2 (len 4, padded to 5)
attention_mask = [[1, 1, 1, 0, 0],
                  [1, 1, 1, 1, 0]]
```

**MAX:**
```python
# Ragged: sequences concatenated, offsets mark boundaries
input_ids = [1, 2, 3, 4, 5, 6, 7]  # all tokens concatenated
input_row_offsets = [0, 3, 7]        # seq 1 starts at 0 (len 3), seq 2 starts at 3 (len 4)
```

This eliminates wasted computation on padding tokens.

### 6.9 Step 8: Tensor Parallelism

Add sharding support to each layer:

1. **Implement `Shardable` protocol** on layers that need sharding
2. **Define sharding strategies** for each weight:
   - QKV projections: `ShardingStrategy.rowwise` (split heads across devices)
   - Output projection: `ShardingStrategy.head_aware_columnwise`
   - Embeddings: `VocabParallelEmbedding` or `ShardingStrategy.replicate`
   - Normalization: `ShardingStrategy.replicate` (all devices need full norm weights)
   - MoE experts: Split expert dimensions across devices
3. **Add `Allreduce`** after attention and MLP blocks for result aggregation
4. **Use `Signals`** for communication buffer management

### 6.10 Step 9: Architecture Registration

```python
my_model_arch = SupportedArchitecture(
    name="MyModelForCausalLM_Legacy",  # Must match HF architectures + _Legacy suffix
    task=PipelineTask.TEXT_GENERATION,
    example_repo_ids=["org/model-name"],
    default_encoding=SupportedEncoding.bfloat16,
    supported_encodings={SupportedEncoding.bfloat16: [KVCacheStrategy.PAGED]},
    pipeline_model=MyModel,
    tokenizer=TextTokenizer,  # or TextAndVisionTokenizer
    context_type=TextContext,  # or TextAndVisionContext
    default_weights_format=WeightsFormat.safetensors,
    multi_gpu_supported=True,
    weight_adapters={WeightsFormat.safetensors: convert_safetensor_state_dict},
    config=MyModelConfig,
)
```

### 6.11 Step 10: Multimodal Extensions (if applicable)

For multimodal models, additional steps are needed:

1. **Implement vision encoder** as a separate compiled model
2. **Implement multimodal projector** to map vision features to text embedding space
3. **Implement embedding merge** using `scatter_nd_skip_oob_indices`:
   ```python
   from max.nn.legacy.kernels import scatter_nd_skip_oob_indices
   merged = scatter_nd_skip_oob_indices(text_embeds, vision_embeds, image_token_indices)
   ```
4. **Custom tokenizer/processor** for image preprocessing
5. **Two-model pipeline** with separate vision and language compilation
6. **Memory estimation** must account for vision model memory (typically ~128 MiB per image)

---

## 7. Known API Gaps and Workarounds

### 7.1 Missing PyTorch Equivalents

| PyTorch | MAX Status | Workaround |
|---|---|---|
| `torch.nn.functional.*` | No direct equivalent | Use `max.graph.ops.*` or custom kernels |
| `torch.masked_scatter` | Not available | Use `scatter_nd_skip_oob_indices` kernel |
| `F.interpolate(mode='bicubic')` | Available | `ops.resize(method="bicubic")` |
| `nn.Conv2d` | Available in legacy | `Conv2d` from `max.nn.legacy.conv`, or convert to `Linear` |
| `nn.AvgPool2d` | Available | `ops.avg_pool2d` |
| `nn.Dropout` | Not needed | Inference-only, dropout is always disabled |
| `DropPath` (stochastic depth) | Not needed | Inference-only, drop path rate is 0 |
| `torch.compile()` | Different paradigm | MAX graph compilation is mandatory, not optional |
| Dynamic control flow | Not supported in graph | Must be expressed as static graph operations |
| `torch.where` | Available | `ops.where` |

### 7.2 Execution Model Differences

1. **Static shapes at compile time**: All tensor shapes must be expressible symbolically at graph construction time. Dynamic batch sizes use symbolic dimensions.

2. **No in-place operations**: MAX graph operations are functional (no in-place mutation). Operations like `input_embeds[mask] = new_values` must be replaced with scatter operations.

3. **Device placement is explicit**: Every weight and operation must have an explicit device assignment. There's no implicit CUDA device like in PyTorch.

4. **Two-phase execution**: Model construction (graph building) is separate from execution. Weight values are not available during graph construction; only shapes and types are known.

### 7.3 KV Cache Differences

| HuggingFace | MAX |
|---|---|
| `DynamicCache`: grows with sequence | `PagedCacheValues`: fixed-size pages |
| Stored as dense tensors | Page-table-based memory management |
| Updated via `cache.update(key, value, layer_idx)` | Written during `fused_qkv_ragged_matmul` |
| Per-sequence cache | Shared pool with page allocation |
| No prefix caching | Supports prefix caching (shared pages) |
| No quantization | Supports FP8 quantized KV cache |

### 7.4 Training vs Inference

MAX model implementations are **inference-only**. Features only needed for training are removed:
- Dropout (always disabled)
- Gradient checkpointing
- Loss computation (CrossEntropyLoss)
- Auxiliary losses (MoE load balancing)
- Label handling
- Stochastic depth (DropPath)

### 7.5 Sequence Representation

MAX uses ragged tensors instead of padded tensors:
- **Pro**: No wasted computation on padding tokens
- **Pro**: Natural support for variable-length sequences
- **Con**: Requires `input_row_offsets` management
- **Con**: Fused kernels must support ragged layout

### 7.6 Checklist for New Model Migration

- [ ] Identify the HuggingFace model architecture and config structure
- [ ] Create config dataclass mapping all HuggingFace fields
- [ ] Identify which MAX base model (if any) to inherit from
- [ ] Map all layer types to MAX equivalents (see tables above)
- [ ] Identify special attention features (sinks, sliding window, QK norm, etc.)
- [ ] Determine RoPE variant and scaling parameters
- [ ] Create weight adapter with name mapping dictionary
- [ ] Handle any weight shape transformations (Conv2d -> Linear, etc.)
- [ ] Implement KV cache integration using fused kernels
- [ ] Convert padded sequence handling to ragged
- [ ] Add tensor parallel sharding for all layers
- [ ] Implement pipeline model with graph compilation
- [ ] Register architecture in `arch.py`
- [ ] Export from `__init__.py`
- [ ] For multimodal: implement vision encoder, projector, and embedding merge
- [ ] For multimodal: implement custom tokenizer/processor
- [ ] Test weight loading and basic inference
- [ ] Verify output matches HuggingFace within numerical tolerance
