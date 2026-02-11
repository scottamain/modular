# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #
"""Gated Attention for Qwen3-Next full-attention layers.

In Qwen3-Next, "Gated Attention" layers produce a query-dependent sigmoid
gate alongside the normal attention output.  The HF checkpoint stores a
single ``q_proj`` weight of shape ``[num_heads * head_dim * 2, hidden_size]``
(query and gate interleaved per head).  The weight adapter splits this into
two separate weights:

  - ``self_attn.q_proj.weight`` — query half  ``[num_heads * head_dim, hidden]``
  - ``self_attn.attn_gate.weight`` — gate half ``[num_heads * head_dim, hidden]``

so the existing fused QKV kernel can be used for the query path while the
gate is computed as a cheap extra linear projection.
"""

from __future__ import annotations

from collections.abc import Callable

from max.dtype import DType
from max.graph import DeviceRef, ShardingStrategy, TensorValue, ops
from max.nn.legacy.kv_cache import KVCacheParams
from max.nn.legacy.linear import Linear
from max.nn.legacy.attention import MHAMaskVariant
from max.nn.legacy.kernels import (
    flash_attention_ragged,
    fused_qk_ragged_rope,
    fused_qkv_ragged_matmul,
    rms_norm_key_cache,
)
from max.nn.legacy.rotary_embedding import RotaryEmbedding
from max.pipelines.architectures.qwen3.layers.attention import Qwen3Attention


class Qwen3NextGatedAttention(Qwen3Attention):
    """Qwen3Attention with an additional query-dependent sigmoid gate.

    Identical to ``Qwen3Attention`` except for:
    * An extra ``attn_gate`` linear projection.
    * The gate is applied *before* ``o_proj``, matching the HF implementation.
    """

    def __init__(
        self,
        rope: RotaryEmbedding,
        num_attention_heads: int,
        num_key_value_heads: int,
        hidden_size: int,
        kv_params: KVCacheParams,
        layer_idx: int,
        dtype: DType = DType.float32,
        devices: list[DeviceRef] | None = None,
        linear_cls: Callable[..., Linear] = Linear,
        scale: float | None = None,
        has_bias: bool = False,
        qk_norm_eps: float = 1e-6,
    ) -> None:
        if devices is None:
            devices = [DeviceRef.CPU()]
        super().__init__(
            rope=rope,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            hidden_size=hidden_size,
            kv_params=kv_params,
            layer_idx=layer_idx,
            dtype=dtype,
            devices=devices,
            linear_cls=linear_cls,
            scale=scale,
            has_bias=has_bias,
            qk_norm_eps=qk_norm_eps,
        )
        # Gate projection (same shape as q_proj).
        q_weight_dim = kv_params.head_dim * num_attention_heads
        self.attn_gate = linear_cls(
            in_dim=hidden_size,
            out_dim=q_weight_dim,
            dtype=dtype,
            device=devices[0],
            has_bias=has_bias,
        )

    @Qwen3Attention.sharding_strategy.setter  # type: ignore[attr-defined]
    def sharding_strategy(self, strategy: ShardingStrategy) -> None:
        Qwen3Attention.sharding_strategy.fset(self, strategy)  # type: ignore[union-attr]
        if strategy.is_tensor_parallel:
            self.attn_gate.sharding_strategy = ShardingStrategy.rowwise(
                strategy.num_devices
            )
        else:
            self.attn_gate.sharding_strategy = strategy

    def shard(
        self, devices: list[DeviceRef]
    ) -> list["Qwen3NextGatedAttention"]:
        """Shard this gated attention module across devices."""
        parent_shards = super().shard(devices)
        gate_shards = self.attn_gate.shard(devices)

        gated_shards: list[Qwen3NextGatedAttention] = []
        for shard_idx, parent_shard in enumerate(parent_shards):
            sharded = Qwen3NextGatedAttention.__new__(
                Qwen3NextGatedAttention
            )
            sharded.__dict__.update(parent_shard.__dict__)
            sharded.attn_gate = gate_shards[shard_idx]
            gated_shards.append(sharded)
        return gated_shards

    def __call__(
        self,
        layer_idx: TensorValue,
        x: TensorValue,
        kv_collection,
        freqs_cis: TensorValue,
        input_row_offsets: TensorValue,
    ) -> TensorValue:
        """Forward with gated attention.

        Same signature as ``Qwen3Attention.__call__`` but applies
        ``sigmoid(gate(x))`` element-wise to the attention output *before*
        the output projection (``o_proj``).
        """
        # Compute gate from the raw input hidden states.
        gate = self.attn_gate(x)  # [total_seq, num_heads * head_dim]
        gate = ops.sigmoid(gate.cast(DType.float32))

        # ---- begin standard attention (mirrors Qwen3Attention.__call__) ----
        total_seq_len = x.shape[0]

        wqkv = self.wqkv
        xq = fused_qkv_ragged_matmul(
            self.kv_params,
            input=x,
            wqkv=wqkv,
            bias=self.wqkv_bias,
            input_row_offsets=input_row_offsets,
            kv_collection=kv_collection,
            layer_idx=layer_idx,
            n_heads=self.n_heads,
        )

        xq = xq.reshape((-1, self.n_heads, self.kv_params.head_dim))
        xq = self.q_norm(xq)

        rms_norm_key_cache(
            self.kv_params,
            kv_collection=kv_collection,
            gamma=self.k_norm.weight.cast(self.kv_params.dtype).to(
                self.devices[0]
            ),
            epsilon=self.qk_norm_eps,
            layer_idx=layer_idx,
            total_seq_len=total_seq_len,
            input_row_offsets=input_row_offsets,
            weight_offset=0.0,
            multiply_before_cast=False,
            per_head_norm=True,
        )

        freqs_cis = ops.cast(freqs_cis, xq.dtype).to(xq.device)
        xq = fused_qk_ragged_rope(
            self.kv_params,
            xq,
            input_row_offsets,
            kv_collection,
            freqs_cis,
            layer_idx,
            interleaved=self.rope.interleaved,
        )

        attn_out = flash_attention_ragged(
            self.kv_params,
            input=xq,
            kv_collection=kv_collection,
            layer_idx=layer_idx,
            input_row_offsets=input_row_offsets,
            mask_variant=MHAMaskVariant.CAUSAL_MASK,
            scale=self.scale,
        )
        # ---- end standard attention ----

        # Apply gate before output projection (matching HF).
        attn_out = ops.reshape(attn_out, shape=[total_seq_len, -1])
        attn_out = ops.mul(attn_out.cast(DType.float32), gate).cast(
            attn_out.dtype
        )
        return self.o_proj(attn_out)
