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
"""Qwen3-Next hybrid decoder: full-attention and linear-attention (Gated DeltaNet) layers."""

from __future__ import annotations

import functools

from max.dtype import DType
from max.graph import BufferType, DeviceRef, TensorType, TensorValue, ops
from max.nn.legacy.kv_cache import KVCacheParams
from max.nn.legacy.layer import LayerList
from max.nn.legacy.linear import Linear
from max.nn.legacy.norm import RMSNorm

from max.nn.legacy.transformer import ReturnLogits

from max.graph import ShardingStrategy

from max.pipelines.architectures.qwen3.qwen3 import (
    Qwen3,
    distribute_value,
    forward_sharded_layers,
)
from max.pipelines.architectures.qwen3_next.model_config import Qwen3NextConfig
from max.pipelines.architectures.qwen3_next.layers.gated_attention import (
    Qwen3NextGatedAttention,
)
from max.pipelines.architectures.qwen3_next.layers.linear_block import (
    Qwen3NextLinearBlock,
)


class Qwen3Next(Qwen3):
    """Qwen3-Next hybrid decoder: full-attention and linear-attention layers in order.

    Builds total_num_layers blocks (e.g. 48). Full-attention blocks use KV cache;
    linear blocks use Gated DeltaNet with conv/recurrent state (zeros for first step).
    """

    def __init__(self, config: Qwen3NextConfig) -> None:
        if not isinstance(config, Qwen3NextConfig):
            raise TypeError("Qwen3Next requires Qwen3NextConfig")
        super().__init__(config)

        # TODO(partial-rope): Qwen3-Coder-Next uses partial_rotary_factor=0.25
        # (only 64 of 256 head dims get RoPE). The fused QK RoPE kernel applies
        # partial RoPE to the LAST dims (DeepSeek-style) with interleaved
        # layout, while Qwen3-Next needs it on the FIRST dims with
        # non-interleaved layout.  A proper fix requires rearranging Q/K weight
        # columns so rotary dims come last in interleaved order.  For now we
        # use full-dim RoPE so the model compiles; accuracy will be addressed
        # in a follow-up.
        if config.partial_rotary_factor < 1.0:
            import logging

            logging.getLogger(__name__).warning(
                "partial_rotary_factor=%.2f is not yet supported by the "
                "fused QK RoPE kernel; using full-dim RoPE as a workaround. "
                "This will reduce accuracy until partial RoPE is properly "
                "implemented.",
                config.partial_rotary_factor,
            )

        create_norm = functools.partial(
            RMSNorm,
            config.hidden_size,
            dtype=config.norm_dtype or DType.float32,
            eps=config.rms_norm_eps,
            multiply_before_cast=False,
        )
        linear_cls = functools.partial(
            Linear, float8_config=config.float8_config
        )

        # Upgrade full-attention blocks from Qwen3Attention to
        # Qwen3NextGatedAttention (query-dependent sigmoid gate).
        num_devices = len(config.devices)
        for block in self.layers:
            gated_attn = Qwen3NextGatedAttention(
                num_attention_heads=config.num_attention_heads,
                num_key_value_heads=config.num_key_value_heads,
                hidden_size=config.hidden_size,
                kv_params=config.kv_params,
                layer_idx=block.self_attn.layer_idx,
                dtype=config.dtype,
                rope=self.rope,
                linear_cls=linear_cls,
                devices=config.devices,
                scale=config.attention_multiplier,
                has_bias=config.attention_bias,
            )
            gated_attn.sharding_strategy = ShardingStrategy.tensor_parallel(
                num_devices
            )
            # Copy existing projection weights (q/k/v/o) from the original
            # attention -- they share the same names so load_state_dict will
            # fill them in.  Replace the block's attention with the gated one.
            block.self_attn = gated_attn
            block.self_attn_shards = gated_attn.shard(config.devices)

        new_layers = []
        kv_idx = 0
        for i in range(config.total_num_layers):
            if config.layer_types[i] == "full_attention":
                new_layers.append(self.layers[kv_idx])
                kv_idx += 1
            else:
                new_layers.append(
                    Qwen3NextLinearBlock(
                        config, i, create_norm, linear_cls
                    )
                )
        self.layers = LayerList(new_layers)
        self._layer_types = config.layer_types
        self._linear_layer_indices = [
            i for i, t in enumerate(config.layer_types)
            if t == "linear_attention"
        ]

    def __call__(
        self,
        tokens,
        kv_collections,
        return_n_logits,
        input_row_offsets,
        signal_buffers,
        conv_states=None,
        recurrent_states=None,
    ) -> tuple:
        """Forward with hybrid layers. Uses conv_states/recurrent_states when provided, else zeros."""
        h = self.embed_tokens(tokens, signal_buffers)
        if self.embedding_multiplier != 1.0:
            h = [hi * self.embedding_multiplier for hi in h]

        freqs_cis = distribute_value(self.rope.freqs_cis, self.devices)
        input_row_offsets_list = distribute_value(
            input_row_offsets, self.devices
        )

        num_linear = len(self._linear_layer_indices)
        assert isinstance(self.config, Qwen3NextConfig)
        if conv_states is None or recurrent_states is None:
            # GatedDeltaNet internally lifts ragged 2D input to [1, total_seq, hidden],
            # so conv/recurrent states must have batch_dim=1.
            batch_dim = 1
            conv_shape = [batch_dim] + list(self.config.get_linear_conv_state_shape())
            rec_shape = [batch_dim] + list(self.config.get_linear_recurrent_state_shape())
            zero = ops.constant(0.0, self.config.dtype or DType.float32, DeviceRef.CPU())
            device = self.devices[0]
            zeros_conv = ops.broadcast_to(zero.to(device), conv_shape)
            zeros_rec = ops.broadcast_to(zero.to(device), rec_shape)
            conv_states = [zeros_conv] * num_linear
            recurrent_states = [zeros_rec] * num_linear

        linear_idx = 0
        for i, layer in enumerate(self.layers):
            layer_idx = ops.constant(i, DType.uint32, device=DeviceRef.CPU())
            if self._layer_types[i] == "full_attention":
                h = layer(
                    layer_idx,
                    h,
                    kv_collections,
                    freqs_cis,
                    input_row_offsets_list,
                    signal_buffers,
                )
            else:
                h, new_conv, new_rec = layer(
                    layer_idx,
                    h,
                    kv_collections,
                    freqs_cis,
                    input_row_offsets_list,
                    signal_buffers,
                    conv_state=conv_states[linear_idx],
                    recurrent_state=recurrent_states[linear_idx],
                )
                conv_states[linear_idx] = new_conv
                recurrent_states[linear_idx] = new_rec
                linear_idx += 1

        h0 = h[0]
        last_token_indices = input_row_offsets[1:] - 1
        last_token_h = ops.gather(h0, last_token_indices, axis=0)
        last_token_distributed = distribute_value(last_token_h, self.devices)
        norm_last_token = forward_sharded_layers(
            self.norm_shards, last_token_distributed
        )
        last_logits = ops.cast(
            self.lm_head(norm_last_token, signal_buffers)[0],
            DType.float32,
        )

        logits = None
        offsets = None
        if self.return_logits == ReturnLogits.VARIABLE:
            return_n_logits_range = ops.range(
                start=return_n_logits[0],
                stop=0,
                step=-1,
                out_dim="return_n_logits_range",
                dtype=DType.int64,
                device=self.devices[0],
            )
            computed_offsets = (
                ops.unsqueeze(input_row_offsets[1:], -1) - return_n_logits_range
            )
            last_indices = ops.reshape(computed_offsets, shape=(-1,))
            variable_tokens = [
                ops.gather(h_device, last_indices, axis=0) for h_device in h
            ]
            variable_normed = forward_sharded_layers(
                self.norm_shards, variable_tokens
            )
            logits = ops.cast(
                self.lm_head(variable_normed, signal_buffers)[0],
                DType.float32,
            )
            offsets = ops.range(
                0,
                TensorValue(last_indices.shape[0]) + return_n_logits[0],
                return_n_logits[0],
                out_dim="logit_offsets",
                dtype=DType.int64,
                device=self.devices[0],
            )
        elif self.return_logits == ReturnLogits.ALL:
            all_normalized = forward_sharded_layers(self.norm_shards, h)
            logits = ops.cast(
                self.lm_head(all_normalized, signal_buffers)[0],
                DType.float32,
            )
            offsets = input_row_offsets

        if logits is not None and offsets is not None:
            logits_tuple = (last_logits, logits, offsets)
        else:
            logits_tuple = (last_logits,)
        return (logits_tuple, conv_states, recurrent_states)

    def input_types(
        self, kv_params: KVCacheParams
    ) -> tuple[TensorType | BufferType, ...]:
        """Add linear conv/recurrent state input types for graph I/O."""
        base = super().input_types(kv_params)
        num_linear = len(self._linear_layer_indices)
        assert isinstance(self.config, Qwen3NextConfig)
        dtype = self.config.dtype or DType.float32
        device = self.devices[0]
        conv_shape = self.config.get_linear_conv_state_shape()
        rec_shape = self.config.get_linear_recurrent_state_shape()
        state_types: list[TensorType | BufferType] = []
        # Use concrete batch=1 since the model unsqueezes ragged 2D input
        # to [1, total_seq, hidden] internally. Symbolic "batch_size" would
        # conflict with the concrete 1 during concat.
        for _ in range(num_linear):
            state_types.append(
                TensorType(
                    dtype,
                    shape=[1, conv_shape[0], conv_shape[1]],
                    device=device,
                )
            )
        for _ in range(num_linear):
            state_types.append(
                TensorType(
                    dtype,
                    shape=[1, rec_shape[0], rec_shape[1], rec_shape[2]],
                    device=device,
                )
            )
        return base + tuple(state_types)
