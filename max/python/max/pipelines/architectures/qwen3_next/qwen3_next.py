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
from max.graph import DeviceRef, TensorValue, ops
from max.nn.legacy.layer import LayerList
from max.nn.legacy.linear import Linear
from max.nn.legacy.norm import RMSNorm

from max.nn.legacy.transformer import ReturnLogits

from max.pipelines.architectures.qwen3.qwen3 import (
    Qwen3,
    distribute_value,
    forward_sharded_layers,
)
from max.pipelines.architectures.qwen3_next.model_config import Qwen3NextConfig
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
    ) -> tuple:
        """Forward with hybrid layers. Creates zeros for linear state when not provided."""
        h = self.embed_tokens(tokens, signal_buffers)
        if self.embedding_multiplier != 1.0:
            h = [hi * self.embedding_multiplier for hi in h]

        freqs_cis = distribute_value(self.rope.freqs_cis, self.devices)
        input_row_offsets_list = distribute_value(
            input_row_offsets, self.devices
        )

        batch_dim = h[0].shape[0]
        conv_shape = [batch_dim] + list(self.config.get_linear_conv_state_shape())
        rec_shape = [batch_dim] + list(self.config.get_linear_recurrent_state_shape())
        zero = ops.constant(0.0, self.config.dtype or DType.float32, DeviceRef.CPU())
        device = self.devices[0]
        zeros_conv = ops.broadcast_to(zero.to(device), conv_shape)
        zeros_rec = ops.broadcast_to(zero.to(device), rec_shape)
        num_linear = len(self._linear_layer_indices)
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
            return (last_logits, logits, offsets)
        return (last_logits,)
