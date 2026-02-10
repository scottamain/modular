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
"""Linear attention block (Gated DeltaNet + MLP) for Qwen3-Next hybrid decoder."""

from __future__ import annotations

from collections.abc import Callable

from max.graph import BufferValue, DeviceRef, TensorValue, ops
from max.nn.legacy.layer import Module
from max.nn.legacy.linear import Linear, MLP
from max.nn.legacy.norm import RMSNorm

from max.pipelines.architectures.qwen3_next.model_config import Qwen3NextConfig

from .linear_attention import GatedDeltaNet


class Qwen3NextLinearBlock(Module):
    """One decoder block: input norm, Gated DeltaNet, residual, norm, MLP, residual.

    Returns (h_list, new_conv_state, new_recurrent_state). Not sharded; runs on first device and replicates.
    """

    def __init__(
        self,
        config: Qwen3NextConfig,
        layer_idx: int,
        create_norm: Callable[..., RMSNorm],
        linear_cls: Callable[..., Linear],
    ) -> None:
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.devices = config.devices
        num_devices = len(config.devices)

        self.linear_attn = GatedDeltaNet(config, layer_idx)
        self.mlp = self._get_mlp(config, layer_idx, linear_cls)
        self.input_layernorm = create_norm()
        self.post_attention_layernorm = create_norm()
        self.residual_multiplier = config.residual_multiplier
        self.num_devices = num_devices

    def _get_mlp(
        self,
        config: Qwen3NextConfig,
        layer_idx: int,
        linear_cls: Callable[..., Linear],
    ) -> MLP:
        """MLP only (no MoE for linear blocks in this implementation)."""
        return MLP(
            config.dtype,
            config.model_quantization_encoding,
            config.hidden_size,
            config.intermediate_size,
            config.devices,
            linear_cls,
            float8_config=config.float8_config,
        )

    def __call__(
        self,
        layer_idx: TensorValue,
        xs: list[TensorValue],
        kv_collections: list,
        freqs_cis: list[TensorValue],
        input_row_offsets: list[TensorValue],
        signal_buffers: list[BufferValue],
        conv_state: TensorValue | None = None,
        recurrent_state: TensorValue | None = None,
    ) -> tuple[list[TensorValue], TensorValue, TensorValue]:
        """Forward. Uses first device hidden state; replicates output. Returns (h_list, new_conv_state, new_recurrent_state)."""
        h0 = xs[0]
        normed = self.input_layernorm(h0)
        attn_out, new_conv_state, new_recurrent_state = self.linear_attn(
            normed, conv_state=conv_state, recurrent_state=recurrent_state
        )
        h0 = ops.add(h0, ops.mul(attn_out, self.residual_multiplier))
        normed_mlp = self.post_attention_layernorm(h0)
        mlp_out = self.mlp(normed_mlp)
        if isinstance(mlp_out, list):
            mlp_out = mlp_out[0]
        h0 = ops.add(h0, mlp_out)
        h_list = [h0.to(device) for device in self.devices]
        return (h_list, new_conv_state, new_recurrent_state)
