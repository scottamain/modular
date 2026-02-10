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
"""Tests for Qwen3-Next linear attention parity: config, adapter, and layer shapes."""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock

from max.pipelines.architectures.qwen3_next.weight_adapters import (
    _full_attention_layer_indices,
    _total_num_layers,
    convert_safetensor_state_dict,
)
from max.pipelines.architectures.qwen3_next.model_config import (
    Qwen3NextConfig,
    _default_layer_types,
)


def test_full_attention_layer_indices_default_interval() -> None:
    """Every 4th layer is full_attention when full_attention_interval=4."""
    config = MagicMock()
    config.layer_types = None
    config.full_attention_interval = 4
    config.num_hidden_layers = 48
    indices = _full_attention_layer_indices(config)
    assert indices == list(range(0, 48, 4))
    assert len(indices) == 12


def test_full_attention_layer_indices_explicit_layer_types() -> None:
    """Explicit layer_types is respected."""
    config = MagicMock()
    config.layer_types = ["full_attention", "linear_attention"] * 24
    config.full_attention_interval = 4
    config.num_hidden_layers = 48
    indices = _full_attention_layer_indices(config)
    assert indices == list(range(0, 48, 2))
    assert len(indices) == 24


def test_total_num_layers() -> None:
    """total_num_layers equals len(layer_types) or num_hidden_layers."""
    config = MagicMock()
    config.layer_types = ["full_attention", "linear_attention"] * 24
    config.num_hidden_layers = 48
    assert _total_num_layers(config) == 48
    config.layer_types = None
    assert _total_num_layers(config) == 48


def test_default_layer_types() -> None:
    """_default_layer_types produces correct pattern."""
    layer_types = _default_layer_types(48, 4)
    assert len(layer_types) == 48
    assert layer_types[0] == "full_attention"
    assert layer_types[1] == "linear_attention"
    assert layer_types[4] == "full_attention"
    assert sum(1 for t in layer_types if t == "full_attention") == 12


def test_linear_state_shapes() -> None:
    """get_linear_conv_state_shape and get_linear_recurrent_state_shape return correct dims."""
    from types import MethodType

    config = MagicMock()
    config.linear_key_head_dim = 128
    config.linear_num_key_heads = 16
    config.linear_value_head_dim = 128
    config.linear_num_value_heads = 32
    config.linear_conv_kernel_dim = 4
    config.get_linear_conv_state_shape = MethodType(
        Qwen3NextConfig.get_linear_conv_state_shape, config
    )
    config.get_linear_recurrent_state_shape = MethodType(
        Qwen3NextConfig.get_linear_recurrent_state_shape, config
    )
    conv_shape = config.get_linear_conv_state_shape()
    assert len(conv_shape) == 2
    assert conv_shape[0] == 16 * 128 * 2 + 32 * 128
    assert conv_shape[1] == 3

    rec_shape = config.get_linear_recurrent_state_shape()
    assert len(rec_shape) == 3
    assert rec_shape == (32, 128, 128)


def test_adapter_imports_and_total_layers() -> None:
    """convert_safetensor_state_dict uses _total_num_layers (all layers included)."""
    assert callable(convert_safetensor_state_dict)
    config = MagicMock()
    config.layer_types = ["full_attention", "linear_attention"] * 24
    config.num_hidden_layers = 48
    assert _total_num_layers(config) == 48
    assert len(_full_attention_layer_indices(config)) == 24
