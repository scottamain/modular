# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #
"""Weight adapters for Qwen3-Next: map HF state dict to full-attention layers only."""

from __future__ import annotations

import re
from collections import defaultdict

import numpy as np
from max.graph.weights import WeightData, Weights
from max.pipelines.lib import PipelineConfig
from transformers import AutoConfig

from ..qwen3.weight_adapters import (
    _numpy_to_weight_data,
    _weight_data_to_numpy,
    QWEN3_MOE_SAFETENSOR_MAPPING,
)


def _full_attention_layer_indices(huggingface_config: AutoConfig) -> list[int]:
    """Return list of layer indices that use full (standard) attention."""
    layer_types = getattr(huggingface_config, "layer_types", None)
    if layer_types is not None:
        return [i for i, t in enumerate(layer_types) if t == "full_attention"]
    interval = getattr(huggingface_config, "full_attention_interval", 4)
    n = huggingface_config.num_hidden_layers
    return [i for i in range(n) if i % interval == 0]


def convert_safetensor_state_dict(
    state_dict: dict[str, Weights],
    huggingface_config: AutoConfig,
    pipeline_config: PipelineConfig,
    **unused_kwargs,
) -> dict[str, WeightData]:
    """Convert Qwen3-Next HF state dict to MAX format (full-attention layers only).

    Copies only weights for full_attention layers; linear_attention layer
    weights are skipped. Remaps layer indices so our layers.0, layers.1, ...
    correspond to HF full_attention layers. Handles MoE expert stacking
    the same way as Qwen3-MoE for layers that use MoE.
    """
    full_attention_indices = _full_attention_layer_indices(huggingface_config)
    hf_to_our = {hf_idx: our_idx for our_idx, hf_idx in enumerate(full_attention_indices)}

    # Pattern for expert weights (same as Qwen3-MoE)
    expert_pattern = re.compile(
        r"model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight"
    )

    new_state_dict: dict[str, WeightData] = {}
    expert_weights: dict[int, dict[int, dict[str, WeightData]]] = defaultdict(
        lambda: defaultdict(dict)
    )

    for safetensor_name, value in state_dict.items():
        rest = (
            safetensor_name[6:]
            if safetensor_name.startswith("model.")
            else safetensor_name
        )

        # Layer key: layers.{hf_idx}.xxx
        layer_match = re.match(r"layers\.(\d+)\.", rest)
        if layer_match:
            hf_idx = int(layer_match.group(1))
            if hf_idx not in hf_to_our:
                continue  # Skip linear_attention layers
            our_idx = hf_to_our[hf_idx]
            max_name = "layers." + str(our_idx) + "." + rest[len(layer_match.group(0)) :]
        else:
            max_name = rest

        # Expert weights: collect for stacking later (skip adding here)
        expert_match = expert_pattern.match(safetensor_name)
        if expert_match:
            layer_idx = int(expert_match.group(1))
            if layer_idx not in hf_to_our:
                continue
            our_layer_idx = hf_to_our[layer_idx]
            expert_idx = int(expert_match.group(2))
            proj_type = expert_match.group(3)
            expert_weights[our_layer_idx][expert_idx][proj_type] = value.data()
            continue

        # Apply Qwen3 MoE name mapping for mlp (e.g. gate -> gate.gate_score)
        for before, after in QWEN3_MOE_SAFETENSOR_MAPPING.items():
            if before in max_name:
                max_name = max_name.replace(before, after)
                break

        new_state_dict[max_name] = value.data()

    # Stack expert weights for each (our) layer that has MoE (same as Qwen3)
    for our_idx in sorted(expert_weights.keys()):
        experts = expert_weights[our_idx]
        num_experts = len(experts)
        gate_projs = []
        up_projs = []
        down_projs = []
        first_expert = experts[0]
        original_dtype = first_expert["gate_proj"].dtype

        for expert_idx in range(num_experts):
            expert_data = experts[expert_idx]
            gate_projs.append(_weight_data_to_numpy(expert_data["gate_proj"]))
            up_projs.append(_weight_data_to_numpy(expert_data["up_proj"]))
            down_projs.append(_weight_data_to_numpy(expert_data["down_proj"]))

        stacked_gate = np.stack(gate_projs, axis=0)
        stacked_up = np.stack(up_projs, axis=0)
        stacked_gate = np.transpose(stacked_gate, (0, 2, 1))
        stacked_up = np.transpose(stacked_up, (0, 2, 1))
        gate_up_proj = np.concatenate([stacked_gate, stacked_up], axis=2)
        gate_up_proj = np.ascontiguousarray(gate_up_proj)
        gate_up_name = f"layers.{our_idx}.mlp.experts.gate_up_proj"
        new_state_dict[gate_up_name] = _numpy_to_weight_data(
            gate_up_proj, gate_up_name, original_dtype
        )

        stacked_down = np.stack(down_projs, axis=0)
        stacked_down = np.transpose(stacked_down, (0, 2, 1))
        stacked_down = np.ascontiguousarray(stacked_down)
        down_name = f"layers.{our_idx}.mlp.experts.down_proj"
        new_state_dict[down_name] = _numpy_to_weight_data(
            stacked_down, down_name, original_dtype
        )

    return new_state_dict
