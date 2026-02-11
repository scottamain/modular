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
"""Qwen3-Next pipeline model with hybrid state management.

Handles conv/recurrent state I/O for GatedDeltaNet linear-attention layers
alongside the standard KV cache for full-attention layers.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

import numpy as np
from max.driver import Buffer
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph
from max.graph.weights import Weights, WeightsAdapter
from max.nn.legacy.kv_cache import KVCacheInputs, KVCacheParams
from max.nn.legacy.transformer import ReturnHiddenStates, ReturnLogits
from max.pipelines.core import TextContext
from max.pipelines.lib import (
    KVCacheConfig,
    ModelInputs,
    ModelOutputs,
    PipelineConfig,
)
from max.support.algorithm import flatten2d
from transformers import AutoConfig

from ..llama3.model import Llama3Inputs, LlamaModelBase
from ..qwen3.model import Qwen3Model
from .model_config import Qwen3NextConfig
from .qwen3_next import Qwen3Next

logger = logging.getLogger("max.pipelines")


# Map MAX DType to numpy dtype for zero-buffer creation.
_DTYPE_TO_NP = {
    DType.float32: np.float32,
    DType.bfloat16: np.float32,  # Create as float32, will be cast on device.
    DType.float16: np.float16,
}


class Qwen3NextModel(Qwen3Model):
    """Qwen3-Next pipeline model with hybrid state management.

    Manages both KV cache (full-attention layers) and conv/recurrent state
    (linear-attention layers via GatedDeltaNet).
    """

    # Number of linear-attention layers in the model.
    _num_linear_layers: int = 0

    # Conv and recurrent state shapes (without batch dim).
    _conv_state_shape: tuple[int, ...] = ()
    _recurrent_state_shape: tuple[int, ...] = ()

    # Stored state buffers from the last execution step. Each list has
    # _num_linear_layers elements. Initialized to zero on first use.
    _conv_states: list[Buffer] | None = None
    _recurrent_states: list[Buffer] | None = None

    @classmethod
    def get_kv_params(
        cls,
        huggingface_config: AutoConfig,
        pipeline_config: PipelineConfig,
        devices: list[DeviceRef],
        kv_cache_config: KVCacheConfig,
        cache_dtype: DType,
    ) -> KVCacheParams:
        return Qwen3NextConfig.construct_kv_params(
            huggingface_config=huggingface_config,
            pipeline_config=pipeline_config,
            devices=devices,
            kv_cache_config=kv_cache_config,
            cache_dtype=cache_dtype,
        )

    def _build_graph(
        self,
        weights: Weights,
        adapter: WeightsAdapter | None = None,
        session: InferenceSession | None = None,
    ) -> Graph:
        state_dict = self._get_state_dict(weights, adapter)
        model_config = Qwen3NextConfig.initialize_from_config(
            self.pipeline_config, self.huggingface_config
        )
        model_config.finalize(
            huggingface_config=self.huggingface_config,
            state_dict=state_dict,
            return_logits=self.return_logits,
            norm_method=self.norm_method,
            attention_bias=self.attention_bias,
        )

        nn_model = Qwen3Next(model_config)
        graph_inputs = nn_model.input_types(self.kv_params)

        nn_model.load_state_dict(
            state_dict,
            override_quantization_encoding=True,
            weight_alignment=1,
            strict=(
                not getattr(
                    self.huggingface_config, "tie_word_embeddings", False
                )
            ),
        )

        self.state_dict = nn_model.state_dict()

        # Cache state metadata for execute/prepare methods.
        num_linear = sum(
            1 for t in model_config.layer_types if t == "linear_attention"
        )
        self._num_linear_layers = num_linear
        self._conv_state_shape = model_config.get_linear_conv_state_shape()
        self._recurrent_state_shape = (
            model_config.get_linear_recurrent_state_shape()
        )

        num_devices = len(self.devices)
        kv_inputs = self.kv_params.get_symbolic_inputs()
        flattened_kv = [
            kv_type for sublist in kv_inputs for kv_type in sublist
        ]
        num_kv_inputs = len(flattened_kv)
        num_state_inputs = 2 * num_linear

        with Graph("qwen3_next", input_types=graph_inputs) as graph:
            tokens, input_row_offsets, return_n_logits, *variadic_args = (
                graph.inputs
            )
            signal_buffers = [v.buffer for v in variadic_args[:num_devices]]
            kv_cache_inputs = variadic_args[
                num_devices : num_devices + num_kv_inputs
            ]
            state_inputs = variadic_args[num_devices + num_kv_inputs :]
            kv_collections = self._unflatten_kv_inputs(kv_cache_inputs)

            conv_states = [v.tensor for v in state_inputs[:num_linear]]
            recurrent_states = [
                v.tensor
                for v in state_inputs[num_linear:num_state_inputs]
            ]

            logits_tuple, new_conv_states, new_recurrent_states = nn_model(
                tokens.tensor,
                kv_collections,
                return_n_logits.tensor,
                input_row_offsets.tensor,
                signal_buffers,
                conv_states=conv_states,
                recurrent_states=recurrent_states,
            )
            graph.output(
                *logits_tuple,
                *new_conv_states,
                *new_recurrent_states,
            )
        return graph

    def _create_zero_states(self, batch_dim: int = 1) -> None:
        """Create zero-filled conv and recurrent state buffers on GPU.

        The dtype matches the graph's state input type (config.dtype, typically
        bfloat16). Since numpy doesn't support bfloat16, we create float32
        buffers and cast via the device transfer.
        """
        device = self.devices[0]
        conv_shape = (batch_dim,) + self._conv_state_shape
        rec_shape = (batch_dim,) + self._recurrent_state_shape

        self._conv_states = []
        self._recurrent_states = []
        for _ in range(self._num_linear_layers):
            buf = Buffer.from_numpy(
                np.zeros(conv_shape, dtype=np.float32)
            )
            self._conv_states.append(buf.to(device))
        for _ in range(self._num_linear_layers):
            buf = Buffer.from_numpy(
                np.zeros(rec_shape, dtype=np.float32)
            )
            self._recurrent_states.append(buf.to(device))

    def execute(self, model_inputs: ModelInputs) -> ModelOutputs:
        """Execute the model with conv/recurrent state management.

        Passes stored state buffers as additional inputs after the KV cache
        inputs. Captures new state outputs and stores them for the next step.
        """
        curr_kv_cache_inputs = model_inputs.kv_cache_inputs or ()
        assert isinstance(model_inputs, Llama3Inputs)

        # Ensure states are initialized.
        if self._conv_states is None or self._recurrent_states is None:
            self._create_zero_states(batch_dim=1)
        assert self._conv_states is not None
        assert self._recurrent_states is not None

        # Build execute args: tokens, row_offsets, return_n_logits,
        # *signal_buffers, *kv_cache, *conv_states, *recurrent_states
        model_outputs = self.model.execute(
            model_inputs.tokens,
            model_inputs.input_row_offsets,
            model_inputs.return_n_logits,
            *model_inputs.signal_buffers,
            *curr_kv_cache_inputs,
            *self._conv_states,
            *self._recurrent_states,
        )

        # Parse outputs: logits first, then states.
        # Logit outputs: 1 (last_logits only) or 3 (last_logits, logits, offsets)
        has_offsets = self.return_logits in (
            ReturnLogits.VARIABLE,
            ReturnLogits.ALL,
        )
        has_hidden_states = (
            self.return_hidden_states != ReturnHiddenStates.NONE
        )

        if has_offsets and has_hidden_states:
            num_logit_outputs = 4
        elif has_offsets:
            num_logit_outputs = 3
        elif has_hidden_states:
            num_logit_outputs = 2
        else:
            num_logit_outputs = 1

        logit_outputs = model_outputs[:num_logit_outputs]
        state_outputs = model_outputs[num_logit_outputs:]

        # Store new states for next step.
        n = self._num_linear_layers
        if len(state_outputs) >= 2 * n:
            self._conv_states = list(state_outputs[:n])
            self._recurrent_states = list(state_outputs[n : 2 * n])

        # Build ModelOutputs from logit outputs (same logic as base class).
        assert isinstance(logit_outputs[0], Buffer)
        if has_offsets and has_hidden_states:
            return ModelOutputs(
                logits=logit_outputs[1],
                next_token_logits=logit_outputs[0],
                logit_offsets=logit_outputs[2],
                hidden_states=logit_outputs[3],
            )
        elif has_offsets:
            return ModelOutputs(
                logits=logit_outputs[1],
                next_token_logits=logit_outputs[0],
                logit_offsets=logit_outputs[2],
            )
        elif has_hidden_states:
            return ModelOutputs(
                logits=logit_outputs[0],
                next_token_logits=logit_outputs[0],
                hidden_states=logit_outputs[1],
            )
        else:
            return ModelOutputs(
                logits=logit_outputs[0],
                next_token_logits=logit_outputs[0],
            )

    def prepare_initial_token_inputs(
        self,
        replica_batches: Sequence[Sequence[TextContext]],
        kv_cache_inputs: KVCacheInputs | None = None,
        return_n_logits: int = 1,
    ) -> Llama3Inputs:
        """Prepare initial inputs and reset conv/recurrent states to zero."""
        # Reset states to zero at the start of a new generation.
        self._create_zero_states(batch_dim=1)
        return super().prepare_initial_token_inputs(
            replica_batches, kv_cache_inputs, return_n_logits
        )
