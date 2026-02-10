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

from __future__ import annotations

import logging

from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph
from max.graph.weights import Weights, WeightsAdapter
from max.nn.legacy.kv_cache import KVCacheParams
from max.pipelines.lib import KVCacheConfig, PipelineConfig
from max.pipelines.lib.interfaces import AlwaysSignalBuffersMixin
from transformers import AutoConfig

from ..qwen3.model import Qwen3Model
from .model_config import Qwen3NextConfig
from .qwen3_next import Qwen3Next

logger = logging.getLogger("max.pipelines")


class Qwen3NextModel(AlwaysSignalBuffersMixin, Qwen3Model):
    """Qwen3-Next pipeline model: full-attention layers only, same pipeline as Qwen3."""

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
                not getattr(self.huggingface_config, "tie_word_embeddings", False)
            ),
        )

        self.state_dict = nn_model.state_dict()
        num_devices = len(self.devices)
        kv_inputs = self.kv_params.get_symbolic_inputs()
        flattened_kv = [
            kv_type for sublist in kv_inputs for kv_type in sublist
        ]
        num_kv_inputs = len(flattened_kv)
        num_linear = sum(
            1 for t in model_config.layer_types if t == "linear_attention"
        )
        num_state_inputs = 2 * num_linear

        with Graph("qwen3_next", input_types=graph_inputs) as graph:
            tokens, input_row_offsets, return_n_logits, *variadic_args = (
                graph.inputs
            )
            signal_buffers = [v.buffer for v in variadic_args[:num_devices]]
            kv_cache_inputs = variadic_args[num_devices : num_devices + num_kv_inputs]
            state_inputs = variadic_args[num_devices + num_kv_inputs :]
            kv_collections = self._unflatten_kv_inputs(kv_cache_inputs)

            conv_states = [v.tensor for v in state_inputs[:num_linear]]
            recurrent_states = [
                v.tensor for v in state_inputs[num_linear:num_state_inputs]
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
