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
"""Config for Qwen3-Next (Qwen3NextForCausalLM) models."""

from __future__ import annotations

from dataclasses import dataclass, field

from max.dtype import DType
from max.graph import DeviceRef
from max.nn.legacy.kv_cache import KVCacheParams
from max.pipelines.lib import KVCacheConfig, PipelineConfig
from transformers.models.auto.configuration_auto import AutoConfig
from typing_extensions import Self, override

from ..qwen3.model_config import Qwen3Config


def _default_layer_types(num_hidden_layers: int, full_attention_interval: int) -> list[str]:
    """Build layer_types when not in config: every full_attention_interval-th layer is full_attention."""
    return [
        "full_attention" if i % full_attention_interval == 0 else "linear_attention"
        for i in range(num_hidden_layers)
    ]


@dataclass(kw_only=True)
class Qwen3NextConfig(Qwen3Config):
    """Configuration for Qwen3-Next (hybrid attention + MoE) models."""

    # Linear / hybrid attention (Qwen3-Next specific)
    linear_key_head_dim: int = 128
    linear_num_key_heads: int = 16
    linear_value_head_dim: int = 128
    linear_num_value_heads: int = 32
    linear_conv_kernel_dim: int = 4
    full_attention_interval: int = 4
    layer_types: list[str] = field(default_factory=list)
    shared_expert_intermediate_size: int = 512
    partial_rotary_factor: float = 1.0

    # Indices of layers that use full (standard) attention; used for decoder and KV cache.
    full_attention_layer_indices: list[int] = field(default_factory=list)

    # Total number of layers (full + linear). Used by hybrid decoder; num_hidden_layers stays full-attention count until then.
    total_num_layers: int = 0

    @staticmethod
    def get_num_layers(huggingface_config: AutoConfig) -> int:
        """Number of decoder layers (full-attention only for current implementation)."""
        layer_types = getattr(huggingface_config, "layer_types", None)
        if layer_types is not None:
            return sum(1 for t in layer_types if t == "full_attention")
        interval = getattr(huggingface_config, "full_attention_interval", 4)
        n = huggingface_config.num_hidden_layers
        return sum(1 for i in range(n) if i % interval == 0)

    @staticmethod
    def construct_kv_params(
        huggingface_config: AutoConfig,
        pipeline_config: PipelineConfig,
        devices: list[DeviceRef],
        kv_cache_config: KVCacheConfig,
        cache_dtype: DType,
    ) -> KVCacheParams:
        """KV params with num_layers = number of full-attention layers."""
        data_parallel_degree = pipeline_config.model.data_parallel_degree
        if data_parallel_degree > 1:
            raise ValueError(
                "Data parallelism is not supported for Qwen3-Next models"
            )
        num_layers = Qwen3NextConfig.get_num_layers(huggingface_config)
        return KVCacheParams(
            dtype=cache_dtype,
            n_kv_heads=huggingface_config.num_key_value_heads,
            head_dim=huggingface_config.head_dim,
            num_layers=num_layers,
            page_size=kv_cache_config.kv_cache_page_size,
            cache_strategy=kv_cache_config.cache_strategy,
            enable_prefix_caching=kv_cache_config.enable_prefix_caching,
            enable_kvcache_swapping_to_host=kv_cache_config.enable_kvcache_swapping_to_host,
            host_kvcache_swap_space_gb=kv_cache_config.host_kvcache_swap_space_gb,
            devices=devices,
            data_parallel_degree=data_parallel_degree,
        )

    @override
    @classmethod
    def initialize_from_config(
        cls, pipeline_config: PipelineConfig, huggingface_config: AutoConfig
    ) -> Self:
        """Initialize Qwen3NextConfig from pipeline and HuggingFace configs."""
        base_config = Qwen3Config.initialize_from_config(
            pipeline_config, huggingface_config
        )

        # Layer types: from config or derive from full_attention_interval
        layer_types = getattr(huggingface_config, "layer_types", None)
        if layer_types is None:
            interval = getattr(huggingface_config, "full_attention_interval", 4)
            layer_types = _default_layer_types(
                huggingface_config.num_hidden_layers, interval
            )
        full_attention_indices = [
            i for i, t in enumerate(layer_types) if t == "full_attention"
        ]
        total_num_layers = len(layer_types)

        linear_key_head_dim = getattr(huggingface_config, "linear_key_head_dim", 128)
        linear_num_key_heads = getattr(huggingface_config, "linear_num_key_heads", 16)
        linear_value_head_dim = getattr(
            huggingface_config, "linear_value_head_dim", 128
        )
        linear_num_value_heads = getattr(
            huggingface_config, "linear_num_value_heads", 32
        )
        linear_conv_kernel_dim = getattr(
            huggingface_config, "linear_conv_kernel_dim", 4
        )
        full_attention_interval = getattr(
            huggingface_config, "full_attention_interval", 4
        )
        shared_expert_intermediate_size = getattr(
            huggingface_config, "shared_expert_intermediate_size", 512
        )
        partial_rotary_factor = getattr(
            huggingface_config, "partial_rotary_factor", 1.0
        )

        # KV params use reduced num_layers (full-attention only)
        kv_cache_config = pipeline_config.model.kv_cache
        quantization_encoding = pipeline_config.model.quantization_encoding
        if quantization_encoding is None:
            raise ValueError("quantization_encoding must not be None")
        cache_dtype = pipeline_config.model.kv_cache.cache_dtype
        n_devices = len(pipeline_config.model.device_specs)
        device_refs = [
            DeviceRef(spec.device_type, spec.id)
            for spec in pipeline_config.model.device_specs[:n_devices]
        ]
        qwen3_next_kv_params = Qwen3NextConfig.construct_kv_params(
            huggingface_config=huggingface_config,
            pipeline_config=pipeline_config,
            devices=device_refs,
            kv_cache_config=kv_cache_config,
            cache_dtype=cache_dtype,
        )

        # mlp_only_layers in reduced space: original HF indices -> our layer index
        hf_to_our = {hf_idx: our_idx for our_idx, hf_idx in enumerate(full_attention_indices)}
        mlp_only_reduced = [
            hf_to_our[hf_idx]
            for hf_idx in base_config.mlp_only_layers
            if hf_idx in hf_to_our
        ]

        return cls(
            hidden_size=base_config.hidden_size,
            num_attention_heads=base_config.num_attention_heads,
            num_key_value_heads=base_config.num_key_value_heads,
            num_hidden_layers=len(full_attention_indices),
            rope_theta=base_config.rope_theta,
            rope_scaling_params=base_config.rope_scaling_params,
            rms_norm_eps=base_config.rms_norm_eps,
            intermediate_size=base_config.intermediate_size,
            interleaved_rope_weights=base_config.interleaved_rope_weights,
            vocab_size=base_config.vocab_size,
            dtype=base_config.dtype,
            model_quantization_encoding=base_config.model_quantization_encoding,
            quantization_config=base_config.quantization_config,
            max_seq_len=base_config.max_seq_len,
            kv_params=qwen3_next_kv_params,
            attention_multiplier=base_config.attention_multiplier,
            embedding_multiplier=base_config.embedding_multiplier,
            residual_multiplier=base_config.residual_multiplier,
            devices=base_config.devices,
            clip_qkv=base_config.clip_qkv,
            use_subgraphs=base_config.use_subgraphs,
            dist_gemm_config=base_config.dist_gemm_config,
            num_experts=base_config.num_experts,
            num_experts_per_tok=base_config.num_experts_per_tok,
            moe_intermediate_size=base_config.moe_intermediate_size,
            mlp_only_layers=mlp_only_reduced,
            norm_topk_prob=base_config.norm_topk_prob,
            decoder_sparse_step=base_config.decoder_sparse_step,
            linear_key_head_dim=linear_key_head_dim,
            linear_num_key_heads=linear_num_key_heads,
            linear_value_head_dim=linear_value_head_dim,
            linear_num_value_heads=linear_num_value_heads,
            linear_conv_kernel_dim=linear_conv_kernel_dim,
            full_attention_interval=full_attention_interval,
            layer_types=layer_types,
            shared_expert_intermediate_size=shared_expert_intermediate_size,
            partial_rotary_factor=partial_rotary_factor,
            full_attention_layer_indices=full_attention_indices,
            total_num_layers=total_num_layers,
        )

    def get_linear_conv_state_shape(self, batch_dim: int | None = None) -> tuple[int, ...]:
        """Shape of conv_state for one linear layer: (batch, conv_dim, conv_kernel_size-1)."""
        conv_dim = (
            self.linear_key_head_dim * self.linear_num_key_heads * 2
            + self.linear_value_head_dim * self.linear_num_value_heads
        )
        state_len = self.linear_conv_kernel_dim - 1
        if batch_dim is not None:
            return (batch_dim, conv_dim, state_len)
        return (conv_dim, state_len)

    def get_linear_recurrent_state_shape(self, batch_dim: int | None = None) -> tuple[int, ...]:
        """Shape of recurrent_state for one linear layer: (batch, num_v_heads, head_k_dim, head_v_dim)."""
        if batch_dim is not None:
            return (
                batch_dim,
                self.linear_num_value_heads,
                self.linear_key_head_dim,
                self.linear_value_head_dim,
            )
        return (
            self.linear_num_value_heads,
            self.linear_key_head_dim,
            self.linear_value_head_dim,
        )
