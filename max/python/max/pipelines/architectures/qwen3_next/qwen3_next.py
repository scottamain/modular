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
"""Qwen3-Next decoder: full-attention layers only, same block structure as Qwen3.

The Hugging Face Qwen3-Next architecture alternates full-attention and
linear-attention layers. This implementation currently uses only the
full-attention layers (same as Qwen3 blocks) so that the model can load
and run; linear-attention layers are skipped and their weights are not
loaded. For full parity with HF, linear attention (Gated DeltaNet) would
need to be implemented separately.
"""

from __future__ import annotations

from max.pipelines.architectures.qwen3.qwen3 import Qwen3
from max.pipelines.architectures.qwen3_next.model_config import Qwen3NextConfig


class Qwen3Next(Qwen3):
    """Qwen3-Next decoder: full-attention layers only.

    Uses the same transformer block as Qwen3 (standard attention + MoE).
    Config must be Qwen3NextConfig with num_hidden_layers set to the
    number of full-attention layers and kv_params matching.
    """

    def __init__(self, config: Qwen3NextConfig) -> None:
        if not isinstance(config, Qwen3NextConfig):
            raise TypeError("Qwen3Next requires Qwen3NextConfig")
        super().__init__(config)
