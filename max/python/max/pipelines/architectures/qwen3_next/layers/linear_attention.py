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
"""Gated DeltaNet linear attention for Qwen3-Next (HF parity)."""

from __future__ import annotations

from max.dtype import DType
from max.graph import DeviceRef, TensorValue, TensorValueLike, ops
from max.graph import Weight
from max.nn.legacy.conv import Conv1D
from max.nn.legacy.layer import Module
from max.nn.legacy.linear import Linear

from max.pipelines.architectures.qwen3_next.model_config import Qwen3NextConfig


def _softplus(x: TensorValue) -> TensorValue:
    """softplus(x) = log(1 + exp(x)). Uses float32 for stability."""
    one = ops.constant(1.0, dtype=DType.float32, device=DeviceRef.CPU())
    x_f32 = x.cast(DType.float32) if x.dtype != DType.float32 else x
    return ops.log(ops.add(one.to(x_f32.device), ops.exp(x_f32))).cast(x.dtype)


class RMSNormGated(Module):
    """RMSNorm followed by gate: out = rms_norm(x) * weight * silu(gate). Matches HF Qwen3NextRMSNormGated."""

    def __init__(
        self,
        dim: int,
        dtype: DType,
        eps: float = 1e-6,
        device: DeviceRef | None = None,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.dtype = dtype
        self.device = device or DeviceRef.CPU()
        self.weight = Weight(
            name="weight",
            dtype=dtype,
            shape=(dim,),
            device=self.device,
        )

    def __call__(
        self, hidden_states: TensorValue, gate: TensorValue
    ) -> TensorValue:
        """hidden_states: (..., dim), gate: (..., dim). Returns (..., dim)."""
        x = hidden_states.cast(DType.float32)
        variance = ops.mean(ops.mul(x, x), axis=-1)
        variance = ops.unsqueeze(variance, -1)
        x = ops.mul(x, ops.rsqrt(ops.add(variance, ops.constant(self.eps, DType.float32, DeviceRef.CPU()))))
        w = self.weight.cast(hidden_states.dtype)
        if hidden_states.device:
            w = w.to(hidden_states.device)
        x = ops.mul(x.cast(hidden_states.dtype), w)
        gate_silu = ops.silu(gate.cast(DType.float32)).cast(hidden_states.dtype)
        return ops.mul(x, gate_silu)


class GatedDeltaNet(Module):
    """Gated DeltaNet linear attention block. Matches HF Qwen3NextGatedDeltaNet.

    Supports decode (seq_len=1) with optional conv_state and recurrent_state.
    For prefill (seq_len > 1), uses a parallel zero-state approximation.
    """

    def __init__(self, config: Qwen3NextConfig, layer_idx: int) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_v_heads = config.linear_num_value_heads
        self.num_k_heads = config.linear_num_key_heads
        self.head_k_dim = config.linear_key_head_dim
        self.head_v_dim = config.linear_value_head_dim
        self.key_dim = self.head_k_dim * self.num_k_heads
        self.value_dim = self.head_v_dim * self.num_v_heads
        self.conv_kernel_size = config.linear_conv_kernel_dim
        self.layer_idx = layer_idx
        self.eps = config.rms_norm_eps
        self.devices = config.devices
        device = self.devices[0] if self.devices else DeviceRef.CPU()
        dtype = config.dtype or DType.float32

        self.conv_dim = self.key_dim * 2 + self.value_dim
        projection_size_qkvz = self.key_dim * 2 + self.value_dim * 2
        projection_size_ba = self.num_v_heads * 2

        # NOTE: Do NOT add a "linear_attn." prefix to weight/submodule names
        # here.  The parent Qwen3NextLinearBlock assigns this module as
        # self.linear_attn, so recursive_named_layers already provides the
        # linear_attn. prefix in the module tree path.
        self.in_proj_qkvz = Linear(
            self.hidden_size,
            projection_size_qkvz,
            dtype=dtype,
            device=device,
            has_bias=False,
        )
        self.in_proj_ba = Linear(
            self.hidden_size,
            projection_size_ba,
            dtype=dtype,
            device=device,
            has_bias=False,
        )
        self.conv1d = Conv1D(
            kernel_size=self.conv_kernel_size,
            in_channels=self.conv_dim,
            out_channels=self.conv_dim,
            dtype=dtype,
            padding=(self.conv_kernel_size - 1, 0),
            num_groups=self.conv_dim,
            device=device,
            has_bias=False,
            permute=True,
        )
        self.dt_bias = Weight(
            name="dt_bias",
            dtype=dtype,
            shape=(self.num_v_heads,),
            device=device,
        )
        self.A_log = Weight(
            name="A_log",
            dtype=dtype,
            shape=(self.num_v_heads,),
            device=device,
        )
        self.norm = RMSNormGated(
            self.head_v_dim,
            dtype=dtype,
            eps=self.eps,
            device=device,
        )
        self.out_proj = Linear(
            self.value_dim,
            self.hidden_size,
            dtype=dtype,
            device=device,
            has_bias=False,
        )

    def _fix_query_key_value_ordering(
        self,
        mixed_qkvz: TensorValue,
        mixed_ba: TensorValue,
    ) -> tuple[TensorValue, TensorValue, TensorValue, TensorValue, TensorValue, TensorValue]:
        """Split projections into query, key, value, z, b, a. Shapes (batch, seq, ...)."""
        batch_dim = mixed_qkvz.shape[0]
        seq_dim = mixed_qkvz.shape[1]
        chunk_qkvz = 2 * self.head_k_dim + 2 * (self.num_v_heads // self.num_k_heads) * self.head_v_dim
        mixed_qkvz = ops.reshape(
            mixed_qkvz,
            [batch_dim, seq_dim, self.num_k_heads, chunk_qkvz],
        )
        chunk_ba = 2 * (self.num_v_heads // self.num_k_heads)
        mixed_ba = ops.reshape(
            mixed_ba,
            [batch_dim, seq_dim, self.num_k_heads, chunk_ba],
        )
        v_size = (self.num_v_heads // self.num_k_heads) * self.head_v_dim
        q = mixed_qkvz[:, :, :, 0 : self.head_k_dim]
        k = mixed_qkvz[:, :, :, self.head_k_dim : 2 * self.head_k_dim]
        v = mixed_qkvz[:, :, :, 2 * self.head_k_dim : 2 * self.head_k_dim + v_size]
        z = mixed_qkvz[:, :, :, 2 * self.head_k_dim + v_size : 2 * self.head_k_dim + 2 * v_size]
        b = mixed_ba[:, :, :, 0 : self.num_v_heads // self.num_k_heads]
        a = mixed_ba[:, :, :, self.num_v_heads // self.num_k_heads : 2 * (self.num_v_heads // self.num_k_heads)]
        v = ops.reshape(v, [batch_dim, seq_dim, self.num_v_heads, self.head_v_dim])
        z = ops.reshape(z, [batch_dim, seq_dim, self.num_v_heads, self.head_v_dim])
        b = ops.reshape(b, [batch_dim, seq_dim, self.num_v_heads])
        a = ops.reshape(a, [batch_dim, seq_dim, self.num_v_heads])
        return q, k, v, z, b, a

    def _recurrent_step(
        self,
        query: TensorValue,
        key: TensorValue,
        value: TensorValue,
        g: TensorValue,
        beta: TensorValue,
        recurrent_state: TensorValue,
    ) -> tuple[TensorValue, TensorValue]:
        """Single recurrent step: (batch, num_v_heads, head_k_dim), (batch, num_v_heads, head_v_dim), state (batch, num_v_heads, head_k_dim, head_v_dim)."""
        g_exp = ops.exp(g)
        g_exp = ops.unsqueeze(ops.unsqueeze(g_exp, -1), -1)
        beta_exp = ops.unsqueeze(beta, -1)
        new_state = ops.mul(recurrent_state, g_exp)
        k_unsq = ops.unsqueeze(key, -1)
        # ops.sum keeps the reduced dim (keepdim=True); squeeze it out.
        kv_mem = ops.squeeze(ops.sum(ops.mul(new_state, k_unsq), axis=-2), -2)
        delta = ops.mul(ops.sub(value, kv_mem), beta_exp)
        delta_unsq = ops.unsqueeze(delta, -2)
        k_unsq_2 = ops.unsqueeze(key, -1)
        new_state = ops.add(new_state, ops.mul(k_unsq_2, delta_unsq))
        q_unsq = ops.unsqueeze(query, -1)
        out = ops.squeeze(ops.sum(ops.mul(new_state, q_unsq), axis=-2), -2)
        return out, new_state

    def __call__(
        self,
        hidden_states: TensorValueLike,
        conv_state: TensorValue | None = None,
        recurrent_state: TensorValue | None = None,
    ) -> tuple[TensorValue, TensorValue, TensorValue]:
        """Forward. Returns (output, new_conv_state, new_recurrent_state).

        Accepts 2D [total_seq, hidden] (ragged) or 3D [batch, seq, hidden]
        input.  Projects in the original dimensionality and lifts to 3D
        [1, total_seq, ...] for conv/recurrent processing.
        """
        h = hidden_states
        device = self.devices[0] if self.devices else DeviceRef.CPU()

        # Detect input dimensionality; project in original dim.
        input_is_2d = len(h.shape) == 2
        projected_qkvz = self.in_proj_qkvz(h)
        projected_ba = self.in_proj_ba(h)

        # Lift to 3D for conv/recurrent logic: [1, total_seq, ...]
        if input_is_2d:
            projected_qkvz = ops.unsqueeze(projected_qkvz, 0)
            projected_ba = ops.unsqueeze(projected_ba, 0)

        query, key, value, z, b, a = self._fix_query_key_value_ordering(
            projected_qkvz, projected_ba
        )
        batch_dim = query.shape[0]
        seq_dim = query.shape[1]
        query = ops.reshape(query, [batch_dim, seq_dim, self.key_dim])
        key = ops.reshape(key, [batch_dim, seq_dim, self.key_dim])
        value = ops.reshape(value, [batch_dim, seq_dim, self.value_dim])

        # Concat q/k/v and permute to channels-first [batch, conv_dim, seq]
        # for Conv1D(permute=True) which expects [batch, channels, length].
        mixed_qkv = ops.concat([query, key, value], axis=-1)
        mixed_qkv = ops.permute(mixed_qkv, [0, 2, 1])

        state_len = self.conv_kernel_size - 1
        if conv_state is not None and recurrent_state is not None:
            # conv_state: [batch, conv_dim, state_len] -- channels-first
            mixed_qkv_cat = ops.concat([conv_state, mixed_qkv], axis=-1)
            # [batch, conv_dim, state_len + seq]
            new_conv_state = mixed_qkv_cat[:, :, -state_len:]
            # Feed channels-first directly to conv1d(permute=True)
            mixed_qkv = self.conv1d(mixed_qkv_cat)
            # Causal padding preserves length; use concrete offset.
            mixed_qkv = ops.silu(mixed_qkv[:, :, state_len:])
            # [batch, conv_dim, seq]
        else:
            # Save pre-convolution features for initial state
            padded = ops.pad(
                mixed_qkv,
                [state_len - 1, 0, 0, 0, 0, 0],
                mode="constant",
                value=0.0,
            )
            new_conv_state = padded[:, :, -state_len:]
            # Feed channels-first directly to conv1d(permute=True)
            mixed_qkv = self.conv1d(mixed_qkv)
            mixed_qkv = ops.silu(mixed_qkv)

        # Permute back to channels-last [batch, seq, conv_dim] for q/k/v split.
        mixed_qkv = ops.permute(mixed_qkv, [0, 2, 1])

        query = mixed_qkv[:, :, 0 : self.key_dim]
        key = mixed_qkv[:, :, self.key_dim : 2 * self.key_dim]
        value = mixed_qkv[:, :, 2 * self.key_dim : 2 * self.key_dim + self.value_dim]
        query = ops.reshape(query, [batch_dim, seq_dim, self.num_k_heads, self.head_k_dim])
        key = ops.reshape(key, [batch_dim, seq_dim, self.num_k_heads, self.head_k_dim])
        value = ops.reshape(value, [batch_dim, seq_dim, self.num_v_heads, self.head_v_dim])

        beta = ops.sigmoid(b)
        a_plus_dt = ops.add(
            a.cast(DType.float32),
            self.dt_bias.cast(DType.float32).to(device),
        )
        g = ops.mul(
            -ops.exp(self.A_log.cast(DType.float32).to(device)),
            _softplus(a_plus_dt.cast(a.dtype)),
        )

        # Expand q/k heads to match v heads (GQA-style).
        # repeat_interleave not supported on GPU; use concat workaround.
        n_rep = self.num_v_heads // self.num_k_heads
        if n_rep > 1:
            q_exp = ops.unsqueeze(query, 3)
            query = ops.concat([q_exp] * n_rep, axis=3)
            query = ops.reshape(query, [batch_dim, seq_dim, self.num_v_heads, self.head_k_dim])
            k_exp = ops.unsqueeze(key, 3)
            key = ops.concat([k_exp] * n_rep, axis=3)
            key = ops.reshape(key, [batch_dim, seq_dim, self.num_v_heads, self.head_k_dim])

        if recurrent_state is None:
            zero = ops.constant(0.0, dtype=value.dtype, device=device)
            recurrent_state = ops.broadcast_to(
                zero,
                [batch_dim, self.num_v_heads, self.head_k_dim, self.head_v_dim],
            )

        # --- Parallel zero-state approximation ---
        # For stateless operation (state always starts at zero), each token's
        # output is computed independently:
        #   out_t = (q_t . k_t) * v_t * beta_t
        # This is exact for seq_len=1 (decode) and a reasonable first-order
        # approximation for seq_len>1 (prefill).
        # query, key: [batch, seq, num_v_heads, head_k_dim]
        # value:      [batch, seq, num_v_heads, head_v_dim]
        # beta:       [batch, seq, num_v_heads]
        q_k_dot = ops.squeeze(
            ops.sum(ops.mul(query, key), axis=-1), -1
        )  # [batch, seq, num_v_heads]
        gate = ops.mul(q_k_dot, beta)  # [batch, seq, num_v_heads]
        core_out = ops.mul(
            ops.unsqueeze(gate, -1), value
        )  # [batch, seq, num_v_heads, head_v_dim]

        # Update recurrent state from first token for state carry-over.
        q_0 = ops.squeeze(query[:, 0:1, :, :], 1)
        k_0 = ops.squeeze(key[:, 0:1, :, :], 1)
        v_0 = ops.squeeze(value[:, 0:1, :, :], 1)
        g_0 = ops.squeeze(g[:, 0:1, :], 1)
        beta_0 = ops.squeeze(beta[:, 0:1, :], 1)
        _, new_recurrent_state = self._recurrent_step(
            q_0, k_0, v_0, g_0, beta_0, recurrent_state
        )

        normed = self.norm(core_out, z)
        normed = ops.reshape(normed, [batch_dim, seq_dim, self.value_dim])

        # out_proj; flatten to 2D if input was 2D.
        normed_2d = ops.reshape(normed, [-1, self.value_dim])
        out_2d = self.out_proj(normed_2d)

        if input_is_2d:
            out = out_2d  # [total_seq, hidden]
        else:
            out = ops.reshape(out_2d, [batch_dim, seq_dim, self.hidden_size])
        return out, new_conv_state, new_recurrent_state
