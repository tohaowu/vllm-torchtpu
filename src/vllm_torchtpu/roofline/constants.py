# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Physical TPU hardware, torus collective, and architecture taxonomy constants."""

from __future__ import annotations

# Collective & TPU topology constants.
AR_PHASES: int = 2
DEFAULT_CORES_PER_CHIP: int = 2
MIN_TORUS_DIM: int = 4
MIN_SLICE_CHIPS: int = 4
WRAPPED_LANES_PER_AXIS: int = 2

DEFAULT_ICI_LINK_BW: float = 100.0e9
DEFAULT_ICI_BW: float = 300.0e9
DEFAULT_ICI_HOP_LATENCY_S: float = 1.0e-6
TPU7X_ICI_LINKS_PER_CHIP: int = 6
TPU7X_ICI_BIDIR_LINK_BW: float = 200.0e9

# Fallback (hbm_bytes, hbm_bw, bf16_flops, fp8_flops, vmem_bytes) per TensorCore
# matching jax._src.tpu_info.get_tpu_info_for_chip(ChipVersion.TPU_7X, 1).
FALLBACK_TPU_SPECS: dict[str, tuple[int, float, float, float, int]] = {
    "tpu7x": (103_000_000_000, 3.70e12, 1.155e15, 2.30e15, 67_108_864),
}

# Supported Cloud TPU7x 3D physical slice shapes (minimum provisionable slice is 2x2x1).
TOPOLOGY_AXES: dict[str, tuple[int, int, int]] = {
    "2x2x1": (2, 2, 1),
    "2x2x2": (2, 2, 2),
    "2x2x4": (2, 2, 4),
    "2x4x4": (2, 4, 4),
    "4x4x4": (4, 4, 4),
}

# Transformer layer taxonomy shared across geometry parsing and cost modeling.
LAYER_FAMILIES: tuple[str, ...] = (
    "gqa",
    "mla",
    "dsa",
    "swa",
    "gdn",
    "kda",
    "dense_mlp",
    "moe",
    "mtp",
    "ple",
)

VALID_LAYER_TOKENS: frozenset[str] = frozenset(
    {
        "full_attention",
        "attention",
        "gqa",
        "mla",
        "mla_attention",
        "dsa",
        "dsa_attention",
        "sparse_mla",
        "sliding_attention",
        "swa",
        "sliding_window_attention",
        "linear_attention",
        "gdn",
        "gated_deltanet",
        "mamba",
        "kda",
        "kimi_linear",
        "kda_attention",
        "dense_mlp",
        "mlp",
        "moe",
        "mtp",
        "ple",
    }
)

LINEAR_ATTENTION_LAYER_TOKENS: frozenset[str] = frozenset(
    {
        "linear_attention",
        "gdn",
        "gated_deltanet",
        "mamba",
        "kda",
        "kimi_linear",
        "kda_attention",
    }
)
