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
"""TPU hardware, torus collective, architecture, dtype, and solver constants."""

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

# Transformer layer taxonomy shared across geometry parsing, memory, and cost modeling.
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

NON_FULL_ATTN_TOKENS: frozenset[str] = frozenset(
    {
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
    }
)

LAYER_TYPE_MAP: dict[str, str] = {
    "mla": "mla",
    "mla_attention": "mla",
    "dsa": "dsa",
    "dsa_attention": "dsa",
    "sparse_mla": "dsa",
    "sliding_attention": "swa",
    "swa": "swa",
    "sliding_window_attention": "swa",
    "kda": "kda",
    "kimi_linear": "kda",
    "kda_attention": "kda",
    "linear_attention": "gdn",
    "gdn": "gdn",
    "gated_deltanet": "gdn",
    "mamba": "gdn",
    "gqa": "gqa",
}

# Unit scales and math constants.
ACTIVATION_BYTES_BF16: float = 2.0
MATMUL_FLOP_PER_MAC: float = 2.0
UNIT_KILO: float = 1.0e3
UNIT_MEGA: float = 1.0e6
UNIT_GIGA: float = 1.0e9
UNIT_MILLIS_PER_SECOND: float = 1.0e3
UNIT_MICROS_PER_SECOND: float = 1.0e-6

# HF config loader & quantization constants.
FOUR_BITS: int = 4
EIGHT_BITS: int = 8
SIXTEEN_BITS: int = 16
WEIGHT_BLOCK_DIM_LEN: int = 2
FP8_BLOCK_SCALE_BYTES: float = 4.0
HTTP_OK: int = 200
HTTP_TIMEOUT_S: int = 15
DEFAULT_HTTP_PORT: int = 80
DEFAULT_HTTPS_PORT: int = 443

FP32_BYTES: float = 4.0
BF16_BYTES: float = 2.0
FP8_BYTES: float = 1.0
DEFAULT_SSM_STATE_BYTES: float = 4.0
MXFP4_BYTES: float = 0.5 + 1.0 / 32.0
NVFP4_BYTES: float = 0.5 + 1.0 / 16.0

BASE_DTYPE_BYTES: dict[str, float] = {
    "float32": FP32_BYTES,
    "fp32": FP32_BYTES,
    "bfloat16": BF16_BYTES,
    "bf16": BF16_BYTES,
    "float16": BF16_BYTES,
    "fp16": BF16_BYTES,
    "float8_e4m3fn": FP8_BYTES,
    "fp8": FP8_BYTES,
}

WEIGHT_DTYPE_BYTES: dict[str, float] = {"bf16": BF16_BYTES, "fp8": FP8_BYTES}
EXPERT_DTYPE_BYTES: dict[str, float] = {
    "bf16": BF16_BYTES,
    "fp8": FP8_BYTES,
    "nvfp4": NVFP4_BYTES,
    "mxfp4": MXFP4_BYTES,
}
KV_DTYPE_BYTES: dict[str, float] = {"bf16": BF16_BYTES, "fp8": FP8_BYTES}
STATE_DTYPE_BYTES: dict[str, float] = {"fp32": FP32_BYTES, "bf16": BF16_BYTES}

WEIGHT_ONLY_QUANT_METHODS: frozenset[str] = frozenset(
    {"awq", "gptq", "gptq_marlin", "awq_marlin", "bitsandbytes", "hqq", "eetq"}
)

# Solver, HBM capacity, and SMEM page-table constants.
DEFAULT_GPU_MEMORY_UTILIZATION: float = 0.92
MAX_CTX_TOKENS_FOR_SMALL_PAGE: int = 8192
PAGE_TABLE_BYTE_BUDGET: int = 524_288
BYTES_PER_PAGE_ENTRY: int = 4
MIN_MOE_INTERMEDIATE_ALIGNMENT: int = 128
RPA_MIN_PAGE_TOKENS: int = 16
RPA_MAX_PAGE_TOKENS: int = 256
PHYSICAL_ALIGNMENT_ELEMS: int = 128
HBM_PACKING_NUMERATOR: float = 4.0
HBM_MIN_PACKING_DIVISOR: float = 0.5
SPARSE_MLA_SCALE_BYTES: float = 4.0
HEAD_DIM_PACKING_THRESHOLD: int = 64
GLU_MATMUL_COUNT: int = 3
DEFAULT_MAX_INSTANCES: int = 16

EXAMPLE_CONCURRENCIES: tuple[int, ...] = (1, 4, 8, 16, 32, 64, 128, 256)
EXAMPLE_SLO_TIERS: dict[str, tuple[float | None, float | None]] = {
    "none": (None, None),
    "interactive": (5000.0, 33.33),
    "throughput": (5000.0, 66.67),
}

# Versioned solve JSON payload and top-K diversity selection constants.
SOLVE_SCHEMA: str = "vllm_torchtpu.roofline.solve/v1"
MIN_TOP_LAYOUTS_FOR_FASTEST: int = 2
