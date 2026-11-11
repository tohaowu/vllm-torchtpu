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
"""Centralized constants and hardware/model lookup tables for advisor."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping

ADVISOR_DIR = Path(__file__).resolve().parent
CALIBRATION_DIR = ADVISOR_DIR / "data" / "calibration"
PRESETS_DIR = ADVISOR_DIR / "data" / "presets"

MEASUREMENTS_SCHEMA = "advisor.measurements.v1"

DTYPE_BYTES: Mapping[str, float] = {
    "fp32": 4.0,
    "bf16": 2.0,
    "fp16": 2.0,
    "fp8": 1.0,
    "fp8_e4m3": 1.0,
    "fp8_e4m3fn": 1.0,
    "fp8_e5m2": 1.0,
    "int8": 1.0,
    "int4": 0.5,
    "nvfp4": 0.5625,
    "mxfp4": 0.53125,
    "fp4": 0.5625,
}

EFFICIENCY_BRACKETS: Mapping[str, tuple[float, float]] = {
    "eta_hbm": (0.70, 0.90),
    "eta_dense": (0.60, 0.85),
    "eta_moe_grouped": (0.35, 0.70),
    "eta_ici": (0.60, 0.90),
}

MEGACORE_CORES = 2
BRACKET_LEN = 2
USABLE_VMEM_RESERVE_BYTES = 2 << 20
DEFAULT_ROW_TILE_FP8 = 32
DEFAULT_ROW_TILE_WIDE = 16
DEFAULT_ROW_TILE_NARROW = 8
GOOGLE_PCI_VENDOR_ID = "0x1ae0"
SYSFS_PCI_DEVICES_DIR = Path("/sys/bus/pci/devices")
PCI_DEVICE_TO_CHIP: Mapping[str, str] = {
    "0x0027": "v4",
    "0x0056": "v5p",
    "0x0063": "v5e",
    "0x006f": "v6e",
    "0x0070": "v6e",
    "0x0076": "tpu7x",
    "0x0077": "tpu7x",
    "0x0078": "tpu7x",
}

REQUIRED_CHIP_SPEC_KEYS: tuple[str, ...] = (
    "vmem_capacity_bytes",
    "hbm_capacity_bytes",
    "hbm_bw_bytes_per_s",
    "bf16_flops_per_s",
)
OPTIONAL_CHIP_SPEC_KEYS: frozenset[str] = frozenset(
    {
        "name",
        "cores_per_device",
        "vmem_usable_bytes",
        "fp8_flops_per_s",
        "int8_flops_per_s",
        "native_ocp_fp8",
        "ici_egress_bw_bytes_per_s",
        "ici_link_bw_bytes_per_s",
        "collective_latency_s",
        "collective_latency_us",
        "collective_latency_us_bracket",
        "dcn_bw_gbps",
        "dcn_bw_gbps_bracket",
        "min_slice_cores",
        "devices_per_chip",
        "supports_megacore",
        "matmul_row_tile_by_dtype",
        "topologies",
        "sources",
    }
)
ALLOWED_CHIP_SPEC_KEYS: frozenset[str] = (
    frozenset(REQUIRED_CHIP_SPEC_KEYS) | OPTIONAL_CHIP_SPEC_KEYS
)

TPU7X_TOPOLOGIES: Mapping[int, str] = {
    8: "2x2x1 (4 chips, 8 cores)",
    16: "2x2x2 (8 chips, 16 cores)",
    32: "2x2x4 (16 chips, 32 cores)",
    64: "2x4x4 (32 chips, 64 cores)",
    128: "4x4x4 (64 chips, 128 cores)",
}

TPU6E_TOPOLOGIES: Mapping[int, str] = {
    1: "1x1 (1 chip, 1 core)",
    4: "2x2 (4 chips, 4 cores)",
    8: "2x4 (8 chips, 8 cores)",
    16: "4x4 (16 chips, 16 cores)",
    32: "4x8 (32 chips, 32 cores)",
    64: "8x8 (64 chips, 64 cores)",
}

V5P_TOPOLOGIES: Mapping[int, str] = dict(TPU7X_TOPOLOGIES)
V5E_TOPOLOGIES: Mapping[int, str] = dict(TPU6E_TOPOLOGIES)
V4_TOPOLOGIES: Mapping[int, str] = dict(TPU7X_TOPOLOGIES)

ROW_TILES_7X: Mapping[str, int] = {
    "fp8": DEFAULT_ROW_TILE_FP8,
    "nvfp4": DEFAULT_ROW_TILE_FP8,
    "mxfp4": DEFAULT_ROW_TILE_FP8,
    "fp4": DEFAULT_ROW_TILE_FP8,
    "bf16": DEFAULT_ROW_TILE_WIDE,
    "fp16": DEFAULT_ROW_TILE_WIDE,
    "int8": DEFAULT_ROW_TILE_WIDE,
    "fp32": DEFAULT_ROW_TILE_WIDE,
}
ROW_TILES_V6E: Mapping[str, int] = dict.fromkeys(ROW_TILES_7X, DEFAULT_ROW_TILE_WIDE)
ROW_TILES_NARROW: Mapping[str, int] = dict.fromkeys(
    ROW_TILES_7X, DEFAULT_ROW_TILE_NARROW
)

CHIP_ALIASES: Mapping[str, str] = {
    "7x": "tpu7x",
    "v7x": "tpu7x",
    "tpu7x": "tpu7x",
    "6e": "v6e",
    "v6e": "v6e",
    "tpu6e": "v6e",
    "5p": "v5p",
    "v5p": "v5p",
    "tpu5p": "v5p",
    "5e": "v5e",
    "v5e": "v5e",
    "tpu5e": "v5e",
    "4": "v4",
    "v4": "v4",
    "tpu4": "v4",
}

DEFAULT_SOURCES_7X: Mapping[str, str] = {
    "cores_per_device": (
        "FACT: jax.experimental.pallas.tpu tpu_info (7x dual-core: 1 core/device)"
    ),
    "devices_per_chip": (
        "FACT: Google Cloud TPU v7x Ironwood docs (2 logical devices/chip)"
    ),
    "vmem_capacity_bytes": (
        "FACT: jax.experimental.pallas.tpu tpu_info (7x: 64 MiB/core)"
    ),
    "vmem_usable_bytes": (
        "CHOSEN: 64 MiB physical minus 2 MiB compiler reserve (62 MiB/core)"
    ),
    "hbm_capacity_bytes": (
        "FACT: jax.experimental.pallas.tpu tpu_info (7x: 103 GB/core)"
    ),
    "hbm_bw_bytes_per_s": (
        "FACT: jax.experimental.pallas.tpu tpu_info (7x: 3.70 TB/s/core)"
    ),
    "bf16_flops_per_s": (
        "FACT: jax.experimental.pallas.tpu tpu_info (7x: 1.155 PF/s/core)"
    ),
    "fp8_flops_per_s": (
        "FACT: jax.experimental.pallas.tpu tpu_info (7x: 2.30 PF/s/core)"
    ),
    "int8_flops_per_s": (
        "DERIVED: jax.experimental.pallas.tpu tpu_info int8=0 on 7x; bf16 rate"
    ),
    "native_ocp_fp8": (
        "FACT: jax.experimental.pallas.tpu tpu_info (7x native OCP FP8)"
    ),
    "ici_egress_bw_bytes_per_s": (
        "ASSUMED: 300 GB/s/core unidirectional ICI egress prior"
    ),
    "ici_link_bw_bytes_per_s": ("ASSUMED: 100 GB/s/link unidirectional ICI link prior"),
    "collective_latency_s": (
        "ASSUMED: [1.0, 5.13] us/hop floor prior; calibrate via trace"
    ),
    "collective_latency_us_bracket": (
        "ASSUMED: [1.0, 5.13] us/hop bracket; calibrate via trace"
    ),
    "dcn_bw_gbps": "ASSUMED: 100 Gbps/chip nominal host DCN egress prior",
    "dcn_bw_gbps_bracket": (
        "ASSUMED: [50.0, 200.0] Gbps/chip host DCN prior (12.5 GB/s)"
    ),
    "min_slice_cores": (
        "FACT: smallest TPU v7x 2x2x1 cube topology has 4 chips (8 cores)"
    ),
    "supports_megacore": (
        "FACT: TPU v7x exposes independent logical cores (supports_megacore=False)"
    ),
    "matmul_row_tile_by_dtype": (
        "FACT: TPU v7x matmul LHS unpadded row tile floor (32 FP8/FP4, 16 BF16/INT8)"
    ),
}

DEFAULT_SOURCES_PUBLIC: Mapping[str, str] = {
    "cores_per_device": "FACT: jax.experimental.pallas.tpu tpu_info (1 core/device)",
    "devices_per_chip": (
        "FACT: Google Cloud TPU architecture docs (devices per physical chip)"
    ),
    "vmem_capacity_bytes": "FACT: jax.experimental.pallas.tpu tpu_info",
    "vmem_usable_bytes": (
        "CHOSEN: physical VMEM capacity minus 2 MiB compiler reserve"
    ),
    "hbm_capacity_bytes": "FACT: jax.experimental.pallas.tpu tpu_info",
    "hbm_bw_bytes_per_s": "FACT: jax.experimental.pallas.tpu tpu_info",
    "bf16_flops_per_s": "FACT: jax.experimental.pallas.tpu tpu_info",
    "fp8_flops_per_s": (
        "DERIVED: jax.experimental.pallas.tpu tpu_info (bf16 rate when fp8=0)"
    ),
    "int8_flops_per_s": (
        "DERIVED: jax.experimental.pallas.tpu tpu_info (bf16 rate when int8=0)"
    ),
    "native_ocp_fp8": "FACT: jax.experimental.pallas.tpu tpu_info native OCP FP8",
    "ici_egress_bw_bytes_per_s": (
        "ASSUMED: Cloud TPU interconnect unidirectional ICI egress prior"
    ),
    "ici_link_bw_bytes_per_s": (
        "ASSUMED: Cloud TPU interconnect unidirectional per-link ICI prior"
    ),
    "collective_latency_s": (
        "ASSUMED: [1.0, 5.0] us/hop floor prior; calibrate via trace"
    ),
    "collective_latency_us_bracket": (
        "ASSUMED: [1.0, 5.0] us/hop bracket; calibrate via trace"
    ),
    "dcn_bw_gbps": "ASSUMED: 100 Gbps/chip nominal host DCN egress prior",
    "dcn_bw_gbps_bracket": (
        "ASSUMED: [50.0, 200.0] Gbps/chip host DCN prior (12.5 GB/s)"
    ),
    "min_slice_cores": "FACT: minimum slice core count from Cloud TPU topology docs",
    "supports_megacore": "FACT: Cloud TPU Megacore support flag per generation",
    "matmul_row_tile_by_dtype": "FACT: matmul LHS unpadded row tile floor per dtype",
}

MIN_INTERVAL = 2
BITS_FP8 = 8
BITS_FP4 = 4
MIN_HEAD_DIM = 8

UNSUPPORTED_NON_CAUSAL_TYPES: frozenset[str] = frozenset(
    {"bert", "roberta", "t5", "whisper", "vit", "clip", "dit", "flux", "sdxl"}
)

UNSUPPORTED_SSM_MODEL_TYPES: frozenset[str] = frozenset(
    {
        "mamba",
        "mamba2",
        "rwkv",
        "rwkv6",
        "falcon_mamba",
        "xlstm",
        "retnet",
        "hyena",
        "stripedhyena",
    }
)

MULTIMODAL_KEYS: tuple[str, ...] = (
    "vision_config",
    "audio_config",
    "video_config",
    "mm_projector_type",
    "image_token_index",
)

VALID_LAYER_TOKENS: frozenset[str] = frozenset(
    {
        "attention",
        "csa",
        "dense_mlp",
        "hca",
        "indexed_attention",
        "dsa",
        "dsa_attention",
        "full_attention",
        "gated_deltanet",
        "gdn",
        "gqa",
        "kda",
        "kda_attention",
        "kimi_linear",
        "linear_attention",
        "mamba",
        "mla",
        "mla_attention",
        "mlp",
        "moe",
        "mtp",
        "ple",
        "sliding_attention",
        "sliding_window_attention",
        "sparse_mla",
        "swa",
    }
)

KDA_MODEL_TYPES: frozenset[str] = frozenset({"kimi_k2", "kimi_k3", "kimi_linear"})
