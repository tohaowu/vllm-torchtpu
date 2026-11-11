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

from dataclasses import dataclass
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


@dataclass(frozen=True)
class CoreLimits:
    """Verified limits mirrored from `megakernel/core/exec_spec.py` and `profile.py`."""

    max_batch_rows: int = 128
    verified_ranks: tuple[int, ...] = (1, 2, 4, 8, 16)
    chip_kinds: tuple[str, ...] = ("TPU7x",)
    max_ar_slots_per_layer: int = 2
    nbd: int = 14
    nbe: int = 10
    bank: int = 2 << 20
    ebank: int = 1 << 20
    vmem_mib: int = 62
    default_decode_bucket_top: int = 32
    default_verify_bucket_top: int = 64


CORE_LIMITS = CoreLimits()
CHIP_KIND_MAP: Mapping[str, str] = {"tpu7x": "TPU7x"}

MIN_VMEM_MIB = 2
DEFAULT_USABLE_VMEM_MIB = CORE_LIMITS.vmem_mib
DOUBLE_BUFFER_SLOTS = 2
GROUPED_MOE_MODES: frozenset[str] = frozenset({"grouped_on_chip"})
VALID_DMA_BUFFERS = (2, 3)
WIDE_DENSE_RING_SLOTS = CORE_LIMITS.nbd
WIDE_EXPERT_RING_SLOTS = CORE_LIMITS.nbe
DENSE_SLOT_DEFAULT_BYTES = CORE_LIMITS.bank
EXPERT_SLOT_DEFAULT_BYTES = CORE_LIMITS.ebank
ARENA_ALIGN_RESERVE_BYTES = 2 << 20
ATTN_TILE_TOKENS = 128
GDN_STREAMED_HEADS_CAP = 8
DENSE_FFN_TILE_CAP = 1024
FP32_BYTES = 4
CHIPS_PER_HOST = 8

DEFAULT_LAUNCH_GAP_FLOOR_US = 1.0
DEFAULT_LAUNCH_GAP_VAL_US = 2.0
DEFAULT_LAUNCH_GAP_UPPER_US = 8.0
COMPILED_XLA_GAP_P50_US = 1.0
COMPILED_XLA_GAP_P90_US = 1.5
UNFUSED_EAGER_GAP_US = 6.0
PER_OP_COLLECTIVE_CALL_FACTOR = 2
BOUNDARY_SYNC_HOP_FACTOR = 2.0

DOT_FP8 = "fp8"
DOT_BF16 = "bf16"
AB_KNOB = "ab_step_time"
SETTLE_CMD = "Pass --measurements measurements.json (schema advisor.measurements.v1)"
CORNER_AXES: Mapping[str, int] = {
    "launch_gap_us": 0,
    "collective_latency_us": 1,
    "prefill_attn_dot_dtype": 2,
}

DEFAULT_CHIP = "tpu7x"
DEFAULT_PREFILL_ISL = 4096
DEFAULT_SPEC_TOKENS = 3
RECOMMEND_TP_CANDIDATES: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64)
RECOMMEND_DEFAULT_BATCH = 16
RECOMMEND_HIGH_BATCH = 64
RECOMMEND_HIGH_CTX = 32768
SATURATION_BASE_BATCH = 256
SATURATION_BATCH_CAP = 600
INVALID_CONFIG_EXIT_CODE = 2

SVG_MIN_FONT_PX = 13
# Width of one glyph as a fraction of font size, a conservative upper bound for
# the system sans-serif stack on digits, Latin letters, and punctuation.
SVG_GLYPH_EM = 0.62
SVG_TICKS: tuple[int, ...] = (1, 8, 64, 512, 4096, 32768)
# Okabe-Ito colorblind-safe palette; each scope also has its own marker shape.
SCOPE_MARKS: Mapping[str, tuple[str, str]] = {
    "whole_step": ("#009E73", "circle"),
    "layer_group": ("#56B4E9", "diamond"),
    "single_layer": ("#E69F00", "diamond"),
    "per_op": ("#CC79A7", "square"),
}
MOE_MODE_SHORT_HINTS: Mapping[str, str] = {
    "masked_on_chip": "Routed experts stream into VMEM; token rows are masked "
    "per expert inside the fused call; no reordering",
    "grouped_on_chip": "Tokens are grouped by expert inside VMEM, then each "
    "expert runs on its rows inside the fused call",
    "grouped_in_hbm": "Tokens are sorted by expert in HBM; the MoE is its own "
    "call, so the kernel boundary falls at each MoE",
    "dense_mlp": "Dense feed-forward; no routed experts",
}
SCOPE_HINTS: Mapping[str, str] = {
    "whole_step": "one call for the whole step; [T, H] stays in VMEM. Rows past a "
    "declared single-call limit run as separate calls that stream weights again; "
    "tiling inside each layer past the VMEM ceiling is not implemented",
    "layer_group": "one call per group of N layers, tiled like whole_step",
    "single_layer": "one call per layer",
    "per_op": "one call per operator; activations hand off through HBM",
}

DEFAULT_PHASE = "decode"
PS_PER_US = 1_000_000.0
NS_PER_US = 1_000.0
# Gaps above 500 us reflect idle pauses rather than kernel launch overhead.
MAX_INTRA_STEP_GAP_US = 500.0
MIN_COLLECTIVE_GROUP_SIZE = 2
MIN_STEPS_FOR_WARMUP_DROP = 3
MIN_WARMUP_DROP_STEPS = 2
DEFAULT_CALIBRATE_LAYERS = 60
SUBLAYER_BOUNDARIES_PER_LAYER = 2
MIN_STEP_OPS_FOR_EXPLICIT_NORM = 2
CONSERVED_GAP_DECIMALS = 10
QUANTILE_DECIMALS = 4
WARMUP_OUTLIER_RATIO = 2.0
MIN_EVENTS_FOR_WARMUP_DROP = 4
CONSERVED_GAP_UNIT = "us_per_sublayer_boundary"
STEP_COVERAGE_BASIS = "trace"
FUSED_ELIGIBLE_PHASES: tuple[str, ...] = ("decode", "verify")
XPLANE_DEVICE_PLANE_PREFIX = "/device:"
XPLANE_SPARSE_FILTER = "sparse"
COLLECTIVE_IGNORE_SUFFIXES: tuple[str, ...] = ("-done", ".done")
XLA_MODULE_MARKERS: tuple[str, ...] = ("xla module", "xla_module")
STEP_EVENT_MARKERS: tuple[str, ...] = ("step", "model_step", "step_num")
COLLECTIVE_PREFIXES: tuple[tuple[str, str], ...] = (
    ("all-reduce", "all-reduce"),
    ("allreduce", "all-reduce"),
    ("all-gather", "all-gather"),
    ("allgather", "all-gather"),
    ("reduce-scatter", "reduce-scatter"),
    ("reducescatter", "reduce-scatter"),
    ("all-to-all", "all-to-all"),
    ("alltoall", "all-to-all"),
)
