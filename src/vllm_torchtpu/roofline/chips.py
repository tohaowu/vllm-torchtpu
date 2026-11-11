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
"""Per-core hardware specifications and physical 3D slice topologies."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from .constants import (
    DEFAULT_CORES_PER_CHIP,
    DEFAULT_ICI_BW,
    DEFAULT_ICI_HOP_LATENCY_S,
    DEFAULT_ICI_LINK_BW,
    DEFAULT_MXU_M_TILE,
    FALLBACK_TPU_SPECS,
    MIN_TORUS_DIM,
    TOPOLOGY_AXES,
)

try:
    from jax.experimental.pallas import tpu as _pltpu
except ImportError:
    pltpu: Any = None
else:
    pltpu = _pltpu


@dataclass(frozen=True)
class ChipOverrides:
    """Optional ICI and tile overrides applied when constructing a ChipSpec."""

    ici_link_bw: float = DEFAULT_ICI_LINK_BW
    ici_bw: float = DEFAULT_ICI_BW
    ici_hop_latency_s: float = DEFAULT_ICI_HOP_LATENCY_S
    cores_per_chip: int = DEFAULT_CORES_PER_CHIP
    mxu_m_tile: int = DEFAULT_MXU_M_TILE


@dataclass(frozen=True)
class ChipSpec:
    """Hardware ceiling parameters for one TensorCore."""

    name: str
    hbm_bytes: int
    hbm_bw: float
    bf16_flops: float
    fp8_flops: float
    vmem_bytes: int
    ici_bw: float = math.inf
    cores_per_chip: int = DEFAULT_CORES_PER_CHIP
    ici_link_bw: float = DEFAULT_ICI_LINK_BW
    ici_hop_latency_s: float = DEFAULT_ICI_HOP_LATENCY_S
    mxu_m_tile: int = DEFAULT_MXU_M_TILE


def from_jax(
    name: str,
    version: object,
    overrides: ChipOverrides | None = None,
) -> ChipSpec:
    """Construct ChipSpec from JAX TPU info table or stdlib fallback table."""
    ovr = overrides or ChipOverrides()
    if pltpu is None:
        if name not in FALLBACK_TPU_SPECS:
            msg = f"Unknown fallback TPU chip {name!r}"
            raise KeyError(msg)
        hbm_b, hbm_bw, bf16, fp8, vmem_b = FALLBACK_TPU_SPECS[name]
        return ChipSpec(
            name=name,
            hbm_bytes=hbm_b,
            hbm_bw=hbm_bw,
            bf16_flops=bf16,
            fp8_flops=fp8,
            vmem_bytes=vmem_b,
            cores_per_chip=ovr.cores_per_chip,
            ici_link_bw=ovr.ici_link_bw,
            ici_bw=ovr.ici_bw,
            ici_hop_latency_s=ovr.ici_hop_latency_s,
            mxu_m_tile=ovr.mxu_m_tile,
        )
    # Querying with logical cores_per_chip=1 returns per-TensorCore HBM, bandwidth,
    # MXU peaks, and VMEM; JAX's table omits ICI rates, so ChipOverrides supplies them.
    info = pltpu.get_tpu_info_for_chip(version, 1)
    return ChipSpec(
        name=name,
        hbm_bytes=info.hbm_capacity_bytes,
        hbm_bw=float(info.mem_bw_bytes_per_second),
        bf16_flops=float(info.bf16_ops_per_second),
        fp8_flops=float(info.fp8_ops_per_second),
        vmem_bytes=info.vmem_capacity_bytes,
        cores_per_chip=ovr.cores_per_chip,
        ici_link_bw=ovr.ici_link_bw,
        ici_bw=ovr.ici_bw,
        ici_hop_latency_s=ovr.ici_hop_latency_s,
        mxu_m_tile=ovr.mxu_m_tile,
    )


@dataclass(frozen=True)
class SliceTopology:
    """Physical 3D chip slice topology and torus wraparound rules."""

    name: str
    chip_axes: tuple[int, int, int]
    cores_per_chip: int = DEFAULT_CORES_PER_CHIP

    @property
    def num_chips(self) -> int:
        """Total number of physical chips in the 3D topology."""
        a, b, c = self.chip_axes
        return a * b * c

    @property
    def num_cores(self) -> int:
        """Total number of TensorCores across all chips in the topology."""
        return self.num_chips * self.cores_per_chip

    @property
    def has_torus_wraparound(self) -> bool:
        """True when all physical chip axes are multiples of MIN_TORUS_DIM (4)."""
        # Scaling Book (https://jax-ml.github.io/scaling-book/tpus/):
        # "Any multiple of a full cube ... will have wraparounds."
        # Google Cloud TPU7x doc slice table (https://cloud.google.com/tpu/docs/tpu7x):
        # "AxBxC where A, B, and C are multiples of 4".
        return all(
            d >= MIN_TORUS_DIM and d % MIN_TORUS_DIM == 0 for d in self.chip_axes
        )

    def axis_wrapped(self, axis: int, submesh_extent: int) -> bool:
        """True when a collective spans the full extent of a wrapped physical axis."""
        return (
            self.has_torus_wraparound
            and submesh_extent == self.chip_axes[axis]
            and submesh_extent >= MIN_TORUS_DIM
        )


# Peak ICI rate: 6 links x 200 GB/s bidirectional = 1,200 GB/s per chip, so
# 100 GB/s per link per direction and 600 GB/s egress per chip (300 GB/s per
# core): Google Cloud TPU7x docs (https://cloud.google.com/tpu/docs/tpu7x).
# Unloaded per-hop ICI latency (~1 us) follows Scaling Book TPU v5e estimates.
CHIPS: dict[str, ChipSpec] = {
    "tpu7x": from_jax(
        "tpu7x",
        pltpu.ChipVersion.TPU_7X if pltpu is not None else "tpu7x",
        ChipOverrides(),
    ),
}

TOPOLOGIES: dict[str, SliceTopology] = {
    name: SliceTopology(name, axes) for name, axes in TOPOLOGY_AXES.items()
}
