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
"""Wire bytes, serial hops, and 4-axis torus floors for collectives."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from .constants import (
    AR_PHASES,
    DEFAULT_CORES_PER_CHIP,
    MIN_TORUS_DIM,
    WRAPPED_LANES_PER_AXIS,
)

if TYPE_CHECKING:
    from .chips import ChipSpec, SliceTopology

Kind = Literal["ar", "ag", "rs", "push", "a2a", "p2p"]
Algo = Literal["one_shot", "two_shot", "ring"]


@dataclass(frozen=True)
class CollectiveConfig:
    """Algorithm and per-chip rank packing parameters for a collective."""

    algo: Algo = "one_shot"
    ranks_per_chip: int = DEFAULT_CORES_PER_CHIP


@dataclass(frozen=True)
class Collective:
    """One collective as seen by one rank."""

    name: str
    group: int
    wire_bytes: float
    hops: int
    offchip: float
    kind: Kind
    payload: float


def offchip_share(group: int, ranks_per_chip: int) -> float:
    """Share of peers outside this rank's chip, assuming adjacent ranks share a chip."""
    if group <= 1:
        return 0.0
    return max(0, group - min(group, ranks_per_chip)) / (group - 1)


def make(
    name: str,
    kind: Kind,
    payload: float,
    group: int,
    config: CollectiveConfig | None = None,
) -> Collective:
    """Price one collective; payload is AR/RS input, AG output, or push/p2p wire."""
    cfg = config or CollectiveConfig()
    if group <= 1:
        return Collective(name, group, 0.0, 0, 0.0, kind, payload)
    n = group
    if kind in ("push", "p2p"):
        wire, hops = payload, 1
    elif kind == "ar":
        if cfg.algo == "one_shot":
            wire, hops = (n - 1) * payload, 1
        elif cfg.algo == "two_shot":
            wire, hops = 2 * (n - 1) / n * payload, AR_PHASES
        else:
            wire, hops = 2 * (n - 1) / n * payload, AR_PHASES * (n - 1)
    else:
        wire = (n - 1) / n * payload
        hops = n - 1 if cfg.algo == "ring" else 1
    return Collective(
        name, n, wire, hops, offchip_share(n, cfg.ranks_per_chip), kind, payload
    )


def padded_rows(rows: int, tp: int, residual: str) -> int:
    """Rows a collective carries: sequence parallelism pads to a multiple of tp."""
    return rows if residual == "replicated" else math.ceil(rows / tp) * tp


def submesh_for_group(
    group: int,
    topology: SliceTopology | None,
    ranks_per_chip: int = DEFAULT_CORES_PER_CHIP,
    mesh_axes: tuple[int, int, int, int] | None = None,
) -> tuple[int, int, int, int]:
    """Pack `group` ranks inner-to-outer across (d2d, x, y, z) physical bounds."""
    if group <= 1:
        return (1, 1, 1, 1)
    if topology is not None and group > topology.num_cores:
        msg = (
            f"Collective group {group} exceeds topology {topology.name} "
            f"({topology.num_cores} cores)"
        )
        raise ValueError(msg)
    if mesh_axes is not None:
        cap = mesh_axes[0] * mesh_axes[1] * mesh_axes[2] * mesh_axes[3]
        if group <= cap and cap % group == 0:
            bounds = mesh_axes
        elif topology is not None:
            bounds = (topology.cores_per_chip, *topology.chip_axes)
        else:
            msg = f"Collective group {group} exceeds submesh bounds {mesh_axes}"
            raise ValueError(msg)
    elif topology is not None:
        bounds = (topology.cores_per_chip, *topology.chip_axes)
    else:
        d2d = math.gcd(group, ranks_per_chip)
        return (d2d, group // d2d, 1, 1)

    rem = group
    out: list[int] = []
    for lim in bounds:
        take = math.gcd(rem, max(1, lim))
        out.append(take)
        rem //= take
    if rem > 1:
        msg = f"Collective group {group} cannot pack within bounds {bounds}"
        raise ValueError(msg)
    return (out[0], out[1], out[2], out[3])


def collective_ici_hops(
    c: Collective,
    topology: SliceTopology | None = None,
    ranks_per_chip: int = DEFAULT_CORES_PER_CHIP,
    mesh_axes: tuple[int, int, int, int] | None = None,
) -> int:
    """Unloaded one-way or two-phase physical ICI hops traversed by a collective."""
    if c.group <= 1 or c.offchip <= 0.0:
        return 0
    if topology is not None or mesh_axes is not None:
        _, ax, ay, az = submesh_for_group(c.group, topology, ranks_per_chip, mesh_axes)
        if ax * ay * az <= 1:
            return 0
        hops_1p = sum(
            (
                ext // 2
                if (topology is not None and topology.axis_wrapped(i, ext))
                else (ext - 1)
            )
            for i, ext in enumerate((ax, ay, az))
            if ext > 1
        )
        phases = AR_PHASES if c.kind == "ar" else 1
        return phases * hops_1p
    return c.hops


def _has_finite_ici(chip: ChipSpec) -> bool:
    """Return whether chip specifies finite positive ICI bandwidths."""
    return (
        chip.ici_bw > 0.0
        and math.isfinite(chip.ici_bw)
        and chip.ici_link_bw > 0.0
        and math.isfinite(chip.ici_link_bw)
    )


def torus_collective_s(
    kind: Kind,
    payload_bytes: float,
    submesh_axes: tuple[int, int, int, int],
    chip: ChipSpec,
    topology: SliceTopology | None = None,
) -> float:
    """Analytical two-phase torus alpha-beta floor across (d2d, a, b, c) axes."""
    d2d, ax, ay, az = submesh_axes
    group = d2d * ax * ay * az
    if group <= 1 or payload_bytes <= 0.0 or not _has_finite_ici(chip):
        return 0.0

    # Pure intra-chip collective (zero external ICI hops).
    # Google Cloud TPU7x doc (https://cloud.google.com/tpu/docs/tpu7x):
    # D2D interconnect is six times faster than one ICI link (600 GB/s per chip).
    if ax * ay * az == 1:
        intra_bw = (
            chip.d2d_bw
            if (math.isfinite(chip.d2d_bw) and chip.d2d_bw > 0.0)
            else chip.ici_bw * chip.cores_per_chip
        )
        xfer = (
            payload_bytes
            if kind in ("push", "p2p")
            else payload_bytes * (group - 1) / group
        )
        phase_s = xfer / intra_bw
        return float(AR_PHASES) * phase_s if kind == "ar" else phase_s

    ici_hops = 0
    lanes_sum = 0
    for axis_idx, extent in enumerate((ax, ay, az)):
        if extent <= 1:
            continue
        wrapped = (
            topology.axis_wrapped(axis_idx, extent) if topology is not None else False
        )
        if wrapped:
            ici_hops += extent // 2
            lanes_sum += WRAPPED_LANES_PER_AXIS
        else:
            ici_hops += extent - 1
            lanes_sum += 1

    contention = float(max(1, d2d))
    cap_bw = (
        chip.ici_bw
        if d2d >= DEFAULT_CORES_PER_CHIP
        else chip.ici_bw * chip.cores_per_chip
    )
    raw_bw = lanes_sum * chip.ici_link_bw / contention
    eff_bw = min(raw_bw, cap_bw) if math.isfinite(cap_bw) and cap_bw > 0.0 else raw_bw

    hop_lat = (
        chip.ici_hop_latency_s
        if chip.ici_hop_latency_s > 0.0 and math.isfinite(chip.ici_hop_latency_s)
        else 0.0
    )
    lat_1p = ici_hops * hop_lat

    if kind == "p2p":
        return lat_1p + payload_bytes / eff_bw

    xfer = payload_bytes if kind == "push" else payload_bytes * (group - 1) / group
    phase_s = lat_1p + xfer / eff_bw
    return float(AR_PHASES) * phase_s if kind == "ar" else phase_s


def _count_cut_links(
    axes: tuple[int, int, int],
    topology: SliceTopology | None,
) -> int:
    """Count unidirectional ICI cut links per direction across active mesh axes."""
    links_in = 0
    for axis_idx, ext in enumerate(axes):
        if ext <= 1:
            continue
        if (
            ext >= MIN_TORUS_DIM
            and topology is not None
            and topology.axis_wrapped(axis_idx, ext)
        ):
            links_in += WRAPPED_LANES_PER_AXIS
        else:
            links_in += 1
    return max(1, links_in)


def collective_cut_bound_s(
    kind: Kind,
    payload_bytes: float,
    submesh_axes: tuple[int, int, int, int],
    chip: ChipSpec,
    topology: SliceTopology | None = None,
) -> float:
    """Physical cut-set lower bound in seconds for one collective."""
    d2d, ax, ay, az = submesh_axes
    g = d2d * ax * ay * az
    if g <= 1 or payload_bytes <= 0.0 or not _has_finite_ici(chip):
        return 0.0
    if ax * ay * az == 1:
        intra_bw = (
            chip.d2d_bw
            if (math.isfinite(chip.d2d_bw) and chip.d2d_bw > 0.0)
            else chip.ici_bw * float(chip.cores_per_chip)
        )
        cut_bytes = (
            payload_bytes * float(g - 1) / float(g)
            if kind in ("ag", "rs")
            else payload_bytes
        )
        return cut_bytes / intra_bw
    r = max(1, min(g, d2d))
    if kind in ("ag", "rs"):
        # AG imports (g - r) / g across the cut; RS exports (g - r) / g
        # symmetrically across the same cut (max with local shard r / g).
        cut_bytes = payload_bytes * float(max(g - r, r)) / float(g)
    else:
        cut_bytes = payload_bytes
    cut_links = _count_cut_links((ax, ay, az), topology)
    return cut_bytes / (float(cut_links) * chip.ici_link_bw)
