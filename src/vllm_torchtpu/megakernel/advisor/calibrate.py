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
"""Trace calibration engine for TPU megakernel advisor priors."""

from __future__ import annotations

import gzip
import json
import math
from bisect import bisect_left
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from math import log2
from pathlib import Path
from typing import TYPE_CHECKING, TypedDict, cast

from vllm_torchtpu.megakernel.core import errors

from .constants import (
    COLLECTIVE_IGNORE_SUFFIXES,
    COLLECTIVE_PREFIXES,
    CONSERVED_GAP_DECIMALS,
    CONSERVED_GAP_UNIT,
    DEFAULT_CALIBRATE_LAYERS,
    DEFAULT_CHIP,
    DEFAULT_PHASE,
    FUSED_ELIGIBLE_PHASES,
    MAX_INTRA_STEP_GAP_US,
    MEASUREMENTS_SCHEMA,
    MIN_COLLECTIVE_GROUP_SIZE,
    MIN_EVENTS_FOR_WARMUP_DROP,
    MIN_STEP_OPS_FOR_EXPLICIT_NORM,
    MIN_STEPS_FOR_WARMUP_DROP,
    MIN_WARMUP_DROP_STEPS,
    NS_PER_US,
    PS_PER_US,
    QUANTILE_DECIMALS,
    STEP_COVERAGE_BASIS,
    SUBLAYER_BOUNDARIES_PER_LAYER,
    WARMUP_OUTLIER_RATIO,
    XLA_MODULE_MARKERS,
    XPLANE_DEVICE_PLANE_PREFIX,
    XPLANE_SPARSE_FILTER,
)
from .models import resolve_model_profile

try:
    from google.protobuf.message import DecodeError

    from vllm_torchtpu.tools import merge_xplane, xplane_pb2
except (ImportError, RuntimeError):
    DecodeError = ValueError
    merge_xplane = None
    xplane_pb2 = None


class XPlaneToolUnavailableError(errors.MegakernelConfigError):
    """Raised when xplane tools (merge_xplane or xplane_pb2) are unavailable."""

    def __init__(
        self,
        message: str = "merge_xplane or xplane_pb2 tool not available",
    ) -> None:
        """Initialize XPlaneToolUnavailableError with a default message."""
        super().__init__(message)


if TYPE_CHECKING:
    from typing import Protocol

    class _XStatProto(Protocol):
        """Structural protocol for an XStat protobuf message."""

        metadata_id: int
        int64_value: int
        uint64_value: int
        double_value: float
        str_value: str

    class _XStatMetaProto(Protocol):
        """Structural protocol for an XStatMetadata protobuf message."""

        id: int
        name: str

    class _XEventMetaProto(Protocol):
        """Structural protocol for an XEventMetadata protobuf message."""

        id: int
        name: str
        display_name: str

    class _XEventProto(Protocol):
        """Structural protocol for an XEvent protobuf message."""

        metadata_id: int
        offset_ps: int
        duration_ps: int
        stats: Sequence[_XStatProto]

    class _XLineProto(Protocol):
        """Structural protocol for an XLine protobuf message."""

        id: int
        name: str
        display_name: str
        timestamp_ns: int
        events: Sequence[_XEventProto]

    class _XPlaneProto(Protocol):
        """Structural protocol for an XPlane protobuf message."""

        id: int
        name: str
        lines: Sequence[_XLineProto]
        event_metadata: Mapping[int, _XEventMetaProto]
        stat_metadata: Mapping[int, _XStatMetaProto]

    class _XSpaceProto(Protocol):
        """Structural protocol for an XSpace protobuf message."""

        planes: Sequence[_XPlaneProto]
        ParseFromString: Callable[[bytes], int]

    class _XPlaneModuleProto(Protocol):
        """Structural protocol for the generated `xplane_pb2` module."""

        XSpace: Callable[[], _XSpaceProto]


@dataclass(frozen=True)
class CalibrateOptions:
    """Typed calibration provenance and layer-normalization options."""

    chip: str = DEFAULT_CHIP
    model: str | None = None
    phase: str = DEFAULT_PHASE
    batch: int | None = None
    tp: int | None = None
    num_layers: int | None = None


class QuantileSummary(TypedDict, total=False):
    """Summary statistics (`p50`, `p90`, sample count `n`) for a latency series."""

    p50: float
    p90: float
    n: int
    step_idle_gap_us: float
    num_layers_norm: int
    unit: str


class CollectiveSummary(TypedDict):
    """Calibrated latency summary for one `(op, group_size, bytes)` bucket."""

    op: str
    group_size: int
    bytes: int
    latency_us: float
    hop_latency_us: float
    n: int


class StepCoverageSummary(TypedDict):
    """Calibrated fraction of tokens eligible for fused megakernel execution."""

    fused_eligible_fraction: float
    basis: str
    n: int


class CalibrationSourceMeta(TypedDict, total=False):
    """Provenance metadata for the input trace used during calibration."""

    kind: str
    file: str
    chip: str
    phase: str
    model: str
    batch: int
    tp: int


class CalibrationPayload(TypedDict, total=False):
    """Schema `advisor.measurements.v1` JSON payload returned by `calibrate_trace`."""

    schema: str
    source: CalibrationSourceMeta
    launch_gap_us: QuantileSummary
    collectives: list[CollectiveSummary]
    step_coverage: StepCoverageSummary
    step_time_us: QuantileSummary


@dataclass(frozen=True)
class _StepWindow:
    """Normalized execution step interval with optional token-eligibility counts."""

    start_us: float
    end_us: float
    dur_us: float
    phase: str | None = None
    active_tokens: int | None = None
    eligible_tokens: int | None = None


@dataclass(frozen=True)
class _ComputeEvent:
    """Normalized compute or collective trace event scoped to a rank stream."""

    rank_key: str
    stream_key: str
    start_us: float
    end_us: float
    dur_us: float
    collective_op: str | None = None
    group_size: int | None = None
    byte_size: int = 0


@dataclass(frozen=True)
class _ParsedTrace:
    """Intermediate normalized representation of an XPlane or Perfetto trace."""

    kind: str
    file_label: str
    steps_by_rank: dict[str, list[_StepWindow]]
    events_by_stream: dict[str, list[_ComputeEvent]]


def _percentile_prec(
    sorted_vals: Sequence[float], q: float, ndigits: int = QUANTILE_DECIMALS
) -> float:
    """Compute linear-interpolated quantile `q` rounded to `ndigits` decimals."""
    if not sorted_vals:
        return 0.0
    pos = q * (len(sorted_vals) - 1)
    lo = int(pos)
    hi = min(len(sorted_vals) - 1, lo + 1)
    frac = pos - lo
    return round(sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac, ndigits)


def _percentile(sorted_vals: Sequence[float], q: float) -> float:
    """Compute linear-interpolated quantile `q` in `[0, 1]` from sorted floats."""
    return _percentile_prec(sorted_vals, q, QUANTILE_DECIMALS)


def _classify_collective(name: str) -> str | None:
    """Return canonical collective op name if `name` is a collective event."""
    norm = name.lower().replace("_", "-")
    if any(sfx in norm for sfx in COLLECTIVE_IGNORE_SUFFIXES):
        return None
    for prefix, canonical in COLLECTIVE_PREFIXES:
        if prefix in norm:
            return canonical
    return None


def _is_module_wrapper(name: str, cat: str, line_name: str) -> bool:
    """Return True when an event or line wraps full-step XLA modules."""
    low_cat = cat.lower()
    low_line = line_name.lower()
    low_name = name.lower()
    return any(
        m in low_line or m in low_cat for m in XLA_MODULE_MARKERS
    ) or low_name.startswith("xla_module")


def _is_step_event(
    name: str, cat: str, line_name: str, args: Mapping[str, object]
) -> bool:
    """Return True when an event marks a model execution step window."""
    low_name = name.lower()
    return (
        "step" in cat.lower()
        or "step" in line_name.lower()
        or low_name.startswith("step")
        or low_name == "model_step"
        or "step" in args
        or "step_num" in args
    )


def _int_or_none(val: object) -> int | None:
    """Convert a JSON/proto integer-like scalar (excluding bool) to int."""
    match val:
        case bool():
            return None
        case int():
            return val
        case float():
            return int(val) if math.isfinite(val) else None
        case str():
            stripped = val.strip()
            return int(stripped) if stripped.isdigit() else None
        case _:
            return None


def _finite_float_or_none(val: object) -> float | None:
    """Convert an int/float scalar (excluding bool) to a finite float."""
    match val:
        case bool():
            return None
        case int() | float():
            out = float(val)
            return out if math.isfinite(out) else None
        case _:
            return None


def _build_step_window(
    start_us: float, dur_us: float, args: Mapping[str, object]
) -> _StepWindow:
    """Construct a `_StepWindow` from microsecond timing and event metadata."""
    match args.get("phase"):
        case str() as phase_raw if phase_raw:
            phase_str = phase_raw.lower()
        case _:
            phase_str = None
    act_tok = _int_or_none(args.get("active_tokens"))
    elig_tok = _int_or_none(args.get("fused_eligible_tokens"))
    return _StepWindow(
        start_us=start_us,
        end_us=start_us + dur_us,
        dur_us=dur_us,
        phase=phase_str,
        active_tokens=act_tok,
        eligible_tokens=elig_tok,
    )


def _build_compute_event(
    stream_ids: tuple[str, str],
    start_us: float,
    dur_us: float,
    name: str,
    args: Mapping[str, object],
) -> _ComputeEvent:
    """Construct a `_ComputeEvent` from microsecond timing and event metadata."""
    rank_key, stream_key = stream_ids
    coll_op = _classify_collective(name)
    grp_raw = _int_or_none(args.get("group_size"))
    grp = (
        grp_raw
        if grp_raw is not None and grp_raw >= MIN_COLLECTIVE_GROUP_SIZE
        else None
    )
    raw_bytes = args.get("bytes")
    if raw_bytes is None:
        raw_bytes = args.get("byte_size")
    nbytes = _int_or_none(raw_bytes) or 0
    return _ComputeEvent(
        rank_key=rank_key,
        stream_key=stream_key,
        start_us=start_us,
        end_us=start_us + dur_us,
        dur_us=dur_us,
        collective_op=coll_op,
        group_size=grp,
        byte_size=max(0, nbytes),
    )


def _propagate_perfetto_host_steps(
    steps_by_rank: dict[str, list[_StepWindow]],
    events_by_stream: Mapping[str, Sequence[_ComputeEvent]],
) -> None:
    """Attach host-PID step markers to device PIDs that lack local step tracks."""
    compute_ranks = {ev.rank_key for evs in events_by_stream.values() for ev in evs}
    if not compute_ranks:
        return
    host_steps = [
        sw
        for rk, steps in sorted(steps_by_rank.items())
        if rk not in compute_ranks
        for sw in steps
    ]
    if not host_steps:
        return
    for rk in sorted(compute_ranks):
        if not steps_by_rank.get(rk):
            steps_by_rank[rk] = list(host_steps)
    for rk in list(steps_by_rank):
        if rk not in compute_ranks:
            del steps_by_rank[rk]


def _read_perfetto_events(path: Path) -> list[object]:
    """Read, decompress if gzipped, and decode a Perfetto trace event list."""
    try:
        raw_bytes = path.read_bytes()
        if path.name.endswith(".gz"):
            raw_bytes = gzip.decompress(raw_bytes)
        text = raw_bytes.decode("utf-8")
        doc = json.loads(text)
    except (OSError, EOFError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        msg = f"Failed to read Perfetto trace {path}: {exc}"
        raise errors.MegakernelConfigError(msg) from exc
    match doc:
        case Mapping() as doc_map:
            events_raw = doc_map.get("traceEvents", [])
        case _:
            events_raw = doc
    match events_raw:
        case list() as event_list:
            return event_list
        case _:
            msg = f"Invalid Perfetto trace in {path}: expected traceEvents list"
            raise errors.MegakernelConfigError(msg)


def _record_perfetto_item(
    item: object,
    steps_by_rank: dict[str, list[_StepWindow]],
    events_by_stream: dict[str, list[_ComputeEvent]],
) -> None:
    """Record one raw Perfetto `X` event into step or compute stream buckets."""
    match item:
        case Mapping() as item_map if item_map.get("ph") == "X":
            pass
        case _:
            return
    start_us = _finite_float_or_none(item_map.get("ts"))
    dur_us = _finite_float_or_none(item_map.get("dur"))
    if start_us is None or dur_us is None or dur_us <= 0.0:
        return
    name = str(item_map.get("name", ""))
    cat = str(item_map.get("cat", ""))
    pid = _int_or_none(item_map.get("pid")) or 0
    tid = _int_or_none(item_map.get("tid")) or 0
    match item_map.get("args"):
        case Mapping() as args_map:
            args: Mapping[str, object] = args_map
        case _:
            args = {}
    rank_key = f"pid:{pid}"
    if _is_step_event(name, cat, "", args):
        steps_by_rank.setdefault(rank_key, []).append(
            _build_step_window(start_us, dur_us, args)
        )
    elif not _is_module_wrapper(name, cat, ""):
        stream_key = f"{rank_key}:tid:{tid}"
        events_by_stream.setdefault(stream_key, []).append(
            _build_compute_event((rank_key, stream_key), start_us, dur_us, name, args)
        )


def _parse_perfetto_file(path: Path) -> _ParsedTrace:
    """Parse a `.json` or `.json.gz` Perfetto trace into normalized streams."""
    steps_by_rank: dict[str, list[_StepWindow]] = {}
    events_by_stream: dict[str, list[_ComputeEvent]] = {}
    for item in _read_perfetto_events(path):
        _record_perfetto_item(item, steps_by_rank, events_by_stream)
    _propagate_perfetto_host_steps(steps_by_rank, events_by_stream)
    return _ParsedTrace(
        kind="perfetto",
        file_label=path.name,
        steps_by_rank=steps_by_rank,
        events_by_stream=events_by_stream,
    )


def _extract_xevent_stats(
    ev: _XEventProto, stat_meta: Mapping[int, _XStatMetaProto]
) -> dict[str, object]:
    """Extract a dictionary of stat values from an XEvent proto."""
    out: dict[str, object] = {}
    for st in ev.stats:
        meta = stat_meta.get(st.metadata_id)
        if meta is None:
            continue
        key = str(meta.name)
        if st.str_value:
            out[key] = str(st.str_value)
        elif st.double_value != 0.0:
            out[key] = float(st.double_value)
        elif st.uint64_value != 0:
            out[key] = int(st.uint64_value)
        else:
            out[key] = int(st.int64_value)
    return out


def _ingest_xplane(
    plane: _XPlaneProto,
    rank_key: str,
    steps_by_rank: dict[str, list[_StepWindow]],
    events_by_stream: dict[str, list[_ComputeEvent]],
) -> None:
    """Ingest all lines from a device XPlane into step or compute streams."""
    ev_meta = dict(plane.event_metadata.items())
    stat_meta = dict(plane.stat_metadata.items())
    for line in plane.lines:
        line_name = str(line.name or line.display_name or "")
        line_id = int(line.id)
        line_base_us = float(line.timestamp_ns) / NS_PER_US
        stream_key = f"{rank_key}:line:{line_id}:{line_name}"
        for ev in line.events:
            dur_us = float(ev.duration_ps) / PS_PER_US
            if dur_us <= 0.0:
                continue
            start_us = line_base_us + float(ev.offset_ps) / PS_PER_US
            meta = ev_meta.get(ev.metadata_id)
            ev_name = (
                str(meta.display_name or meta.name)
                if meta is not None
                else f"ev_{ev.metadata_id}"
            )
            args = _extract_xevent_stats(ev, stat_meta)
            if _is_step_event(ev_name, "", line_name, args):
                steps_by_rank.setdefault(rank_key, []).append(
                    _build_step_window(start_us, dur_us, args)
                )
            elif not _is_module_wrapper(ev_name, "", line_name):
                events_by_stream.setdefault(stream_key, []).append(
                    _build_compute_event(
                        (rank_key, stream_key), start_us, dur_us, ev_name, args
                    )
                )


def _discover_xplane_files(path: Path) -> list[Path]:
    """Discover `.xplane.pb` files from a single path or profile directory."""
    if path.is_file():
        return [path]
    if merge_xplane is None:
        raise XPlaneToolUnavailableError
    sessions = merge_xplane.discover_sessions([str(path)])
    if len(sessions) > 1:
        dirs = ", ".join(str(s.directory) for s in sessions)
        msg = (
            f"Multiple XPlane sessions found under {path} ({dirs}); "
            "pass a single session directory"
        )
        raise errors.MegakernelConfigError(msg)
    if len(sessions) == 1:
        return [Path(cap.path) for cap in sessions[0].captures]
    return sorted(path.rglob("*.xplane.pb"))


def _parse_xplane_target(path: Path) -> _ParsedTrace:
    """Parse one or more `.xplane.pb` files using `vllm_torchtpu.tools.xplane_pb2`."""
    if xplane_pb2 is None:
        msg = (
            "XPlane parsing requires vllm_torchtpu.tools.xplane_pb2, "
            "which could not be loaded"
        )
        raise errors.MegakernelConfigError(msg)
    pb_mod = cast("_XPlaneModuleProto", xplane_pb2)
    files = _discover_xplane_files(path)
    if not files:
        msg = f"No .xplane.pb files found under {path}"
        raise errors.MegakernelConfigError(msg)

    steps_by_rank: dict[str, list[_StepWindow]] = {}
    events_by_stream: dict[str, list[_ComputeEvent]] = {}
    for file_idx, pb_path in enumerate(files):
        space = pb_mod.XSpace()
        try:
            space.ParseFromString(pb_path.read_bytes())
        except (OSError, DecodeError) as exc:
            msg = f"Failed to parse XPlane trace {pb_path}: {exc}"
            raise errors.MegakernelConfigError(msg) from exc
        for plane in space.planes:
            pname = str(plane.name)
            if (
                not pname.startswith(XPLANE_DEVICE_PLANE_PREFIX)
                or XPLANE_SPARSE_FILTER in pname.lower()
            ):
                continue
            rank_key = f"file:{file_idx}:plane:{plane.id}:{pname}"
            _ingest_xplane(plane, rank_key, steps_by_rank, events_by_stream)
    return _ParsedTrace(
        kind="xplane",
        file_label=path.name,
        steps_by_rank=steps_by_rank,
        events_by_stream=events_by_stream,
    )


def _steady_state_steps(steps: Sequence[_StepWindow]) -> list[_StepWindow]:
    """Sort step windows and drop compilation/warmup step 0 under both thresholds.

    Drops step 0 when at least `MIN_STEPS_FOR_WARMUP_DROP` (3) steps exist, or when
    at least `MIN_WARMUP_DROP_STEPS` (2) steps exist and step 0 exceeds
    `WARMUP_OUTLIER_RATIO` times the minimum duration of subsequent steps.
    """
    ordered = sorted(steps, key=lambda sw: sw.start_us)
    if len(ordered) >= MIN_WARMUP_DROP_STEPS:
        min_steady_step_dur_us = min(s.dur_us for s in ordered[1:])
        if ordered[0].dur_us > WARMUP_OUTLIER_RATIO * min_steady_step_dur_us:
            return ordered[1:]
        if len(ordered) >= MIN_STEPS_FOR_WARMUP_DROP:
            return ordered[1:]
    return ordered


def _scan_event_window(
    window_evs: Sequence[_ComputeEvent],
) -> tuple[list[float], list[_ComputeEvent]]:
    """Collect positive consecutive gaps and collectives within one event window."""
    gaps: list[float] = []
    collectives: list[_ComputeEvent] = []
    for idx, cur_ev in enumerate(window_evs):
        if cur_ev.collective_op is not None:
            collectives.append(cur_ev)
        if idx > 0:
            gap = cur_ev.start_us - window_evs[idx - 1].end_us
            if 0.0 < gap <= MAX_INTRA_STEP_GAP_US:
                gaps.append(gap)
    return gaps, collectives


def _collect_stream_gaps_and_collectives(
    evs: Sequence[_ComputeEvent],
    steady_steps: Sequence[_StepWindow],
) -> tuple[list[float], list[_ComputeEvent]]:
    """Extract intra-step positive launch gaps and steady-state collective events."""
    ordered = sorted(evs, key=lambda e: e.start_us)
    if steady_steps:
        starts = [ev.start_us for ev in ordered]
        gaps: list[float] = []
        collectives: list[_ComputeEvent] = []
        for sw in steady_steps:
            lo = bisect_left(starts, sw.start_us)
            hi = bisect_left(starts, sw.end_us)
            w_gaps, w_colls = _scan_event_window(ordered[lo:hi])
            gaps.extend(w_gaps)
            collectives.extend(w_colls)
        return gaps, collectives

    trimmed = ordered[1:] if len(ordered) >= MIN_EVENTS_FOR_WARMUP_DROP else ordered
    return _scan_event_window(trimmed)


def _summarize_collectives(
    collectives: Sequence[_ComputeEvent],
    default_group_size: int | None = None,
) -> list[CollectiveSummary]:
    """Group collective events by `(op, group_size, bytes)` and compute medians."""
    buckets: dict[tuple[str, int, int], list[float]] = {}
    for ev in collectives:
        grp = ev.group_size if ev.group_size is not None else default_group_size
        if ev.collective_op is None or grp is None or grp < MIN_COLLECTIVE_GROUP_SIZE:
            continue
        key = (ev.collective_op, grp, ev.byte_size)
        buckets.setdefault(key, []).append(ev.dur_us)

    rows: list[CollectiveSummary] = []
    for (op_name, grp, nbytes), durs in sorted(buckets.items()):
        s_durs = sorted(durs)
        med_us = _percentile(s_durs, 0.5)
        hops = max(1, round(log2(max(MIN_COLLECTIVE_GROUP_SIZE, grp))))
        rows.append(
            {
                "op": op_name,
                "group_size": grp,
                "bytes": nbytes,
                "latency_us": med_us,
                "hop_latency_us": round(med_us / hops, QUANTILE_DECIMALS),
                "n": len(s_durs),
            }
        )
    return rows


def _summarize_step_coverage(
    steady_steps: Sequence[_StepWindow],
) -> StepCoverageSummary | None:
    """Compute `step_coverage` when step token or phase metadata is present."""
    total_active = 0
    total_elig = 0
    meta_count = 0
    for sw in steady_steps:
        if (
            sw.active_tokens is not None
            and sw.eligible_tokens is not None
            and sw.active_tokens > 0
        ):
            total_active += sw.active_tokens
            total_elig += sw.eligible_tokens
            meta_count += 1
        elif sw.phase is not None:
            total_active += 1
            total_elig += 1 if sw.phase in FUSED_ELIGIBLE_PHASES else 0
            meta_count += 1
    if meta_count == 0 or total_active == 0:
        return None
    return {
        "fused_eligible_fraction": round(total_elig / total_active, QUANTILE_DECIMALS),
        "basis": STEP_COVERAGE_BASIS,
        "n": meta_count,
    }


def _load_parsed_trace(trace_path: str | Path) -> _ParsedTrace:
    """Validate `trace_path` and dispatch to the XPlane or Perfetto parser."""
    path = Path(trace_path)
    if not path.exists():
        msg = f"Missing trace path: {path}"
        raise errors.MegakernelConfigError(msg)
    if path.is_dir() or path.name.endswith(".xplane.pb"):
        return _parse_xplane_target(path)
    if path.name.endswith((".json", ".json.gz")):
        return _parse_perfetto_file(path)
    msg = f"Unsupported trace format: {path.name}"
    raise errors.MegakernelConfigError(msg)


def _build_source_meta(
    parsed: _ParsedTrace,
    opts: CalibrateOptions,
) -> CalibrationSourceMeta:
    """Assemble provenance metadata for the calibrated output payload."""
    source_meta: CalibrationSourceMeta = {
        "kind": parsed.kind,
        "file": parsed.file_label,
        "chip": opts.chip,
        "phase": opts.phase,
    }
    if opts.model is not None:
        source_meta["model"] = opts.model
    if opts.batch is not None:
        source_meta["batch"] = opts.batch
    if opts.tp is not None:
        source_meta["tp"] = opts.tp
    return source_meta


def _merged_busy_duration_us(
    intervals: Sequence[tuple[float, float]],
) -> float:
    """Compute union duration across sorted or overlapping `(start, end)` spans."""
    if not intervals:
        return 0.0
    ordered = sorted(intervals, key=lambda pair: pair[0])
    cur_s, cur_e = ordered[0]
    total = 0.0
    for s_us, e_us in ordered[1:]:
        if s_us <= cur_e:
            cur_e = max(cur_e, e_us)
        else:
            total += cur_e - cur_s
            cur_s, cur_e = s_us, e_us
    return total + (cur_e - cur_s)


def _resolve_norm_layers(model: str | None, explicit_layers: int | None) -> int:
    """Resolve layer count for sublayer boundary normalization."""
    if explicit_layers is not None and explicit_layers > 0:
        return explicit_layers
    if model is not None:
        return resolve_model_profile(model).num_layers
    return DEFAULT_CALIBRATE_LAYERS


def _compute_conserved_launch_gap(
    events_by_stream: Mapping[str, Sequence[_ComputeEvent]],
    eval_steps_by_rank: Mapping[str, Sequence[_StepWindow]],
    norm_layers: int,
    *,
    explicit_norm: bool,
) -> QuantileSummary | None:
    """Normalize interval-merged step idle time by `2 * norm_layers` boundaries."""
    events_by_rank: dict[str, list[_ComputeEvent]] = {}
    for evs in events_by_stream.values():
        for ev in evs:
            events_by_rank.setdefault(ev.rank_key, []).append(ev)

    boundaries = SUBLAYER_BOUNDARIES_PER_LAYER * norm_layers
    step_idles: list[float] = []
    max_step_ops = 0
    for rk in sorted(eval_steps_by_rank):
        rk_evs = events_by_rank.get(rk, ())
        for sw in eval_steps_by_rank[rk]:
            clipped = [
                (max(sw.start_us, ev.start_us), min(sw.end_us, ev.end_us))
                for ev in rk_evs
                if ev.end_us > sw.start_us and ev.start_us < sw.end_us
            ]
            max_step_ops = max(max_step_ops, len(clipped))
            if clipped:
                busy_us = _merged_busy_duration_us(clipped)
                step_idles.append(max(0.0, sw.dur_us - busy_us))

    use_conserved = bool(step_idles) and (
        (explicit_norm and max_step_ops >= MIN_STEP_OPS_FOR_EXPLICIT_NORM)
        or max_step_ops >= boundaries
    )
    if not use_conserved:
        return None
    sorted_idles = sorted(step_idles)
    p50_idle = _percentile_prec(sorted_idles, 0.5, CONSERVED_GAP_DECIMALS)
    p90_idle = _percentile_prec(sorted_idles, 0.9, CONSERVED_GAP_DECIMALS)
    return {
        "p50": round(p50_idle / boundaries, CONSERVED_GAP_DECIMALS),
        "p90": round(p90_idle / boundaries, CONSERVED_GAP_DECIMALS),
        "n": len(sorted_idles),
        "step_idle_gap_us": round(p50_idle, QUANTILE_DECIMALS),
        "num_layers_norm": norm_layers,
        "unit": CONSERVED_GAP_UNIT,
    }


def _filter_phase_steps(
    steady_by_rank: Mapping[str, list[_StepWindow]],
    phase: str,
) -> tuple[dict[str, list[_StepWindow]], list[_StepWindow]]:
    """Filter steady-state step windows by requested serving phase."""
    if phase == "mixed":
        all_steps = [sw for rk in sorted(steady_by_rank) for sw in steady_by_rank[rk]]
        return {rk: list(steps) for rk, steps in steady_by_rank.items()}, all_steps

    allowed = FUSED_ELIGIBLE_PHASES if phase == "decode" else (phase,)
    filtered = {
        rk: [sw for sw in steps if sw.phase is None or sw.phase in allowed]
        for rk, steps in steady_by_rank.items()
    }
    matched = [sw for steps in filtered.values() for sw in steps]
    if not matched and any(steady_by_rank.values()):
        found_phases = {
            sw.phase
            for steps in steady_by_rank.values()
            for sw in steps
            if sw.phase is not None
        }
        msg = (
            f"No steps match requested phase '{phase}'; found phases: "
            f"{sorted(found_phases)}"
        )
        raise errors.MegakernelConfigError(msg)
    return filtered, matched


def _compute_launch_gap_and_collectives(
    parsed: _ParsedTrace,
    eval_steps_by_rank: Mapping[str, Sequence[_StepWindow]],
    opts: CalibrateOptions,
) -> tuple[QuantileSummary, list[_ComputeEvent]]:
    """Collect intra-step gaps and collective events and compute launch gap summary."""
    all_gaps: list[float] = []
    all_collectives: list[_ComputeEvent] = []
    for stream_key in sorted(parsed.events_by_stream):
        evs = parsed.events_by_stream[stream_key]
        if not evs:
            continue
        rk = evs[0].rank_key
        gaps, colls = _collect_stream_gaps_and_collectives(
            evs, eval_steps_by_rank.get(rk, ())
        )
        all_gaps.extend(gaps)
        all_collectives.extend(colls)

    norm_layers = _resolve_norm_layers(opts.model, opts.num_layers)
    conserved_gap = _compute_conserved_launch_gap(
        parsed.events_by_stream,
        eval_steps_by_rank,
        norm_layers,
        explicit_norm=opts.num_layers is not None,
    )
    sorted_gaps = sorted(all_gaps)
    launch_gap: QuantileSummary = conserved_gap or (
        {
            "p50": _percentile(sorted_gaps, 0.5),
            "p90": _percentile(sorted_gaps, 0.9),
            "n": len(sorted_gaps),
        }
        if sorted_gaps
        else {"n": 0}
    )
    return launch_gap, all_collectives


def calibrate_trace(
    trace_path: str | Path,
    out_path: str | Path | None = None,
    *,
    options: CalibrateOptions | None = None,
) -> CalibrationPayload:
    """Calibrate advisor priors from an XPlane or Perfetto trace file or directory."""
    opts = options or CalibrateOptions()
    parsed = _load_parsed_trace(trace_path)
    steady_by_rank = {
        rk: _steady_state_steps(steps) for rk, steps in parsed.steps_by_rank.items()
    }
    all_steady_steps = [
        sw for rk in sorted(steady_by_rank) for sw in steady_by_rank[rk]
    ]
    eval_steps_by_rank, eval_steady_steps = _filter_phase_steps(
        steady_by_rank, opts.phase
    )
    launch_gap, all_collectives = _compute_launch_gap_and_collectives(
        parsed, eval_steps_by_rank, opts
    )
    payload: CalibrationPayload = {
        "schema": MEASUREMENTS_SCHEMA,
        "source": _build_source_meta(parsed, opts),
        "launch_gap_us": launch_gap,
        "collectives": _summarize_collectives(all_collectives, opts.tp),
    }
    cov = _summarize_step_coverage(all_steady_steps)
    if cov is not None:
        payload["step_coverage"] = cov
    if eval_steady_steps:
        step_durs = sorted(sw.dur_us for sw in eval_steady_steps)
        payload["step_time_us"] = {
            "p50": _percentile(step_durs, 0.5),
            "p90": _percentile(step_durs, 0.9),
            "n": len(step_durs),
        }
    if out_path is not None:
        dest = Path(out_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload
