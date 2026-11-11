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
"""Unit tests for the megakernel advisor trace calibration engine."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from vllm_torchtpu.megakernel import advisor
from vllm_torchtpu.megakernel.advisor import calibrate, constants
from vllm_torchtpu.megakernel.core import errors
from vllm_torchtpu.tools import xplane_pb2

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from typing import Protocol

    class _MutableXStatProto(Protocol):
        """Mutable XStat protobuf message protocol for test fixture construction."""

        metadata_id: int
        int64_value: int
        double_value: float
        str_value: str

    class _StatContainerProto(Protocol):
        """Repeated XStat container protocol."""

        def add(self) -> _MutableXStatProto:
            """Append and return a new XStat entry."""

    class _MutableXEventProto(Protocol):
        """Mutable XEvent protobuf message protocol for test fixture construction."""

        metadata_id: int
        offset_ps: int
        duration_ps: int
        stats: _StatContainerProto

    class _EventContainerProto(Protocol):
        """Repeated XEvent container protocol."""

        def add(self) -> _MutableXEventProto:
            """Append and return a new XEvent entry."""

    class _MutableXLineProto(Protocol):
        """Mutable XLine protobuf message protocol for test fixture construction."""

        id: int
        name: str
        timestamp_ns: int
        events: _EventContainerProto


PERFETTO_FIXTURE = (
    Path(__file__).resolve().parent / "data" / "advisor" / "synthetic_perfetto.json.gz"
)
FIXTURE_AR_BYTES = 32768
FIXTURE_A2A_BYTES = 65536
FIXTURE_GROUP_SIZE = 8
PERFETTO_EXPECTED_GAPS = 9
MULTI_RANK_EXPECTED_GAPS = 18
STEADY_DECODE_VERIFY_STEPS = 3


def _add_int_stat(ev: _MutableXEventProto, meta_id: int, value: int) -> None:
    """Append an int64 XStat entry to `ev`."""
    st = ev.stats.add()
    st.metadata_id = meta_id
    st.int64_value = value


def _add_str_stat(ev: _MutableXEventProto, meta_id: int, value: str) -> None:
    """Append a string XStat entry to `ev`."""
    st = ev.stats.add()
    st.metadata_id = meta_id
    st.str_value = value


def _write_json_trace(tmp_path: Path, filename: str, payload: object) -> Path:
    """Write a JSON payload to `tmp_path / filename` and return its path."""
    trace_file = tmp_path / filename
    trace_file.write_text(json.dumps(payload), encoding="utf-8")
    return trace_file


def _make_step_events(
    steps: Sequence[tuple[int, str, int, int]],
) -> list[dict[str, object]]:
    """Build Perfetto `ModelStep` events from `(step_num, phase, ts, dur)` tuples."""
    return [
        {
            "ph": "X",
            "name": f"Step {step_num}",
            "cat": "ModelStep",
            "ts": ts,
            "dur": dur,
            "pid": 1,
            "tid": 1,
            "args": {"step_num": step_num, "phase": phase},
        }
        for step_num, phase, ts, dur in steps
    ]


def _write_xplane_dir(root: Path, captures: Mapping[str, bytes]) -> Path:
    """Create a profile capture directory tree populated with `.xplane.pb` bytes."""
    session_dir = root / "plugins" / "profile" / "2026_11_04"
    session_dir.mkdir(parents=True, exist_ok=True)
    for name, raw in captures.items():
        (session_dir / name).write_bytes(raw)
    return root


def _advise_qwen3_30b_decode(meas_path: Path) -> advisor.AdvisorReport:
    """Run `advisor.advise` for Qwen3-30B-A3B decode on tpu7x with calibration."""
    profile = advisor.resolve_model_profile("Qwen/Qwen3-30B-A3B-FP8")
    chip = advisor.resolve_chip_spec("tpu7x")
    return advisor.advise(
        profile,
        advisor.WorkloadSpec(phase="decode", batch=16, tp=8, ep=8),
        chip,
        measurements=meas_path,
    )


def _append_steady_step_events(
    step_line: _MutableXLineProto,
    ops_line: _MutableXLineProto,
    step_spec: tuple[int, int, int, str, int, int],
    time_shift_us: int,
) -> None:
    """Append one steady-state step and its 4 compute ops to `step_line`/`ops_line`."""
    ps_per_us = int(constants.PS_PER_US)
    step_num, base_us, dur_us, phase_str, act_tok, elig_tok = step_spec
    t0 = time_shift_us + base_us
    st_ev = step_line.events.add()
    st_ev.metadata_id = 1
    st_ev.offset_ps = t0 * ps_per_us
    st_ev.duration_ps = dur_us * ps_per_us
    _add_int_stat(st_ev, 1, step_num)
    _add_str_stat(st_ev, 2, phase_str)
    _add_int_stat(st_ev, 3, act_tok)
    _add_int_stat(st_ev, 4, elig_tok)

    for meta_id, rel_us, op_dur_us, coll_bytes in (
        (2, 5.0, 10, 0),
        (3, 17.0, 12, FIXTURE_AR_BYTES),
        (4, 31.5, 15, 0),
        (5, 50.0, 18, FIXTURE_A2A_BYTES),
    ):
        op_ev = ops_line.events.add()
        op_ev.metadata_id = meta_id
        op_ev.offset_ps = int((t0 + rel_us) * ps_per_us)
        op_ev.duration_ps = op_dur_us * ps_per_us
        if coll_bytes > 0:
            _add_int_stat(op_ev, 5, FIXTURE_GROUP_SIZE)
            _add_int_stat(op_ev, 6, coll_bytes)


def _build_synthetic_xspace_bytes(time_shift_us: int = 0) -> bytes:
    """Build a deterministic XSpace proto with device step and XLA op lines."""
    ps_per_us = int(constants.PS_PER_US)
    space = xplane_pb2.XSpace()
    space.hostnames.append("tpu-worker-0")

    for pid, pname in ((99, "/host:metadata"), (98, "/host:CPU")):
        plane = space.planes.add()
        plane.id = pid
        plane.name = pname

    dev_plane = space.planes.add()
    dev_plane.id = 1
    dev_plane.name = "/device:TPU:0"
    for ev_id, ev_name in (
        (1, "Step"),
        (2, "qkv_proj"),
        (3, "all-reduce.1"),
        (4, "attn_core"),
        (5, "all-to-all.2"),
    ):
        dev_plane.event_metadata[ev_id].id = ev_id
        dev_plane.event_metadata[ev_id].name = ev_name

    for st_id, st_name in (
        (1, "step_num"),
        (2, "phase"),
        (3, "active_tokens"),
        (4, "fused_eligible_tokens"),
        (5, "group_size"),
        (6, "bytes"),
    ):
        dev_plane.stat_metadata[st_id].id = st_id
        dev_plane.stat_metadata[st_id].name = st_name

    step_line = dev_plane.lines.add()
    step_line.id = 10
    step_line.name = "Steps"
    ops_line = dev_plane.lines.add()
    ops_line.id = 1
    ops_line.name = "XLA Ops"

    w_step = step_line.events.add()
    w_step.metadata_id = 1
    w_step.offset_ps = time_shift_us * ps_per_us
    w_step.duration_ps = 140 * ps_per_us
    _add_int_stat(w_step, 1, 0)
    _add_str_stat(w_step, 2, "decode")

    for rel_us, meta_id in ((10, 2), (55, 3)):
        w_op = ops_line.events.add()
        w_op.metadata_id = meta_id
        w_op.offset_ps = (time_shift_us + rel_us) * ps_per_us
        w_op.duration_ps = 20 * ps_per_us

    for spec in (
        (1, 200, 98, "decode", 16, 16),
        (2, 400, 100, "verify", 16, 16),
        (3, 600, 100, "decode", 16, 16),
        (4, 800, 102, "prefill", 16, 0),
    ):
        _append_steady_step_events(step_line, ops_line, spec, time_shift_us)
    return bytes(space.SerializeToString())


def test_calibrate_perfetto_extracts_launch_gap_and_collectives(
    tmp_path: Path,
) -> None:
    """Verify Perfetto trace calibration extracts steady-state gaps and collectives."""
    assert PERFETTO_FIXTURE.is_file()
    out_json = tmp_path / "perfetto_measurements.json"

    req_batch = 16
    req_tp = 8
    rc = advisor.main(
        [
            "calibrate",
            "--trace",
            str(PERFETTO_FIXTURE),
            "--chip",
            "tpu7x",
            "--model",
            "Qwen/Qwen3-30B-A3B-FP8",
            "--batch",
            str(req_batch),
            "--tp",
            str(req_tp),
            "--out",
            str(out_json),
        ]
    )
    assert rc == 0
    assert out_json.is_file()

    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["schema"] == constants.MEASUREMENTS_SCHEMA
    assert payload["source"]["kind"] == "perfetto"
    assert payload["source"]["batch"] == req_batch
    assert payload["source"]["tp"] == req_tp

    lg = payload["launch_gap_us"]
    assert lg["p50"] == pytest.approx(2.5)
    assert lg["p90"] == pytest.approx(3.5)
    assert lg["n"] == PERFETTO_EXPECTED_GAPS

    coll_by_op = {item["op"]: item for item in payload["collectives"]}
    assert set(coll_by_op) == {"all-reduce", "all-to-all"}
    assert coll_by_op["all-reduce"]["group_size"] == FIXTURE_GROUP_SIZE
    assert coll_by_op["all-reduce"]["bytes"] == FIXTURE_AR_BYTES
    assert coll_by_op["all-reduce"]["latency_us"] == pytest.approx(12.0)
    assert coll_by_op["all-reduce"]["hop_latency_us"] == pytest.approx(4.0)
    assert coll_by_op["all-reduce"]["n"] == STEADY_DECODE_VERIFY_STEPS
    assert coll_by_op["all-to-all"]["latency_us"] == pytest.approx(18.0)
    assert coll_by_op["all-to-all"]["n"] == STEADY_DECODE_VERIFY_STEPS

    cov = payload["step_coverage"]
    assert cov["fused_eligible_fraction"] == pytest.approx(0.75)
    assert cov["basis"] == constants.STEP_COVERAGE_BASIS

    st = payload["step_time_us"]
    assert st["p50"] == pytest.approx(100.0)
    assert st["n"] == STEADY_DECODE_VERIFY_STEPS

    report = _advise_qwen3_30b_decode(out_json)
    assert report.confidence == "validated_end_to_end"
    assert report.metrics["launch_gap_us"]["tag"] == "MEASURED"
    assert report.metrics["launch_gap_us"]["value"] == pytest.approx(2.5)
    assert report.metrics["collective_latency_us"]["tag"] == "MEASURED"
    assert report.metrics["collective_latency_us"]["value"] == pytest.approx(4.0)
    assert report.metrics["step_coverage"]["tag"] == "MEASURED"
    assert report.metrics["step_coverage"]["value"] == pytest.approx(0.75)


def test_calibrate_xplane_roundtrip_with_in_repo_parser(tmp_path: Path) -> None:
    """Verify single-file and multi-rank XPlane calibration via xplane_pb2."""
    single_pb = tmp_path / "single_rank.xplane.pb"
    single_pb.write_bytes(_build_synthetic_xspace_bytes(time_shift_us=0))

    out_single = tmp_path / "xplane_single_measurements.json"
    opts = advisor.CalibrateOptions(chip="tpu7x", model="Qwen/Qwen3-30B-A3B-FP8")
    payload = advisor.calibrate_trace(single_pb, out_single, options=opts)
    assert out_single.is_file()
    assert payload["schema"] == constants.MEASUREMENTS_SCHEMA
    assert payload["source"]["kind"] == "xplane"
    assert payload["launch_gap_us"]["p50"] == pytest.approx(2.5)
    assert payload["launch_gap_us"]["p90"] == pytest.approx(3.5)
    assert payload["launch_gap_us"]["n"] == PERFETTO_EXPECTED_GAPS
    assert payload["step_coverage"]["fused_eligible_fraction"] == pytest.approx(0.75)
    assert payload["step_time_us"]["p50"] == pytest.approx(100.0)

    decode_root = _write_xplane_dir(
        tmp_path / "decode_trace",
        {
            "rank0_worker0.xplane.pb": _build_synthetic_xspace_bytes(time_shift_us=0),
            "rank1_worker1.xplane.pb": _build_synthetic_xspace_bytes(time_shift_us=7),
        },
    )
    out_multi = tmp_path / "xplane_multi_measurements.json"
    rc = advisor.main(
        ["calibrate", "--trace", str(decode_root), "--out", str(out_multi)]
    )
    assert rc == 0
    multi_payload = json.loads(out_multi.read_text(encoding="utf-8"))
    assert multi_payload["launch_gap_us"]["p50"] == pytest.approx(2.5)
    assert multi_payload["launch_gap_us"]["p90"] == pytest.approx(3.5)
    assert multi_payload["launch_gap_us"]["n"] == MULTI_RANK_EXPECTED_GAPS

    report = _advise_qwen3_30b_decode(out_multi)
    assert report.confidence == "validated_end_to_end"
    assert report.metrics["launch_gap_us"]["tag"] == "MEASURED"
    assert report.metrics["collective_latency_us"]["tag"] == "MEASURED"


def test_calibrate_float_stats_and_empty_gaps_preserve_assumed_prior(
    tmp_path: Path,
) -> None:
    """Verify float-valued token stats parse cleanly and n=0 gaps stay ASSUMED."""
    trace_file = _write_json_trace(
        tmp_path,
        "single_op_float_stats.json",
        {
            "traceEvents": [
                {
                    "ph": "X",
                    "name": "Step",
                    "cat": "step",
                    "pid": 1,
                    "tid": 1,
                    "ts": 100.0,
                    "dur": 50.0,
                    "args": {
                        "step_num": 1.0,
                        "phase": "decode",
                        "active_tokens": 16.0,
                        "fused_eligible_tokens": 12.0,
                    },
                },
                {
                    "ph": "X",
                    "name": "all-reduce.1",
                    "cat": "op",
                    "pid": 1,
                    "tid": 2,
                    "ts": 110.0,
                    "dur": 9.0,
                    "args": {"group_size": 8.0, "bytes": 32768.0},
                },
            ]
        },
    )
    out_json = tmp_path / "empty_gap_measurements.json"
    payload = advisor.calibrate_trace(trace_file, out_json)
    assert payload["launch_gap_us"]["n"] == 0
    assert "p50" not in payload["launch_gap_us"]
    assert payload["step_coverage"]["fused_eligible_fraction"] == pytest.approx(0.75)

    for model_id in ("qwen3.5-397b", "kimi-k3"):
        prof = advisor.resolve_model_profile(model_id)
        rep = advisor.advise(
            prof,
            advisor.WorkloadSpec(phase="decode", batch=16, tp=8, ep=8),
            advisor.resolve_chip_spec("tpu7x"),
            measurements=out_json,
        )
        assert rep.metrics["launch_gap_us"]["tag"] == "ASSUMED"
        assert rep.metrics["collective_latency_us"]["tag"] == "MEASURED"
        assert rep.metrics["step_coverage"]["tag"] == "MEASURED"


def test_calibrate_rejects_multi_session_and_unannotated_collectives(
    tmp_path: Path,
) -> None:
    """Verify multi-session error guard and collective group_size fallback to tp."""
    phased_root = tmp_path / "phased_run"
    raw = _build_synthetic_xspace_bytes(time_shift_us=0)
    for phase_dir in ("prefill_only", "decode_only"):
        _write_xplane_dir(phased_root / phase_dir, {"rank0_worker0.xplane.pb": raw})
    with pytest.raises(
        errors.MegakernelConfigError, match="Multiple XPlane sessions found"
    ):
        advisor.calibrate_trace(phased_root)

    with pytest.raises(errors.MegakernelConfigError, match="Missing trace path"):
        advisor.calibrate_trace(tmp_path / "nonexistent.json")

    bad_ext = tmp_path / "trace.txt"
    bad_ext.write_text("plain text", encoding="utf-8")
    with pytest.raises(errors.MegakernelConfigError, match="Unsupported trace format"):
        advisor.calibrate_trace(bad_ext)

    empty_dir = tmp_path / "empty_xplane_dir"
    empty_dir.mkdir()
    with pytest.raises(errors.MegakernelConfigError, match=r"No \.xplane\.pb files"):
        advisor.calibrate_trace(empty_dir)

    no_grp_trace = _write_json_trace(
        tmp_path,
        "unannotated_collective.json",
        {
            "traceEvents": [
                {
                    "ph": "X",
                    "name": "all-reduce-start.1",
                    "cat": "op",
                    "pid": 1,
                    "tid": 2,
                    "ts": 100.0,
                    "dur": 9.0,
                    "args": {"bytes": 16384},
                },
                {
                    "ph": "X",
                    "name": "all-reduce-done.1",
                    "cat": "op",
                    "pid": 1,
                    "tid": 2,
                    "ts": 112.0,
                    "dur": 1.0,
                    "args": {"group_size": 4, "bytes": 16384},
                },
            ]
        },
    )
    skipped_payload = advisor.calibrate_trace(no_grp_trace)
    assert skipped_payload["collectives"] == []

    fallback_tp = 4
    tp_payload = advisor.calibrate_trace(
        no_grp_trace, options=advisor.CalibrateOptions(tp=fallback_tp)
    )
    assert len(tp_payload["collectives"]) == 1
    assert tp_payload["collectives"][0]["group_size"] == fallback_tp
    assert tp_payload["collectives"][0]["hop_latency_us"] == pytest.approx(4.5)


def test_steady_state_steps_drops_compilation_warmup_outlier(tmp_path: Path) -> None:
    """Verifies 2-step trace drops step 0 if it is a severe compilation outlier."""
    events = _make_step_events([(0, "decode", 0, 30000), (1, "decode", 30000, 5000)])
    trace_file = _write_json_trace(tmp_path, "outlier_trace.json", events)
    cal = calibrate.calibrate_trace(
        trace_file, options=calibrate.CalibrateOptions(phase="decode")
    )
    assert cal["step_time_us"]["n"] == 1
    assert cal["step_time_us"]["p50"] == pytest.approx(5000.0)


def test_steady_state_steps_keeps_normal_two_step_trace(tmp_path: Path) -> None:
    """Verifies 2-step trace preserves step 0 if within normal variation."""
    events = _make_step_events([(0, "decode", 0, 5200), (1, "decode", 5200, 5000)])
    trace_file = _write_json_trace(tmp_path, "normal_trace.json", events)
    cal = calibrate.calibrate_trace(
        trace_file, options=calibrate.CalibrateOptions(phase="decode")
    )
    expected_kept = 2
    assert cal["step_time_us"]["n"] == expected_kept
    assert cal["step_time_us"]["p50"] == pytest.approx(5100.0)


def test_steady_state_steps_drops_step_zero_when_three_steps_exist(
    tmp_path: Path,
) -> None:
    """Verifies 3-step trace drops normal step 0 under MIN_STEPS_FOR_WARMUP_DROP."""
    events = _make_step_events(
        [
            (0, "decode", 0, 5100),
            (1, "decode", 6000, 5000),
            (2, "decode", 12000, 5050),
        ]
    )
    trace_file = _write_json_trace(tmp_path, "three_step_trace.json", events)
    cal = calibrate.calibrate_trace(
        trace_file, options=calibrate.CalibrateOptions(phase="decode")
    )
    expected_kept = 2
    assert cal["step_time_us"]["n"] == expected_kept
    assert cal["step_time_us"]["p50"] == pytest.approx(5025.0)


def test_calibrate_trace_filters_by_requested_phase(tmp_path: Path) -> None:
    """Verifies calibrate_trace isolates step metrics by requested phase."""
    events = _make_step_events(
        [
            (0, "decode", 1000, 2000),
            (1, "decode", 5000, 2000),
            (2, "decode", 8000, 2100),
            (3, "prefill", 12000, 8000),
        ]
    )
    trace_file = _write_json_trace(tmp_path, "filter_trace.json", events)

    cal_decode = calibrate.calibrate_trace(
        trace_file, options=calibrate.CalibrateOptions(phase="decode")
    )
    expected_decode = 2
    expected_mixed = 3
    assert cal_decode["step_time_us"]["n"] == expected_decode
    assert cal_decode["step_time_us"]["p50"] == pytest.approx(2050.0)
    assert cal_decode["source"]["phase"] == "decode"

    cal_prefill = calibrate.calibrate_trace(
        trace_file, options=calibrate.CalibrateOptions(phase="prefill")
    )
    assert cal_prefill["step_time_us"]["n"] == 1
    assert cal_prefill["step_time_us"]["p50"] == pytest.approx(8000.0)
    assert cal_prefill["source"]["phase"] == "prefill"

    cal_mixed = calibrate.calibrate_trace(
        trace_file, options=calibrate.CalibrateOptions(phase="mixed")
    )
    assert cal_mixed["step_time_us"]["n"] == expected_mixed
    assert cal_mixed["source"]["phase"] == "mixed"

    with pytest.raises(errors.MegakernelConfigError, match="found phases:"):
        calibrate.calibrate_trace(
            trace_file, options=calibrate.CalibrateOptions(phase="saturation")
        )


def test_calibrate_wraps_corrupt_json_and_xplane_errors(tmp_path: Path) -> None:
    """Verify malformed Perfetto JSON/gzip and corrupt XPlane pb wrap cleanly."""
    bad_json = tmp_path / "corrupt.json"
    bad_json.write_text("{not valid json", encoding="utf-8")
    with pytest.raises(
        errors.MegakernelConfigError, match="Failed to read Perfetto trace"
    ):
        advisor.calibrate_trace(bad_json)

    bad_gz = tmp_path / "corrupt.json.gz"
    bad_gz.write_bytes(b"not a gzip payload")
    with pytest.raises(
        errors.MegakernelConfigError, match="Failed to read Perfetto trace"
    ):
        advisor.calibrate_trace(bad_gz)

    bad_pb = tmp_path / "corrupt.xplane.pb"
    bad_pb.write_bytes(b"\x08\x80\x80\x80")
    with pytest.raises(
        errors.MegakernelConfigError, match="Failed to parse XPlane trace"
    ):
        advisor.calibrate_trace(bad_pb)


def test_calibrate_ignores_bool_scalars_in_json_events(tmp_path: Path) -> None:
    """Verify JSON booleans are not coerced to 0/1 ints in collective/token stats."""
    bool_trace = _write_json_trace(
        tmp_path,
        "bool_scalars.json",
        {
            "traceEvents": [
                {
                    "ph": "X",
                    "name": "Step",
                    "cat": "step",
                    "pid": 1,
                    "tid": 1,
                    "ts": 100.0,
                    "dur": 50.0,
                    "args": {
                        "phase": "decode",
                        "active_tokens": True,
                        "fused_eligible_tokens": False,
                    },
                },
                {
                    "ph": "X",
                    "name": "all-reduce.1",
                    "cat": "op",
                    "pid": 1,
                    "tid": 2,
                    "ts": 110.0,
                    "dur": 8.0,
                    "args": {"group_size": True, "bytes": True},
                },
            ]
        },
    )
    payload = advisor.calibrate_trace(bool_trace)
    assert payload["collectives"] == []
    assert payload["step_coverage"]["fused_eligible_fraction"] == pytest.approx(1.0)


def test_calibrate_cli_validates_chip_tp_and_batch_upfront(tmp_path: Path) -> None:
    """Verify `advisor calibrate` rejects bad chip/tp/batch before parsing traces."""
    empty_trace = _write_json_trace(tmp_path, "empty.json", {"traceEvents": []})
    out_json = tmp_path / "out.json"
    base_argv = ["calibrate", "--trace", str(empty_trace), "--out", str(out_json)]

    with pytest.raises(errors.MegakernelConfigError, match="Unknown chip"):
        advisor.main([*base_argv, "--chip", "invalid_chip_xyz"])

    with pytest.raises(errors.MegakernelConfigError, match="power of 2 >= 1"):
        advisor.main([*base_argv, "--tp", "-4"])

    with pytest.raises(errors.MegakernelConfigError, match="power of 2 >= 1"):
        advisor.main([*base_argv, "--tp", "3"])

    with pytest.raises(errors.MegakernelConfigError, match="batch must be >= 1"):
        advisor.main([*base_argv, "--batch", "0"])
