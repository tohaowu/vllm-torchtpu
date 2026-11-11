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
"""Unit tests for megakernel advisor sub-term decomposition and JS engine."""

from __future__ import annotations

import json
import os
import re
import shutil
import signal
import time
from pathlib import Path
from typing import cast

import pytest

from vllm_torchtpu.megakernel import advisor
from vllm_torchtpu.megakernel.advisor import workbench_engine
from vllm_torchtpu.megakernel.core import errors

DENYLIST_RE = re.compile(
    r"gmm|ragged_dot|megablox|sparsecore|mxu|tensorcore|vpu|\bllo\b",
    re.IGNORECASE,
)
# Each round(x, 2) term carries at most 0.5 * 1e-2 = 0.005 us half-ULP rounding error:
# roof_floor_us sums 3 rounded terms + roof_floor_us (4 * 0.005 = 0.020 us);
# step_bound_us sums 5 rounded terms + step_bound_us (6 * 0.005 = 0.030 us).
DECIMAL_HALF_ULP_US = 0.005
ROOF_ROUNDING_BOUND_US = 4 * DECIMAL_HALF_ULP_US
STEP_ROUNDING_BOUND_US = 6 * DECIMAL_HALF_ULP_US
# A single-call row limit a kernel could declare; any value the sweep must carry.
ROW_LIMIT = 80
EXPECTED_ACCORDION_COUNT = 4
EXPECTED_VERIFY_SEQ_TOKENS = 4
EXPECTED_PREFILL_SEQ_TOKENS = 256
EXPECTED_PHASE_MAP_TOK_128 = 128
EXPECTED_PHASE_MAP_TOK_512 = 512
WATERFALL_ROW_COUNT = 5


@pytest.mark.parametrize("model_name", ["qwen3.5-397b", "kimi-k3", "qwen3.8-2.4t"])
def test_scope_estimate_subterms_conserve_step_bound(model_name: str) -> None:
    """Verify ScopeEstimate sub-terms reconstruct roof_floor_us and step_bound_us."""
    sweep_report = advisor.advise_regimes(
        model_name,
        "tpu7x",
        options=advisor.RegimeSweepOptions(
            base_workload=advisor.WorkloadSpec(op_overhead_us=1.5)
        ),
    )
    assert len(sweep_report.regimes) > 0
    for regime in sweep_report.regimes:
        assert regime.workload["phase"] == regime.phase
        for scope_dict in regime.scopes:
            chain_us = cast("float", scope_dict["chain_us"])
            hbm_us = cast("float", scope_dict["hbm_us"])
            comm_us = cast("float", scope_dict["comm_us"])
            fill_drain_us = cast("float", scope_dict["fill_drain_us"])
            repass_spill_us = cast("float", scope_dict["repass_spill_us"])
            launch_us = cast("float", scope_dict["launch_us"])
            roof_floor = cast("float", scope_dict["roof_floor_us"])
            step_bound = cast("float", scope_dict["step_bound_us"])
            is_fused_multi = scope_dict["scope"] in ("whole_step", "layer_group")
            roof_core = (
                max(hbm_us, chain_us + comm_us)
                if is_fused_multi
                else max(chain_us, hbm_us) + comm_us
            )
            roof_reconstructed = roof_core + fill_drain_us
            step_reconstructed = roof_reconstructed + repass_spill_us + launch_us
            assert roof_reconstructed == pytest.approx(
                roof_floor, abs=ROOF_ROUNDING_BOUND_US
            )
            assert step_reconstructed == pytest.approx(
                step_bound, abs=STEP_ROUNDING_BOUND_US
            )


def test_workbench_carries_row_limit_and_chain_rule() -> None:
    """Verify the workbench reads a declared row limit and shows the chain rule."""
    sweep_report = advisor.advise_regimes(
        "qwen3.5-397b",
        "tpu7x",
        options=advisor.RegimeSweepOptions(
            base_workload=advisor.WorkloadSpec(max_rows_per_call=ROW_LIMIT)
        ),
    )
    assert len(sweep_report.regimes) > 0
    for regime in sweep_report.regimes:
        assert regime.workload["max_rows_per_call"] == ROW_LIMIT

    html_text = advisor.render_advisor_html(sweep_report)
    for required_text in (
        "maxRowsPerCall: rw.max_rows_per_call",
        "t_chain = t_dense + max(t_routed, t_moe)",
        "max(t_hbm, t_chain + t_comm)",
        "max(t_chain, t_hbm) + t_comm",
        "s.roof_floor_us - s.fill_drain_us",
        "Ideal 1-pass",
        "IDEAL TMAX",
    ):
        assert required_text in html_text
    for stale_text in ("t_comp + t_comm", "ICI Collective Overlap"):
        assert stale_text not in html_text


def test_workbench_exposes_zero_snapped_collective_helper() -> None:
    """Verify workbench.js exports exposedCommUs snapping 1-ULP rounding steps."""
    sweep_report = advisor.advise_regimes("qwen3.5-397b", "tpu7x")
    html_text = advisor.render_advisor_html(sweep_report)
    for required_text in (
        "function exposedCommUs(s)",
        "if (!(s.comm_us > 0)) return 0;",
        "exposedCommUs,",
    ):
        assert required_text in html_text


def test_workbench_scales_collective_snap_by_calls() -> None:
    """Verify exposedCommUs scales its rounding snap by nCalls and syncs scope."""
    sweep_report = advisor.advise_regimes("qwen3.5-397b", "tpu7x")
    html_text = advisor.render_advisor_html(sweep_report)
    for required_text in (
        "const snap = 0.01 * (s.nCalls || 1) + 0.005;",
        "return raw >= snap ? raw : 0;",
        "STATE.selectedTopic.scope = evalA.winner.scope;",
    ):
        assert required_text in html_text


def test_workbench_engine_js_exposes_cleanroom_physics_api() -> None:
    """Verify ES2022 physics evaluator exposes required functions with zero leaks."""
    engine_js = workbench_engine.render_workbench_engine_js()
    assert "window.__MK_ADVISOR_ENGINE" in engine_js
    for symbol_name in (
        "computePerTokenVmem",
        "evaluateWorkload",
        "allReduceCost",
        "expertsHit",
        "devices_per_chip",
    ):
        assert symbol_name in engine_js
    assert DENYLIST_RE.search(engine_js) is None
    body_lines = [
        line
        for line in engine_js.splitlines()
        if "http://www.apache.org/licenses/LICENSE-2.0" not in line
    ]
    body_js = "\n".join(body_lines)
    assert "https://" not in body_js
    assert "http://" not in body_js


@pytest.mark.parametrize(
    ("chip_name", "max_usable_mib"),
    [
        ("tpu7x", 62.0),
        ("v6e", 126.0),
        ("v5p", 62.0),
        ("v5p_megacore", 124.0),
        ("v4", 14.0),
        ("v4_megacore", 28.0),
    ],
)
def test_vmem_budget_bounded_by_chip_usable_ceiling(
    chip_name: str,
    max_usable_mib: float,
) -> None:
    """Verify advise() accepts usable ceiling and rejects budgets above chip VMEM."""
    within_budget = advisor.WorkloadSpec(batch=4, vmem_budget_mib=max_usable_mib)
    ok_report = advisor.advise("qwen3.5-397b", within_budget, chip=chip_name)
    assert ok_report.chip == chip_name

    over_budget = advisor.WorkloadSpec(batch=4, vmem_budget_mib=max_usable_mib + 0.5)
    with pytest.raises(errors.MegakernelConfigError, match="exceeds usable VMEM"):
        advisor.advise("qwen3.5-397b", over_budget, chip=chip_name)


def test_custom_chip_spec_preserves_explicit_vmem_usable_bytes(
    tmp_path: Path,
) -> None:
    """Verify --chip-spec JSON preserves explicit vmem_usable_bytes."""
    spec_file = tmp_path / "custom_chip.json"
    spec_file.write_text(
        json.dumps(
            {
                "name": "custom_xpu",
                "devices_per_chip": 2,
                "cores_per_device": 1,
                "vmem_capacity_bytes": 64 * (1 << 20),
                "vmem_usable_bytes": 50 * (1 << 20),
                "hbm_capacity_bytes": 96_000_000_000,
                "hbm_bw_bytes_per_s": 3_600_000_000_000.0,
                "bf16_flops_per_s": 1_150_000_000_000_000.0,
                "ici_egress_bw_bytes_per_s": 300_000_000_000.0,
            }
        ),
        encoding="utf-8",
    )
    resolved = advisor.resolve_chip_spec("custom_xpu", chip_spec_path=spec_file)
    assert resolved.vmem_usable_bytes == 50 * (1 << 20)
    with pytest.raises(errors.MegakernelConfigError, match=r"50\.0 MiB usable"):
        advisor.advise(
            "qwen3.5-397b",
            advisor.WorkloadSpec(batch=4, vmem_budget_mib=50.5),
            chip=resolved,
        )


class _UniqueIdCollector:
    """Collect HTML id attributes from rendered HTML markup."""

    def __init__(self) -> None:
        """Initialize empty id list."""
        self.ids: list[str] = []

    def feed_markup(self, html_text: str) -> list[str]:
        """Extract all id attribute values via regex over static tags."""
        self.ids = re.findall(r'\bid="([^"]+)"', html_text)
        return self.ids


@pytest.mark.parametrize("model_name", ["qwen3.5-397b", "kimi-k3", "qwen3.8-2.4t"])
def test_workbench_html_three_pane_landmarks_and_unique_ids(model_name: str) -> None:
    """Verify 3-pane Fusion Workbench landmarks, unique DOM IDs, and zero CDN links."""
    sweep_report = advisor.advise_regimes(model_name, "tpu7x")
    html_text = advisor.render_advisor_html(sweep_report)

    for required_id in (
        "wb-topbar",
        "view-workbench",
        "wb-left-rail",
        "wb-center-stage",
        "wb-right-inspector",
        "wb-vmem-arena-svg",
        "wb-scope-timeline-svg",
        "wb-waterfall-svg",
        "wb-deriv-drawer",
        "view-audit",
        "btn-swap-ab",
        "btn-vs-perop",
        "btn-reset-all",
        "wb-ab-status",
        "ctl-phase",
        "ctl-batch",
        "ctl-seq-tokens",
        "ctl-context",
        "ctl-vmem",
        "ctl-dma",
        "ctl-alias",
        "ctl-launch-gap",
        "ctl-coll-hop",
        "advisor-data",
    ):
        assert f'id="{required_id}"' in html_text

    collector = _UniqueIdCollector()
    all_ids = collector.feed_markup(html_text)
    assert len(all_ids) == len(set(all_ids))
    assert "ph === 'decode' ? 1" in html_text
    assert "phase === 'decode' ? 1" in html_text
    assert "wbOnHwControl()" in html_text
    assert "wbOnPhaseChange()" in html_text
    assert "wbOnSeqChange()" in html_text
    assert "max-height:164px" in html_text
    assert "wb-knob-pair" in html_text
    assert "overflow-wrap:anywhere" in html_text
    assert 'max="1024"' in html_text
    assert 'max="8192"' in html_text
    assert 'step="0.01"' in html_text
    assert 'step="0.001"' in html_text
    assert DENYLIST_RE.search(html_text) is None
    assert "https://" not in html_text
    assert 'href="http://' not in html_text
    assert 'src="http://' not in html_text
    for sel in (
        ".wb-verdict-wrap",
        ".wb-verdict-pill",
        ".wb-verdict-meta",
        ".wb-invalid-banner",
        ".wb-dr-group",
        ".wb-mesh-grid",
        ".wb-eta-details summary",
    ):
        assert sel in html_text


@pytest.mark.parametrize(
    ("chip_name", "expected_max_mib"),
    [
        ("tpu7x", "62.0"),
        ("v6e", "126.0"),
        ("v5p_megacore", "124.0"),
        ("v4", "14.0"),
        ("v4_megacore", "28.0"),
    ],
)
def test_workbench_vmem_slider_bounded_by_chip_usable_ceiling(
    chip_name: str,
    expected_max_mib: str,
) -> None:
    """Verify #ctl-vmem range slider min/max reflects each chip's usable VMEM."""
    sweep_report = advisor.advise_regimes("qwen3.5-397b", chip_name)
    html_text = advisor.render_advisor_html(sweep_report)
    vmem_tag = (
        f'<input id="ctl-vmem" type="range" min="2.0" max="{expected_max_mib}" '
        f'step="0.5" value="{expected_max_mib}" oninput="wbOnHwControl()"/>'
    )
    assert vmem_tag in html_text
    assert 'max="96"' not in html_text


def test_workbench_mesh_dropdowns_contain_only_physical_powers_of_two() -> None:
    """Verify mesh dropdowns exclude non-physical axis sizes such as 3, 6, or 7."""
    sweep_report = advisor.advise_regimes("qwen3.5-397b", "tpu7x")
    html_text = advisor.render_advisor_html(sweep_report)
    tp_block = html_text.split('<select id="ctl-tp"', 1)[1].split("</select>", 1)[0]
    options = re.findall(r'value="([^"]+)"', tp_block)
    assert options == ["1", "2", "4", "8", "16", "32", "64", "128", "256", "512"]
    for bad in ("3", "6", "7"):
        assert bad not in options


def test_workbench_verdict_mesh_regime_and_proof_drawer_controls() -> None:
    """Verify verdict bar, feasibility banner, mesh knobs, and proof drawer."""
    sweep_report = advisor.advise_regimes("qwen3.5-397b", "tpu7x")
    html_text = advisor.render_advisor_html(sweep_report)
    for snippet in (
        'id="wb-verdict-pill"',
        'id="wb-verdict-summary"',
        'id="wb-invalid-banner"',
        'id="btn-dr-zero"',
        'id="btn-dr-xla"',
        'id="btn-dr-eager"',
        'id="ctl-tp"',
        'id="ctl-ep"',
        'id="ctl-dp"',
        'id="ctl-pp"',
        'id="ctl-dcp"',
        'id="ctl-pcp"',
        'id="ctl-eta-dense"',
        'id="ctl-eta-moe"',
        'id="ctl-eta-hbm"',
        'id="ctl-eta-ici"',
        "TUNE XLA FLAGS FIRST",
        "INVALID CONFIG",
        "Fusion gain (estimate)",
        "expertSlotBytes: 1048576",
        "window.showProof",
        'max="1048576"',
    ):
        assert snippet in html_text
    assert "vmemMip" not in html_text


def test_workbench_phase_map_and_calibration_dropzone_markup() -> None:
    """Verify 2D phase map tab, local FileReader dropzone, and bucket copy UI."""
    sweep_report = advisor.advise_regimes("qwen3.5-397b", "tpu7x")
    html_text = advisor.render_advisor_html(sweep_report)
    for required_snippet in (
        'id="btn-stage-phase"',
        'id="wb-stage-phase"',
        'id="wb-phase-map-svg"',
        'id="wb-cal-dropzone"',
        'id="cal-file-input"',
        'id="wb-bucket-schedule-pre"',
        "wbFocusedTopicCard",
        "Selected VMEM:",
        "Waterfall:",
        "Copied!",
        "new FileReader()",
        "new DecompressionStream('gzip')",
        "traceEventsToMeas",
        "minHop.toFixed(3)",
        "advisor.measurements.v1",
        "advisor.bucket_schedule.v1",
        "advisory_offline_v1",
    ):
        assert required_snippet in html_text
    assert DENYLIST_RE.search(html_text) is None


def test_build_bucket_schedule_and_cli_export(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Verify build_bucket_schedule coalescing and --bucket-schedule CLI export."""
    sweep_report = advisor.advise_regimes("qwen3.5-397b", "tpu7x")
    schedule = advisor.build_bucket_schedule(sweep_report)
    distinct_phases = {reg.phase for reg in sweep_report.regimes}
    assert schedule.schema == "advisor.bucket_schedule.v1"
    assert schedule.consumer_status == "advisory_offline_v1"
    assert schedule.model == "qwen3.5-397b"
    assert len(distinct_phases) <= len(schedule.buckets) <= len(sweep_report.regimes)
    for rule in schedule.buckets:
        assert rule.min_active_tokens <= rule.max_active_tokens
        assert rule.recommended_scope in {
            "whole_step",
            "layer_group",
            "single_layer",
            "per_op",
        }

    out_file = tmp_path / "bucket_schedule.json"
    exit_code = advisor.main(
        ["--model", "qwen3.5-397b", "--bucket-schedule", str(out_file)]
    )
    assert exit_code == 0
    assert out_file.is_file()
    captured = capsys.readouterr()
    assert captured.out == ""

    exit_stdout = advisor.main(["--model", "qwen3.5-397b", "--bucket-schedule", "-"])
    assert exit_stdout == 0
    stdout_json = capsys.readouterr().out
    assert '"schema": "advisor.bucket_schedule.v1"' in stdout_json
    assert '"consumer_status": "advisory_offline_v1"' in stdout_json


def test_phase_map_long_context_oom_hatch_and_conserved_trace_idle(
    tmp_path: Path,
) -> None:
    """Verify 256K/1M phase grid, OOM hatch, and step-idle trace parity."""
    sweep_report = advisor.advise_regimes("qwen3.5-397b", "tpu7x")
    html_text = advisor.render_advisor_html(sweep_report)
    assert "262144, 1048576" in html_text
    assert 'id="wb-oom-hatch"' in html_text
    assert "OOM &gt; 80% HBM" in html_text
    assert "tVal * epDp >= rFloor" in html_text
    assert "tVal >= 32" not in html_text
    assert "idle / (2.0 * numLayers)" in html_text
    assert "us_per_sublayer_boundary" in html_text

    # 4 steps x 400 non-overlapping 2.0 us ops -> 200 us idle;
    # for 60 layers (120 boundaries), p50 == 200 / 120 == 1.6667 us.
    events: list[dict[str, object]] = []
    for step_idx in range(4):
        base_ts = float(step_idx * 1200)
        events.append(
            {
                "name": f"Step {step_idx}",
                "cat": "step_marker",
                "ph": "X",
                "pid": 1,
                "tid": 1,
                "ts": base_ts,
                "dur": 1000.0,
            }
        )
        events.extend(
            [
                {
                    "name": f"fusion.{op_idx}",
                    "cat": "hlo_op",
                    "ph": "X",
                    "pid": 2,
                    "tid": 1,
                    "ts": base_ts + op_idx * 2.5,
                    "dur": 2.0,
                }
                for op_idx in range(400)
            ]
        )
    trace_path = tmp_path / "perfetto_steps.json"
    out_meas = tmp_path / "cal_out.json"
    trace_path.write_text(json.dumps({"traceEvents": events}), encoding="utf-8")
    cal_payload = advisor.calibrate_trace(
        trace_path,
        out_meas,
        options=advisor.CalibrateOptions(chip="tpu7x", num_layers=60),
    )
    py_gap = cal_payload["launch_gap_us"]
    assert "p50" in py_gap
    assert py_gap["p50"] == pytest.approx(200.0 / 120.0, abs=1e-3)
    assert py_gap["step_idle_gap_us"] == pytest.approx(200.0, abs=1e-2)


def _run_workbench_ui_runner(
    tmp_path: Path,
    sweep_report: advisor.AdvisorSweepReport,
    steps: list[dict[str, object]],
    *,
    extra_kpis: bool = False,
) -> list[dict[str, object]]:
    """Execute advisor_workbench_runner.js via gjs/node and return UI snapshots."""
    js_runtime = shutil.which("gjs") or shutil.which("node")
    if js_runtime is None:
        pytest.skip("Neither gjs nor node runtime found")
    runner_path = Path(__file__).with_name("advisor_workbench_runner.js")
    if extra_kpis:
        runner_src = runner_path.read_text(encoding="utf-8").replace(
            '    winnerScope: evA.winner ? evA.winner.scope : "",\n',
            '    winnerScope: evA.winner ? evA.winner.scope : "",\n'
            '    recScope: evA.recScope || "",\n'
            '    kpiScope: elems.get("wb-kpi-scope").textContent,\n'
            '    kpiFloor: elems.get("wb-kpi-floor").textContent,\n'
            '    kpiTmax: elems.get("wb-kpi-tmax").textContent,\n'
            '    kpiTps: elems.get("wb-kpi-tps").textContent,\n'
            '    regimeListHtml: elems.get("wb-regime-list").innerHTML,\n'
            '    abStatus: elems.get("wb-ab-status").innerHTML,\n'
            '    waterfallTitle: elems.get("wb-waterfall-title").textContent,\n',
            1,
        )
        runner_path = tmp_path / "kpi_probe_runner.js"
        runner_path.write_text(runner_src, encoding="utf-8")
    engine_path = (
        Path(workbench_engine.__file__).parent / "assets" / "workbench_engine.js"
    )
    html_text = advisor.render_advisor_html(sweep_report)
    m_seed = re.search(
        r'<script id="advisor-data" type="application/json">(.*?)</script>',
        html_text,
        re.DOTALL,
    )
    assert m_seed is not None
    seed_data = {
        **workbench_engine.build_engine_seed_defaults(),
        **json.loads(m_seed.group(1)),
    }

    in_json = tmp_path / "wb_ui_in.json"
    out_json = tmp_path / "wb_ui_out.json"
    in_json.write_text(
        json.dumps(
            {
                "engine_path": str(engine_path),
                "seed_data": seed_data,
                "steps": steps,
            }
        ),
        encoding="utf-8",
    )
    argv = [js_runtime, str(runner_path), str(in_json), str(out_json)]
    pid = os.posix_spawnp(js_runtime, argv, os.environ)
    deadline = time.monotonic() + 60.0
    while time.monotonic() < deadline:
        wpid, status = os.waitpid(pid, os.WNOHANG)
        if wpid == pid:
            if os.waitstatus_to_exitcode(status) != 0:
                msg = "Workbench UI runner exited non-zero"
                raise RuntimeError(msg)
            payload = json.loads(out_json.read_text(encoding="utf-8"))
            return cast("list[dict[str, object]]", payload["snapshots"])
        time.sleep(0.05)
    os.kill(pid, signal.SIGKILL)
    os.waitpid(pid, 0)
    msg = "Workbench UI runner timed out"
    raise TimeoutError(msg)


def test_workbench_pin_b_preserves_pin_a_and_sliders(tmp_path: Path) -> None:
    """Verify [A]/[B] pins, Swap, vs per_op, arena scope sync, and decode seq lock."""
    sweep_report = advisor.advise_regimes("qwen3.5-397b", "tpu7x")
    pinned_row_two = 2
    snaps = _run_workbench_ui_runner(
        tmp_path,
        sweep_report,
        [
            {"op": "set_batch", "value": 64, "tag": "dirty_a"},
            {"op": "pin_regime", "idx": 0, "slot": "B", "tag": "pin_b_same_idx"},
            {
                "op": "pin_regime",
                "idx": pinned_row_two,
                "slot": "B",
                "tag": "pin_b_row2",
            },
            {"op": "swap_ab", "tag": "swapped"},
            {"op": "clear_b", "tag": "cleared_b"},
            {"op": "clear_b", "tag": "cleared_b_again"},
            {"op": "set_arena_scope", "scope": "per_op", "tag": "arena_per_op"},
            {"op": "set_phase", "phase": "verify", "tag": "phase_verify"},
            {"op": "set_phase", "phase": "prefill", "tag": "phase_prefill"},
            {"op": "set_phase", "phase": "decode", "tag": "phase_decode"},
            {"op": "set_seq_tokens", "value": 32, "tag": "seq_drag_on_decode"},
        ],
    )
    by_tag = {str(s["tag"]): s for s in snaps}

    assert by_tag["initial"]["swapDisabled"] is True
    assert by_tag["initial"]["seqDisabled"] is True
    assert by_tag["initial"]["lblSeqTokens"] == "1 (decode)"

    assert by_tag["pin_b_same_idx"]["pinA"] == 0
    assert by_tag["pin_b_same_idx"]["pinB"] == 0
    assert by_tag["pin_b_same_idx"]["customDirty"] is True
    assert by_tag["pin_b_same_idx"]["ctlBatch"] == "64"
    assert by_tag["pin_b_same_idx"]["swapDisabled"] is False

    assert by_tag["pin_b_row2"]["pinA"] == 0
    assert by_tag["pin_b_row2"]["pinB"] == pinned_row_two
    assert by_tag["pin_b_row2"]["ctlBatch"] == "64"

    assert by_tag["swapped"]["pinA"] == pinned_row_two
    assert by_tag["swapped"]["pinB"] == 0

    scope_before = by_tag["swapped"]["ctlArenaScope"]
    assert by_tag["cleared_b"]["pinB"] == -1
    assert by_tag["cleared_b"]["swapDisabled"] is True
    assert by_tag["cleared_b"]["ctlArenaScope"] == scope_before
    assert by_tag["cleared_b_again"]["ctlArenaScope"] == scope_before

    assert by_tag["arena_per_op"]["selectedScope"] == "per_op"
    assert "Mechanism (per_op)" in str(by_tag["arena_per_op"]["derivDrawer"])

    assert by_tag["phase_verify"]["seqDisabled"] is False
    assert (
        int(str(by_tag["phase_verify"]["ctlSeqTokens"])) == EXPECTED_VERIFY_SEQ_TOKENS
    )
    assert (
        int(str(by_tag["phase_prefill"]["ctlSeqTokens"])) == EXPECTED_PREFILL_SEQ_TOKENS
    )
    assert by_tag["phase_decode"]["seqDisabled"] is True
    assert by_tag["phase_decode"]["ctlSeqTokens"] == "1"
    assert by_tag["seq_drag_on_decode"]["ctlPhase"] == "decode"
    assert by_tag["seq_drag_on_decode"]["ctlSeqTokens"] == "1"


def test_workbench_dp_phase_map_scales_batch(tmp_path: Path) -> None:
    """Verify 2D Phase Map and live bucket schedule scale batch by DP on DP4xTP2."""
    sweep_report = advisor.advise_regimes("qwen3.5-397b", "tpu7x")
    sat_idx = next(
        idx
        for idx, reg in enumerate(sweep_report.regimes)
        if int(reg.workload.get("dp", 1)) > 1
    )
    snaps = _run_workbench_ui_runner(
        tmp_path,
        sweep_report,
        [
            {"op": "pin_regime", "idx": sat_idx, "slot": "A", "tag": "pin_dp4"},
            {"op": "select_phase_cell", "t": 128, "s": 32768, "tag": "cell_128"},
            {"op": "select_phase_cell", "t": 512, "s": 32768, "tag": "cell_512"},
        ],
    )
    by_tag = {str(s["tag"]): s for s in snaps}
    assert by_tag["cell_128"]["activeTokens"] == EXPECTED_PHASE_MAP_TOK_128
    assert by_tag["cell_128"]["ctlBatch"] == "512"
    assert by_tag["cell_512"]["activeTokens"] == EXPECTED_PHASE_MAP_TOK_512
    assert by_tag["cell_512"]["ctlBatch"] == "2048"

    sched = json.loads(str(by_tag["pin_dp4"]["bucketScheduleJson"]))
    assert sched["buckets"][0]["min_active_tokens"] == 1
    assert sched["buckets"][-1]["max_active_tokens"] == EXPECTED_PHASE_MAP_TOK_512


def test_workbench_inspector_renders_progressive_accordions(tmp_path: Path) -> None:
    """Verify progressive-disclosure <details> dropdowns and LOW ROI topbar badge."""
    sweep_report = advisor.advise_regimes("qwen3.5-397b", "tpu7x")
    html_text = advisor.render_advisor_html(sweep_report)
    assert 'id="wb-cuj-hint"' in html_text
    assert 'id="wb-mesh-details"' in html_text
    assert 'id="wb-adv-hw-details"' in html_text
    assert 'onchange="wbOnArenaScopeChange()"' in html_text

    chunked_idx = next(
        idx
        for idx, reg in enumerate(sweep_report.regimes)
        if "Chunked Prefill" in reg.label
    )
    snaps = _run_workbench_ui_runner(
        tmp_path,
        sweep_report,
        [{"op": "pin_regime", "idx": chunked_idx, "slot": "A", "tag": "chunked"}],
    )
    by_tag = {str(s["tag"]): s for s in snaps}
    drawer_html = str(by_tag["initial"]["derivDrawer"])
    assert 'class="wb-exec-card"' in drawer_html
    assert drawer_html.count('class="wb-acc"') == EXPECTED_ACCORDION_COUNT
    assert '<details class="wb-acc" open>' in drawer_html
    assert '<details class="wb-topbar-details">' in str(
        by_tag["initial"]["verdictSummary"]
    )
    assert "LOW ROI (&lt;1% PHYSICS)" in str(by_tag["chunked"]["verdictSummary"])

    k3_sweep = advisor.advise_regimes("kimi-k3", "tpu7x")
    snaps_k3 = _run_workbench_ui_runner(tmp_path, k3_sweep, [], extra_kpis=True)
    init_k3 = snaps_k3[0]
    assert init_k3["winnerScope"] == "whole_step"
    assert init_k3["recScope"] == "per_op"
    assert init_k3["verdictPill"] == "KEEP PER_OP"
    assert str(init_k3["kpiScope"]).startswith("per_op ")
    assert str(init_k3["kpiFloor"]).startswith("4415.7 ~")
    assert init_k3["kpiTps"] == "56.6 tok/s/chip"
    assert "T&le;NaN" not in str(init_k3["verdictSummary"])
    assert "Saves <strong>0.0 us" in str(init_k3["verdictSummary"])
    assert "hypothetical: whole_step" in str(init_k3["verdictSummary"])


def test_workbench_waterfall_card_matches_chain_composition(tmp_path: Path) -> None:
    """Verify waterfall drill-down cards match the chain rule and SVG bar deltas."""
    sweep_report = advisor.advise_regimes("qwen3.5-397b", "tpu7x")
    verify_high_idx = next(
        idx
        for idx, reg in enumerate(sweep_report.regimes)
        if reg.label.startswith("Spec Verify High")
    )
    steps: list[dict[str, object]] = [
        {"op": "pin_regime", "idx": verify_high_idx, "slot": "A", "tag": "verify_hi"},
    ]
    steps.extend(
        {
            "op": "select_topic",
            "kind": "waterfall",
            "scope": "whole_step",
            "sub": str(row_idx),
            "tag": f"wf_{row_idx}",
        }
        for row_idx in range(WATERFALL_ROW_COUNT)
    )
    snaps = _run_workbench_ui_runner(tmp_path, sweep_report, steps)
    by_tag = {str(s["tag"]): s for s in snaps}

    svg_deltas = [
        float(val)
        for val in re.findall(
            r'class="tabular">([+-]?[0-9.]+) us</text>',
            str(by_tag["verify_hi"]["waterfallSvg"]),
        )
    ]
    assert len(svg_deltas) == WATERFALL_ROW_COUNT

    drawer_core = str(by_tag["wf_0"]["derivDrawer"])
    assert "max(t_chain, t_hbm) overlap floor" in drawer_core
    assert "max(t_comp, t_hbm)" not in drawer_core

    drawer_comm = str(by_tag["wf_1"]["derivDrawer"])
    m_comm_a = re.search(r"\[A\] whole_step: ([+-]?[0-9.]+) us", drawer_comm)
    assert m_comm_a is not None
    assert float(m_comm_a.group(1)) > 0.0

    for row_idx, svg_delta in enumerate(svg_deltas):
        drawer = str(by_tag[f"wf_{row_idx}"]["derivDrawer"])
        m_delta = re.search(r"Delta \(\[A\] - \[B\]\): ([+-]?[0-9.]+) us", drawer)
        assert m_delta is not None
        assert float(m_delta.group(1)) == pytest.approx(svg_delta, abs=1e-9)


def test_workbench_waterfall_card_tracks_winner_and_zero_sign(tmp_path: Path) -> None:
    """Verify waterfall card tracks evalA.winner, exact zero sign, and 0-comm snap."""
    sweep_report = advisor.advise_regimes("qwen3.5-397b", "tpu7x")
    verify_small_idx = next(
        idx
        for idx, reg in enumerate(sweep_report.regimes)
        if reg.label.startswith("Spec Verify Small")
    )
    mixed_idx = next(
        idx
        for idx, reg in enumerate(sweep_report.regimes)
        if reg.label.startswith("Mixed Decode+Prefill")
    )
    steps: list[dict[str, object]] = [
        {
            "op": "pin_regime",
            "idx": verify_small_idx,
            "slot": "A",
            "tag": "vsmall",
        },
        {
            "op": "select_topic",
            "kind": "waterfall",
            "scope": "whole_step",
            "sub": "3",
            "tag": "wf_zero_row",
        },
        {
            "op": "select_topic",
            "kind": "waterfall",
            "scope": "whole_step",
            "sub": "1",
            "tag": "wf_comm_before_flip",
        },
        {"op": "set_batch", "value": 1024, "tag": "flipped_winner"},
        {"op": "pin_regime", "idx": mixed_idx, "slot": "A", "tag": "mixed"},
        {"op": "set_batch", "value": 1024, "tag": "mixed_1024"},
        {
            "op": "select_topic",
            "kind": "waterfall",
            "scope": "whole_step",
            "sub": "1",
            "tag": "mixed_comm_card",
        },
    ]
    snaps = _run_workbench_ui_runner(tmp_path, sweep_report, steps)
    by_tag = {str(s["tag"]): s for s in snaps}

    zero_svg = re.findall(
        r'class="tabular">([+-]?[0-9.]+ us)</text>',
        str(by_tag["wf_zero_row"]["waterfallSvg"]),
    )
    zero_row_idx = 3
    assert zero_svg[zero_row_idx] == "0.00 us"
    assert "Delta ([A] - [B]): 0.00 us" in str(by_tag["wf_zero_row"]["derivDrawer"])

    assert by_tag["flipped_winner"]["winnerScope"] == "per_op"
    flip_svg = re.findall(
        r'class="tabular">([+-]?[0-9.]+ us)</text>',
        str(by_tag["flipped_winner"]["waterfallSvg"]),
    )
    flip_drawer = str(by_tag["flipped_winner"]["derivDrawer"])
    assert "[A] per_op:" in flip_drawer
    assert f"Delta ([A] - [B]): {flip_svg[1]}" in flip_drawer

    mixed_drawer = str(by_tag["mixed_comm_card"]["derivDrawer"])
    assert "[A] whole_step: 0.00 us" in mixed_drawer


def test_workbench_multicall_snap_and_inspecting_scope_sync(tmp_path: Path) -> None:
    """Verify multi-call hidden collectives snap to 0.00 us and scope syncs on flip."""
    row_limit_16 = 16
    batch_32 = 32
    sweep_38 = advisor.advise_regimes(
        "qwen3.8-2.4t",
        "tpu7x",
        options=advisor.RegimeSweepOptions(
            base_workload=advisor.WorkloadSpec(max_rows_per_call=row_limit_16)
        ),
    )
    dec_high_idx = next(
        idx
        for idx, reg in enumerate(sweep_38.regimes)
        if reg.label.startswith("Decode High")
    )
    snaps_mc = _run_workbench_ui_runner(
        tmp_path,
        sweep_38,
        [
            {"op": "pin_regime", "idx": dec_high_idx, "slot": "A", "tag": "dec_hi"},
            {"op": "set_batch", "value": batch_32, "tag": "mc_2calls"},
            {
                "op": "select_topic",
                "kind": "waterfall",
                "scope": "whole_step",
                "sub": "1",
                "tag": "mc_wf_comm",
            },
        ],
    )
    mc_by_tag = {str(s["tag"]): s for s in snaps_mc}
    assert "[A] whole_step: 0.00 us" in str(mc_by_tag["mc_wf_comm"]["derivDrawer"])

    sweep_35 = advisor.advise_regimes("qwen3.5-397b", "tpu7x")
    vsmall_idx = next(
        idx
        for idx, reg in enumerate(sweep_35.regimes)
        if reg.label.startswith("Spec Verify Small")
    )
    snaps_flip = _run_workbench_ui_runner(
        tmp_path,
        sweep_35,
        [
            {"op": "pin_regime", "idx": vsmall_idx, "slot": "A", "tag": "vs"},
            {
                "op": "select_topic",
                "kind": "waterfall",
                "scope": "whole_step",
                "sub": "1",
                "tag": "wf",
            },
            {"op": "set_batch", "value": 1024, "tag": "flip"},
        ],
    )
    flip_snap = next(s for s in snaps_flip if s["tag"] == "flip")
    assert flip_snap["selectedScope"] == "per_op"
    assert flip_snap["ctlArenaScope"] == "per_op"
    assert "<strong>per_op (" in str(flip_snap["derivDrawer"])


def test_live_bucket_schedule_uses_recommended_scope_not_hypothetical(
    tmp_path: Path,
) -> None:
    """Verify buildLiveBucketSchedule uses recScope/recWinner over hypothetical."""
    k3_sweep = advisor.advise_regimes("kimi-k3", "tpu7x")
    snaps = _run_workbench_ui_runner(tmp_path, k3_sweep, [])
    sched = json.loads(str(snaps[0]["bucketScheduleJson"]))
    first_bucket = sched["buckets"][0]
    assert first_bucket["recommended_scope"] == "per_op"
    assert first_bucket["min_active_tokens"] == 1


def test_workbench_uncalibrated_regime_list_ab_and_waterfall_mark_hypothetical(
    tmp_path: Path,
) -> None:
    """Verify UI regime list, [B] status, KPI mode, and waterfall agree cleanly."""
    k3_sweep = advisor.advise_regimes("kimi-k3", "tpu7x")
    snaps = _run_workbench_ui_runner(
        tmp_path,
        k3_sweep,
        [{"op": "pin_regime", "idx": 1, "slot": "B", "tag": "pin_b"}],
        extra_kpis=True,
    )
    init_k3, pin_b = snaps[0], snaps[1]
    expected_mode = k3_sweep.regimes[0].recommended_moe_mode
    assert init_k3["kpiScope"] == f"per_op ({expected_mode})"
    regime_html = str(init_k3["regimeListHtml"])
    assert (
        ">per_op</span> &middot; T=4 &middot; 4415.7 us (hyp: whole_step" in regime_html
    )
    assert "<strong>[B]</strong>" in str(pin_b["abStatus"])
    assert str(pin_b["abStatus"]).endswith("(per_op)")
    wf_title = str(pin_b["waterfallTitle"])
    assert "[A] whole_step [hypothetical] vs [B]" in wf_title
    assert "[hypothetical])" in wf_title

    q_sweep = advisor.advise_regimes("qwen3.5-397b", "tpu7x")
    sat_indices = [i for i, r in enumerate(q_sweep.regimes) if "Saturated" in r.label]
    assert [q_sweep.regimes[i].label for i in sat_indices] == [
        "Decode Saturated (B=600 over DP4, T=1, 32K)",
        "Spec Verify Saturated (B=600 over DP4, K=3, 32K)",
    ]
    q_acts = [
        {"op": "pin_regime", "idx": idx, "slot": "A", "tag": f"sat_{idx}"}
        for idx in sat_indices
    ]
    q_snaps = _run_workbench_ui_runner(tmp_path, q_sweep, q_acts, extra_kpis=True)
    first_item_q = str(q_snaps[0]["regimeListHtml"]).split("wbPinRegime(1,")[0]
    assert "(hyp:" not in first_item_q
    assert "[hypothetical]" not in str(q_snaps[0]["waterfallTitle"])
    q_rail = str(q_snaps[0]["regimeListHtml"])
    for pos, idx in enumerate(sat_indices, start=1):
        row_frag = q_rail.split(f"wbPinRegime({idx}, 'A')")[1].split("B</button>")[0]
        assert ">per_op</span>" in row_frag
        assert ">invalid<" not in row_frag
        sat_snap = q_snaps[pos]
        assert sat_snap["recScope"] == "per_op"
        assert sat_snap["verdictPill"] == "KEEP PER_OP"
        assert "(per_op) vs" in str(sat_snap["abStatus"])
        assert "(invalid)" not in str(sat_snap["abStatus"])
