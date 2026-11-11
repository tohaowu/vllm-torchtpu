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
"""Regime sweep, auto-parallelism recommendation, and cross-regime advisor matrix."""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import gcd
from typing import TYPE_CHECKING, Literal

from ..core import errors
from .chips import ChipSpec, resolve_chip_spec
from .constants import (
    DEFAULT_CHIP,
    DEFAULT_PREFILL_ISL,
    DEFAULT_SPEC_TOKENS,
    DTYPE_BYTES,
    RECOMMEND_DEFAULT_BATCH,
    RECOMMEND_HIGH_BATCH,
    RECOMMEND_HIGH_CTX,
    RECOMMEND_TP_CANDIDATES,
    SATURATION_BASE_BATCH,
    SATURATION_BATCH_CAP,
)
from .models import (
    DEPLOYMENT_PRESETS,
    MODEL_PRESET_SOURCES,
    DeploymentPreset,
    ModelProfile,
    ParallelLayout,
    resolve_model_profile,
)
from .specs import (
    AdvisorSweepReport,
    DisaggHandoffSpec,
    MeasurementsData,
    ParallelismRecommendation,
    RegimeEntry,
    WorkloadSpec,
    load_measurements,
)
from .support import CORE_LIMITS
from .verdict import (
    AdvisorReport,
    ScopeEstimate,
    advise,
)
from .vmem import compute_per_token_vmem_bytes, is_slice_topology_supported

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path


@dataclass(frozen=True)
class RegimeSweepOptions:
    """Typed options for multi-regime advisor sweeps without dynamic `**opts`."""

    base_workload: WorkloadSpec | None = None
    layout: ParallelLayout | None = None
    tp: int | None = None
    ep: int | None = None
    dp: int = 1
    pp: int = 1
    dcp: int = 1
    pcp: int = 1
    measurements: MeasurementsData | Mapping[str, object] | Path | str | None = None
    deployment_preset: bool = True
    chip_spec_path: str | Path | None = None


def recommend_parallelism(
    profile: ModelProfile,
    chip: ChipSpec | str = DEFAULT_CHIP,
    *,
    dp: int = 1,
    pp: int = 1,
) -> ParallelismRecommendation:
    """Auto-derive the minimum balanced TPU parallelism mesh for a model and chip."""
    chip_spec = resolve_chip_spec(chip)
    fallback: ParallelismRecommendation | None = None
    for tp_cand in RECOMMEND_TP_CANDIDATES:
        if not is_slice_topology_supported(tp_cand * dp * pp, chip_spec):
            continue
        if profile.num_attention_heads % tp_cand != 0:
            continue
        if (
            profile.num_linear_layers > 0
            and profile.linear_value_heads > 0
            and profile.linear_value_heads % tp_cand != 0
        ):
            continue
        ep_cand = (
            gcd(tp_cand * dp, profile.num_experts) if profile.num_experts > 0 else 1
        )
        wl = WorkloadSpec(
            phase="decode",
            batch=_round_dp_batch(RECOMMEND_DEFAULT_BATCH, dp),
            tp=tp_cand,
            ep=ep_cand,
            dp=dp,
            pp=pp,
        )
        rep = advise(profile, wl, chip_spec)
        if not rep.hbm_fits:
            continue
        sub = rep.scopes[0]
        total_cores = tp_cand * dp * pp
        topo = chip_spec.topologies.get(total_cores, f"{total_cores} cores")
        cand = ParallelismRecommendation(
            tp=tp_cand,
            ep=ep_cand,
            dp=dp,
            pp=pp,
            topology=topo,
            hbm_used_gb=rep.hbm_used_gb,
            hbm_capacity_gb=rep.hbm_capacity_gb,
            w_step_gb=sub.w_step_gb,
            auto_selected=True,
        )
        fallback = cand
        # Verify HBM capacity holds at the high-concurrency 32K regime (B=64, 32K)
        # and that the reference B=16 decode batch fits in one VMEM pass.
        rep_high = advise(
            profile,
            replace(
                wl,
                batch=_round_dp_batch(RECOMMEND_HIGH_BATCH, dp),
                isl=RECOMMEND_HIGH_CTX,
                context=RECOMMEND_HIGH_CTX,
            ),
            chip_spec,
        )
        whole_sc = next(s for s in rep.scopes if s.scope == "whole_step")
        if rep_high.hbm_fits and whole_sc.n_splits == 1:
            return cand
    if fallback is not None:
        return fallback
    msg = (
        f"No balanced parallelism layout fits in HBM for {profile.name} "
        f"on {chip_spec.name} (dp={dp}, pp={pp})"
    )
    raise errors.MegakernelConfigError(msg)


def _round_dp_batch(raw_batch: int, dp: int) -> int:
    """Round raw_batch up to the smallest positive multiple of dp."""
    return dp * max(1, (raw_batch + dp - 1) // dp)


def _layout_recommendation(
    prof: ModelProfile,
    chip_spec: ChipSpec,
    layout: ParallelLayout,
    base: WorkloadSpec,
    source: str = "deployment preset",
) -> ParallelismRecommendation:
    """Evaluate a fixed layout HBM fit at the reference decode workload."""
    base_rep = advise(
        prof,
        replace(
            base,
            phase="decode",
            batch=_round_dp_batch(RECOMMEND_DEFAULT_BATCH, layout.dp),
            tp=layout.tp,
            ep=layout.ep,
            dp=layout.dp,
            pp=layout.pp,
            dcp=layout.dcp,
            pcp=layout.pcp,
            layer_groups=layout.layer_groups,
        ),
        chip_spec,
    )
    total_cores = layout.tp * layout.dp * layout.pp
    return ParallelismRecommendation(
        tp=layout.tp,
        ep=layout.ep,
        dp=layout.dp,
        pp=layout.pp,
        topology=chip_spec.topologies.get(total_cores, f"{total_cores} cores"),
        hbm_used_gb=base_rep.hbm_used_gb,
        hbm_capacity_gb=base_rep.hbm_capacity_gb,
        w_step_gb=base_rep.scopes[0].w_step_gb,
        auto_selected=False,
        dcp=layout.dcp,
        pcp=layout.pcp,
        source=source,
        layer_groups=layout.layer_groups,
    )


def _select_layouts(
    prof: ModelProfile,
    chip_spec: ChipSpec,
    base: WorkloadSpec,
    requested: tuple[int | None, int | None, int, int, int, int],
    dep: DeploymentPreset | None,
) -> tuple[
    ParallelismRecommendation,
    ParallelismRecommendation | None,
    ParallelismRecommendation | None,
]:
    """Return decode, prefill, and saturation layouts: preset, user, or auto-pick."""
    tp, ep, dp, pp, dcp, pcp = requested
    if dep is not None:
        sat_rec = (
            _layout_recommendation(prof, chip_spec, dep.saturation, base)
            if dep.saturation is not None
            else None
        )
        dec_rec = (
            sat_rec
            if base.objective == "throughput" and sat_rec is not None
            else _layout_recommendation(prof, chip_spec, dep.decode, base)
        )
        return (
            dec_rec,
            _layout_recommendation(prof, chip_spec, dep.prefill, base),
            sat_rec,
        )
    if tp is not None:
        ep_val = (
            ep
            if ep is not None
            else (gcd(tp * dp, prof.num_experts) if prof.num_experts > 0 else 1)
        )
        layout = ParallelLayout(tp=tp, ep=ep_val, dp=dp, pp=pp, dcp=dcp, pcp=pcp)
        return (
            _layout_recommendation(prof, chip_spec, layout, base, source=""),
            None,
            None,
        )
    par = recommend_parallelism(prof, chip_spec, dp=dp, pp=pp)
    if ep is not None or dcp > 1 or pcp > 1:
        par = replace(par, ep=ep if ep is not None else par.ep, dcp=dcp, pcp=pcp)
    return par, None, None


def _compute_disagg_handoff(
    prof: ModelProfile,
    chip_spec: ChipSpec,
    dep: DeploymentPreset | None,
    meas: MeasurementsData | None,
    isl: int = DEFAULT_PREFILL_ISL,
) -> DisaggHandoffSpec | None:
    """Compute per-request KV + recurrent state wire bytes and DCN transfer floor."""
    if dep is None:
        return None
    bw_kv = DTYPE_BYTES.get(prof.kv_dtype.lower(), 1.0)
    bw_st = DTYPE_BYTES.get(prof.state_dtype.lower(), 2.0)
    kv_per_tok = (
        (prof.kv_lora_rank + prof.qk_rope_head_dim) * bw_kv
        if prof.kv_lora_rank
        else 2.0 * prof.num_kv_heads * prof.head_dim * bw_kv
    )
    kv_wire_bytes = (
        prof.num_attention_layers * isl * kv_per_tok * prof.kv_compress_factor
    )
    qkv_lin_full = (
        2 * prof.linear_key_heads * prof.linear_key_dim
        + prof.linear_value_heads * prof.linear_value_dim
    )
    z_dim_full = prof.linear_value_heads * prof.linear_value_dim
    state_wire_bytes = (
        prof.num_linear_layers
        * (z_dim_full * prof.linear_key_dim + qkv_lin_full * prof.linear_conv_kernel)
        * bw_st
    )
    total_wire_bytes = kv_wire_bytes + state_wire_bytes
    if meas is not None and meas.dcn_bw_gbps is not None:
        dcn_gbps = meas.dcn_bw_gbps
        dcn_bracket = (dcn_gbps, dcn_gbps)
        dcn_prov: Literal["ASSUMED", "MEASURED"] = "MEASURED"
    else:
        dcn_gbps = chip_spec.dcn_bw_gbps
        dcn_bracket = chip_spec.dcn_bw_gbps_bracket
        dcn_prov = "ASSUMED"
    dcn_bytes_per_s = (dcn_gbps / 8.0) * 1e9
    dcn_lo_bytes_per_s = (dcn_bracket[0] / 8.0) * 1e9
    dcn_hi_bytes_per_s = (dcn_bracket[1] / 8.0) * 1e9
    xfer_ms = round((total_wire_bytes / max(1.0, dcn_bytes_per_s)) * 1e3, 2)
    xfer_lo_ms = round((total_wire_bytes / max(1.0, dcn_hi_bytes_per_s)) * 1e3, 2)
    xfer_hi_ms = round((total_wire_bytes / max(1.0, dcn_lo_bytes_per_s)) * 1e3, 2)
    return DisaggHandoffSpec(
        isl=isl,
        kv_wire_mb=round(kv_wire_bytes / 1e6, 2),
        state_wire_mb=round(state_wire_bytes / 1e6, 2),
        total_wire_mb=round(total_wire_bytes / 1e6, 2),
        dcn_bw_gbps=dcn_gbps,
        dcn_bw_gbps_bracket=dcn_bracket,
        dcn_xfer_ms=xfer_ms,
        dcn_xfer_bracket_ms=(xfer_lo_ms, xfer_hi_ms),
        dcn_provenance=dcn_prov,
    )


def _with_layout(wl: WorkloadSpec, layout: ParallelLayout) -> WorkloadSpec:
    """Copy `wl` onto `layout`'s parallelism degrees and layer groups."""
    return replace(
        wl,
        tp=layout.tp,
        ep=layout.ep,
        dp=layout.dp,
        pp=layout.pp,
        dcp=layout.dcp,
        pcp=layout.pcp,
        layer_groups=layout.layer_groups,
    )


def _with_preset_prefill(
    workloads: tuple[tuple[str, WorkloadSpec], ...],
    dep: DeploymentPreset | None,
) -> tuple[tuple[str, WorkloadSpec], ...]:
    """Move prefill and saturated decode regimes onto preset layouts and add windows."""
    if dep is None:
        return workloads
    pre = dep.prefill
    sat = dep.saturation
    decode_base = _with_layout(workloads[0][1], pre)
    windows = tuple(
        (
            f"Preset Prefill Window (W={win})",
            replace(
                decode_base,
                phase="prefill",
                batch=decode_base.dp,
                isl=win,
                chunk_size=win,
                moe_mode="grouped_on_chip",
            ),
        )
        for win in dep.prefill_windows
    )
    sat_batch = (
        min(SATURATION_BATCH_CAP, SATURATION_BASE_BATCH * sat.dp)
        if sat is not None
        else SATURATION_BASE_BATCH
    )
    sat_k = dep.decode.spec_tokens
    moved_list: list[tuple[str, WorkloadSpec]] = []
    for lbl, wl in workloads:
        if wl.phase == "prefill":
            moved_list.append((lbl, _with_layout(wl, pre)))
        elif sat is not None and lbl.startswith("Decode Saturated"):
            moved_list.append(
                (
                    f"Decode Saturated (B={sat_batch} over DP{sat.dp}, T=1, 32K)",
                    replace(
                        _with_layout(wl, sat),
                        batch=min(SATURATION_BATCH_CAP, wl.batch * sat.dp),
                    ),
                )
            )
        else:
            moved_list.append((lbl, wl))
            if sat is not None and sat_k > 0 and lbl.startswith("Spec Verify High"):
                sat_lbl = (
                    f"Spec Verify Saturated (B={sat_batch} over DP{sat.dp}, "
                    f"K={sat_k}, 32K)"
                )
                moved_list.append(
                    (
                        sat_lbl,
                        replace(
                            _with_layout(wl, sat),
                            phase="verify",
                            batch=sat_batch,
                            spec_tokens=sat_k,
                        ),
                    )
                )
    return tuple(moved_list) + windows


def _build_reference_workloads(
    base: WorkloadSpec, k_spec: int
) -> tuple[tuple[str, WorkloadSpec], ...]:
    """Build the standard 12 reference request regimes with DP-divisible batches."""
    dp = base.dp
    b4 = _round_dp_batch(4, dp)
    b8 = _round_dp_batch(8, dp)
    b16 = _round_dp_batch(16, dp)
    b32 = _round_dp_batch(32, dp)
    b64 = _round_dp_batch(64, dp)
    bsat = _round_dp_batch(SATURATION_BASE_BATCH, dp)
    b1 = _round_dp_batch(1, dp)
    decode_regimes = (
        (
            f"Decode Small (B={b4}, T=1, 4K)",
            replace(base, phase="decode", batch=b4, isl=3072, osl=2048),
        ),
        (
            f"Decode Low (B={b8}, T=1, 4K)",
            replace(base, phase="decode", batch=b8, isl=3072, osl=2048),
        ),
        (
            f"Decode Med (B={b16}, T=1, 4K)",
            replace(base, phase="decode", batch=b16, isl=3072, osl=2048),
        ),
        (
            f"Decode High (B={b64}, T=1, 32K)",
            replace(
                base,
                phase="decode",
                batch=b64,
                isl=RECOMMEND_HIGH_CTX,
                osl=2048,
                context=RECOMMEND_HIGH_CTX,
                moe_mode="grouped_on_chip",
            ),
        ),
        (
            f"Decode Saturated (B={bsat}, T=1, 32K)",
            replace(
                base,
                phase="decode",
                batch=bsat,
                isl=RECOMMEND_HIGH_CTX,
                osl=2048,
                context=RECOMMEND_HIGH_CTX,
                moe_mode="grouped_on_chip",
            ),
        ),
    )
    verify_regimes = (
        (
            f"Spec Verify Small (B={b4}, K={k_spec})",
            replace(
                base, phase="verify", batch=b4, spec_tokens=k_spec, isl=3072, osl=2048
            ),
        ),
        (
            f"Spec Verify Low (B={b8}, K={k_spec})",
            replace(
                base,
                phase="verify",
                batch=b8,
                spec_tokens=k_spec,
                isl=3072,
                osl=2048,
                moe_mode="grouped_on_chip",
            ),
        ),
        (
            f"Spec Verify Med (B={b16}, K={k_spec})",
            replace(
                base,
                phase="verify",
                batch=b16,
                spec_tokens=k_spec,
                isl=3072,
                osl=2048,
                moe_mode="grouped_on_chip",
            ),
        ),
        (
            f"Spec Verify High (B={b32}, K={k_spec})",
            replace(
                base,
                phase="verify",
                batch=b32,
                spec_tokens=k_spec,
                isl=RECOMMEND_HIGH_CTX,
                osl=2048,
                context=RECOMMEND_HIGH_CTX,
                moe_mode="grouped_on_chip",
            ),
        ),
    )
    prefill_mixed_regimes = (
        (
            f"Mixed Decode+Prefill (B={b16}, C=256)",
            replace(
                base,
                phase="mixed",
                batch=b16,
                isl=DEFAULT_PREFILL_ISL,
                osl=2048,
                chunk_size=256,
                prefix_hit_rate=0.90,
                moe_mode="grouped_on_chip",
            ),
        ),
        (
            f"Chunked Prefill (B={b1}, T=256)",
            replace(
                base,
                phase="prefill",
                batch=b1,
                isl=DEFAULT_PREFILL_ISL,
                chunk_size=256,
                prefix_hit_rate=0.9375,
                moe_mode="grouped_on_chip",
            ),
        ),
        (
            f"Long Prefill (B={b1}, T=4096)",
            replace(
                base,
                phase="prefill",
                batch=b1,
                isl=DEFAULT_PREFILL_ISL,
                moe_mode="grouped_on_chip",
            ),
        ),
    )
    return (*decode_regimes, *verify_regimes, *prefill_mixed_regimes)


def _serialize_scope_dicts(
    scopes: tuple[ScopeEstimate, ...],
) -> tuple[dict[str, object], ...]:
    """Convert evaluated ScopeEstimate records into regime ledger dictionaries."""
    return tuple(
        {
            "scope": s.scope,
            "feasible": s.feasible,
            "reason": s.reason,
            "step_bound_us": s.step_bound_us,
            "step_achievable_us": s.step_achievable_us,
            "step_achievable_upper_us": s.step_achievable_upper_us,
            "step_upper_us": s.step_upper_us,
            "roof_floor_us": s.roof_floor_us,
            "compute_us": s.compute_us,
            "chain_us": s.chain_us,
            "hbm_us": s.hbm_us,
            "comm_us": s.comm_us,
            "fill_drain_us": s.fill_drain_us,
            "repass_spill_us": s.repass_spill_us,
            "launch_us": s.launch_us,
            "ops_per_step": s.ops_per_step,
            "n_splits": s.n_splits,
            "t_vmem_max": s.t_vmem_max,
            "selected_nbd": s.selected_nbd,
            "selected_nbe": s.selected_nbe,
            "vmem_static_mib": s.vmem_static_mib,
            "w_step_gb": s.w_step_gb,
            "act_hbm_mb": s.act_hbm_mb,
            "moe_flop_factor": s.moe_flop_factor,
            "w_step_strict_gb": s.w_step_strict_gb,
            "step_floor_strict_us": s.step_floor_strict_us,
            "support": s.support,
            "blockers": list(s.blockers),
        }
        for s in scopes
    )


def _serialize_workload_dict(
    wl: WorkloadSpec, effective_context: int
) -> dict[str, object]:
    """Serialize WorkloadSpec fields into a JSON-compatible dictionary."""
    return {
        "phase": wl.phase,
        "batch": wl.batch,
        "isl": wl.isl,
        "osl": wl.osl,
        "context": effective_context,
        "spec_tokens": wl.spec_tokens,
        "chunk_size": wl.chunk_size,
        "prefix_hit_rate": wl.prefix_hit_rate,
        "microbatches": wl.microbatches,
        "segment_layers": wl.segment_layers,
        "tp": wl.tp,
        "ep": wl.ep,
        "dp": wl.dp,
        "pp": wl.pp,
        "dcp": wl.dcp,
        "pcp": wl.pcp,
        "kv_split": wl.kv_split,
        "layer_groups": list(wl.layer_groups),
        "moe_mode": wl.moe_mode,
        "vmem_alias": wl.vmem_alias,
        "vmem_fit_headroom": wl.vmem_fit_headroom,
        "op_overhead_us": wl.op_overhead_us,
        "dma_buffers": wl.dma_buffers,
        "dispatch_regime": wl.dispatch_regime,
        "hbm_utilization_cap": wl.hbm_utilization_cap,
        "max_rows_per_call": wl.max_rows_per_call,
    }


def _evaluate_sweep_workloads(
    prof: ModelProfile,
    chip_spec: ChipSpec,
    workloads: tuple[tuple[str, WorkloadSpec], ...],
    meas_opt: MeasurementsData | Mapping[str, object] | Path | str | None,
) -> tuple[list[RegimeEntry], list[AdvisorReport]]:
    """Evaluate workloads across all regimes and assemble report entries."""
    entries: list[RegimeEntry] = []
    reports: list[AdvisorReport] = []
    for label, wl in workloads:
        rep = advise(prof, wl, chip_spec, measurements=meas_opt)
        reports.append(rep)
        rec_sc = next(s for s in rep.scopes if s.scope == rep.recommended_scope)
        entries.append(
            RegimeEntry(
                label=label,
                phase=wl.phase,
                active_tokens=rep.active_tokens,
                effective_context=rep.effective_context,
                step_arithmetic_intensity=rep.step_arithmetic_intensity,
                max_vmem_tokens=rec_sc.t_vmem_max,
                num_vmem_splits=rec_sc.n_splits,
                recommended_scope=rep.recommended_scope,
                recommended_moe_mode=rep.recommended_moe_mode,
                step_bound_us=rec_sc.step_bound_us,
                rationale=rep.rationale,
                layout=ParallelLayout(
                    tp=wl.tp, ep=wl.ep, dp=wl.dp, pp=wl.pp, dcp=wl.dcp, pcp=wl.pcp
                ).label(),
                step_upper_us=rec_sc.step_upper_us,
                verdict=rep.verdict,
                confidence=rep.confidence,
                flags=rep.flags,
                metrics=rep.metrics,
                unresolved_assumptions=rep.unresolved_assumptions,
                scopes=_serialize_scope_dicts(rep.scopes),
                workload=_serialize_workload_dict(wl, rep.effective_context),
                fastest_scope=rep.fastest_scope,
                fastest_verdict=rep.fastest_verdict,
                evidence=rep.evidence,
                step_achievable_us=rec_sc.step_achievable_us,
                step_achievable_upper_us=rec_sc.step_achievable_upper_us,
                step_floor_strict_us=rec_sc.step_floor_strict_us,
                blockers=rec_sc.blockers,
                hypothetical_best=rep.hypothetical_best,
            )
        )
    return entries, reports


def _build_tailoring_guide(
    decode_rep: AdvisorReport,
    vmem_parts: Mapping[str, int],
    dma_buffers: int,
) -> tuple[str, ...]:
    """Assemble the three-regime tailoring guidance rules for the sweep."""
    cross_sc = next(s for s in decode_rep.scopes if s.scope == "whole_step")
    one_l_sc = next(s for s in decode_rep.scopes if s.scope == "single_layer")
    ref_sc = cross_sc if cross_sc.feasible else one_l_sc
    t_ceil = ref_sc.t_vmem_max
    t_ridge = decode_rep.ridge_tokens
    step_mode = primary_step_mode(cross_sc)
    slope_kib = vmem_parts["total_per_token_bytes"] / 1024
    ridge_fb = decode_rep.ridge_dense_flop_per_byte
    return (
        f"Small/Med Decode & Spec Verify (T <= {t_ceil} active tokens): "
        f"use scope='{step_mode}' with moe_mode='masked_on_chip' (T < 32) "
        f"or 'grouped_on_chip' (T >= 32) to keep [T, H] in VMEM "
        f"({ref_sc.vmem_static_mib:.1f} MiB {dma_buffers}-slot weight staging + "
        f"{slope_kib:.1f} KiB/tok). The megakernel core's default buckets stop at "
        f"{CORE_LIMITS.default_decode_bucket_top} decode rows and "
        f"{CORE_LIMITS.default_verify_bucket_top} verify rows; a larger call needs "
        "a custom bucket list.",
        f"High-Batch Decode & Chunked Prefill ({t_ceil} < T <= {t_ridge} tokens): "
        "use scope='single_layer' or 'layer_group' "
        f"(single-layer VMEM ceiling T_max={one_l_sc.t_vmem_max}) "
        "so weights are not re-streamed across multi-pass token splits.",
        f"Saturated Decode & Compute-Bound Prefill (T > {t_ridge} tokens, "
        f"AI >= {ridge_fb:.0f} FLOP/B): use scope='per_op' "
        "with moe_mode='grouped_in_hbm': tokens are grouped by expert in HBM "
        "and each operator is its own call, so no VMEM is held for "
        "multi-layer weight staging.",
    )


def advise_regimes(
    profile: ModelProfile | str | Mapping[str, object],
    chip: ChipSpec | str = DEFAULT_CHIP,
    *,
    options: RegimeSweepOptions | None = None,
    measurements: MeasurementsData | Mapping[str, object] | Path | str | None = None,
) -> AdvisorSweepReport:
    """Evaluate the 12-regime Batch Size x Phase & Context matrix."""
    cfg = options or RegimeSweepOptions()
    meas_opt = measurements if measurements is not None else cfg.measurements
    base = cfg.base_workload or WorkloadSpec()
    prof = resolve_model_profile(profile)
    chip_spec = resolve_chip_spec(chip, chip_spec_path=cfg.chip_spec_path)
    layout = cfg.layout
    if layout is not None:
        requested: tuple[int | None, int | None, int, int, int, int] = (
            layout.tp,
            layout.ep,
            layout.dp,
            layout.pp,
            layout.dcp,
            layout.pcp,
        )
    else:
        requested = (cfg.tp, cfg.ep, cfg.dp, cfg.pp, cfg.dcp, cfg.pcp)
    explicit_layout = layout is not None or requested != (None, None, 1, 1, 1, 1)
    dep = (
        DEPLOYMENT_PRESETS.get(prof.name)
        if cfg.deployment_preset and not explicit_layout
        else None
    )
    par, prefill_par, sat_par = _select_layouts(prof, chip_spec, base, requested, dep)
    meas_data = (
        load_measurements(meas_opt, chip_name=chip_spec.name)
        if meas_opt is not None
        else None
    )
    disagg_handoff = _compute_disagg_handoff(prof, chip_spec, dep, meas_data)
    base = replace(
        base, tp=par.tp, ep=par.ep, dp=par.dp, pp=par.pp, dcp=par.dcp, pcp=par.pcp
    )
    k_spec = (
        dep.decode.spec_tokens
        if dep is not None and dep.decode.spec_tokens
        else DEFAULT_SPEC_TOKENS
    )
    workloads = _with_preset_prefill(_build_reference_workloads(base, k_spec), dep)
    entries, reports = _evaluate_sweep_workloads(prof, chip_spec, workloads, meas_opt)
    decode_rep = reports[2]
    vmem_parts = compute_per_token_vmem_bytes(prof, workloads[2][1])
    cross_sc = next(s for s in decode_rep.scopes if s.scope == "whole_step")
    one_l_sc = next(s for s in decode_rep.scopes if s.scope == "single_layer")
    ref_sc = cross_sc if cross_sc.feasible else one_l_sc
    return AdvisorSweepReport(
        model=prof.name,
        chip=chip_spec.name,
        parallelism=par,
        vmem_budget_mib=decode_rep.vmem_budget_mib,
        vmem_static_mib=ref_sc.vmem_static_mib,
        per_token_vmem_bytes=vmem_parts["total_per_token_bytes"],
        vmem_breakdown_bytes=vmem_parts,
        max_vmem_tokens=ref_sc.t_vmem_max,
        ridge_dense_flop_per_byte=decode_rep.ridge_dense_flop_per_byte,
        ridge_tokens=decode_rep.ridge_tokens,
        regimes=tuple(entries),
        tailoring_guide=_build_tailoring_guide(
            decode_rep, vmem_parts, base.dma_buffers
        ),
        dma_buffers=base.dma_buffers,
        single_layer_vmem_tokens=one_l_sc.t_vmem_max,
        prefill_parallelism=prefill_par,
        saturation_parallelism=sat_par,
        disagg_handoff=disagg_handoff,
        chip_sources=chip_spec.sources if chip_spec.sources is not None else {},
        model_sources=_build_model_sources(prof),
        chip_spec=chip_spec,
        model_profile=prof,
    )


def _build_model_sources(prof: ModelProfile) -> dict[str, str]:
    """Extract provenance sources for model architecture facts."""
    base_src = MODEL_PRESET_SOURCES.get(prof.name, f"Model config: {prof.name}")
    sources = {
        "config": base_src,
        "num_layers": f"{prof.num_layers} layers ({base_src})",
        "hidden_size": f"H={prof.hidden_size} ({base_src})",
        "attention_heads": (
            f"NH={prof.num_attention_heads}, NKV={prof.num_kv_heads}, "
            f"HD={prof.head_dim}"
        ),
        "dtypes": (
            f"weight={prof.weight_dtype}, expert={prof.expert_dtype}, "
            f"kv={prof.kv_dtype}"
        ),
    }
    if prof.num_experts > 0:
        sources["moe"] = (
            f"E={prof.num_experts}, TOPK={prof.top_k}, "
            f"I={prof.moe_intermediate_size} ({base_src})"
        )
    return sources


def primary_step_mode(cross_sc: ScopeEstimate) -> str:
    """Return whole_step if whole-step fusion is feasible, else single_layer."""
    return "whole_step" if cross_sc.feasible else "single_layer"
