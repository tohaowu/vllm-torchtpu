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
"""Structured launch flag and solve/v2 payload emission for winning serving plans."""

from __future__ import annotations

import shlex
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Final, TypedDict, cast

from .chips import JAX_VERSION, ChipSpec, chip_catalog
from .constants import (
    KV_DTYPE_BYTES,
    MIN_TOP_LAYOUTS_FOR_FASTEST,
    PERCENT_SCALE,
    SOLVE_SCHEMA,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from .modules import Dtypes
    from .solver import ParetoPoint, PoolMeshPlan, ServingPlan

_POOL_AXES: tuple[tuple[str, str], ...] = (
    ("TP", "tp"),
    ("DP", "dp"),
    ("PP", "pp"),
    ("PCP", "pcp"),
    ("DCP", "kv_split"),
    ("EP", "ep"),
)

MOE_DISPATCH_NOTES: Final[tuple[str, ...]] = (
    "- 'allreduce': src/vllm_torchtpu/layers/adapter/custom_ops/moe_runner.py:128 "
    "(default when USE_MOE_FUSED_EP_KERNEL=0)",
    "- 'a2a_push': src/vllm_torchtpu/layers/adapter/fused_moe_ep.py:794 "
    "(USE_MOE_FUSED_EP_KERNEL=1)",
    "- 'ag_ep': src/vllm_torchtpu/layers/adapter/pipelined_fused_moe.py:261-301 "
    "(default all-gather + reduce-scatter across EP ranks when "
    "USE_MOE_FUSED_EP_KERNEL=0)",
    "- 'fused_rs': src/vllm_torchtpu/platforms/tpu_platform.py:1565-1625 and "
    "src/vllm_torchtpu/models/vllm/qwen3_next_patch.py:190 "
    "(TPU_MOE_SEQUENCE_PARALLEL=1 USE_MOE_FUSED_EP_KERNEL=1; requires "
    "--enable-expert-parallel, DP=1, PP=1, TP>1 per __init__.py:664)",
)


@dataclass(frozen=True)
class LaunchOptions:
    """Typed launch command emission options for ``point_launch_args``."""

    kv_dtype: str = "auto"
    gpu_memory_utilization: float | None = None
    model: str | None = None
    spec_tokens: int = 0
    mtp_layers: int = 0
    hbm_reserve_gb: float = 0.0


class PointLaunchPayload(TypedDict):
    """Launch CLI arguments and formatted commands for a single Pareto point."""

    mode: str
    cli_args: list[str]
    prefill_cli_args: list[str]
    commands: list[str]


class LaunchPayload(TypedDict):
    """Structured vLLM serve CLI payload emitted for one serving plan."""

    mode: str
    cli_args: list[str]
    prefill_cli_args: list[str]
    commands: list[str]
    warnings: list[str]
    serving_plan: Mapping[str, object]


class ProvenancePayload(TypedDict):
    """Analytical model provenance metadata in solve JSON payloads."""

    calibrated: bool
    basis: str


class TopKEntry(TypedDict):
    """Ranked top layout entry with role label, throughput margin, and launch."""

    frontier_role: str
    margin_vs_rank1_pct: float
    point: Mapping[str, object]
    launch: PointLaunchPayload


class SolvePayload(TypedDict, total=False):
    """Versioned ``solve/v2`` JSON payload with provenance and Top-K launches."""

    model_layers: int
    chip: str
    scenario: Mapping[str, object]
    mode: str
    recommended: Mapping[str, object] | None
    frontier: tuple[Mapping[str, object], ...]
    warnings: tuple[str, ...]
    candidates: tuple[Mapping[str, object], ...]
    gpu_memory_utilization: float
    search: Mapping[str, object] | None
    schema: str
    provenance: ProvenancePayload
    chip_spec: Mapping[str, object]
    model: str | None
    top_k: list[TopKEntry]
    winner_prefix_hit_range: tuple[float, float] | None
    prefix_hit_crossover_rate: float | None
    spec_tokens: int
    mtp_layers: int
    hbm_reserve_gb: float


class CeilingRow(TypedDict):
    """Single Pareto ceiling row exported by ``emit_ceiling_rows``."""

    est_interactivity_tok_s: float
    token_throughput_per_chip: float
    output_throughput_per_chip: float
    est_ttft_ms: float
    est_tpot_ms: float
    bound_tpot_ms: float
    concurrency: int
    total_chips: int
    mode: str
    binding_unit: str


def pool_cli_args(
    pool: PoolMeshPlan,
    *,
    kv_dtype: str = "auto",
    gpu_memory_utilization: float | None = None,
    spec_tokens: int = 0,
    mtp_layers: int = 0,
) -> list[str]:
    """Translate one ``PoolMeshPlan`` into ``vllm serve`` CLI flags."""
    args = [
        "--tensor-parallel-size",
        str(pool.tp),
        "--data-parallel-size",
        str(pool.dp),
        "--kv-cache-dtype",
        kv_dtype,
    ]
    if pool.pp > 1:
        args.extend(["--pipeline-parallel-size", str(pool.pp)])
    if pool.pcp > 1:
        args.extend(["--prefill-context-parallel-size", str(pool.pcp)])
    if pool.kv_split > 1:
        args.extend(["--decode-context-parallel-size", str(pool.kv_split)])
    if pool.ep > 1 or pool.moe_dispatch == "fused_rs":
        args.append("--enable-expert-parallel")
    if pool.max_model_len is not None:
        args.extend(["--max-model-len", str(pool.max_model_len)])
    if pool.max_num_seqs is not None:
        args.extend(["--max-num-seqs", str(pool.max_num_seqs)])
    if pool.kv_offload_gb > 0.0:
        args.extend(["--kv-offloading-size", f"{pool.kv_offload_gb:g}"])
    if gpu_memory_utilization is not None:
        args.extend(["--gpu-memory-utilization", f"{gpu_memory_utilization:g}"])
    if spec_tokens > 0 and pool.role != "prefill" and mtp_layers > 0:
        args.extend(
            [
                "--speculative-config",
                f'{{"method":"mtp","num_speculative_tokens":{spec_tokens}}}',
            ]
        )
    return args


def _pool_int(val: object, default: int = 1) -> int:
    """Return integer pool attribute value when present, else default."""
    if isinstance(val, int) and not isinstance(val, bool):
        return val
    return default


def pool_label(pool: Mapping[str, object]) -> str:
    """Return a concise pool label such as ``4x 2x2x2 [TP16 DCP2 EP16]``."""
    axes = " ".join(
        f"{name}{_pool_int(pool.get(key))}"
        for name, key in _POOL_AXES
        if _pool_int(pool.get(key)) > 1
    )
    count = _pool_int(pool.get("instances"))
    prefix = f"{count}\u00d7 " if count > 1 else ""
    return f"{prefix}{pool.get('topology', '')} [{axes or 'TP1'}]"


def _format_server_cmd(
    cli_args: list[str],
    model: str | None,
    *,
    pp_layer_partition: tuple[int, ...] | None = None,
    moe_dispatch: str = "allreduce",
) -> str:
    """Format launch CLI arguments as a standalone bash command string."""
    joined = shlex.join(cli_args)
    prefix = f"vllm serve {model}" if model else "vllm serve"
    cmd = f"{prefix} {joined}".strip()
    env_parts: list[str] = []
    if moe_dispatch == "a2a_push":
        env_parts.append("USE_MOE_FUSED_EP_KERNEL=1")
    elif moe_dispatch == "fused_rs":
        env_parts.append("TPU_MOE_SEQUENCE_PARALLEL=1 USE_MOE_FUSED_EP_KERNEL=1")
    if pp_layer_partition is not None and len(set(pp_layer_partition)) > 1:
        part_str = ",".join(str(x) for x in pp_layer_partition)
        env_parts.append(f"VLLM_PP_LAYER_PARTITION={part_str}")
    return f"{' '.join(env_parts)} {cmd}" if env_parts else cmd


def _pool_key(pool: PoolMeshPlan | None) -> tuple[object, ...] | None:
    """Group layouts by structural parallelism axes yielding distinct serve CLIs."""
    if pool is None:
        return None
    return (
        pool.topology,
        pool.tp,
        pool.dp,
        pool.ep > 1,
        pool.pp,
        pool.pcp,
        pool.kv_split,
        pool.pp_layer_partition,
    )


def _point_struct_key(
    pt: ParetoPoint,
) -> tuple[str, tuple[object, ...] | None, tuple[object, ...] | None]:
    """Build composite structural identity key across mixed or prefill/decode pools."""
    return (pt.mode, _pool_key(pt.prefill_pool), _pool_key(pt.decode_pool))


def top_layouts(plan: ServingPlan, k: int = 3) -> tuple[ParetoPoint, ...]:
    """Select up to k structurally distinct SLO-met layouts spanning Pareto roles."""
    if k <= 0:
        return ()
    best: dict[
        tuple[str, tuple[object, ...] | None, tuple[object, ...] | None],
        ParetoPoint,
    ] = {}
    slo_cands = [pt for pt in plan.candidates if pt.slo_met]
    est_meets = [pt for pt in slo_cands if pt.slo_verdict == "estimate_meets"]
    eligible = est_meets if est_meets else slo_cands
    for pt in eligible:
        key = _point_struct_key(pt)
        cur = best.get(key)
        if cur is None or (
            pt.token_throughput_per_chip,
            pt.est_interactivity_tok_s,
        ) > (
            cur.token_throughput_per_chip,
            cur.est_interactivity_tok_s,
        ):
            best[key] = pt
    ranked = sorted(best.values(), key=lambda p: -p.token_throughput_per_chip)
    if len(ranked) <= 1 or k < MIN_TOP_LAYOUTS_FOR_FASTEST:
        return tuple(ranked[:k])
    selected = list(ranked[:k])
    fastest = max(
        eligible,
        key=lambda p: (p.est_interactivity_tok_s, p.token_throughput_per_chip),
    )
    f_key = _point_struct_key(fastest)
    sel_keys = [_point_struct_key(p) for p in selected]
    if f_key not in sel_keys:
        selected[-1] = fastest
    elif sel_keys.index(f_key) > 0:
        selected[sel_keys.index(f_key)] = fastest
    selected.sort(key=lambda p: -p.token_throughput_per_chip)
    return tuple(selected)


def point_launch_args(
    pt: ParetoPoint,
    options: LaunchOptions | None = None,
) -> PointLaunchPayload:
    """Emit ``vllm serve`` CLI flags and commands for a single ``ParetoPoint``."""
    opts = options or LaunchOptions()
    decode_args = pool_cli_args(
        pt.decode_pool,
        kv_dtype=opts.kv_dtype,
        gpu_memory_utilization=opts.gpu_memory_utilization,
        spec_tokens=opts.spec_tokens,
        mtp_layers=opts.mtp_layers,
    )
    prefill_args = (
        pool_cli_args(
            pt.prefill_pool,
            kv_dtype=opts.kv_dtype,
            gpu_memory_utilization=opts.gpu_memory_utilization,
            spec_tokens=opts.spec_tokens,
            mtp_layers=opts.mtp_layers,
        )
        if pt.prefill_pool is not None
        else []
    )
    decode_cmd = _format_server_cmd(
        decode_args,
        opts.model,
        pp_layer_partition=pt.decode_pool.pp_layer_partition,
        moe_dispatch=pt.decode_pool.moe_dispatch,
    )
    if pt.prefill_pool is not None:
        p_pool = pt.prefill_pool
        d_pool = pt.decode_pool
        prefill_cmd = _format_server_cmd(
            prefill_args,
            opts.model,
            pp_layer_partition=p_pool.pp_layer_partition,
            moe_dispatch=p_pool.moe_dispatch,
        )
        p_chips = p_pool.instances * p_pool.chips_per_instance
        d_chips = d_pool.instances * d_pool.chips_per_instance
        commands = [
            f"# provision: prefill {pool_label(asdict(p_pool))} ({p_chips} chips)",
            prefill_cmd,
            f"# provision: decode {pool_label(asdict(d_pool))} ({d_chips} chips)",
            decode_cmd,
            (
                "# note: disaggregated KV transfer / connector flags are not "
                "emitted; configure for your deployment's KV connector"
            ),
        ]
    else:
        commands = [decode_cmd]

    if (
        "--max-num-seqs" in decode_args or "--max-num-seqs" in prefill_args
    ) and opts.hbm_reserve_gb == 0.0:
        commands.append(
            "# note: --max-num-seqs is an upper bound on what fits when "
            "--hbm-reserve-gb is 0"
        )

    return {
        "mode": pt.mode,
        "cli_args": decode_args,
        "prefill_cli_args": prefill_args,
        "commands": commands,
    }


def _plan_dict(plan: ServingPlan) -> dict[str, object]:
    """Convert ``ServingPlan`` to dict excluding internal ``sweep_candidates``."""
    data: dict[str, object] = dict(asdict(plan).items())
    data.pop("sweep_candidates", None)
    return data


def emit_launch_args(
    plan: ServingPlan,
    *,
    dtypes: Dtypes | None = None,
    model: str | None = None,
) -> LaunchPayload:
    """Emit ``vllm serve`` CLI flags and commands for ``plan``."""
    kv_str = (
        "fp8" if (dtypes is not None and dtypes.kv <= KV_DTYPE_BYTES["fp8"]) else "auto"
    )
    warnings = list(plan.warnings)
    rec = plan.recommended
    if rec is None or not rec.slo_met:
        if not warnings:
            warnings.append("No feasible layout met memory and latency constraints.")
        return {
            "mode": plan.mode,
            "cli_args": [],
            "prefill_cli_args": [],
            "commands": [],
            "warnings": warnings,
            "serving_plan": _plan_dict(plan),
        }

    eff_spec = (
        plan.spec_tokens if plan.spec_tokens > 0 else (plan.scenario.spec_tokens or 0)
    )
    plan_mtp = plan.mtp_layers
    if eff_spec > 0 and plan_mtp <= 0:
        warnings.append(
            "Model declares zero MTP draft layers; omitting --speculative-config."
        )
    launch = point_launch_args(
        rec,
        LaunchOptions(
            kv_dtype=kv_str,
            gpu_memory_utilization=plan.gpu_memory_utilization,
            model=model,
            spec_tokens=eff_spec,
            mtp_layers=plan_mtp,
            hbm_reserve_gb=plan.hbm_reserve_gb,
        ),
    )
    return {
        "mode": launch["mode"],
        "cli_args": launch["cli_args"],
        "prefill_cli_args": launch["prefill_cli_args"],
        "commands": launch["commands"],
        "warnings": warnings,
        "serving_plan": _plan_dict(plan),
    }


def build_solve_payload(
    plan: ServingPlan,
    *,
    chip: ChipSpec,
    dtypes: Dtypes | None = None,
    model: str | None = None,
    top_k: int = 3,
) -> SolvePayload:
    """Build ``solve/v2`` JSON payload with provenance, chip catalog, and Top-K."""
    kv_str = (
        "fp8" if (dtypes is not None and dtypes.kv <= KV_DTYPE_BYTES["fp8"]) else "auto"
    )
    payload: dict[str, object] = _plan_dict(plan)
    payload["schema"] = SOLVE_SCHEMA
    payload["provenance"] = {
        "calibrated": False,
        "basis": (
            "analytical roofline bounds and estimates; est_* values are "
            "modeled estimates, bound_* values are physical lower bounds"
        ),
    }
    chip_dict: dict[str, object] = dict(asdict(chip).items())
    chip_dict["jax_version"] = JAX_VERSION
    chip_dict["catalog"] = chip_catalog(chip)
    payload["chip_spec"] = chip_dict
    payload["model"] = model
    top_pts = top_layouts(plan, top_k)
    top1_tput = top_pts[0].token_throughput_per_chip if top_pts else 0.0
    fastest_idx = (
        max(
            range(len(top_pts)),
            key=lambda i: (
                top_pts[i].est_interactivity_tok_s,
                top_pts[i].token_throughput_per_chip,
            ),
        )
        if len(top_pts) > 1
        else -1
    )
    eff_spec = (
        plan.spec_tokens if plan.spec_tokens > 0 else (plan.scenario.spec_tokens or 0)
    )
    opts = LaunchOptions(
        kv_dtype=kv_str,
        gpu_memory_utilization=plan.gpu_memory_utilization,
        model=model,
        spec_tokens=eff_spec,
        mtp_layers=plan.mtp_layers,
        hbm_reserve_gb=plan.hbm_reserve_gb,
    )
    top_entries: list[TopKEntry] = [
        {
            "frontier_role": (
                "max_throughput"
                if idx == 0
                else ("max_interactivity" if idx == fastest_idx else "balanced_knee")
            ),
            "margin_vs_rank1_pct": round(
                PERCENT_SCALE
                * (pt.token_throughput_per_chip - top1_tput)
                / max(1e-9, top1_tput),
                2,
            ),
            "point": asdict(pt),
            "launch": point_launch_args(pt, opts),
        }
        for idx, pt in enumerate(top_pts)
    ]
    payload["top_k"] = top_entries
    return cast("SolvePayload", payload)


def emit_ceiling_rows(plan: ServingPlan) -> list[CeilingRow]:
    """Export Pareto ceiling rows leading with interactivity and throughput."""
    return [
        {
            "est_interactivity_tok_s": pt.est_interactivity_tok_s,
            "token_throughput_per_chip": pt.token_throughput_per_chip,
            "output_throughput_per_chip": pt.output_throughput_per_chip,
            "est_ttft_ms": pt.est_ttft_ms,
            "est_tpot_ms": pt.est_tpot_ms,
            "bound_tpot_ms": pt.bound_tpot_ms,
            "concurrency": pt.concurrency,
            "total_chips": pt.total_chips,
            "mode": pt.mode,
            "binding_unit": pt.binding,
        }
        for pt in plan.frontier
        if pt.slo_met
    ]
