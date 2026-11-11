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
"""Megakernel core capability limits and default-deny support gating."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .constants import CHIP_KIND_MAP, CORE_LIMITS, CoreLimits
from .models import DEPLOYMENT_PRESETS, DeploymentPreset, ModelProfile
from .specs import (
    SUPPORTED_ONCHIP_FAMILIES,
    FusionScope,
    SupportStatus,
    WorkloadSpec,
)

if TYPE_CHECKING:
    from .chips import ChipSpec
    from .roofline import _ScopeContext

__all__ = [
    "CORE_LIMITS",
    "CoreLimits",
    "evaluate_support",
    "exceeds_default_buckets",
    "single_call_row_limit",
]


def _find_deployment_preset(profile_name: str) -> DeploymentPreset | None:
    """Resolve a DeploymentPreset by exact or normalized model preset name."""
    dep = DEPLOYMENT_PRESETS.get(profile_name)
    if dep is not None:
        return dep
    norm = profile_name.lower().replace(".", "_").replace("-", "_")
    for key, val in DEPLOYMENT_PRESETS.items():
        if key.lower().replace(".", "_").replace("-", "_") == norm:
            return val
    return None


def _declared_max_rows(profile_name: str, scope: FusionScope, eff_tp: int) -> int:
    """Return the row limit deployments.json declares for scope and ranks, else 0."""
    dep = _find_deployment_preset(profile_name)
    if dep is None:
        return 0
    for layout in (dep.decode, dep.prefill, dep.saturation):
        mk = layout.megakernel if layout is not None else None
        if not mk:
            continue
        scopes_raw = mk.get("scopes", ())
        match scopes_raw:
            case list() | tuple():
                scopes = tuple(str(s) for s in scopes_raw)
            case _:
                scopes = ()
        if (
            str(mk.get("source", ""))
            and scope in scopes
            and eff_tp == int(str(mk.get("ranks", 0)))
        ):
            return int(str(mk.get("max_rows", 0)))
    return 0


def _is_declared_in_preset(
    profile_name: str,
    scope: FusionScope,
    ctx: _ScopeContext,
) -> bool:
    """Return True if deployments.json declares megakernel support for scope."""
    declared_rows = _declared_max_rows(profile_name, scope, ctx.eff_tp)
    return 0 < ctx.active_tokens <= declared_rows


def single_call_row_limit(
    profile_name: str, workload: WorkloadSpec, eff_tp: int
) -> int:
    """Return the megakernel's single-call row limit for a decode or verify step.

    The workload's own limit wins; otherwise a deployments.json declaration for
    `whole_step` at this rank count supplies it. 0 means no declared limit.
    """
    if workload.phase not in ("decode", "verify"):
        return 0
    return workload.max_rows_per_call or _declared_max_rows(
        profile_name, "whole_step", eff_tp
    )


def exceeds_default_buckets(workload: WorkloadSpec, rows_per_call: int) -> bool:
    """Return whether one call is larger than the core's default bucket list."""
    tops = {
        "decode": CORE_LIMITS.default_decode_bucket_top,
        "verify": CORE_LIMITS.default_verify_bucket_top,
    }
    return rows_per_call > tops.get(workload.phase, rows_per_call)


def evaluate_support(
    scope: FusionScope,
    profile: ModelProfile,
    workload: WorkloadSpec,
    chip_spec: ChipSpec,
    ctx: _ScopeContext,
) -> tuple[SupportStatus, tuple[str, ...]]:
    """Return `(support_status, blockers)` for `scope` under `CORE_LIMITS`."""
    if scope == "per_op" or workload.batch <= 0:
        return ("baseline", ())

    has_unmodeled = any(
        fam not in SUPPORTED_ONCHIP_FAMILIES for fam in profile.layer_families
    )
    planner_bad = (
        bool(ctx.plan_error)
        or not profile.is_periodic
        or profile.ar_slots_per_layer > CORE_LIMITS.max_ar_slots_per_layer
    )
    checks = (
        (
            scope in ("single_layer", "layer_group"),
            "scope_without_core_counterpart",
        ),
        (
            ctx.active_tokens > CORE_LIMITS.max_batch_rows,
            "rows_exceed_core_max_batch_rows",
        ),
        (
            ctx.eff_tp not in CORE_LIMITS.verified_ranks,
            "rank_count_outside_core_grid",
        ),
        (
            CHIP_KIND_MAP.get(chip_spec.name.lower()) not in CORE_LIMITS.chip_kinds,
            "chip_without_core_budget",
        ),
        (ctx.ep_dp > 1, "all_to_all_not_in_core"),
        (has_unmodeled, "unmodeled_layer_family"),
        (profile.architecture_unverified, "model_type_unrecognized"),
        (planner_bad, "planner_rejected"),
    )
    blockers = tuple(tag for cond, tag in checks if cond)
    if blockers:
        return ("hypothetical", blockers)
    if _is_declared_in_preset(profile.name, scope, ctx):
        return ("declared", ())
    return ("within_core_limits", ())
