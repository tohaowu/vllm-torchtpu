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
"""VMEM floorplan, roofline bounds, and verdict tests for TPU advisor."""

from __future__ import annotations

import ast
import dataclasses
from pathlib import Path

import pytest

from tests.megakernel.core.megakernel_fakes import TINY_TILES, make_fake_model_spec
from tests.megakernel.test_advisor_specs import DEEPSEEK_V3_CFG, QWEN35_397B_CFG
from vllm_torchtpu.megakernel import advisor
from vllm_torchtpu.megakernel.advisor import constants, spec_bridge, support
from vllm_torchtpu.megakernel.advisor.roofline import (
    _build_scope_context,
    _core_time,
    _evaluate_scopes,
    _routed_chain,
)
from vllm_torchtpu.megakernel.core.errors import MegakernelConfigError
from vllm_torchtpu.megakernel.core.exec_spec import ExecSpec, Tiles

QWEN35_08B_CFG = {
    "model_type": "qwen3_5",
    "num_hidden_layers": 24,
    "hidden_size": 1024,
    "intermediate_size": 3584,
    "num_attention_heads": 8,
    "num_key_value_heads": 2,
    "head_dim": 256,
    "full_attention_interval": 4,
    "linear_num_key_heads": 16,
    "linear_num_value_heads": 16,
    "linear_key_head_dim": 128,
    "linear_value_head_dim": 128,
    "linear_conv_kernel_dim": 4,
    "top_k": 50,
    "quantization_config": {"quant_method": "fp8"},
}
MIXTRAL_8X7B_CFG = {
    "model_type": "mixtral",
    "num_hidden_layers": 32,
    "hidden_size": 4096,
    "intermediate_size": 14336,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "num_local_experts": 8,
    "num_experts_per_tok": 2,
    "torch_dtype": "bfloat16",
}

TPU7X_FP8_FLOPS = 2.30e15
TPU7X_FP8_ROW_FLOOR = 32
V6E_FP8_FLOPS = 9.20e14
V5E_HBM_CAP_GB = 24.0
TPU7X_VMEM_MIB = 62.0
CUSTOM_VMEM_MIB = 58.0
STAGE_OPS_PER_STEP = 18
QWEN35_IDEAL_VMEM_ROWS = 448
QWEN35_VERIFY_ROWS_MIN = 64
K3_PREFILL_CALLS = 13


def _preset_workload(
    layout: advisor.ParallelLayout, **changes: object
) -> advisor.WorkloadSpec:
    """Return a WorkloadSpec on a deployment preset layout with `changes` applied."""
    mesh = advisor.WorkloadSpec(
        tp=layout.tp,
        ep=layout.ep,
        dp=layout.dp,
        pp=layout.pp,
        dcp=layout.dcp,
        pcp=layout.pcp,
        layer_groups=layout.layer_groups,
    )
    return dataclasses.replace(mesh, **changes)


def test_per_token_vmem_live_interval_and_sharding() -> None:
    """Verify per-token live VMEM interval math and TP/EP sharding on Kimi-K3.1-Pro."""
    pk3 = advisor.resolve_model_profile("kimi-k3.1-pro")
    wl_grp = advisor.WorkloadSpec(
        phase="verify",
        batch=16,
        spec_tokens=3,
        tp=32,
        ep=32,
        moe_mode="grouped_on_chip",
    )
    parts_grp = advisor.compute_per_token_vmem_bytes(pk3, wl_grp)
    expected_grp_tok = parts_grp["residual_buffer_bytes"] + max(
        parts_grp["attn_scratch_bytes"],
        parts_grp["linear_attn_scratch_bytes"],
        parts_grp["moe_scratch_bytes"],
    )
    assert parts_grp["total_per_token_bytes"] == expected_grp_tok
    assert parts_grp["b_tok"] == expected_grp_tok
    rep_grp = advisor.advise(pk3, wl_grp, "tpu7x")
    cross_sc = next(s for s in rep_grp.scopes if s.scope == "whole_step")
    one_l_sc = next(s for s in rep_grp.scopes if s.scope == "single_layer")
    assert cross_sc.max_vmem_tokens >= QWEN35_VERIFY_ROWS_MIN
    assert one_l_sc.max_vmem_tokens > cross_sc.max_vmem_tokens

    wl_msk = dataclasses.replace(wl_grp, moe_mode="masked_on_chip")
    parts_msk = advisor.compute_per_token_vmem_bytes(pk3, wl_msk)
    assert parts_grp["total_per_token_bytes"] < parts_msk["total_per_token_bytes"]

    wl_no_alias = dataclasses.replace(wl_grp, vmem_alias=False)
    parts_no_alias = advisor.compute_per_token_vmem_bytes(pk3, wl_no_alias)
    assert parts_no_alias["total_per_token_bytes"] > parts_grp["total_per_token_bytes"]


def test_multi_tpu_vmem_and_fp8_dtype_physics() -> None:
    """Verify TPU FP8 physics, VMEM buffer overflow, Mixtral MoE, and DeepSeek MLA."""
    c7x = advisor.resolve_chip_spec("tpu7x")
    cv6e = advisor.resolve_chip_spec("v6e")
    cv5e = advisor.resolve_chip_spec("v5e")
    cv5p = advisor.resolve_chip_spec("v5p")
    cv4 = advisor.resolve_chip_spec("v4")

    assert (c7x.native_ocp_fp8, c7x.peak_flops("fp8"), c7x.lhs_row_floor("fp8")) == (
        True,
        TPU7X_FP8_FLOPS,
        TPU7X_FP8_ROW_FLOOR,
    )
    assert (cv6e.native_ocp_fp8, cv6e.peak_flops("fp8"), cv5p.peak_flops("fp8")) == (
        False,
        V6E_FP8_FLOPS,
        cv5p.bf16_flops_per_s,
    )
    assert (
        c7x.vmem_usable_bytes,
        cv6e.vmem_usable_bytes,
        cv4.vmem_usable_bytes,
    ) == (62 << 20, 126 << 20, 14 << 20)

    p397 = advisor.model_profile_from_hf_config(QWEN35_397B_CFG, name="qwen35-397b")
    rep_v4 = advisor.advise(
        p397,
        advisor.WorkloadSpec(tp=8, batch=4, tiles=Tiles(nbd=8, nbe=4)),
        cv4,
    )
    v4_cross = next(s for s in rep_v4.scopes if s.scope == "whole_step")
    assert rep_v4.hbm_fits is False
    assert rep_v4.build_megakernel is False
    assert v4_cross.t_vmem_max == 0
    assert v4_cross.reason == "Weight buffers exceed VMEM"
    rep_v6e_tp8 = advisor.advise(p397, advisor.WorkloadSpec(tp=8, batch=4), cv6e)
    assert rep_v6e_tp8.hbm_fits is False

    pmix = advisor.model_profile_from_hf_config(MIXTRAL_8X7B_CFG, name="mixtral-8x7b")
    assert pmix.num_moe_layers == MIXTRAL_8X7B_CFG["num_hidden_layers"]
    assert pmix.moe_intermediate_size == MIXTRAL_8X7B_CFG["intermediate_size"]
    rep_mix = advisor.advise(pmix, advisor.WorkloadSpec(tp=4, batch=16), cv5e)
    assert rep_mix.hbm_fits is False
    assert rep_mix.hbm_used_gb > V5E_HBM_CAP_GB
    ffn_dim_test = 2048
    p_ffn = advisor.model_profile_from_hf_config(
        {
            "model_type": "mixtral",
            "num_hidden_layers": 2,
            "hidden_size": 512,
            "ffn_dim": ffn_dim_test,
            "num_attention_heads": 8,
            "num_local_experts": 4,
            "num_experts_per_tok": 2,
        }
    )
    assert p_ffn.intermediate_size == ffn_dim_test
    assert p_ffn.moe_intermediate_size == ffn_dim_test


def test_deepseek_v3_mla_kv_split_physics() -> None:
    """Verify DeepSeek-V3 MLA parameter sizing and kv_split HBM savings."""
    c7x = advisor.resolve_chip_spec("tpu7x")
    pdsv3 = advisor.model_profile_from_hf_config(DEEPSEEK_V3_CFG, name="deepseek-v3")
    assert pdsv3.head_dim == DEEPSEEK_V3_CFG["qk_nope_head_dim"]
    assert (
        pdsv3.shared_expert_intermediate_size
        == DEEPSEEK_V3_CFG["moe_intermediate_size"]
    )
    assert pdsv3.num_moe_layers == (
        DEEPSEEK_V3_CFG["num_hidden_layers"] - DEEPSEEK_V3_CFG["first_k_dense_replace"]
    )
    assert pdsv3.is_periodic is False
    pdsv3_dense_only = dataclasses.replace(
        pdsv3, num_moe_layers=0, num_experts=0, top_k=0, intermediate_size=0
    )
    r_mla_only = advisor.advise(
        pdsv3_dense_only, advisor.WorkloadSpec(tp=8, batch=1, context=4096), c7x
    )
    nh_loc = pdsv3_dense_only.num_attention_heads // 8
    q_tot = nh_loc * (
        pdsv3_dense_only.qk_nope_head_dim + pdsv3_dense_only.qk_rope_head_dim
    )
    q_dim = nh_loc * pdsv3_dense_only.head_dim
    mla_layer_params = (
        pdsv3_dense_only.q_lora_rank * (pdsv3_dense_only.hidden_size + q_tot)
        + pdsv3_dense_only.hidden_size
        * (pdsv3_dense_only.kv_lora_rank + pdsv3_dense_only.qk_rope_head_dim + q_dim)
        + pdsv3_dense_only.kv_lora_rank
        * nh_loc
        * (pdsv3_dense_only.qk_nope_head_dim + pdsv3_dense_only.head_dim)
        + 3 * pdsv3_dense_only.hidden_size
    )
    expected_mla_w_gb = round((pdsv3_dense_only.num_layers * mla_layer_params) / 1e9, 4)
    assert r_mla_only.scopes[0].w_step_gb == expected_mla_w_gb
    r_mla_pre = advisor.advise(
        pdsv3_dense_only,
        advisor.WorkloadSpec(phase="prefill", tp=8, batch=1, isl=4096, chunk_size=1),
        c7x,
    )
    assert r_mla_only.step_arithmetic_intensity > r_mla_pre.step_arithmetic_intensity
    r_ds1 = advisor.advise(
        pdsv3, advisor.WorkloadSpec(tp=8, batch=32, context=8192), c7x
    )
    r_ds2 = advisor.advise(
        pdsv3, advisor.WorkloadSpec(tp=8, kv_split=2, batch=32, context=8192), c7x
    )
    assert r_ds2.hbm_used_gb < r_ds1.hbm_used_gb
    assert [
        (s.feasible, s.support, "unmodeled_layer_family" in s.blockers)
        for s in r_ds1.scopes
        if s.scope == "whole_step"
    ] == [(True, "hypothetical", True)]
    verify_batch, verify_spec = 8, 3
    mixed_chunk = 128
    r_ver = advisor.advise(
        pdsv3,
        advisor.WorkloadSpec(
            phase="verify", tp=8, batch=verify_batch, spec_tokens=verify_spec
        ),
        c7x,
    )
    r_mix = advisor.advise(
        pdsv3,
        advisor.WorkloadSpec(
            phase="mixed",
            tp=8,
            batch=verify_batch,
            spec_tokens=verify_spec,
            chunk_size=mixed_chunk,
        ),
        c7x,
    )
    assert r_ver.active_tokens == verify_batch * (verify_spec + 1)
    assert r_mix.active_tokens == verify_batch * (verify_spec + 1) + mixed_chunk


def test_asymptotic_decode_intensity_and_vmem_fit_headroom() -> None:
    """Verify `asymptotic_decode_intensity`, `fit_vmem_budget_bytes`, `w_crit_gb`."""
    assert advisor.fit_vmem_budget_bytes() == 62 << 20
    assert advisor.fit_vmem_budget_bytes(Tiles(vmem_mib=62)) == 62 << 20
    p08 = advisor.model_profile_from_hf_config(
        QWEN35_08B_CFG, name="qwen35-0.8b", state_dtype="fp32"
    )
    r_b16 = advisor.advise(
        p08,
        advisor.WorkloadSpec(phase="verify", batch=16, spec_tokens=3, tp=8),
        "tpu7x",
    )
    r_b128 = advisor.advise(
        p08,
        advisor.WorkloadSpec(phase="verify", batch=128, spec_tokens=3, tp=8),
        "tpu7x",
    )
    assert r_b16.asymptotic_decode_intensity is not None
    assert r_b16.asymptotic_decode_intensity == r_b128.asymptotic_decode_intensity
    assert r_b16.asymptotic_decode_intensity < r_b16.ridge_dense_flop_per_byte
    assert r_b16.step_arithmetic_intensity < r_b16.asymptotic_decode_intensity

    r_alias = advisor.advise(
        p08, advisor.WorkloadSpec(phase="prefill", batch=1, isl=2048, tp=8), "tpu7x"
    )
    r_no_alias = advisor.advise(
        p08,
        advisor.WorkloadSpec(
            phase="prefill",
            batch=1,
            isl=2048,
            tp=8,
            vmem_alias=False,
            vmem_fit_headroom=True,
        ),
        "tpu7x",
    )
    r_no_alias_disp = advisor.advise(
        p08,
        advisor.WorkloadSpec(
            phase="prefill",
            batch=1,
            isl=2048,
            tp=8,
            vmem_alias=False,
            vmem_fit_headroom=True,
            op_overhead_us=5.0,
        ),
        "tpu7x",
    )
    assert r_alias.vmem_budget_mib == TPU7X_VMEM_MIB
    assert r_no_alias.vmem_budget_mib == advisor.fit_vmem_budget_bytes() / (1 << 20)
    cx_alias = next(s for s in r_alias.scopes if s.scope == "whole_step")
    cx_no_al = next(s for s in r_no_alias.scopes if s.scope == "whole_step")
    cx_disp = next(s for s in r_no_alias_disp.scopes if s.scope == "whole_step")
    assert cx_alias.n_splits == 1
    assert cx_no_al.n_splits > 1
    assert cx_no_al.w_crit_gb is not None
    assert cx_disp.w_crit_gb is not None
    assert cx_disp.w_crit_gb > cx_no_al.w_crit_gb
    assert cx_disp.step_bound_us > cx_no_al.step_bound_us


def test_all_four_fusion_scopes_reachable() -> None:
    """Prove `whole_step`, `single_layer`, `stage`, and `per_op` each win."""
    p08 = advisor.model_profile_from_hf_config(QWEN35_08B_CFG, name="qwen35-0.8b")
    p397 = advisor.model_profile_from_hf_config(QWEN35_397B_CFG, name="qwen35-397b")
    pllama = advisor.resolve_model_profile("llama-3.1-405b")

    # 1. `whole_step` (`whole_step`) wins on 397B TP8 decode and 0.8B prefill.
    r_dec = advisor.advise(
        p397, advisor.WorkloadSpec(phase="decode", batch=16, tp=8), "tpu7x"
    )
    assert r_dec.recommended_scope == "whole_step"
    assert r_dec.recommended_moe_mode == "masked_on_chip"
    assert r_dec.build_megakernel is True
    r_b32 = advisor.advise(
        p397, advisor.WorkloadSpec(phase="decode", batch=32, tp=8), "tpu7x"
    )
    assert r_b32.recommended_scope == "whole_step"
    assert r_b32.recommended_moe_mode == "grouped_on_chip"
    r_08_pre = advisor.advise(
        p08, advisor.WorkloadSpec(phase="prefill", batch=1, isl=2048, tp=8), "tpu7x"
    )
    assert r_08_pre.fastest_scope == "whole_step"
    assert r_08_pre.recommended_moe_mode == "dense_mlp"

    # 2. `single_layer` (`layer`) wins in the B=488 VMEM cliff band on tpu7x,
    # whereas v6e (TP16) and v5p Megacore (TP8) keep `whole_step` in 1 pass.
    r_cliff_7x = advisor.advise(
        p397,
        advisor.WorkloadSpec(phase="decode", batch=488, context=1024, tp=8),
        "tpu7x",
    )
    assert r_cliff_7x.fastest_scope == "single_layer"
    assert r_cliff_7x.recommended_moe_mode == "grouped_on_chip"
    r_cliff_v6e = advisor.advise(
        p397,
        advisor.WorkloadSpec(phase="decode", batch=488, context=1024, tp=16),
        "v6e",
    )
    assert r_cliff_v6e.fastest_scope == "whole_step"
    v5p_mc = advisor.resolve_chip_spec("v5p", megacore=True)
    r_cliff_v5p_mc = advisor.advise(
        p397,
        advisor.WorkloadSpec(phase="decode", batch=488, context=1024, tp=8),
        v5p_mc,
    )
    assert r_cliff_v5p_mc.fastest_scope == "whole_step"
    r_ep8_sat = advisor.advise(
        p397,
        advisor.WorkloadSpec(
            phase="decode", batch=600, tp=2, dp=4, ep=8, tiles=Tiles()
        ),
        "tpu7x",
    )
    # Saturated DP4 x TP2 decode with locked wide ring: 150 tok/core overflow
    # the wide whole-step ceiling while single_layer fits in 1 pass.
    assert r_ep8_sat.fastest_scope == "single_layer"
    assert r_ep8_sat.recommended_moe_mode == "grouped_on_chip"

    r_llama_pre = advisor.advise(
        pllama, advisor.WorkloadSpec(phase="prefill", batch=1, isl=4096, tp=8), "tpu7x"
    )
    assert r_llama_pre.recommended_scope == "per_op"
    assert r_llama_pre.recommended_moe_mode == "dense_mlp"
    assert r_llama_pre.build_megakernel is False

    # 3. `layer_group` wins on PP8 tail and segmented TP8 (`segment_layers=10`).
    r_stg = advisor.advise(
        p397,
        advisor.WorkloadSpec(
            phase="prefill", batch=1, isl=4096, chunk_size=64, tp=1, pp=8
        ),
        "tpu7x",
    )
    assert r_stg.fastest_scope == "layer_group"
    r_seg = advisor.advise(
        p397,
        advisor.WorkloadSpec(phase="decode", batch=16, tp=8, segment_layers=10),
        "tpu7x",
    )
    assert r_seg.fastest_scope == "layer_group"
    assert (
        next(s for s in r_seg.scopes if s.scope == "layer_group").ops_per_step
        == STAGE_OPS_PER_STEP
    )
    r_16k = advisor.advise(
        p397,
        advisor.WorkloadSpec(phase="prefill", batch=8, isl=16384, tp=1, pp=8),
        "tpu7x",
    )
    assert r_16k.recommended_scope == "per_op"
    assert r_16k.build_megakernel is False


def test_vmem_budget_override_and_dma_buffers_triple_buffering() -> None:
    """Verify user-configurable `--vmem-budget-mib` and `--dma-buffers {2,3}`."""
    p397 = advisor.resolve_model_profile("qwen3.5-397b")
    rep_2buf = advisor.advise(
        p397, advisor.WorkloadSpec(tp=8, ep=8, batch=16, dma_buffers=2), "tpu7x"
    )
    rep_3buf = advisor.advise(
        p397, advisor.WorkloadSpec(tp=8, ep=8, batch=16, dma_buffers=3), "tpu7x"
    )
    rep_custom_vmem = advisor.advise(
        p397,
        advisor.WorkloadSpec(
            tp=8, ep=8, batch=16, dma_buffers=2, vmem_budget_mib=CUSTOM_VMEM_MIB
        ),
        "tpu7x",
    )
    cx_2 = next(s for s in rep_2buf.scopes if s.scope == "whole_step")
    cx_3 = next(s for s in rep_3buf.scopes if s.scope == "whole_step")
    cx_vmem = next(s for s in rep_custom_vmem.scopes if s.scope == "whole_step")
    assert rep_2buf.vmem_budget_mib == TPU7X_VMEM_MIB
    assert rep_custom_vmem.vmem_budget_mib == CUSTOM_VMEM_MIB
    assert cx_vmem.max_vmem_tokens < cx_2.max_vmem_tokens
    assert cx_3.vmem_static_mib > cx_2.vmem_static_mib
    assert cx_3.max_vmem_tokens < cx_2.max_vmem_tokens
    assert cx_3.step_bound_us >= cx_2.step_bound_us


@pytest.mark.parametrize(
    ("tp", "ep", "pcp", "accepted"),
    [(4, 8, 2, True), (4, 8, 1, False), (4, 16, 2, False)],
)
def test_expert_parallel_spans_context_parallel_ranks(
    tp: int, ep: int, pcp: int, *, accepted: bool
) -> None:
    """Verify `ep` may span `tp * dp * pcp` ranks and no wider."""
    p397 = advisor.resolve_model_profile("qwen3.5-397b")
    wl = advisor.WorkloadSpec(phase="prefill", batch=1, isl=4096, tp=tp, ep=ep, pcp=pcp)
    if accepted:
        rep = advisor.advise(p397, wl, "tpu7x")
        assert rep.active_tokens == 4096 // pcp
    else:
        with pytest.raises(MegakernelConfigError):
            advisor.advise(p397, wl, "tpu7x")


def test_advise_from_model_spec_and_planner() -> None:
    """Verify `advise_from_spec` bridges periodic and non-periodic `ModelSpec`."""
    ms = make_fake_model_spec(L=4, att_every=4, H=256)
    ex = ExecSpec(B=4, T=1, R=2, ctx=1024, tiles=TINY_TILES)
    rep = advisor.advise_from_spec(
        ms, ex, chip="tpu7x", moe_mode="grouped_on_chip", dp=4
    )
    assert rep.recommended_scope == "whole_step"
    rep_def = advisor.advise_from_spec(
        ms,
        ExecSpec(B=4, T=1, R=2, ctx=1024),
        chip="tpu7x",
        moe_mode="grouped_on_chip",
        dp=4,
    )
    assert rep_def.recommended_scope == "whole_step"
    assert rep_def.vmem_budget_mib == TPU7X_VMEM_MIB

    ms_non_periodic = make_fake_model_spec(L=5, att_every=4, H=256)
    rep_np = advisor.advise_from_spec(
        ms_non_periodic, ex, chip="tpu7x", moe_mode="grouped_on_chip", dp=4
    )
    cross = next(s for s in rep_np.scopes if s.scope == "whole_step")
    assert cross.feasible is False
    assert rep_np.fastest_scope == "single_layer"
    assert rep_np.recommended_scope == "per_op"


def test_spec_bridge_jax_isolation_ast() -> None:
    """Verify spec_bridge.py isolates jax and planner imports inside ast.Try."""
    bridge_path = Path(advisor.__file__).with_name("spec_bridge.py")
    tree = ast.parse(bridge_path.read_text("utf-8"), filename=str(bridge_path))
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = [alias.name for alias in node.names]
            mod = getattr(node, "module", "") or ""
            assert not any("jax" in n for n in names), f"Unguarded jax import: {names}"
            assert "jax" not in mod, f"Unguarded jax module import: {mod}"
            assert not any("planner" in n for n in names), (
                f"Unguarded planner import: {names}"
            )
    try_nodes = [n for n in tree.body if isinstance(n, ast.Try)]
    assert any(
        any(
            (
                isinstance(h.type, ast.Tuple)
                and any(
                    getattr(e, "id", "") in ("ImportError", "RuntimeError")
                    for e in h.type.elts
                )
            )
            or getattr(h.type, "id", "") in ("ImportError", "RuntimeError")
            for h in t.handlers
        )
        for t in try_nodes
    ), "spec_bridge.py must have ast.Try catching (ImportError, RuntimeError)"


def test_advise_from_spec_raises_when_planner_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify advise_from_spec raises MegakernelConfigError when planner is missing."""
    monkeypatch.setattr(spec_bridge, "planner", None)
    ms = make_fake_model_spec(L=4, att_every=4, H=256)
    ex = ExecSpec(B=4, T=1, R=2, ctx=1024, tiles=TINY_TILES)
    with pytest.raises(MegakernelConfigError, match="Megakernel planner requires JAX"):
        advisor.advise_from_spec(ms, ex)


def test_qwen35_decode_preset_whole_step_admits_shipped_verify_rows() -> None:
    """Verify whole_step fits 64 verify rows and reports the ideal VMEM ceiling."""
    dep = advisor.DEPLOYMENT_PRESETS["qwen3.5-397b"]
    rows_per_seq = dep.decode.spec_tokens + 1
    rep = advisor.advise(
        advisor.resolve_model_profile("qwen3.5-397b"),
        _preset_workload(
            dep.decode,
            phase="verify",
            batch=QWEN35_VERIFY_ROWS_MIN // rows_per_seq,
            spec_tokens=dep.decode.spec_tokens,
            moe_mode="grouped_on_chip",
        ),
        "tpu7x",
    )
    whole = next(s for s in rep.scopes if s.scope == "whole_step")
    rows_cap = rows_per_seq * (whole.t_vmem_max // rows_per_seq)
    assert rows_cap >= QWEN35_VERIFY_ROWS_MIN
    assert abs(rows_cap - QWEN35_IDEAL_VMEM_ROWS) <= rows_per_seq


@pytest.mark.parametrize("batch", [1, 2, 4, 8])
def test_kimi_k3_decode_preset_low_batch_prefers_whole_step(batch: int) -> None:
    """Verify Kimi-K3 decode at batch 1..8 on the TP32 mesh is one call."""
    dep = advisor.DEPLOYMENT_PRESETS["kimi-k3"]
    rep = advisor.advise(
        advisor.resolve_model_profile("kimi-k3"),
        _preset_workload(
            dep.decode,
            phase="decode",
            batch=batch,
        ),
        "tpu7x",
    )
    assert rep.fastest_scope == "whole_step"
    assert rep.recommended_scope == "per_op"
    assert (
        "rank_count_outside_core_grid"
        in next(s for s in rep.scopes if s.scope == "whole_step").blockers
    )


@pytest.mark.parametrize("window", [2048, 8192])
def test_kimi_k3_prefill_preset_evaluates_layer_group(window: int) -> None:
    """Verify the shipped 13-call layer grouping is feasible and evaluated."""
    dep = advisor.DEPLOYMENT_PRESETS["kimi-k3"]
    rep = advisor.advise(
        advisor.resolve_model_profile("kimi-k3"),
        _preset_workload(
            dep.prefill,
            phase="prefill",
            batch=1,
            isl=window,
            chunk_size=window,
            moe_mode="grouped_on_chip",
        ),
        "tpu7x",
    )
    group = next(s for s in rep.scopes if s.scope == "layer_group")
    assert group.feasible
    assert group.ops_per_step == K3_PREFILL_CALLS
    assert rep.active_tokens == window // dep.prefill.pcp


def test_layer_groups_must_cover_every_layer() -> None:
    """Verify a layer grouping that does not sum to num_layers is rejected."""
    with pytest.raises(MegakernelConfigError):
        advisor.advise(
            advisor.resolve_model_profile("kimi-k3"),
            advisor.WorkloadSpec(tp=32, ep=32, layer_groups=(8, 8)),
            "tpu7x",
        )


def test_whole_step_tiles_overflow_and_reads_weights_once() -> None:
    """Verify whole_step tiles tokens past the ceiling instead of re-passing."""
    dep = advisor.DEPLOYMENT_PRESETS["qwen3.5-397b"]
    rep = advisor.advise(
        advisor.resolve_model_profile("qwen3.5-397b"),
        _preset_workload(
            dep.decode,
            phase="verify",
            batch=(QWEN35_IDEAL_VMEM_ROWS // (dep.decode.spec_tokens + 1)) + 4,
            spec_tokens=dep.decode.spec_tokens,
            moe_mode="grouped_on_chip",
        ),
        "tpu7x",
    )
    tiled = next(s for s in rep.scopes if s.scope == "whole_step")
    one_l = next(s for s in rep.scopes if s.scope == "single_layer")
    assert tiled.n_splits > 1
    assert tiled.feasible
    assert tiled.w_step_gb == one_l.w_step_gb
    assert tiled.step_bound_us < 1.5 * one_l.step_bound_us


def test_fused_scopes_price_the_recommended_moe_mode() -> None:
    """Verify a full row tile per EP rank prices on-chip grouping either way."""
    prof = advisor.resolve_model_profile("qwen3.5-397b")
    steps = []
    for mode in ("masked_on_chip", "grouped_on_chip"):
        rep = advisor.advise(
            prof,
            advisor.WorkloadSpec(phase="decode", batch=64, tp=8, ep=8, moe_mode=mode),
            "tpu7x",
        )
        steps.append(next(s for s in rep.scopes if s.scope == "whole_step"))
        assert rep.recommended_moe_mode == "grouped_on_chip"
    assert steps[0].moe_flop_factor == steps[1].moe_flop_factor


GROUPED_PHI_CEILING = 5.0
UNGROUPED_PHI_FLOOR = 10.0


def test_intra_host_dp_grouped_moe_consistency() -> None:
    """Verify intra-host DPExchange groups MoE while multi-host DCN stays ungrouped."""
    prof = advisor.resolve_model_profile("qwen3.5-397b")
    rep_grp = advisor.advise(
        prof,
        advisor.WorkloadSpec(
            phase="decode",
            batch=600,
            context=32768,
            tp=2,
            dp=4,
            ep=8,
            moe_mode="grouped_on_chip",
        ),
        "tpu7x",
    )
    ws_grp = next(s for s in rep_grp.scopes if s.scope == "whole_step")
    assert ws_grp.moe_flop_factor < GROUPED_PHI_CEILING
    assert rep_grp.fastest_scope == "whole_step"
    assert rep_grp.recommended_scope == "per_op"
    assert rep_grp.recommended_moe_mode == "grouped_on_chip"
    assert float(rep_grp.metrics["achievable_gain_pct"]["value"]) > 0.0

    rep_msk = advisor.advise(
        prof,
        advisor.WorkloadSpec(
            phase="decode",
            batch=600,
            context=32768,
            tp=2,
            dp=4,
            ep=8,
            moe_mode="masked_on_chip",
        ),
        "tpu7x",
    )
    ws_msk = next(s for s in rep_msk.scopes if s.scope == "whole_step")
    assert ws_msk.moe_flop_factor > UNGROUPED_PHI_FLOOR
    assert rep_msk.recommended_moe_mode != "grouped_on_chip"

    rep_dcn = advisor.advise(
        prof,
        advisor.WorkloadSpec(
            phase="decode",
            batch=600,
            context=32768,
            tp=2,
            dp=8,
            ep=16,
            moe_mode="grouped_on_chip",
        ),
        "tpu7x",
    )
    ws_dcn = next(s for s in rep_dcn.scopes if s.scope == "whole_step")
    assert ws_dcn.moe_flop_factor > UNGROUPED_PHI_FLOOR
    assert rep_dcn.recommended_moe_mode != "grouped_on_chip"


def test_unchunked_pp_prefill_preserves_window() -> None:
    """Verify unchunked B=1 prefill under PP>1 preserves its full stage window."""
    prof = advisor.resolve_model_profile("qwen3.5-397b")
    isl_tokens = 4096
    rep_single = advisor.advise(
        prof,
        advisor.WorkloadSpec(
            phase="prefill",
            batch=1,
            isl=isl_tokens,
            tp=1,
            pp=8,
            ep=1,
            chunk_size=0,
            microbatches=0,
        ),
        "tpu7x",
    )
    assert rep_single.active_tokens == isl_tokens
    assert rep_single.recommended_scope == "per_op"
    assert rep_single.recommended_moe_mode == "grouped_in_hbm"
    assert float(rep_single.metrics["compiler_proof_gain_pct"]["value"]) == 0.0

    rep_microbatched = advisor.advise(
        prof,
        advisor.WorkloadSpec(
            phase="prefill",
            batch=8,
            isl=isl_tokens,
            tp=1,
            pp=8,
            ep=1,
            chunk_size=0,
            microbatches=8,
        ),
        "tpu7x",
    )
    assert rep_microbatched.active_tokens == isl_tokens


TP32_HOPS = 5
# 2 * (31 / 32) * 1 MiB over one 100 GB/s link is 20.3162 us, plus 5 hops of 1 us.
TP32_AR_COST_US = 25.3162
A2A_TIME_US = 98.304
STRICT_W_STEP_GB = 2.17052
STRICT_STEP_FLOOR_US = 851.66
# Routed-expert chain plus upper-band collectives now exceed the HBM stream.
# One-link transfer adds 120 all-reduces x 6.1167 us / eta_ici 0.75 = 978.67 us.
ACH_UPPER_VERIFY_US = 15473.32
DS4_KV_READ_GB = 10.1478
QWEN4_RESIDUAL_BYTES = 49152
# Per-op pays 120 all-reduces x 1.5292 us more on one link than on the egress
# sum (183.5 us); the fused scope still overlaps that transfer with HBM.
GAIN_FACT_US = 358.50
# The fused scope pays 175.83 us of that 183.5 us once collectives no longer
# fit under the HBM stream, so the prior-dependent part shrinks from 954.0.
GAIN_ASSUMED_US = 778.17
CHAIN_T_DENSE_S = 0.05
CHAIN_T_MOE_S = 0.10
CHAIN_ROUTED_HBM_S = 0.80
CHAIN_HBM_S = 1.00
CHAIN_COMM_S = 0.30
SLOW_COLLECTIVE_US = 3.0
SINGLE_PASS_BATCH = 16
MULTI_PASS_BATCH = 512
ROW_LIMIT = 80
ROWS_PER_VERIFY_REQUEST = 4
SPLIT_BATCH = 28
# A launch gap large enough that whole_step is the fastest scope even as
# several calls, so the report-level fields describe the split step.
WHOLE_STEP_FAVORING_GAP_US = 500.0


def test_core_limits_literals_match_megakernel_core() -> None:
    """Verify CORE_LIMITS matches megakernel/core/exec_spec.py and profile.py."""
    mk_dir = Path(advisor.__file__).resolve().parent.parent
    exec_text = (mk_dir / "core" / "exec_spec.py").read_text("utf-8")
    prof_text = (mk_dir / "core" / "profile.py").read_text("utf-8")
    tiles_default = Tiles()
    lim = advisor.CORE_LIMITS
    assert f"MAX_BATCH_ROWS = {lim.max_batch_rows}" in exec_text
    assert "_NUMERICS_GRID_R = (1, 2, 4, 8, 16)" in prof_text

    assert lim.max_batch_rows == constants.CORE_LIMITS.max_batch_rows
    assert lim.verified_ranks == constants.CORE_LIMITS.verified_ranks
    assert lim.chip_kinds == constants.CORE_LIMITS.chip_kinds
    assert lim.max_ar_slots_per_layer == constants.CORE_LIMITS.max_ar_slots_per_layer
    assert (lim.nbd, lim.nbe, lim.bank, lim.ebank, lim.vmem_mib) == (
        tiles_default.nbd,
        tiles_default.nbe,
        tiles_default.bank,
        tiles_default.ebank,
        tiles_default.vmem_mib,
    )


def test_unknown_model_type_is_hypothetical() -> None:
    """Verify an unrecognized HF model_type is hypothetical and never recommended."""
    cfg = {
        "model_type": "brand_new_2027_arch",
        "num_hidden_layers": 32,
        "hidden_size": 4096,
        "intermediate_size": 11008,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "head_dim": 128,
    }
    prof = advisor.model_profile_from_hf_config(cfg, name="brand-new-2027")
    rep = advisor.advise(
        prof, advisor.WorkloadSpec(phase="decode", batch=16, tp=8), "tpu7x"
    )
    whole = next(s for s in rep.scopes if s.scope == "whole_step")
    assert rep.recommended_scope == "per_op"
    assert rep.fastest_scope == "whole_step"
    assert whole.support == "hypothetical"
    assert "model_type_unrecognized" in whole.blockers
    assert rep.hypothetical_best is not None
    assert rep.hypothetical_best["scope"] == "whole_step"


@pytest.mark.parametrize(
    ("workload", "chip", "expected_blocker"),
    [
        (
            advisor.WorkloadSpec(phase="decode", batch=256, tp=8, ep=8, dp=1),
            "tpu7x",
            "rows_exceed_core_max_batch_rows",
        ),
        (
            advisor.WorkloadSpec(phase="decode", batch=64, tp=2, ep=8, dp=4),
            "tpu7x",
            "all_to_all_not_in_core",
        ),
        (
            advisor.WorkloadSpec(phase="decode", batch=16, tp=32, ep=32, dp=1),
            "tpu7x",
            "rank_count_outside_core_grid",
        ),
        (
            advisor.WorkloadSpec(phase="decode", batch=16, tp=16, ep=16, dp=1),
            "v6e",
            "chip_without_core_budget",
        ),
    ],
)
def test_no_fused_recommendation_outside_core_limits(
    workload: advisor.WorkloadSpec,
    chip: str,
    expected_blocker: str,
) -> None:
    """Verify >128 rows, ep_dp>1, 32 ranks, or non-TPU7x chips gate to per_op."""
    prof = advisor.resolve_model_profile("qwen3.5-397b")
    rep = advisor.advise(prof, workload, chip)
    whole = next(s for s in rep.scopes if s.scope == "whole_step")
    assert whole.support == "hypothetical"
    assert expected_blocker in whole.blockers
    assert rep.recommended_scope == "per_op"
    assert rep.build_megakernel is False
    assert rep.fastest_scope in ("whole_step", "single_layer", "layer_group")
    assert rep.hypothetical_best is not None


def test_in_slice_all_reduce_uses_ici_for_every_rank_count() -> None:
    """Verify TP32 All-Reduce uses single-slice ICI rather than DCN (Ledger Row 2)."""
    chip = advisor.resolve_chip_spec("tpu7x")
    msg_bytes = 64 * 8192 * 2
    t_trans, t_lat, hops, _ = advisor.compute_all_reduce_transfer_cost(
        msg_bytes, 32, chip
    )
    assert hops == TP32_HOPS
    assert round((t_trans + t_lat) * 1e6, 4) == TP32_AR_COST_US


def test_all_reduce_transfer_uses_single_link() -> None:
    """Verify all-reduce transfer is priced on one ICI link, not the egress sum."""
    chip = advisor.resolve_chip_spec("tpu7x")
    msg_bytes = 64 * 8192 * 2
    t_trans, _, _, volume = advisor.compute_all_reduce_transfer_cost(msg_bytes, 8, chip)
    assert t_trans == pytest.approx(volume / chip.ici_link_bw_bytes_per_s)
    assert t_trans > volume / chip.ici_egress_bw_bytes_per_s


@pytest.mark.parametrize("batch", [SINGLE_PASS_BATCH, MULTI_PASS_BATCH])
def test_strict_floor_prices_all_reduce_on_egress(batch: int) -> None:
    """Verify the strict floor ignores link bandwidth while the floor pays it."""
    prof = advisor.resolve_model_profile("qwen3.5-397b")
    wl = advisor.WorkloadSpec(phase="decode", batch=batch, tp=8, ep=8)
    chip = advisor.resolve_chip_spec("tpu7x")
    slow_chip = dataclasses.replace(
        chip, ici_link_bw_bytes_per_s=chip.ici_link_bw_bytes_per_s / 2
    )
    fast, slow = (
        next(
            s for s in advisor.advise(prof, wl, spec).scopes if s.scope == "whole_step"
        )
        for spec in (chip, slow_chip)
    )
    assert (fast.n_splits > 1) == (batch == MULTI_PASS_BATCH)
    assert slow.step_floor_strict_us == pytest.approx(fast.step_floor_strict_us)
    assert slow.step_bound_us > fast.step_bound_us


def test_all_to_all_fanout_capped_by_destination_groups() -> None:
    """Verify MoE All-to-All fanout caps at min(top_k, ep_dp) (Ledger Row 3)."""
    prof = advisor.resolve_model_profile("qwen3.5-397b")
    chip = advisor.resolve_chip_spec("tpu7x")
    wl = advisor.WorkloadSpec(
        phase="decode", batch=2400, tp=2, dp=4, ep=8, moe_mode="grouped_on_chip"
    )
    ctx = _build_scope_context(prof, wl, chip)
    assert ctx.ep_dp == wl.ep // wl.tp
    assert ctx.a2a_time * 1e6 == pytest.approx(A2A_TIME_US, abs=0.01)


def test_strict_floor_not_above_uniform_estimate() -> None:
    """Verify strict balanced routing floor vs uniform coupon-collector (Row 4)."""
    prof = advisor.resolve_model_profile("qwen3.5-397b")
    rep = advisor.advise(
        prof,
        advisor.WorkloadSpec(phase="decode", batch=16, tp=8, ep=8),
        "tpu7x",
    )
    whole = next(s for s in rep.scopes if s.scope == "whole_step")
    assert whole.w_step_strict_gb == pytest.approx(STRICT_W_STEP_GB, abs=1e-4)
    assert whole.step_floor_strict_us == pytest.approx(STRICT_STEP_FLOOR_US, abs=0.05)
    assert 0.0 < whole.step_floor_strict_us < whole.step_bound_us


def test_fused_core_is_max_of_hbm_and_compute_plus_comm() -> None:
    """Verify fused critical path uses max(hbm_time_s, t_matmul + comm_s)."""
    prof = advisor.resolve_model_profile("qwen3.5-0.8b")
    rep = advisor.advise(
        prof,
        advisor.WorkloadSpec(phase="prefill", batch=1, isl=512, chunk_size=512, tp=8),
        "tpu7x",
    )
    whole = next(s for s in rep.scopes if s.scope == "whole_step")
    assert whole.compute_us + whole.comm_us > whole.hbm_us
    assert whole.comm_us > 0.0
    assert whole.roof_floor_us == pytest.approx(
        max(whole.hbm_us, whole.compute_us + whole.comm_us) + whole.fill_drain_us,
        abs=0.05,
    )
    assert (
        whole.roof_floor_us
        > max(whole.compute_us, whole.hbm_us, whole.comm_us) + whole.fill_drain_us
    )


def test_core_time_routed_stream_cannot_hide_collectives() -> None:
    """Verify collectives add to the routed-expert chain, not under the HBM stream."""
    chain_s = _routed_chain(CHAIN_T_DENSE_S, CHAIN_ROUTED_HBM_S, CHAIN_T_MOE_S)
    assert chain_s == pytest.approx(CHAIN_T_DENSE_S + CHAIN_ROUTED_HBM_S)
    fused_s = _core_time(CHAIN_HBM_S, chain_s, CHAIN_COMM_S, fused=True)
    assert fused_s == pytest.approx(chain_s + CHAIN_COMM_S)
    assert fused_s > max(CHAIN_HBM_S, CHAIN_T_DENSE_S + CHAIN_T_MOE_S + CHAIN_COMM_S)
    split_s = _core_time(CHAIN_HBM_S, chain_s, CHAIN_COMM_S, fused=False)
    assert split_s == pytest.approx(CHAIN_HBM_S + CHAIN_COMM_S)
    assert _core_time(CHAIN_HBM_S, chain_s, 0.0, fused=True) == CHAIN_HBM_S


def test_whole_step_bound_pays_collectives_after_router() -> None:
    """Verify a slower collective raises the fused MoE floor one for one."""
    prof = advisor.resolve_model_profile("qwen3.5-397b")
    wl = advisor.WorkloadSpec(
        phase="verify", batch=1, spec_tokens=3, context=4096, tp=8, dcp=4, ep=8
    )
    chip = advisor.resolve_chip_spec("tpu7x")
    slow_chip = dataclasses.replace(
        chip,
        collective_latency_s=SLOW_COLLECTIVE_US * 1e-6,
        collective_latency_us_bracket=(SLOW_COLLECTIVE_US, SLOW_COLLECTIVE_US),
    )
    wholes = [
        next(
            s for s in advisor.advise(prof, wl, spec).scopes if s.scope == "whole_step"
        )
        for spec in (chip, slow_chip)
    ]
    fast, slow = wholes
    fast_core_us = fast.roof_floor_us - fast.fill_drain_us
    slow_core_us = slow.roof_floor_us - slow.fill_drain_us
    assert fast.n_splits == slow.n_splits == 1
    assert fast_core_us > fast.hbm_us
    assert slow.comm_us > fast.comm_us
    assert slow_core_us - fast_core_us == pytest.approx(
        slow.comm_us - fast.comm_us, abs=0.05
    )


def test_achievable_upper_band_never_inverts_below_achievable_or_floor() -> None:
    """Verify step_achievable_upper_us (Row 7) and CSA/HC bounds (Rows 8, 9)."""
    prof = advisor.resolve_model_profile("qwen3.5-397b")
    rep = advisor.advise(
        prof,
        advisor.WorkloadSpec(
            phase="verify", batch=16, spec_tokens=3, context=4096, tp=8, dcp=4, ep=8
        ),
        "tpu7x",
    )
    whole = next(s for s in rep.scopes if s.scope == "whole_step")
    assert whole.step_achievable_upper_us == pytest.approx(ACH_UPPER_VERIFY_US, abs=1.0)
    for sc in rep.scopes:
        assert sc.step_achievable_upper_us >= max(
            sc.step_achievable_us, sc.step_upper_us
        )

    p_ds4 = advisor.resolve_model_profile("deepseek-v4")
    c7x = advisor.resolve_chip_spec("tpu7x")
    wl_ds4 = advisor.WorkloadSpec(
        phase="decode", batch=128, context=160000, dp=16, tp=1, ep=16
    )
    ctx_ds4 = _build_scope_context(p_ds4, wl_ds4, c7x)
    assert ctx_ds4.kv_cache_read_bytes / 1e9 == pytest.approx(DS4_KV_READ_GB, abs=0.01)

    p_q4 = advisor.resolve_model_profile("qwen4-exp")
    parts_q4 = advisor.compute_per_token_vmem_bytes(
        p_q4, advisor.WorkloadSpec(phase="decode", batch=16, tp=8)
    )
    assert parts_q4["residual_buffer_bytes"] == QWEN4_RESIDUAL_BYTES


def test_gain_split_sums_to_total_and_fact_part_ignores_priors() -> None:
    """Verify gain_fact_us and gain_assumed_us split and ASSUMED tag (Row 5)."""
    prof = advisor.resolve_model_profile("qwen3.5-397b")
    rep = advisor.advise(
        prof,
        advisor.WorkloadSpec(
            phase="decode",
            batch=16,
            tp=8,
            dcp=4,
            ep=8,
        ),
        "tpu7x",
    )
    assert float(rep.metrics["gain_fact_us"]["value"]) == pytest.approx(
        GAIN_FACT_US, abs=0.05
    )
    assert float(rep.metrics["gain_assumed_us"]["value"]) == pytest.approx(
        GAIN_ASSUMED_US, abs=0.05
    )
    assert rep.metrics["compiler_proof_gain_us"]["tag"] == "ASSUMED"


def test_invalid_sharding_returns_invalid_config() -> None:
    """Verify non-power-of-two TP produces invalid_config verdict and per_op."""
    prof = advisor.resolve_model_profile("qwen3.5-397b")
    rep = advisor.advise(
        prof,
        advisor.WorkloadSpec(phase="decode", batch=16, tp=3, ep=1, objective="both"),
        "tpu7x",
    )
    assert rep.verdict.startswith("invalid_config(")
    assert rep.fastest_verdict == rep.verdict
    assert rep.recommended_scope == "per_op"
    assert rep.build_megakernel is False


def test_bundled_priors_keep_prior_only_confidence() -> None:
    """Verify architectural_prior bundled measurements keep prior_only confidence."""
    prof = advisor.resolve_model_profile("qwen3.5-397b")
    meas: str | dict[str, object] = (
        "bundled"
        if advisor.chips.public_calibration_path("tpu7x") is not None
        else {
            "schema": "advisor.measurements.v1",
            "chip": "tpu7x",
            "source": {"kind": "architectural_prior"},
        }
    )
    rep = advisor.advise(
        prof,
        advisor.WorkloadSpec(phase="decode", batch=16, tp=8, ep=8),
        "tpu7x",
        measurements=meas,
    )
    assert rep.confidence == "prior_only"


def _verify_whole_step(
    batch: int, max_rows_per_call: int, op_overhead_us: float = 0.0
) -> tuple[advisor.AdvisorReport, advisor.ScopeEstimate]:
    """Advise a Qwen3.5-397B verify step; return the report and its whole_step."""
    rep = advisor.advise(
        advisor.resolve_model_profile("qwen3.5-397b"),
        advisor.WorkloadSpec(
            phase="verify",
            batch=batch,
            spec_tokens=ROWS_PER_VERIFY_REQUEST - 1,
            context=4096,
            tp=8,
            dcp=4,
            ep=8,
            max_rows_per_call=max_rows_per_call,
            op_overhead_us=op_overhead_us,
        ),
        "tpu7x",
    )
    return rep, next(s for s in rep.scopes if s.scope == "whole_step")


def test_max_rows_per_call_clamps_single_call_rows() -> None:
    """Verify a declared row limit caps rows per call without changing the price."""
    _, ideal = _verify_whole_step(SINGLE_PASS_BATCH, 0)
    _, capped = _verify_whole_step(SINGLE_PASS_BATCH, ROW_LIMIT)
    assert ideal.t_vmem_max > ROW_LIMIT
    assert capped.t_vmem_max == ROW_LIMIT
    assert capped.n_splits == 1
    assert capped.step_bound_us == ideal.step_bound_us


def test_rows_above_single_call_limit_priced_as_microbatches() -> None:
    """Verify rows past the limit cost a full call plus the remainder call."""
    full_requests = ROW_LIMIT // ROWS_PER_VERIFY_REQUEST
    _, split = _verify_whole_step(SPLIT_BATCH, ROW_LIMIT)
    _, full = _verify_whole_step(full_requests, ROW_LIMIT)
    _, rest = _verify_whole_step(SPLIT_BATCH - full_requests, ROW_LIMIT)
    _, tiled = _verify_whole_step(SPLIT_BATCH, 0)
    assert split.n_splits == full.n_splits + rest.n_splits
    assert split.ops_per_step == full.ops_per_step + rest.ops_per_step
    assert split.step_bound_us == pytest.approx(
        full.step_bound_us + rest.step_bound_us, abs=0.02
    )
    assert split.step_achievable_us == pytest.approx(
        full.step_achievable_us + rest.step_achievable_us, abs=0.02
    )
    assert split.w_step_gb == pytest.approx(full.w_step_gb + rest.w_step_gb, abs=1e-3)
    assert split.step_bound_us > tiled.step_bound_us


def test_max_rows_per_call_splits_whole_requests() -> None:
    """Verify a limit between request boundaries rounds down to whole requests."""
    _, exact = _verify_whole_step(SPLIT_BATCH, ROW_LIMIT)
    _, ragged = _verify_whole_step(SPLIT_BATCH, ROW_LIMIT + 1)
    assert ragged.t_vmem_max == ROW_LIMIT
    assert ragged == exact


def test_max_rows_per_call_below_one_request_is_config_error() -> None:
    """Verify a limit smaller than one verify request is rejected."""
    with pytest.raises(MegakernelConfigError, match="below one verify request"):
        _verify_whole_step(SINGLE_PASS_BATCH, ROWS_PER_VERIFY_REQUEST - 1)


def test_microbatched_whole_step_keeps_support_status() -> None:
    """Verify a split whole_step stays supported and is flagged, not blocked."""
    rep, split = _verify_whole_step(SPLIT_BATCH, ROW_LIMIT)
    assert split.support == "within_core_limits"
    assert split.blockers == ()
    assert "microbatch_rerun" in rep.flags
    assert "multi_call_unmodeled" not in rep.flags
    rep_one, _ = _verify_whole_step(SINGLE_PASS_BATCH, ROW_LIMIT)
    assert "microbatch_rerun" not in rep_one.flags


def test_microbatched_step_intensity_counts_weights_once() -> None:
    """Verify two equal full calls keep the arithmetic intensity of one call."""
    full_requests = ROW_LIMIT // ROWS_PER_VERIFY_REQUEST
    two_calls = 2
    one, _ = _verify_whole_step(full_requests, ROW_LIMIT, WHOLE_STEP_FAVORING_GAP_US)
    two, split = _verify_whole_step(
        two_calls * full_requests, ROW_LIMIT, WHOLE_STEP_FAVORING_GAP_US
    )
    assert one.fastest_scope == two.fastest_scope == "whole_step"
    assert split.n_splits == two_calls
    # Doubling the batch doubles both the FLOPs and the bytes of the step.
    assert two.step_arithmetic_intensity == pytest.approx(
        one.step_arithmetic_intensity, abs=0.02
    )


def test_in_call_passes_stay_flagged_under_row_limit() -> None:
    """Verify a call larger than the VMEM ceiling keeps the unmodeled-pass flag."""
    two_calls = 2
    limit = two_calls * QWEN35_IDEAL_VMEM_ROWS
    requests = (limit + QWEN35_IDEAL_VMEM_ROWS) // ROWS_PER_VERIFY_REQUEST
    rep, split = _verify_whole_step(requests, limit, WHOLE_STEP_FAVORING_GAP_US)
    assert rep.fastest_scope == "whole_step"
    # One full call of two ideal-ceiling passes, then a remainder call of one.
    assert split.n_splits == two_calls + 1
    assert {"microbatch_rerun", "multi_call_unmodeled"} <= set(rep.flags)
    single_pass_calls, _ = _verify_whole_step(
        two_calls * (ROW_LIMIT // ROWS_PER_VERIFY_REQUEST),
        ROW_LIMIT,
        WHOLE_STEP_FAVORING_GAP_US,
    )
    assert single_pass_calls.fastest_scope == "whole_step"
    assert "multi_call_unmodeled" not in single_pass_calls.flags


def test_microbatch_zero_map_is_sum_of_calls() -> None:
    """Verify overridden latency priors reach every call of a split whole_step."""
    prof = advisor.resolve_model_profile("qwen3.5-397b")
    chip = advisor.resolve_chip_spec("tpu7x")

    def zero_latency_bound(batch: int) -> float:
        """Return the whole_step bound with every latency prior set to zero."""
        wl = advisor.WorkloadSpec(
            phase="verify",
            batch=batch,
            spec_tokens=ROWS_PER_VERIFY_REQUEST - 1,
            context=4096,
            tp=8,
            dcp=4,
            ep=8,
            max_rows_per_call=ROW_LIMIT,
        )
        ctx = dataclasses.replace(
            _build_scope_context(prof, wl, chip),
            lat_coll=0.0,
            lat_coll_upper=0.0,
            ar_base_lat_s=0.0,
            ar_inter_latency_s=0.0,
            launch_gap_upper_us=0.0,
        )
        estimates, _ = _evaluate_scopes(ctx, prof, wl)
        return next(e for e in estimates if e.scope == "whole_step").step_bound_us

    full_requests = ROW_LIMIT // ROWS_PER_VERIFY_REQUEST
    split_us = zero_latency_bound(SPLIT_BATCH)
    assert split_us == pytest.approx(
        zero_latency_bound(full_requests)
        + zero_latency_bound(SPLIT_BATCH - full_requests),
        abs=0.02,
    )
    _, priced = _verify_whole_step(SPLIT_BATCH, ROW_LIMIT)
    assert split_us < priced.step_bound_us


def test_declared_preset_status_reachable(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify a deployments.json megakernel declaration yields declared support."""
    dep = advisor.DEPLOYMENT_PRESETS["qwen3.5-397b"]
    declaration = {
        "source": "unit test",
        "scopes": ["whole_step"],
        "max_rows": ROW_LIMIT,
        "ranks": dep.decode.tp,
    }
    declared = dataclasses.replace(
        dep, decode=dataclasses.replace(dep.decode, megakernel=declaration)
    )
    monkeypatch.setattr(support, "DEPLOYMENT_PRESETS", {"qwen3.5-397b": declared})
    _, one_call = _verify_whole_step(SINGLE_PASS_BATCH, 0)
    assert one_call.support == "declared"
    assert one_call.t_vmem_max == ROW_LIMIT
    rep, split = _verify_whole_step(SPLIT_BATCH, 0)
    assert split.support == "declared"
    assert split.n_splits > 1
    assert "microbatch_rerun" in rep.flags


def test_preset_row_limit_below_one_request_is_config_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify a preset row limit below one verify request is rejected, not ignored."""
    dep = advisor.DEPLOYMENT_PRESETS["qwen3.5-397b"]
    declaration = {
        "source": "unit test",
        "scopes": ["whole_step"],
        "max_rows": ROWS_PER_VERIFY_REQUEST - 1,
        "ranks": dep.decode.tp,
    }
    declared = dataclasses.replace(
        dep, decode=dataclasses.replace(dep.decode, megakernel=declaration)
    )
    monkeypatch.setattr(support, "DEPLOYMENT_PRESETS", {"qwen3.5-397b": declared})
    with pytest.raises(MegakernelConfigError, match="below one verify request"):
        _verify_whole_step(SINGLE_PASS_BATCH, 0)


def test_validate_workload_names_row_limit_below_one_request() -> None:
    """Verify the non-raising validator reports a row limit below one request."""
    workload = advisor.WorkloadSpec(
        phase="verify",
        batch=SINGLE_PASS_BATCH,
        spec_tokens=ROWS_PER_VERIFY_REQUEST - 1,
        context=4096,
        tp=8,
        dcp=4,
        ep=8,
        max_rows_per_call=ROWS_PER_VERIFY_REQUEST - 1,
    )
    reasons = advisor.validate_workload("qwen3.5-397b", "tpu7x", workload)
    assert reasons == ("row_limit_below_request",)
    one_request = dataclasses.replace(
        workload, max_rows_per_call=ROWS_PER_VERIFY_REQUEST
    )
    assert advisor.validate_workload("qwen3.5-397b", "tpu7x", one_request) == ()


def test_rows_above_default_buckets_flagged() -> None:
    """Verify CORE_LIMITS mirrors the core bucket defaults and flags larger calls."""
    core_dir = Path(advisor.__file__).resolve().parent.parent / "core"
    policy_text = (core_dir / "runtime" / "policy.py").read_text("utf-8")
    tops = [
        max(
            ast.literal_eval(
                policy_text.split(f"{name}: tuple[int, ...] = ", 1)[1].split("\n", 1)[0]
            )
        )
        for name in ("DEFAULT_BUCKETS", "DEFAULT_VERIFY_BUCKETS")
    ]
    lim = advisor.CORE_LIMITS
    assert [lim.default_decode_bucket_top, lim.default_verify_bucket_top] == tops

    prof = advisor.resolve_model_profile("qwen3.5-397b")
    for batch, flagged in (
        (lim.default_decode_bucket_top, False),
        (lim.default_decode_bucket_top + 1, True),
    ):
        rep = advisor.advise(
            prof, advisor.WorkloadSpec(phase="decode", batch=batch, tp=8, ep=8), "tpu7x"
        )
        assert ("above_default_buckets" in rep.flags) is flagged
    verify_top_requests = lim.default_verify_bucket_top // ROWS_PER_VERIFY_REQUEST
    rep_at_top, _ = _verify_whole_step(verify_top_requests, 0)
    rep_above, _ = _verify_whole_step(verify_top_requests + 1, 0)
    assert "above_default_buckets" not in rep_at_top.flags
    assert "above_default_buckets" in rep_above.flags


def test_ffn_scratch_includes_fp32_bytes() -> None:
    """Verify FFN scratch uses FP32 bytes so Qwen3.5-397B VMEM cap is 448 rows."""
    prof = advisor.resolve_model_profile("qwen3.5-397b")
    wl = advisor.WorkloadSpec(phase="decode", batch=16, tp=8, ep=8)
    chip = advisor.resolve_chip_spec("tpu7x")
    ctx = _build_scope_context(prof, wl, chip)
    assert ctx.ffn_scratch == (
        constants.DOUBLE_BUFFER_SLOTS
        * constants.ATTN_TILE_TOKENS
        * prof.moe_intermediate_size
        * constants.FP32_BYTES
    )
    rep = advisor.advise(prof, wl, chip)
    whole = next(s for s in rep.scopes if s.scope == "whole_step")
    assert whole.t_vmem_max == QWEN35_IDEAL_VMEM_ROWS


def test_physical_config_rejects_undersized_slice_and_invalid_combos() -> None:
    """Verify undersized slices are flagged and unphysical combos raise cleanly."""
    undersized = advisor.validate_workload(
        "qwen3.5-397b",
        "tpu7x",
        advisor.WorkloadSpec(phase="decode", batch=4, tp=2, ep=2, dp=1),
    )
    assert "slice_topology_unsupported" in undersized
    valid_slice = advisor.validate_workload(
        "qwen3.5-397b",
        "tpu7x",
        advisor.WorkloadSpec(phase="decode", batch=4, tp=2, ep=2, dp=4),
    )
    assert "slice_topology_unsupported" not in valid_slice

    prof = advisor.resolve_model_profile("qwen3.5-397b")
    invalid_workloads = (
        advisor.WorkloadSpec(phase="decode", batch=2, tp=2, dp=4, ep=8),
        advisor.WorkloadSpec(phase="prefill", batch=1, isl=1024, tp=2, pp=2, pcp=2),
        advisor.WorkloadSpec(phase="verify", batch=4, tp=8, ep=8, spec_tokens=0),
        advisor.WorkloadSpec(
            phase="decode", batch=4, tp=8, ep=8, hbm_utilization_cap=1.5
        ),
        advisor.WorkloadSpec(phase="decode", batch=4, tp=8, ep=8, dma_buffers=4),
        advisor.WorkloadSpec(
            phase="decode", batch=4, tp=8, ep=8, vmem_budget_mib=TPU7X_VMEM_MIB + 1.0
        ),
    )
    for bad_wl in invalid_workloads:
        with pytest.raises(MegakernelConfigError):
            advisor.advise(prof, bad_wl, "tpu7x")

    kvless = dataclasses.replace(prof, num_kv_heads=0)
    kv_reasons = advisor.validate_workload(
        kvless,
        "tpu7x",
        advisor.WorkloadSpec(phase="decode", batch=8, tp=8, kv_split=2),
    )
    assert "kv_heads_not_divisible_by_tp" in kv_reasons


def test_advisor_core_limits_match_core_package() -> None:
    """Verify spec_bridge AST drift check reports zero differences vs core/."""
    assert spec_bridge.verify_core_limits_sync() == []


def test_per_op_vmem_overflow_marks_infeasible() -> None:
    """Verify static VMEM overflow on per_op marks the baseline infeasible."""
    prof = advisor.resolve_model_profile("qwen3.5-397b")
    rep = advisor.advise(
        prof,
        advisor.WorkloadSpec(
            phase="decode",
            tp=8,
            ep=8,
            batch=4,
            vmem_budget_mib=float(constants.MIN_VMEM_MIB),
        ),
        "tpu7x",
    )
    per_op = next(s for s in rep.scopes if s.scope == "per_op")
    assert (per_op.feasible, per_op.t_vmem_max, per_op.reason) == (
        False,
        0,
        "per_op_vmem_overflow",
    )


def test_recommended_scope_metrics_and_hbm_oom_zero_tps() -> None:
    """Verify summary metrics follow recommended scope and zero out on HBM OOM."""
    prof_bad = advisor.resolve_model_profile("qwen3.5-0.8b")
    rep_bad = advisor.advise(
        prof_bad, advisor.WorkloadSpec(phase="decode", batch=16, tp=2), "tpu7x"
    )
    assert rep_bad.verdict == "invalid_config(slice_topology_unsupported)"
    per_op_bad = next(s for s in rep_bad.scopes if s.scope == "per_op")
    assert float(rep_bad.metrics["step_floor_us"]["value"]) == pytest.approx(
        per_op_bad.step_bound_us, abs=0.01
    )
    assert float(rep_bad.metrics["throughput_tok_s_per_chip"]["value"]) == 0.0
    cmp_res = advisor.verdict.compare_parallelism_candidates(
        "qwen3.5-0.8b",
        (advisor.ParallelLayout(tp=2, ep=1), advisor.ParallelLayout(tp=8, ep=1)),
        chip="tpu7x",
    )
    assert cmp_res["best_step_layout"] == "TP8"
    assert cmp_res["best_throughput_layout"] == "TP8"
    assert cmp_res["verdict"] == "TP8"

    pk3 = advisor.resolve_model_profile("kimi-k3")
    rep_k3 = advisor.advise(
        pk3,
        advisor.WorkloadSpec(phase="decode", batch=4, tp=32, ep=32),
        "tpu7x",
    )
    assert rep_k3.fastest_scope == "whole_step"
    assert rep_k3.recommended_scope == "per_op"
    per_op_k3 = next(s for s in rep_k3.scopes if s.scope == "per_op")
    assert float(rep_k3.metrics["step_floor_us"]["value"]) == pytest.approx(
        per_op_k3.step_bound_us, abs=0.01
    )
    assert float(rep_k3.metrics["throughput_tok_s_per_chip"]["value"]) > 0.0

    p397 = advisor.resolve_model_profile("qwen3.5-397b")
    rep_oom = advisor.advise(
        p397,
        advisor.WorkloadSpec(phase="decode", batch=4, tp=2, ep=2, dp=2),
        "tpu7x",
    )
    assert rep_oom.hbm_fits is False
    assert float(rep_oom.metrics["throughput_tok_s_per_chip"]["value"]) == 0.0
