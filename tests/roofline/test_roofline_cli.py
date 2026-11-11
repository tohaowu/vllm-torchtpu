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
"""CLI unit tests for plan, explain, solve, emit, and ceiling subcommands."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import TYPE_CHECKING

import pytest

from vllm_torchtpu.roofline import (
    CHIPS,
    Dtypes,
    Parallel,
    ServingRequest,
    cli,
    from_hf_config,
    plan,
)

if TYPE_CHECKING:
    from pathlib import Path

QWEN35_CFG: dict[str, object] = {
    "text_config": {
        "hidden_size": 4096,
        "num_hidden_layers": 60,
        "layer_types": (["linear_attention"] * 3 + ["full_attention"]) * 15,
        "num_attention_heads": 32,
        "num_key_value_heads": 2,
        "head_dim": 256,
        "attn_output_gate": True,
        "linear_num_key_heads": 16,
        "linear_num_value_heads": 64,
        "linear_key_head_dim": 128,
        "linear_value_head_dim": 128,
        "linear_conv_kernel_dim": 4,
        "num_experts": 512,
        "num_experts_per_tok": 10,
        "moe_intermediate_size": 1024,
        "shared_expert_intermediate_size": 1024,
        "vocab_size": 248320,
        "mtp_num_hidden_layers": 1,
        "dtype": "bfloat16",
    }
}


def _iter_subcommand_options(
    parser: argparse.ArgumentParser,
) -> list[tuple[str, tuple[str, ...], str | None]]:
    """Collect ``(subcommand, option_strings, help)`` across root and subparsers."""
    items: list[tuple[str, tuple[str, ...], str | None]] = []

    def _walk(cmd_name: str, subparser: argparse.ArgumentParser) -> None:
        """Walk parser actions via stdlib attributes lacking public accessors."""
        actions: Sequence[argparse.Action] = getattr(subparser, "_actions", ())
        for action in actions:
            if action.option_strings:
                items.append((cmd_name, tuple(action.option_strings), action.help))
            choices = getattr(action, "choices", None)
            if isinstance(choices, Mapping):
                for sub_name, child in choices.items():
                    if isinstance(child, argparse.ArgumentParser):
                        _walk(str(sub_name), child)

    _walk("root", parser)
    return items


def test_cli_plan_json_schema(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Verify JSON output schema and hard floor for plan command."""
    expected_concurrency_rows = 2
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(json.dumps(QWEN35_CFG))
    rc = cli.main(
        [
            "plan",
            "--config",
            str(cfg_path),
            "--tp",
            "8",
            "--context",
            "1024",
            "--concurrency",
            "1,32",
            "--json",
        ]
    )
    assert rc == 0
    rows = json.loads(capsys.readouterr().out)
    assert len(rows) == expected_concurrency_rows
    expected_keys = {
        "c",
        "rows",
        "binding",
        "hard_floor_ms",
        "serial_estimate_ms",
        "tpot_floor_ms",
        "interactivity_ceil_tok_s",
        "output_tput_ceil_per_chip",
        "flag",
    }
    assert set(rows[0].keys()) == expected_keys
    assert rows[0]["c"] == 1
    assert rows[0]["hard_floor_ms"] <= rows[1]["hard_floor_ms"]


def test_cli_explain_spec_window_includes_draft_modules(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Verify explain command includes speculative draft modules."""
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(json.dumps(QWEN35_CFG))
    rc = cli.main(
        [
            "explain",
            "--config",
            str(cfg_path),
            "--tp",
            "8",
            "--context",
            "90036",
            "--spec-window",
            "3",
            "--json",
        ]
    )
    assert rc == 0
    mods = json.loads(capsys.readouterr().out)
    names = [m["name"] for m in mods]
    assert "gqa" in names
    assert "draft_gqa" in names
    assert "draft_fc" in names


def test_cli_rejects_non_positive_tp(tmp_path: Path) -> None:
    """Verify explain command rejects non-positive tensor parallel values."""
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(json.dumps(QWEN35_CFG))
    with pytest.raises(ValueError, match="--tp must be >= 1"):
        cli.main(["explain", "--config", str(cfg_path), "--tp", "0"])


def test_cli_plan_axes_flags(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Verify explain command with MoE expert parallel and FP8 KV cache."""
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(json.dumps(QWEN35_CFG))
    rc = cli.main(
        [
            "explain",
            "--config",
            str(cfg_path),
            "--tp",
            "8",
            "--context",
            "90036",
            "--moe",
            "ep",
            "--kv-dtype",
            "fp8",
            "--json",
        ]
    )
    assert rc == 0
    mods = json.loads(capsys.readouterr().out)
    total_phys = sum(m["hbm_bytes"] for m in mods)
    assert total_phys == pytest.approx(5_168_280_024.77, abs=1.0)
    gqa_mod = next(m for m in mods if m["name"] == "gqa")
    assert gqa_mod.get("hbm_bytes_min") is not None
    logical_total = total_phys - (gqa_mod["hbm_bytes"] - gqa_mod["hbm_bytes_min"])
    assert logical_total == pytest.approx(4_476_803_544.77, abs=1.0)


def test_plan_requires_acceptance_length_for_spec_window(tmp_path: Path) -> None:
    """Verify acceptance length requirement when speculative window is positive."""
    geom = from_hf_config(QWEN35_CFG)
    with pytest.raises(
        ValueError, match="acceptance_length is required when spec_window > 0"
    ):
        plan(
            geom,
            CHIPS["tpu7x"],
            Parallel(tp=8),
            Dtypes(),
            ServingRequest(context=1024, concurrencies=[1], spec_window=3),
        )

    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(json.dumps(QWEN35_CFG))
    with pytest.raises(SystemExit) as excinfo:
        cli.main(
            [
                "plan",
                "--config",
                str(cfg_path),
                "--tp",
                "8",
                "--spec-window",
                "3",
                "--context",
                "1024",
            ]
        )
    assert excinfo.value.code != 0


def test_cli_from_serve_and_bench_cmd_roundtrip(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Verify CLI solve roundtrip from serve and bench command strings."""
    expected_isl = 4096
    expected_osl = 512
    max_chips = 4
    bf16_cfg_path = tmp_path / "synth_bf16_config.json"
    bf16_cfg_path.write_text(json.dumps(QWEN35_CFG))
    cfg_path = tmp_path / "synth_config.json"
    cfg_path.write_text(
        json.dumps({**QWEN35_CFG, "quantization_config": {"quant_method": "fp8"}})
    )
    html_path = tmp_path / "roundtrip.html"
    rc = cli.main(
        [
            "solve",
            "--from-serve-cmd",
            (
                f"vllm serve {bf16_cfg_path} --quantization fp8 "
                "--kv-cache-dtype fp8 --tensor-parallel-size 8 "
                "--data-parallel-size 1 --enable-expert-parallel "
                "--gpu-memory-utilization 0.90 --kv-offloading-size 32"
            ),
            "--from-bench-cmd",
            (
                "vllm bench serve --dataset-name random "
                "--random-prefix-len 3072 --random-input-len 1024 "
                "--random-output-len 512 --max-concurrency 16"
            ),
            "--dcn-gbit-per-s",
            "200",
            "--html",
            str(html_path),
            "--json",
        ]
    )
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["scenario"]["isl"] == expected_isl
    assert payload["scenario"]["osl"] == expected_osl
    assert payload["scenario"]["prefix_hit_rate"] == pytest.approx(0.75)
    assert payload["scenario"]["concurrencies"] == [16]
    assert payload["scenario"]["kv_offload_gb"] == pytest.approx(32.0)
    assert payload["gpu_memory_utilization"] == pytest.approx(0.90)
    assert payload["chip_spec"]["dcn_bw"] == pytest.approx(25.0e9)
    # TP=8, DP=1 on tpu7x (2 TensorCores/chip) budgets 4 physical chips, not 8.
    assert payload["recommended"]["total_chips"] <= max_chips
    assert payload["winner_prefix_hit_range"] is not None
    assert "sweep_candidates" not in payload
    assert "--kv-offloading-size 32" in payload["top_k"][0]["launch"]["commands"][0]
    html_text = html_path.read_text()
    assert "sweep_candidates" not in html_text
    assert "Prefix-Hit Regime" in html_text
    assert ("crossover=" in html_text) or ("stable" in html_text)

    # Sonnet dataset --sonnet-input-len already includes --sonnet-prefix-len, and
    # direct --shared-prefix-len / --kv-offload-gb flags work standalone.
    stable_html = tmp_path / "sonnet_direct.html"
    sonnet_isl = 550
    rc_sonnet = cli.main(
        [
            "solve",
            "--config",
            str(cfg_path),
            "--from-bench-cmd",
            (
                "vllm bench serve --dataset-name sonnet "
                "--sonnet-input-len 550 --sonnet-prefix-len 200 "
                "--sonnet-output-len 150 --max-concurrency 4"
            ),
            "--kv-offload-gb",
            "16",
            "--max-chips",
            "4",
            "--mode",
            "mixed",
            "--html",
            str(stable_html),
            "--json",
        ]
    )
    assert rc_sonnet == 0
    sonnet_payload = json.loads(capsys.readouterr().out)
    assert sonnet_payload["scenario"]["isl"] == sonnet_isl
    assert sonnet_payload["scenario"]["prefix_hit_rate"] == pytest.approx(200.0 / 550.0)
    assert sonnet_payload["scenario"]["kv_offload_gb"] == pytest.approx(16.0)
    stable_text = stable_html.read_text()
    assert ("crossover=" in stable_text) or ("stable" in stable_text)

    rc_direct = cli.main(
        [
            "solve",
            "--config",
            str(cfg_path),
            "--isl",
            "4096",
            "--shared-prefix-len",
            "1024",
            "--osl",
            "256",
            "--concurrency",
            "8",
            "--max-chips",
            "4",
            "--json",
        ]
    )
    assert rc_direct == 0
    direct_payload = json.loads(capsys.readouterr().out)
    assert direct_payload["scenario"]["isl"] == expected_isl
    assert direct_payload["scenario"]["prefix_hit_rate"] == pytest.approx(0.25)

    # Generic --input-len under default random dataset adds --random-prefix-len.
    rc_gen_rand = cli.main(
        [
            "solve",
            "--config",
            str(cfg_path),
            "--from-bench-cmd",
            (
                "vllm bench serve --random-prefix-len 3072 "
                "--input-len 1024 --output-len 256 --max-concurrency 8"
            ),
            "--max-chips",
            "4",
            "--json",
        ]
    )
    assert rc_gen_rand == 0
    gen_rand_payload = json.loads(capsys.readouterr().out)
    assert gen_rand_payload["scenario"]["isl"] == expected_isl
    assert gen_rand_payload["scenario"]["prefix_hit_rate"] == pytest.approx(0.75)


def test_cli_module_dunder_main_help(capsys: pytest.CaptureFixture[str]) -> None:
    """Ensure CLI help output is emitted and exits cleanly."""
    with pytest.raises(SystemExit) as excinfo:
        cli.main(["--help"])
    out = capsys.readouterr().out
    assert excinfo.value.code == 0
    assert "usage:" in out.lower()


def test_cli_forwards_spec_tokens_and_token_caps(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Verify CLI forwards spec_tokens and token caps from serve cmd."""
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(json.dumps(QWEN35_CFG))
    expected_spec_tokens = 3
    expected_prefill_chunk = 2048
    cmd = [
        "solve",
        "--config",
        str(cfg_path),
        "--from-serve-cmd",
        (
            f"vllm serve {cfg_path} --tensor-parallel-size 8 --quantization fp8 "
            "--max-model-len 16384 --max-num-batched-tokens 2048 "
            '--speculative-config \'{"method":"mtp","num_speculative_tokens":3}\''
        ),
        "--json",
    ]
    rc = cli.main(cmd)
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["scenario"]["spec_tokens"] == expected_spec_tokens
    assert payload["scenario"]["prefill_chunk"] == expected_prefill_chunk
    launch_args = payload["top_k"][0]["launch"]["cli_args"]
    assert "--max-model-len" in launch_args
    assert "16384" in launch_args
    assert "--speculative-config" in launch_args


def test_every_cli_flag_has_help(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Verify every CLI flag has help text with unit/scope, aliases, and assumptions."""
    parser = cli.build_parser()
    sub_opts = _iter_subcommand_options(parser)
    assert len(sub_opts) > 0
    for sub_name, option_strings, raw_help in sub_opts:
        if not option_strings or "-h" in option_strings:
            continue
        assert raw_help is not None, f"{sub_name} {option_strings} missing help"
        help_str = raw_help.strip()
        assert bool(help_str), f"{sub_name} {option_strings} empty help"
        assert help_str != "==SUPPRESS==", (
            f"{sub_name} {option_strings} suppressed help"
        )
        assert "scope:" in help_str.lower(), (
            f"{sub_name} {option_strings} help missing scope: {help_str}"
        )
        assert "unit:" in help_str.lower(), (
            f"{sub_name} {option_strings} help missing unit: {help_str}"
        )

    rc_assump = cli.main(["--dcn-gbps", "200", "--assumptions", "--json"])
    assert rc_assump == 0
    assump_payload = json.loads(capsys.readouterr().out)
    assert "catalog" in assump_payload
    assert assump_payload["catalog"]["dcn_bw"]["value"] == pytest.approx(25.0e9)

    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(json.dumps(QWEN35_CFG))
    rc_alias = cli.main(
        [
            "explain",
            "--config",
            str(cfg_path),
            "--tp",
            "8",
            "--ici-bw-gbytes-per-s",
            "250",
            "--json",
        ]
    )
    assert rc_alias == 0
    _ = capsys.readouterr()


def test_solve_exits_nonzero_on_empty_frontier(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Verify solve exits 1 on empty frontier and 2 on invalid subhost or loader err."""
    expected_error_exit = 2
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(json.dumps(QWEN35_CFG))

    # Empty frontier (BF16 397B weights cannot fit on 2x2x1 = 4 chips) exits 1.
    rc_empty = cli.main(
        [
            "solve",
            "--config",
            str(cfg_path),
            "--max-chips",
            "4",
            "--example-scenario",
            "chat-8k1k",
        ]
    )
    captured_empty = capsys.readouterr()
    assert rc_empty == 1
    assert "WARNING:" in captured_empty.err

    # Subhost --max-chips 1 exits cleanly with single error line and code 2.
    rc_subhost = cli.main(
        [
            "solve",
            "--config",
            str(cfg_path),
            "--max-chips",
            "1",
            "--example-scenario",
            "chat-8k1k",
        ]
    )
    captured_subhost = capsys.readouterr()
    assert rc_subhost == expected_error_exit
    assert "error:" in captured_subhost.err

    # Missing config path exits 2 with a single error line.
    rc_err = cli.main(
        [
            "solve",
            "--config",
            str(tmp_path / "missing_config.json"),
            "--max-chips",
            "4",
            "--example-scenario",
            "chat-8k1k",
        ]
    )
    captured_err = capsys.readouterr()
    assert rc_err == expected_error_exit
    err_lines = captured_err.err.strip().splitlines()
    assert len(err_lines) == 1
    assert err_lines[0].startswith("error:")


def test_plan_row_serial_estimate_ms_and_ici_measure_flag() -> None:
    """Verify plan emits serial_estimate_ms and flags ICI-bound configs to MEASURE."""
    geom = from_hf_config(QWEN35_CFG)
    low_ici_chip = replace(
        CHIPS["tpu7x"],
        ici_link_bw=1.0e6,
        ici_bw=1.0e6,
        d2d_bw=2.0e6,
    )
    rows = plan(
        geom,
        low_ici_chip,
        Parallel(tp=8),
        Dtypes(),
        ServingRequest(context=128, concurrencies=[1]),
    )
    assert len(rows) == 1
    assert rows[0].binding == "ici"
    assert rows[0].flag == "MEASURE"
    assert rows[0].serial_estimate_ms >= rows[0].hard_floor_ms
