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
"""HuggingFace config.json resolution and quantization dtype extraction."""

from __future__ import annotations

import contextlib
import http.client
import json
import urllib.parse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from .. import envs
from .constants import (
    BASE_DTYPE_BYTES,
    DEFAULT_HTTP_PORT,
    DEFAULT_HTTPS_PORT,
    DEFAULT_SSM_STATE_BYTES,
    EIGHT_BITS,
    FOUR_BITS,
    FP8_BLOCK_SCALE_BYTES,
    FP8_BYTES,
    HTTP_OK,
    HTTP_TIMEOUT_S,
    MXFP4_BYTES,
    NVFP4_BYTES,
    SIXTEEN_BITS,
    WEIGHT_BLOCK_DIM_LEN,
    WEIGHT_ONLY_QUANT_METHODS,
)
from .modules import Dtypes


def _as_mapping(value: object) -> Mapping[str, Any]:
    """Return value as a mapping when compatible with HF config JSON, else empty."""
    return value if isinstance(value, Mapping) else {}


def _read_cached_hub_repo(repo_dir: Path) -> dict[str, Any] | None:
    """Read cached config.json from one HuggingFace Hub repo cache directory."""
    snap_dir = repo_dir / "snapshots"
    ref_main = repo_dir / "refs" / "main"
    if ref_main.is_file():
        rev = ref_main.read_text().strip()
        ref_cfg = snap_dir / rev / "config.json"
        if ref_cfg.is_file():
            return dict(json.loads(ref_cfg.read_text()))
    if snap_dir.is_dir():
        snaps = list(snap_dir.glob("*/config.json"))
        if snaps:
            latest = max(snaps, key=lambda item: item.stat().st_mtime)
            return dict(json.loads(latest.read_text()))
    return None


def _resolve_local_or_hub_url(raw_str: str) -> tuple[dict[str, Any] | None, str]:
    """Resolve local JSON or cached Hub snapshot, else return remote config.json URL."""
    p = Path(raw_str)
    if p.is_file():
        return dict(json.loads(p.read_text())), ""
    if p.is_dir() and (p / "config.json").is_file():
        return dict(json.loads((p / "config.json").read_text())), ""

    if p.is_absolute() or raw_str.endswith(".json") or raw_str.startswith((".", "/")):
        msg = f"Could not resolve HF config path: {raw_str}"
        raise FileNotFoundError(msg)

    if "/" in raw_str and not raw_str.startswith(("http://", "https://")):
        org_repo = raw_str.strip("/").replace("/", "--")
        roots: list[Path] = []
        if envs.HF_HUB_CACHE:
            roots.append(Path(envs.HF_HUB_CACHE))
        hf_home = Path(envs.HF_HOME or (Path.home() / ".cache" / "huggingface"))
        roots.extend((hf_home / "hub", hf_home))
        for root in roots:
            cached = _read_cached_hub_repo(root / f"models--{org_repo}")
            if cached is not None:
                return cached, ""
        return (
            None,
            f"https://huggingface.co/{raw_str.strip('/')}/resolve/main/config.json",
        )

    if raw_str.startswith(("http://", "https://")):
        return None, raw_str

    msg = f"Could not resolve HF config path: {raw_str}"
    raise FileNotFoundError(msg)


def _fetch_remote_hf_config(url: str, token: str) -> dict[str, Any]:
    """Fetch remote HF config via http.client to avoid urllib scheme auditing flags."""
    parsed_url = urllib.parse.urlparse(url)
    if parsed_url.scheme not in ("https", "http") or not parsed_url.hostname:
        msg = f"Unsupported URL scheme for HF config: {url!r}"
        raise ValueError(msg)

    headers = {"Accept": "application/json"}
    if (
        token
        and parsed_url.scheme == "https"
        and parsed_url.hostname == "huggingface.co"
    ):
        headers["Authorization"] = f"Bearer {token}"

    is_https = parsed_url.scheme == "https"
    port = parsed_url.port or (DEFAULT_HTTPS_PORT if is_https else DEFAULT_HTTP_PORT)
    conn_cls = http.client.HTTPSConnection if is_https else http.client.HTTPConnection
    path = parsed_url.path or "/"
    if parsed_url.query:
        path = f"{path}?{parsed_url.query}"
    try:
        with contextlib.closing(
            conn_cls(parsed_url.hostname, port=port, timeout=HTTP_TIMEOUT_S)
        ) as conn:
            conn.request("GET", path, headers=headers)
            resp = conn.getresponse()
            if resp.status != HTTP_OK:
                msg = f"HTTP {resp.status} fetching HF config from {url}"
                raise FileNotFoundError(msg)
            data = resp.read()
    except (TimeoutError, OSError) as exc:
        msg = f"Network timeout or error fetching HF config from {url}: {exc}"
        raise RuntimeError(msg) from exc
    return dict(json.loads(data.decode("utf-8")))


def load_hf_config(
    model_or_path: str | Path | Mapping[str, Any],
    *,
    offline: bool = False,
) -> dict[str, Any]:
    """Load HF config.json from mapping, local path, HF cache, or Hub URL."""
    if isinstance(model_or_path, Mapping):
        return dict(model_or_path)
    raw_str = str(model_or_path)
    cached_cfg, remote_url = _resolve_local_or_hub_url(raw_str)
    if cached_cfg is not None:
        return cached_cfg

    if offline or envs.HF_HUB_OFFLINE or envs.TRANSFORMERS_OFFLINE:
        msg = f"Offline mode is enabled and {raw_str!r} was not found in local cache"
        raise FileNotFoundError(msg)

    token = envs.HF_TOKEN or envs.HUGGING_FACE_HUB_TOKEN
    return _fetch_remote_hf_config(remote_url, token)


def _extract_base_and_ssm_dtypes(
    tc: Mapping[str, Any],
    cfg: Mapping[str, Any],
    *,
    has_qcfg: bool = False,
) -> tuple[float, float]:
    """Extract standard base model and SSM state byte sizes from config."""
    raw_dtype = (
        tc.get("dtype")
        or cfg.get("dtype")
        or tc.get("torch_dtype")
        or cfg.get("torch_dtype")
        or ("bfloat16" if has_qcfg else None)
    )
    if not raw_dtype:
        msg = "HF config missing required field: 'dtype' or 'torch_dtype'"
        raise ValueError(msg)
    base_name = str(raw_dtype).lower().removeprefix("torch.")
    if base_name not in BASE_DTYPE_BYTES:
        supp = sorted(BASE_DTYPE_BYTES)
        msg = f"Unknown base dtype: {base_name!r}; supported: {supp}"
        raise ValueError(msg)
    base_b = BASE_DTYPE_BYTES[base_name]

    ssm_dtype_raw = (
        tc.get("mamba_ssm_dtype")
        or cfg.get("mamba_ssm_dtype")
        or tc.get("ssm_dtype")
        or cfg.get("ssm_dtype")
    )
    if ssm_dtype_raw:
        ssm_name = str(ssm_dtype_raw).lower().removeprefix("torch.")
        if ssm_name not in BASE_DTYPE_BYTES:
            msg = f"Unknown mamba_ssm_dtype: {ssm_dtype_raw!r}"
            raise ValueError(msg)
        state_b = BASE_DTYPE_BYTES[ssm_name]
    else:
        state_b = DEFAULT_SSM_STATE_BYTES
    return base_b, state_b


@dataclass(frozen=True)
class _QuantParams:
    """Extracted quantization parameters."""

    q_method: str
    q_algo: str
    bits: int
    group_size: int | None
    scale_bytes: float | None
    act_bits: int | None


def _extract_config_groups_params(
    qcfg: Mapping[str, Any],
) -> tuple[int, int | None, float | None, int | None]:
    """Extract bits, group_size, scale_bytes, and input act bits from config_groups."""
    bits = 0
    group_size: int | None = None
    scale_bytes: float | None = None
    act_bits: int | None = None
    cfg_groups = _as_mapping(qcfg.get("config_groups"))
    for grp_raw in cfg_groups.values():
        grp = _as_mapping(grp_raw)
        w = _as_mapping(grp.get("weights"))
        if w:
            nb = int(w.get("num_bits", 0) or 0)
            if nb > 0 and bits == 0:
                bits = nb
            gs = w.get("group_size")
            if gs is not None and group_size is None:
                group_size = int(gs)
            sb = w.get("scale_bytes")
            if sb is not None and scale_bytes is None:
                scale_bytes = float(sb)
        ia = _as_mapping(grp.get("input_activations"))
        if ia and act_bits is None:
            ia_bits = int(ia.get("num_bits", 0) or 0)
            if ia_bits > 0:
                act_bits = ia_bits
    return bits, group_size, scale_bytes, act_bits


def _extract_scale_bytes(
    qcfg: Mapping[str, Any],
    fallback_sb: float | None,
) -> float | None:
    """Extract scale_bytes from scale_bytes or scale_dtype."""
    if qcfg.get("scale_bytes") is not None:
        return float(qcfg["scale_bytes"])
    if qcfg.get("scale_dtype") is not None:
        sd = str(qcfg["scale_dtype"]).lower().removeprefix("torch.")
        if sd not in BASE_DTYPE_BYTES:
            msg = f"Unknown 4-bit scale_dtype: {sd!r}"
            raise ValueError(msg)
        return BASE_DTYPE_BYTES[sd]
    return fallback_sb


def _extract_quant_params(
    qcfg: Mapping[str, Any],
) -> _QuantParams:
    """Extract quantization params from quantization_config."""
    q_method = str(qcfg.get("quant_method") or qcfg.get("format") or "").lower()
    q_algo = str(qcfg.get("quant_algo") or "").lower()
    bits = int(qcfg.get("bits", 0) or 0)
    group_size: int | None = None
    if "group_size" in qcfg and qcfg.get("group_size") is not None:
        group_size = int(qcfg["group_size"])

    g_bits, g_size, g_scale, act_bits = _extract_config_groups_params(qcfg)
    if bits == 0 and g_bits > 0:
        bits = g_bits
    if group_size is None and g_size is not None:
        group_size = g_size

    scale_bytes = _extract_scale_bytes(qcfg, g_scale)
    return _QuantParams(q_method, q_algo, bits, group_size, scale_bytes, act_bits)


def _extract_ignored_modules(
    qcfg: Mapping[str, Any],
) -> tuple[bool, bool, bool]:
    """Detect whether attention, lm_head, and aux layers skip quantization."""
    ign_raw = (
        qcfg.get("modules_to_not_convert")
        or qcfg.get("ignore")
        or qcfg.get("ignored_layers")
    )
    has_ign_list = ign_raw is not None and len(tuple(ign_raw)) > 0
    ignored = tuple(str(x).lower() for x in (ign_raw or ()))
    attn_ignored = any(
        "linear_attn" not in x
        and any(
            seg in ("self_attn", "attn", "q_proj", "k_proj", "v_proj", "o_proj")
            for seg in x.replace("/", ".").split(".")
        )
        for x in ignored
    )
    lm_ignored = any("lm_head" in x for x in ignored) or not bool(
        qcfg.get("quantize_lm_head", False)
    )
    aux_ignored = (
        any(k in x for x in ignored for k in ("mtp", "gate", "ba", "router"))
        if has_ign_list
        else not bool(qcfg.get("quantize_aux", False))
    )
    return attn_ignored, lm_ignored, aux_ignored


def _derive_nvfp4_dense(
    qcfg: Mapping[str, Any],
    fp8_b: float,
    base_b: float,
    *,
    attn_ignored: bool,
) -> float:
    """Derive dense weight bytes for NVFP4."""
    dense_override = str(qcfg.get("dense_dtype", "")).lower()
    if dense_override == "fp8":
        return fp8_b
    if attn_ignored:
        return base_b
    return NVFP4_BYTES


def _derive_four_bit_bytes(
    group_size: int,
    scale_bytes: float | None,
) -> float:
    """Derive weight bytes for 4-bit grouped quantization."""
    if scale_bytes is None or scale_bytes <= 0.0:
        msg = (
            "4-bit quantization config missing required field: "
            "'scale_bytes' or 'scale_dtype'"
        )
        raise ValueError(msg)
    return 0.5 + scale_bytes / float(group_size)


def _derive_quantized_weight_bytes(
    qp: _QuantParams,
    fp8_b: float,
    base_b: float,
    qcfg: Mapping[str, Any],
    *,
    attn_ignored: bool,
) -> tuple[float, float]:
    """Derive (w_dense, w_expert) byte footprints across quantization schemes."""
    if qp.bits == FOUR_BITS:
        if qp.group_size is None:
            msg = "4-bit quantization config missing required field: 'group_size'"
            raise ValueError(msg)
        if qp.group_size <= 0:
            msg = f"Invalid 4-bit group_size: {qp.group_size}"
            raise ValueError(msg)

    if "mxfp4" in qp.q_method or "mxfp4" in qp.q_algo:
        w_expert = MXFP4_BYTES
        w_dense = (
            base_b
            if (attn_ignored or not qcfg.get("quantize_attn", False))
            else MXFP4_BYTES
        )
    elif (
        "nvfp4" in qp.q_method
        or "nvfp4" in qp.q_algo
        or ("compressed" in qp.q_method and "nvfp4" in str(qcfg).lower())
        or ("modelopt" in qp.q_method and "nvfp4" in str(qcfg).lower())
    ):
        w_expert = NVFP4_BYTES
        w_dense = _derive_nvfp4_dense(qcfg, fp8_b, base_b, attn_ignored=attn_ignored)
    elif qp.bits == FOUR_BITS and qp.group_size is not None:
        w_expert = _derive_four_bit_bytes(qp.group_size, qp.scale_bytes)
        w_dense = base_b if attn_ignored else w_expert
    elif "fp8" in qp.q_method or qp.bits == EIGHT_BITS:
        w_expert = fp8_b
        w_dense = base_b if attn_ignored else fp8_b
    else:
        w_expert = base_b
        w_dense = base_b
    return w_dense, w_expert


def _derive_activation_mode(
    qp: _QuantParams, qcfg: Mapping[str, Any]
) -> Literal["a8", "a16"]:
    """Derive compute activation peak ('a8' vs 'a16') from quantization_config."""
    if qp.act_bits == EIGHT_BITS:
        return "a8"
    if qp.act_bits == SIXTEEN_BITS or qp.q_method in WEIGHT_ONLY_QUANT_METHODS:
        return "a16"
    if qcfg.get("weight_only") or ("config_groups" in qcfg and qp.act_bits is None):
        return "a16"
    qcfg_lower = str(qcfg).lower()
    if any(tok in qcfg_lower for tok in ("w4a16", "w8a16")):
        return "a16"
    if (
        "fp8" in qp.q_method
        or "fp8" in qp.q_algo
        or any(tok in qcfg_lower for tok in ("nvfp4", "mxfp4", "w4a8", "w8a8", "a8"))
    ):
        return "a8"
    return "a16"


def dtypes_from_hf_config(cfg: Mapping[str, Any]) -> Dtypes:
    """Derive Dtypes from torch_dtype, quantization_config, and ignored module lists."""
    tc = _as_mapping(cfg.get("text_config")) or cfg
    qcfg = _as_mapping(tc.get("quantization_config") or cfg.get("quantization_config"))
    has_qcfg = len(qcfg) > 0

    base_b, state_b = _extract_base_and_ssm_dtypes(tc, cfg, has_qcfg=has_qcfg)
    if not has_qcfg:
        act_mode = "a8" if base_b <= FP8_BYTES else "a16"
        return Dtypes(
            w_dense=base_b,
            w_expert=base_b,
            w_aux=base_b,
            kv=base_b,
            state=state_b,
            conv=base_b,
            router=base_b,
            lm_head=base_b,
            act=act_mode,
        )

    qp = _extract_quant_params(qcfg)
    attn_ignored, lm_ignored, aux_ignored = _extract_ignored_modules(qcfg)

    block_sz = qcfg.get("weight_block_size")
    fp8_b = FP8_BYTES
    if (
        isinstance(block_sz, Sequence)
        and not isinstance(block_sz, (str, bytes))
        and len(block_sz) == WEIGHT_BLOCK_DIM_LEN
    ):
        b0, b1 = int(block_sz[0]), int(block_sz[1])
        if b0 > 0 and b1 > 0:
            fp8_b = FP8_BYTES + FP8_BLOCK_SCALE_BYTES / float(b0 * b1)

    w_dense, w_expert = _derive_quantized_weight_bytes(
        qp,
        fp8_b,
        base_b,
        qcfg,
        attn_ignored=attn_ignored,
    )

    kv_scheme = str(
        qcfg.get("kv_cache_scheme") or qcfg.get("kv_cache_dtype") or ""
    ).lower()
    kv_b = FP8_BYTES if "fp8" in kv_scheme or "8" in kv_scheme else base_b

    quant_conv = bool(qcfg.get("quantize_conv", False))
    conv_b = w_dense if quant_conv else base_b

    quant_router = bool(qcfg.get("quantize_router", False)) or (
        bool(qcfg.get("quantize_aux", False)) and not aux_ignored
    )
    router_b = w_dense if quant_router else base_b

    act_mode = _derive_activation_mode(qp, qcfg)
    return Dtypes(
        w_dense=w_dense,
        w_expert=w_expert,
        w_aux=base_b if aux_ignored else w_dense,
        kv=kv_b,
        state=state_b,
        conv=conv_b,
        router=router_b,
        lm_head=base_b if lm_ignored else w_dense,
        act=act_mode,
    )
