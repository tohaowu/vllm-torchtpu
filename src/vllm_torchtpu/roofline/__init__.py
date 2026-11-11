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
"""Analytical TPU serving roofline, step floors, and CLI entrypoint."""

from __future__ import annotations

from .chips import CHIPS, TOPOLOGIES, ChipOverrides, ChipSpec, SliceTopology
from .collectives import Collective, CollectiveConfig, torus_collective_s
from .geometry import ModelGeometry, from_hf_config
from .hf_loader import dtypes_from_hf_config, load_hf_config
from .modules import Dtypes, ModuleCost, Parallel, StepShape, pass_costs
from .serving import ConcurrencyRow, ServingRequest, plan
from .step import FloorOptions, SpecShape, StepFloor, spec_cycle_floor, step_floor

__all__ = [
    "CHIPS",
    "TOPOLOGIES",
    "ChipOverrides",
    "ChipSpec",
    "Collective",
    "CollectiveConfig",
    "ConcurrencyRow",
    "Dtypes",
    "FloorOptions",
    "ModelGeometry",
    "ModuleCost",
    "Parallel",
    "ServingRequest",
    "SliceTopology",
    "SpecShape",
    "StepFloor",
    "StepShape",
    "dtypes_from_hf_config",
    "from_hf_config",
    "load_hf_config",
    "pass_costs",
    "plan",
    "spec_cycle_floor",
    "step_floor",
    "torus_collective_s",
]
