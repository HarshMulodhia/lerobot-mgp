#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

"""
Markov Generative Policies for SO-101 with LeRobot

Complete implementation of unified Markov Generative Policies framework:
- Probability paths (Gaussian CondOT, Section 3.1)
- Markov decomposition: Flow, Diffusion, Jump, CTMC
- Conditional Generator Matching (CGM) loss
- Reward alignment (inference-time + post-training)
- Hardware safety constraints
- Multi-camera vision support

Reference: "Markov Generative Policies for SO-101 with LeRobot" (2026)
"""

from .configuration_mgp import MGPConfig
from .modeling_mgp import MarkovGenerativePolicy, MGPPolicy, MGPRgbEncoder, MGPDiffusionHead
from ._gm_utils import (
    GaussianCondOTPath,
    GeneratorMatchingLoss,
    FlowMatchingGenerator,
    JumpProcessGenerator,
    CTMCGenerator,
    SafetyConstrainedSampler,
)

__all__ = [
    "MGPConfig",
    "MarkovGenerativePolicy",
    "MGPPolicy",
    "MGPRgbEncoder",
    "MGPDiffusionHead",
    "GaussianCondOTPath",
    "GeneratorMatchingLoss",
    "FlowMatchingGenerator",
    "JumpProcessGenerator",
    "CTMCGenerator",
    "SafetyConstrainedSampler",
]
