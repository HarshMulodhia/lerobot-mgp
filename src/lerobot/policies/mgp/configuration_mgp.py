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
Markov Generator Policy (MGP) - Configuration

Extends DiffusionConfig with all Markov Superposition components:
- Probability paths (Gaussian CondOT)
- Generator decomposition (Flow, Diffusion, Jump, CTMC)
- Conditional Generator Matching (CGM) losses
- Reward alignment (inference-time and post-training)
- Multi-camera observation support
"""

from dataclasses import dataclass, field
from typing import Dict, Optional

from lerobot.configs import PreTrainedConfig

try:
    from ..diffusion.configuration_diffusion import DiffusionConfig
except ImportError:
    DiffusionConfig = None


@PreTrainedConfig.register_subclass("mgp")
@dataclass
class MGPConfig(DiffusionConfig if DiffusionConfig else object):
    """Configuration for Markov Generator Policy (MGP).

    Implements unified Markov Generative Policies with:
    - Flow, Diffusion, Jump, and CTMC components
    - Markov superposition for component blending
    - Full Generator Matching theory
    - Multi-camera vision support
    - Reward alignment (inference-time and post-training)

    All components are independently tunable via command-line arguments.
    
    Usage examples:
        # Default (balanced)
        lerobot-train --policy.type=mgp
        
        # Custom loss weights via JSON dict
        lerobot-train --policy.type=mgp \\
          --policy.loss_weights='{"diffusion": 1.5, "gm": 0.3, "flow": 0.1, "jump": 0.2, "ctmc": 0.0}'
        
        # Enable all components with Markov superposition
        lerobot-train --policy.type=mgp \\
          --policy.enable_jump_component=true \\
          --policy.enable_ctmc_component=true \\
          --policy.enable_markov_superposition=true \\
          --policy.loss_weights='{"diffusion": 1.0, "gm": 0.1, "flow": 0.05, "jump": 0.1, "ctmc": 0.05, "reward": 0.01}'
        
        # Focus on jump processes for mode switching
        lerobot-train --policy.type=mgp \\
          --policy.enable_jump_component=true \\
          --policy.loss_weights='{"jump": 0.5, "diffusion": 1.0}'
    """

    # ===== Generator Matching Core (Section 3.1-3.4) =====
    use_generator_matching: bool = True
    """Enable explicit Generator Matching theory components."""

    gm_loss_type: str = "score_matching"
    """Type of GM loss: 'score_matching', 'flow_matching', 'bregman'."""

    # ===== Combined Loss Weights (Section 4.3, 5.1) =====
    # Total Loss: L = α*L_DP + β*L_GM + γ*L_FM + δ*L_JUMP + ε*L_CTMC + λ*L_reward
    loss_weights: Optional[Dict[str, float]] = field(
        default_factory=lambda: {
            "diffusion": 1.0,      # α: Primary imitation (DDPM MSE)
            "gm": 0.1,             # β: Generator Matching (multi-camera)
            "flow": 0.05,          # γ: Flow/ODE baseline (behavior cloning)
            "jump": 0.0,           # δ: Jump process (mode switching)
            "ctmc": 0.0,           # ε: CTMC (discrete skills)
            "reward": 0.01,        # λ: Reward alignment
        }
    )
    """
    Markov Superposition loss weights for unified framework.
    
    Keys:
    - 'diffusion' (α): DDPM noise-prediction MSE (Eq. 4.2)
        Handles multimodal action distributions.
        Tune: 1.0-2.0 for imitation-focused learning.
    
    - 'gm' (β): Conditional Generator Matching (Eq. 4.3, 3.4)
        Leverages multi-camera observations concatenated in action space.
        Tune: 0.1-0.5 for stronger visual grounding.
    
    - 'flow' (γ): Flow/ODE generator (Eq. 3.3, L^flow_t)
        Deterministic behavior cloning, stabilizing baseline.
        Tune: 0.05-0.2 for smoother, more predictable motions.
    
    - 'jump' (δ): Jump process generator (Eq. 3.3, L^jump_t)
        Models abrupt strategy switches, regrasp attempts, mode changes.
        Tune: 0.05-0.3 for discrete behavior transitions.
        Requires: enable_jump_component=true
    
    - 'ctmc' (ε): CTMC generator on discrete modes (Eq. 3.3, L^CTMC_t)
        High-level skill or behavior mode switching.
        Tune: 0.01-0.2 for hierarchical policies.
        Requires: enable_ctmc_component=true
    
    - 'reward' (λ): Reward alignment loss (Section 6, Flow-GRPO)
        Post-training generator retargeting toward high-reward regions.
        Tune: 0.01-0.1 when reliable reward signals available.
        Requires: enable_reward_alignment=true
    
    Example via command line:
        --policy.loss_weights='{"diffusion": 1.5, "gm": 0.3, "flow": 0.1, "jump": 0.2}'
    """

    # ===== Probability Path Configuration (Section 3.1) =====
    beta_schedule: str = "linear"
    """Noise schedule: 'linear', 'cosine', 'exponential'."""

    trajectory_horizon: int = 10
    """Horizon T_p for action sequence prediction."""

    chunk_size: int = 1
    """Action chunk size for receding-horizon control."""

    # ===== Flow Component (L^flow_t) - Section 3.3, 5.3 =====
    enable_flow_component: bool = True
    """Enable deterministic flow/ODE component."""

    flow_hidden_dim: int = 128
    """Hidden dimension for flow velocity network."""

    # ===== Diffusion Component (L^diff_t) - Section 4, 3.3 =====
    enable_diffusion_component: bool = True
    """Enable stochastic diffusion component (always on for base DP)."""

    # ===== Jump Process Component (L^jump_t) - Section 3.3, Table 7 =====
    enable_jump_component: bool = False
    """Enable jump process component for discrete mode switches."""

    jump_num_modes: int = 4
    """Number of discrete modes for jump process."""

    jump_rate: float = 0.1
    """Poisson jump rate parameter λ_t."""

    jump_loss_weight: float = 0.0
    """Jump-specific loss weight (overridden by loss_weights['jump'] if set)."""

    # ===== CTMC Component (L^CTMC_t) - Section 3.3, Table 7 =====
    enable_ctmc_component: bool = False
    """Enable CTMC (continuous-time Markov chain) for discrete skills."""

    ctmc_num_skills: int = 8
    """Number of discrete skills/modes for CTMC."""

    ctmc_skill_dim: int = 64
    """Embedding dimension for skill representations."""

    ctmc_loss_weight: float = 0.0
    """CTMC-specific loss weight (overridden by loss_weights['ctmc'] if set)."""

    # ===== Markov Superposition (Section 3.5, 5.3) =====
    enable_markov_superposition: bool = False
    """Enable Markov superposition with learned gating weights."""

    superposition_hidden_dim: int = 128
    """Hidden dimension for gating network g(h_t)."""

    superposition_learn_weights: bool = True
    """Learn gating weights vs fixed initialization."""

    # ===== Multi-Camera Support =====
    enable_multi_camera_concat: bool = True
    """Enable concatenation of remapped camera observations."""

    camera_concat_dim: int = -3
    """Dimension for camera concatenation (default: -3 for channel dim)."""

    # ===== Multimodality =====
    enable_multimodal_sampling: bool = False
    """Enable sampling multiple diverse trajectories."""

    num_sample_candidates: int = 8
    """Number of trajectory samples for selection."""

    # ===== Reward Alignment (Section 6) =====
    enable_reward_alignment: bool = False
    """Enable reward alignment for inference or post-training."""

    reward_alignment_type: str = "inference_time"
    """'inference_time' (Gibbs tilt, SMC) or 'post_training' (Flow-GRPO, EGM)."""

    reward_temperature: float = 1.0
    """Temperature β for reward-tilted sampling (Eq. 6.1)."""

    use_sequential_monte_carlo: bool = False
    """Use SMC for progressive refinement during sampling (Section 6.3)."""

    smc_particles: int = 16
    """Number of particles for SMC."""

    # ===== Distribution Shift & Curriculum =====
    enable_distribution_shift_adaptation: bool = False
    """Enable online adaptation for distribution shift."""

    use_curriculum_learning: bool = False
    """Use curriculum learning with progressive difficulty."""

    # ===== Hardware Safety (Section 6.1) =====
    enable_hardware_safety_checks: bool = True
    """Enable safety constraints for real hardware."""

    max_action_step_size: float = 0.1
    """Maximum action magnitude for safety (SO-101 constraint)."""

    target_hardware: str = "so101"
    """Target hardware: 'so101', 'generic', 'arm'."""

    # ===== Computational Efficiency =====
    use_fast_inference_mode: bool = False
    """Use faster inference with fewer diffusion steps."""

    fast_inference_steps: int = 5
    """Number of denoising steps for fast inference."""
