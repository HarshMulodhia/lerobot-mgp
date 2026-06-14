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

Extends DiffusionConfig with Generator Matching theory components,
reward alignment options, and hardware safety constraints.
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

    Extends DiffusionPolicy with explicit Generator Matching theory:
    - Probability paths for conditional generative modeling
    - Markov generator decomposition (flow, diffusion, jump, CTMC)
    - Conditional Generator Matching (CGM) losses
    - Reward alignment (inference-time and post-training)
    - Distribution shift adaptation and safety constraints

    Fully backward compatible with DiffusionPolicy configurations.
    
    Example usage:
    
    1. Default (balanced):
        lerobot-train --policy.type=mgp
    
    2. Custom loss weights via dict:
        lerobot-train --policy.type=mgp \\
          --policy.loss_weights='{"diffusion": 1.5, "gm": 0.3, "flow": 0.1, "reward": 0.05}'
    
    3. Focus on multi-camera learning:
        lerobot-train --policy.type=mgp \\
          --policy.loss_weights='{"gm": 0.5}'  # Other weights use defaults
    
    4. Smooth baseline prioritization:
        lerobot-train --policy.type=mgp \\
          --policy.loss_weights='{"diffusion": 1.0, "flow": 0.2, "gm": 0.05}'
    """

    # ===== Generator Matching Components =====
    use_generator_matching: bool = True
    """Enable explicit Generator Matching theory components."""

    gm_loss_type: str = "score_matching"
    """Type of GM loss: 'score_matching', 'flow_matching', 'bregman'."""

    # ===== Combined Loss Weights (Section 4.3, 5.1) =====
    # Total Loss: L_total = α*L_DP + β*L_GM + γ*L_FM + λ*L_reward
    loss_weights: Optional[Dict[str, float]] = field(
        default_factory=lambda: {
            "diffusion": 1.0,
            "gm": 0.1,
            "flow": 0.05,
            "reward": 0.01,
        }
    )
    """
    Combined loss weights for multi-objective training.
    
    Keys:
    - 'diffusion' (α): Primary imitation loss (default=1.0)
        Higher values prioritize matching demonstration trajectories.
        Tune: increase to 1.5-2.0 for behavior cloning focused training.
    
    - 'gm' (β): Generator matching loss for multi-camera conditioning (default=0.1)
        Encourages learning multi-camera visual representations.
        Tune: increase to 0.2-0.5 for stronger visual understanding.
    
    - 'flow' (γ): Flow matching loss for deterministic baseline (default=0.05)
        Adds smooth, deterministic reaching motions as regularization.
        Tune: increase to 0.1-0.2 for smoother, more predictable policies.
    
    - 'reward' (λ): Reward alignment loss (default=0.01)
        Only active when enable_reward_alignment=True.
        Tune: increase to 0.05-0.1 when reliable reward signals available.
    
    Example via command line:
        --policy.loss_weights='{"diffusion": 1.5, "gm": 0.3, "flow": 0.1}'
    """

    # ===== Trajectory Sampling =====
    trajectory_horizon: int = 10
    """Horizon for action trajectory prediction."""

    chunk_size: int = 1
    """Action chunk size for receding-horizon control."""

    # ===== Multimodality =====
    enable_multimodal_sampling: bool = False
    """Enable sampling multiple diverse trajectories."""

    num_sample_candidates: int = 8
    """Number of trajectory samples for selection."""

    # ===== Reward Alignment =====
    enable_reward_alignment: bool = False
    """Enable reward alignment for inference or post-training."""

    reward_alignment_type: str = "inference_time"
    """'inference_time' (Gibbs tilt) or 'post_training' (RL)."""

    reward_temperature: float = 1.0
    """Temperature β for reward-tilted sampling."""

    use_sequential_monte_carlo: bool = False
    """Use SMC for progressive refinement during sampling."""

    smc_particles: int = 16
    """Number of particles for SMC."""

    # ===== Distribution Shift =====
    enable_distribution_shift_adaptation: bool = False
    """Enable online adaptation for distribution shift."""

    use_curriculum_learning: bool = False
    """Use curriculum learning with progressive difficulty."""

    # ===== Hardware Safety =====
    enable_hardware_safety_checks: bool = True
    """Enable safety constraints for real hardware."""

    max_action_step_size: float = 0.1
    """Maximum action magnitude for safety."""

    target_hardware: str = "so101"
    """Target hardware: 'so101', 'generic', 'arm'."""

    # ===== Computational Efficiency =====
    use_fast_inference_mode: bool = False
    """Use faster inference with fewer diffusion steps."""

    fast_inference_steps: int = 5
    """Number of denoising steps for fast inference."""
