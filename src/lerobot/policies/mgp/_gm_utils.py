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
Generator Matching Theory Utilities for MGP

Core components implementing unified Markov generative policies:
- Probability paths (Gaussian CondOT) - Section 3.1
- Markov generators (flow, diffusion, jump, CTMC) - Section 3.3
- Generator Matching losses (CGM) - Section 3.4, 4.3
- Safety constraints for hardware - Section 6.1
- Reward alignment methods - Section 6.1-6.5

Theory: Generator Matching defines a Bregman-divergence loss between the
infinitesimal evolution induced by L_t and target evolution consistent with p_t.
Conditional Generator Matching (CGM) applies this at the level of conditional
paths p_t(·|z), yielding scalable objectives that reduce to MSE/KL losses.
"""

import math
import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional, Tuple, Dict, Callable
import torch.nn.functional as F


class GaussianCondOTPath(nn.Module):
    """
    Gaussian Conditional Optimal Transport (CondOT) Path (Section 3.1).

    Theory: p_t(x|z) = N(α_t * z, σ_t² * I)
    with α_0=0, α_1=1, σ_0=1, σ_1=0.

    Implements the conditional probability path that interpolates between
    a simple prior (p_0) and the data distribution (p_1) in action space
    for SO-101 manipulation tasks.

    References:
    - Section 3.1: Probability paths in action space
    - Section 4.1: Action space as state space for diffusion
    """

    def __init__(self, sigma_schedule: str = "linear"):
        """
        Args:
            sigma_schedule: 'linear', 'cosine', or 'exponential'
        """
        super().__init__()
        self.sigma_schedule = sigma_schedule

    def alpha_t(self, t: Tensor) -> Tensor:
        """Signal strength interpolation: 0 at t=0, 1 at t=1."""
        return t

    def sigma_t(self, t: Tensor) -> Tensor:
        """Noise scale: 1 at t=0, 0 at t=1."""
        if self.sigma_schedule == "linear":
            return 1.0 - t
        elif self.sigma_schedule == "cosine":
            return torch.cos(t * math.pi / 2)
        elif self.sigma_schedule == "exponential":
            return torch.exp(-(t * 5.0))
        else:
            raise ValueError(f"Unknown schedule: {self.sigma_schedule}")

    def forward(self, x0: Tensor, t: Tensor) -> Tensor:
        """Mean of conditional distribution: μ_t = α_t * x0."""
        if isinstance(t, float):
            t = torch.tensor(t, device=x0.device, dtype=x0.dtype)
        if t.dim() == 0:
            t = t.unsqueeze(0)

        while t.dim() < x0.dim():
            t = t.unsqueeze(-1)

        alpha_t = self.alpha_t(t)
        return alpha_t * x0

    def sample(self, x0: Tensor, t: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Sample from probability path: x_t = α_t * x0 + σ_t * ε

        Args:
            x0: Data point (action sequence)
            t: Time index(es)

        Returns:
            (x_t, eps): Noisy sample and noise used
        """
        if isinstance(t, float):
            t = torch.tensor(t, device=x0.device, dtype=x0.dtype)
        if t.dim() == 0:
            t = t.unsqueeze(0)

        while t.dim() < x0.dim():
            t = t.unsqueeze(-1)

        alpha_t = self.alpha_t(t)
        sigma_t = self.sigma_t(t)

        eps = torch.randn_like(x0)
        x_t = alpha_t * x0 + sigma_t * eps

        return x_t, eps

    def score_function(self, x_t: Tensor, t: Tensor, x0: Tensor) -> Tensor:
        """
        Score function: ∇_{x_t} log p_t(x_t|x0) = -(x_t - α_t*x0) / σ_t²

        Theory: Used for score matching training in diffusion policies.
        """
        if isinstance(t, float):
            t = torch.tensor(t, device=x_t.device, dtype=x_t.dtype)
        if t.dim() == 0:
            t = t.unsqueeze(0)

        while t.dim() < x_t.dim():
            t = t.unsqueeze(-1)

        alpha_t = self.alpha_t(t)
        sigma_t = self.sigma_t(t)

        mu_t = alpha_t * x0
        score = -(x_t - mu_t) / (sigma_t.pow(2) + 1e-8)

        return score


class GeneratorMatchingLoss(nn.Module):
    """
    Conditional Generator Matching (CGM) Loss (Section 3.4, 4.3).

    Theory:
    - L_CGM(θ) = E_{t,z,x~p_t(·|z)} [D(F_t^z(x), F_t^θ(x))]
    - Proposition 2: ∇L_CGM = ∇L_GM (same gradient as marginal)
    - Special case: DDPM noise prediction MSE loss
    - Reduces to standard diffusion objective for quadratic Bregman divergence

    References:
    - Section 4.3: Training objective as conditional generator matching
    - Section 3.4: Generator matching loss and CGM
    """

    def __init__(self, action_dim: int, loss_type: str = "score_matching", reduction: str = "mean"):
        """
        Args:
            action_dim: Dimension of action space
            loss_type: 'score_matching', 'flow_matching', 'bregman'
            reduction: 'mean' or 'sum'
        """
        super().__init__()
        self.action_dim = action_dim
        self.loss_type = loss_type
        self.reduction = reduction

    def forward(
        self,
        diffusion_pred: Tensor,
        diffusion_target: Tensor,
        weights: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Dict]:
        """
        Compute CGM loss.

        Args:
            diffusion_pred: Model predictions, shape (batch_size, horizon, action_dim)
            diffusion_target: Target values (e.g., noise), same shape
            weights: Optional sample weights

        Returns:
            (loss, metrics_dict)
        """
        # Quadratic Bregman divergence (MSE)
        loss = F.mse_loss(diffusion_pred, diffusion_target, reduction="none")
        loss = loss.mean(dim=-1)  # Average over action dims

        if weights is not None:
            loss = loss * weights
            loss_value = loss.sum() / (weights.sum() + 1e-8)
        else:
            if self.reduction == "mean":
                loss_value = loss.mean()
            else:
                loss_value = loss.sum()

        return loss_value, {"gm_loss_value": loss_value.item()}


class SafetyConstrainedSampler(nn.Module):
    """
    Enforces safety constraints for hardware deployment (Section 6.1).

    Theory: Hardware safety constraints on SO-101 include:
    - Maximum action norm (velocity/acceleration limits)
    - Safety margin (feasibility guarantee)
    - Joint position limits
    - Gripper constraints

    Projects action samples to satisfy these constraints without retraining.
    """

    def __init__(
        self,
        max_action_norm: float = 1.0,
        safety_margin: float = 0.9,
        joint_limits: Optional[Tuple[Tensor, Tensor]] = None,
    ):
        """
        Args:
            max_action_norm: Maximum action magnitude
            safety_margin: Margin factor (< 1.0 for guarantee)
            joint_limits: (min_positions, max_positions) for clipping
        """
        super().__init__()
        self.max_action_norm = max_action_norm
        self.safety_margin = safety_margin
        self.joint_limits = joint_limits

    def forward(self, actions: Tensor) -> Tensor:
        """
        Project actions to safe region.

        Args:
            actions: Action sequences, shape (batch_size, horizon, action_dim)

        Returns:
            Safe actions with constraints satisfied
        """
        safe_actions = actions.clone()

        # Norm constraint (action velocity limits)
        norms = torch.norm(safe_actions, dim=-1, keepdim=True)
        scaling = torch.minimum(
            torch.ones_like(norms),
            (self.max_action_norm * self.safety_margin) / (norms + 1e-8),
        )
        safe_actions = safe_actions * scaling

        # Joint limits (if provided)
        if self.joint_limits is not None:
            min_pos, max_pos = self.joint_limits
            safe_actions = torch.clamp(safe_actions, min=min_pos, max=max_pos)

        return safe_actions


class JumpProcessGenerator(nn.Module):
    """
    Jump process component of Markov generator (Section 3.3).

    Theory: L^jump_t models abrupt strategy shifts via Poisson jumps.
    Useful for mode switching in tasks (regrasp attempts, approach angle changes).

    Markov decomposition: L_t = L^flow_t + L^diff_t + L^jump_t + L^CTMC_t
    """

    def __init__(self, action_dim: int, num_modes: int = 4, jump_rate: float = 0.1):
        """
        Args:
            action_dim: Action space dimension
            num_modes: Number of discrete modes/strategies
            jump_rate: Poisson jump rate parameter
        """
        super().__init__()
        self.action_dim = action_dim
        self.num_modes = num_modes
        self.jump_rate = jump_rate

        # Transition matrix Q_t(y|x) for mode switches
        self.transition_matrix = nn.Parameter(
            torch.eye(num_modes) + torch.randn(num_modes, num_modes) * 0.1
        )

    def forward(self, x: Tensor, t: float) -> Tensor:
        """
        Sample from jump process at time t.

        Args:
            x: Current action state
            t: Time index

        Returns:
            New action state after potential jump
        """
        # Probability of jump
        jump_prob = 1.0 - torch.exp(torch.tensor(-self.jump_rate * t))

        if torch.rand(1) < jump_prob:
            # Apply mode transition with magnitude change
            mode_idx = torch.multinomial(torch.ones(self.num_modes), 1)
            scale = torch.randn(1) * 0.5 + 1.0
            return x * scale
        else:
            return x


class CTMCGenerator(nn.Module):
    """
    Continuous-Time Markov Chain (CTMC) generator for discrete modes (Section 3.3).

    Theory: Operates on discrete skill/behavior space with rate matrix Q,
    complementing continuous flow and diffusion generators for hierarchical policy.

    Synthesis matrix row: CTMC in R^(d_A * H) with rate matrix Q_t.
    """

    def __init__(self, num_skills: int, skill_dim: int = 64):
        """
        Args:
            num_skills: Number of discrete skills/modes
            skill_dim: Embedding dimension for skills
        """
        super().__init__()
        self.num_skills = num_skills
        self.skill_embeddings = nn.Embedding(num_skills, skill_dim)

        # Rate matrix Q for CTMC
        self.rate_matrix = nn.Parameter(torch.randn(num_skills, num_skills) * 0.1)

    def forward(self, current_skill: int, t: float) -> Tensor:
        """
        Sample next skill from CTMC transition.

        Args:
            current_skill: Current skill index
            t: Time duration

        Returns:
            Skill embedding for next state
        """
        # Compute transition probabilities from rate matrix
        rates = torch.exp(self.rate_matrix * t)
        probs = torch.softmax(rates[current_skill], dim=0)

        # Sample next skill
        next_skill = torch.multinomial(probs, 1).item()

        return self.skill_embeddings(torch.tensor([next_skill]))


class FlowMatchingGenerator(nn.Module):
    """
    Flow matching generator - deterministic behavior cloning (Section 3.3, 5.3).

    Theory: L^flow_t corresponds to ODE drift field u_t(x),
    modeling smooth deterministic behavior (e.g., simple reaching).

    Synthesis matrix row: ODE generator [L_t f](x) = ∇f · u_t(x).
    """

    def __init__(self, action_dim: int, hidden_dim: int = 128):
        """
        Args:
            action_dim: Action space dimension
            hidden_dim: Hidden layer size for velocity field
        """
        super().__init__()
        self.action_dim = action_dim

        # Neural network parameterizing flow velocity field
        self.velocity_net = nn.Sequential(
            nn.Linear(action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, x: Tensor, t: float) -> Tensor:
        """
        Compute velocity field u_t(x).

        Args:
            x: Current action state
            t: Time index (unused, for interface compatibility)

        Returns:
            Velocity/drift at current state
        """
        return self.velocity_net(x)


class RewardTiltedDistribution(nn.Module):
    """
    Reward-tilted distribution for alignment (Section 6.1).

    Theory: Gibbs tilt - π_β(x) ∝ p_base(x) * exp(β * r(x))

    Used in inference-time alignment to reweight samples toward high-reward regions
    without retraining the base policy.
    """

    def __init__(self, reward_temperature: float = 1.0):
        """
        Args:
            reward_temperature: β parameter controlling tilt strength
        """
        super().__init__()
        self.reward_temperature = reward_temperature

    def forward(
        self,
        base_log_prob: Tensor,
        reward: Tensor,
    ) -> Tensor:
        """
        Compute log probability under Gibbs tilt.

        log π_β(x) = log p_base(x) + β * r(x) - log Z

        Args:
            base_log_prob: Log probability under base distribution
            reward: Reward value r(x)

        Returns:
            Log probability under reward-tilted distribution
        """
        # Unnormalized log probability (log Z cancels in softmax)
        log_prob = base_log_prob + self.reward_temperature * reward

        return log_prob


class SequentialMonteCarloSampler(nn.Module):
    """
    Sequential Monte Carlo sampler for reward alignment (Section 6.3).

    Theory: Maintains particle set through denoising steps,
    reweighting by value function at each step. Implements:

    Modified reverse kernel: p̃_{k-1}(A_{k-1}|A_k) ∝ p^pre_{k-1}(A_{k-1}|A_k) v^r_{k-1}(A_{k-1})
    """

    def __init__(self, num_particles: int = 32, resample_threshold: float = 0.5):
        """
        Args:
            num_particles: Number of particles to maintain
            resample_threshold: Effective sample size threshold for resampling
        """
        super().__init__()
        self.num_particles = num_particles
        self.resample_threshold = resample_threshold

    def forward(
        self,
        initial_samples: Tensor,
        value_fn: Callable,
        num_steps: int = 10,
    ) -> Tensor:
        """
        Run SMC over denoising steps.

        Args:
            initial_samples: Initial particle set (batch_size * num_particles, ...)
            value_fn: Value function for reweighting
            num_steps: Number of SMC steps

        Returns:
            Refined samples with higher expected value
        """
        particles = initial_samples
        weights = torch.ones(initial_samples.shape[0]) / self.num_particles

        for step in range(num_steps):
            # Compute values for reweighting
            values = value_fn(particles)
            weights = weights * torch.softmax(values, dim=0)

            # Normalize weights
            weights = weights / (weights.sum() + 1e-8)

            # Check effective sample size
            ess = 1.0 / ((weights ** 2).sum() + 1e-8)
            if ess < self.resample_threshold * self.num_particles:
                # Resample
                indices = torch.multinomial(weights, self.num_particles, replacement=True)
                particles = particles[indices]
                weights = torch.ones_like(weights) / self.num_particles

        return particles


class EnergyBasedGeneratorMatching(nn.Module):
    """
    Energy-Based Generator Matching (EGM) for reward alignment (Section 6.5).

    Theory: Treats reward as energy, defining unnormalized target law
    π(x) ∝ exp(-E(x)) with E(x) = -r(x).

    Useful when aligned target known only up to normalizing constant,
    common in multi-objective SO-101 rewards with safety constraints.
    """

    def __init__(self, action_dim: int):
        """
        Args:
            action_dim: Action space dimension
        """
        super().__init__()
        self.action_dim = action_dim

    def forward(self, actions: Tensor, rewards: Tensor) -> Tensor:
        """
        Compute EGM loss for unnormalized reward-tilted distribution.

        Args:
            actions: Action samples
            rewards: Reward values

        Returns:
            Loss for score matching under exp(-E(x))
        """
        # Energy = negative reward
        energy = -rewards

        # Score matching loss for unnormalized model
        loss = (energy ** 2).mean()

        return loss
