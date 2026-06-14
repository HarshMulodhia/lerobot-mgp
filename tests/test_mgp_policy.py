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
Comprehensive test suite for MGP policy.

Tests cover:
- Generator Matching theory components
- Distribution shift handling
- Multi-modal sampling
- Reward alignment
- Hardware safety constraints
- Training stability
"""

import logging
import tempfile
from pathlib import Path
from typing import Dict, Tuple

import pytest
import torch
import torch.nn as nn

# Assume MGP modules can be imported
from lerobot.policies.mgp.configuration_mgp import MGPConfig
from lerobot.policies.mgp.generator_matching import (
    ConditionalGeneratorMatching,
    DiffusionGenerator,
    GaussianCondOTPath,
    GeneratorMatchingLoss,
)
from lerobot.policies.mgp.mgp_training import (
    CurriculumScheduler,
    EnergyBasedGeneratorMatching,
    SafetyConstrainedSampler,
    TrajectoryImportanceWeighter,
)


class TestProbabilityPath:
    """Test probability path sampling."""

    def test_condot_path_interpolation(self):
        """Test Gaussian CondOT path interpolation."""
        path = GaussianCondOTPath()

        x0 = torch.randn(4, 6, 10)  # (B, D, T)
        x1 = torch.randn(4, 6, 10)
        t = 0.5

        x_t = path.sample_path(x0, x1, t)

        assert x_t.shape == x0.shape
        # At t=0.5, should be approximately (x0+x1)/sqrt(2)
        expected = (x0 / torch.sqrt(torch.tensor(2.0)) + x1 / torch.sqrt(torch.tensor(2.0)))
        assert torch.allclose(x_t, expected, atol=1e-5)

    def test_marginal_prob_bounds(self):
        """Test marginal probabilities are well-bounded."""
        path = GaussianCondOTPath()

        t = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0])
        alpha, sigma = path.marginal_prob(t)

        # alpha^2 + sigma^2 should equal 1 (CondOT property)
        total = alpha**2 + sigma**2
        assert torch.allclose(total, torch.ones_like(total), atol=1e-5)

        # alpha should decrease, sigma should increase
        assert (alpha[1:] <= alpha[:-1]).all()
        assert (sigma[1:] >= sigma[:-1]).all()


class TestGeneratorMatching:
    """Test Generator Matching losses and components."""

    def test_gm_loss_score_matching(self):
        """Test score matching loss computation."""
        loss_fn = GeneratorMatchingLoss(loss_type="score_matching")

        pred_score = torch.randn(8, 6, 10)
        target_score = pred_score + 0.01 * torch.randn_like(pred_score)

        loss, metrics = loss_fn(pred_score, target_score)

        assert loss.item() > 0
        assert "score_l2_norm" in metrics
        assert metrics["score_l2_norm"] >= 0

    def test_gm_loss_with_weights(self):
        """Test loss computation with sample weights."""
        loss_fn = GeneratorMatchingLoss(loss_type="score_matching", reduction="mean")

        pred_score = torch.randn(8, 6, 10)
        target_score = torch.randn(8, 6, 10)
        weights = torch.softmax(torch.randn(8), dim=0)

        loss_weighted, _ = loss_fn(pred_score, target_score, weights=weights)
        loss_unweighted, _ = loss_fn(pred_score, target_score, weights=None)

        assert loss_weighted.shape == torch.Size([])
        assert loss_unweighted.shape == torch.Size([])

    def test_diffusion_generator_components(self):
        """Test diffusion generator drift and diffusion computation."""

        def dummy_score_fn(x, t):
            return torch.zeros_like(x)

        gen = DiffusionGenerator(score_fn=dummy_score_fn)

        x = torch.randn(8, 6, 10)
        t = torch.tensor([0.5] * 8)

        drift = gen.compute_drift(x, t)
        diffusion = gen.compute_diffusion(t)

        assert drift.shape == x.shape
        assert diffusion.shape == torch.Size([8])


class TestDistributionShiftHandling:
    """Test distribution shift and compounding error handling."""

    def test_curriculum_scheduler_linear(self):
        """Test linear curriculum scheduling."""
        scheduler = CurriculumScheduler(
            total_steps=100,
            start_difficulty=0.0,
            end_difficulty=1.0,
            curriculum_type="linear",
        )

        difficulties = []
        for _ in range(100):
            difficulties.append(scheduler.get_difficulty())
            scheduler.step()

        # Should be monotonically increasing
        for i in range(1, len(difficulties)):
            assert difficulties[i] >= difficulties[i - 1]

        # Should start near 0 and end near 1
        assert difficulties[0] < 0.2
        assert difficulties[-1] > 0.8

    def test_curriculum_sample_weighting(self):
        """Test curriculum sample weighting."""
        scheduler = CurriculumScheduler(total_steps=100)

        # Trajectories with varying difficulty
        traj_diffs = torch.tensor([0.1, 0.3, 0.5, 0.7, 0.9])

        for step in range(50):
            weights = scheduler.get_sample_weights(traj_diffs)
            scheduler.step()

            assert weights.shape == traj_diffs.shape
            assert torch.allclose(weights.sum(), torch.tensor(1.0), atol=1e-5)
            assert (weights >= 0).all()

    def test_importance_weighter(self):
        """Test trajectory importance weighting."""
        weighter = TrajectoryImportanceWeighter(action_dim=6)

        actions = torch.randn(8, 10, 6)
        weights = weighter.compute_importance_weights(actions)

        assert weights.shape == torch.Size([8])
        assert torch.allclose(weights.sum(), torch.tensor(1.0), atol=1e-5)
        assert (weights >= 0).all()


class TestSafetyConstraints:
    """Test hardware safety constraint enforcement."""

    def test_step_size_constraint(self):
        """Test action step size constraint."""
        sampler = SafetyConstrainedSampler(max_action_step_size=0.1)

        # Create actions with large jumps
        actions = torch.zeros(2, 10, 6)
        actions[:, 1, :] = 1.0  # Big jump
        actions[:, 2:, :] = 1.0  # Hold

        constrained = sampler.enforce_constraints(actions)

        # Check step sizes
        for t in range(1, constrained.shape[1]):
            step = constrained[:, t] - constrained[:, t - 1]
            step_size = torch.norm(step, dim=1)
            assert (step_size <= 0.1 + 1e-5).all()

    def test_joint_limit_enforcement(self):
        """Test joint limit enforcement."""
        min_q = torch.tensor([-1.0, -1.0, -1.0, -1.0, -1.0, -1.0])
        max_q = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

        sampler = SafetyConstrainedSampler(joint_limits=(min_q, max_q))

        # Create actions outside limits
        actions = torch.ones(2, 10, 6) * 2.0

        constrained = sampler.enforce_constraints(actions)

        # Check limits
        assert (constrained >= min_q.view(1, 1, -1)).all()
        assert (constrained <= max_q.view(1, 1, -1)).all()


class TestRewardAlignment:
    """Test reward alignment mechanisms."""

    def test_energy_based_gm(self):
        """Test energy-based generator matching."""
        ebm = EnergyBasedGeneratorMatching(temperature=1.0)

        trajectories = torch.randn(8, 10, 6)
        rewards = torch.randn(8)
        learnt_scores = torch.randn(8, 10)

        loss, metrics = ebm.compute_ebm_loss(trajectories, rewards, learnt_scores)

        assert loss.item() is not None
        assert "mean_reward" in metrics
        assert "energy_term" in metrics


class TestMGPConfiguration:
    """Test MGP configuration options."""

    def test_mgp_config_creation(self):
        """Test MGP config instantiation."""
        config = MGPConfig(
            use_generator_matching=True,
            enable_multimodal_sampling=True,
            trajectory_horizon=10,
        )

        assert config.type == "mgp"
        assert config.use_generator_matching is True
        assert config.trajectory_horizon == 10

    def test_mgp_config_defaults(self):
        """Test MGP config default values."""
        config = MGPConfig()

        assert config.use_generator_matching is True
        assert config.enable_multimodal_sampling is True
        assert config.enable_distribution_shift_adaptation is True
        assert config.max_action_step_size == 0.1
        assert config.target_hardware == "so101"


class TestIntegration:
    """Integration tests for MGP components."""

    def test_path_to_loss_pipeline(self):
        """Test probability path -> GM loss pipeline."""
        path = GaussianCondOTPath()
        loss_fn = GeneratorMatchingLoss()

        # Simulated batch
        batch_size, action_dim, horizon = 4, 6, 10

        x0 = torch.randn(batch_size, action_dim, horizon)
        x1 = torch.randn(batch_size, action_dim, horizon)
        t = torch.rand(batch_size) * 0.5 + 0.25

        # Sample path
        x_t = path.sample_path(x0, x1, t)
        assert x_t.shape == x0.shape

        # Compute target score
        alpha_t, sigma_t = path.marginal_prob(t)
        target_score = (x1 - x0) / (sigma_t.reshape(-1, 1, 1) + 1e-6)

        # Predict score (mock)
        pred_score = target_score + 0.01 * torch.randn_like(target_score)

        # Compute loss
        loss, metrics = loss_fn(pred_score, target_score)

        assert loss.item() >= 0
        assert metrics is not None

    def test_full_training_pipeline(self):
        """Test full training pipeline with all components."""
        from lerobot.policies.mgp.mgp_training import MGPTrainingHelper

        config = MGPConfig(
            use_curriculum_learning=True,
            enable_distribution_shift_adaptation=True,
            use_offline_rl_alignment=True,
            steps=1000,
        )

        helper = MGPTrainingHelper(config)

        # Mock batch
        batch = {
            "action": torch.randn(8, 10, 6),
        }

        # Compute weights
        traj_diffs = torch.rand(8)
        weights = helper.compute_sample_weights(batch, traj_diffs)

        assert weights.shape == torch.Size([8])
        assert torch.allclose(weights.sum(), torch.tensor(1.0), atol=1e-5)

        # Step curriculum
        helper.step_curriculum()

        # Enforce safety
        actions = torch.randn(4, 10, 6)
        safe_actions = helper.enforce_safety_constraints(actions)

        assert safe_actions.shape == actions.shape


class TestPerformance:
    """Performance and numerical stability tests."""

    def test_numerical_stability(self):
        """Test numerical stability of core components."""
        path = GaussianCondOTPath()
        loss_fn = GeneratorMatchingLoss()

        # Edge case: very small t
        x0 = torch.randn(4, 6, 10)
        x1 = torch.randn(4, 6, 10)

        for t_val in [1e-6, 0.0, 1.0, 1.0 - 1e-6]:
            x_t = path.sample_path(x0, x1, t_val)
            assert not torch.isnan(x_t).any()
            assert not torch.isinf(x_t).any()

    def test_batch_size_robustness(self):
        """Test robustness across different batch sizes."""
        weighter = TrajectoryImportanceWeighter(action_dim=6)

        for batch_size in [1, 2, 8, 32, 128]:
            actions = torch.randn(batch_size, 10, 6)
            weights = weighter.compute_importance_weights(actions)

            assert weights.shape == torch.Size([batch_size])
            assert not torch.isnan(weights).any()


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
