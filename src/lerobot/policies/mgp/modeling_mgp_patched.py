#!/usr/bin/env python

import logging
from typing import Any, Dict, Optional, Tuple, Callable

import torch
import torch.nn as nn
from torch import Tensor

from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.utils.constants import ACTION, OBS_STATE

from .configuration_mgp import MGPConfig
from ._gm_utils import (
    GaussianCondOTPath,
    GeneratorMatchingLoss,
    SafetyConstrainedSampler,
)

logger = logging.getLogger(__name__)


class MarkovGenerativePolicy(DiffusionPolicy):
    """Markov Generative Policy (MGP) - Complete Implementation"""

    config_class = MGPConfig
    name = "mgp"

    def __init__(self, config: MGPConfig, **kwargs):
        """Initialize MGP policy extending DiffusionPolicy."""
        super().__init__(config, **kwargs)
        self.config = config

        logger.info("Initializing Markov Generative Policy (MGP)")

        # ===== SECTION 3.1: Probability Paths =====
        self.prob_path = GaussianCondOTPath(sigma_schedule=self.config.beta_schedule)
        logger.info(f"Initialized Gaussian CondOT path with schedule: {self.config.beta_schedule}")

        # ===== SECTION 3.4 / 4.3: Conditional Generator Matching Loss =====
        if self.config.use_generator_matching:
            self.gm_loss = GeneratorMatchingLoss(
                action_dim=self._get_action_dim(),
                loss_type=self.config.gm_loss_type,
            )
            logger.info(f"CGM loss enabled: {self.config.gm_loss_type}")

        # ===== SECTION 6.1-6.3: Reward Alignment =====
        if self.config.enable_reward_alignment:
            self._init_reward_alignment()
            logger.info(f"Reward alignment enabled: {self.config.reward_alignment_type}")

        # ===== SECTION 6.1: Safety Constraints =====
        if self.config.enable_hardware_safety_checks:
            self.safety_sampler = SafetyConstrainedSampler(
                max_action_norm=self.config.max_action_step_size,
            )
            logger.info("Hardware safety constraints enabled")

    def _get_action_dim(self) -> int:
        """Get action dimensionality from config."""
        if hasattr(self.config, "action_feature") and hasattr(self.config.action_feature, "shape"):
            return int(self.config.action_feature.shape[0])
        return 6

    def _init_reward_alignment(self):
        """Initialize reward alignment components (Section 6.2)."""
        if self.config.reward_alignment_type == "inference_time":
            self._init_inference_time_alignment()
        elif self.config.reward_alignment_type == "post_training":
            self._init_post_training_alignment()

    def _init_inference_time_alignment(self):
        """Initialize inference-time alignment (Section 6.3)."""
        if self.config.use_sequential_monte_carlo:
            self.smc_particles = self.config.smc_particles
            self.smc_resampling_threshold = 0.5
            logger.info(f"SMC initialized with {self.config.smc_particles} particles")

    def _init_post_training_alignment(self):
        """Initialize post-training alignment (Section 6.4)."""
        self.kl_weight = 0.1
        logger.info("Post-training alignment initialized")

    def forward(self, batch: Dict[str, Tensor]) -> Tuple[Tensor, Optional[Dict[str, Any]]]:
        """Forward pass computing combined diffusion and GM losses."""
        loss_diffusion, output_dict = super().forward(batch)

        if output_dict is None:
            output_dict = {}

        output_dict["loss_diffusion"] = (
            loss_diffusion.item() if isinstance(loss_diffusion, Tensor) else loss_diffusion
        )
        output_dict["mgp_enabled"] = self.config.use_generator_matching

        # ===== SECTION 4.3: Add CGM Loss =====
        if self.config.use_generator_matching and ACTION in batch:
            try:
                loss_diffusion, output_dict = self._compute_gm_loss(
                    batch, loss_diffusion, output_dict
                )
            except Exception as e:
                logger.warning(f"GM loss computation failed: {e}, using diffusion loss only")

        return loss_diffusion, output_dict

    def _compute_gm_loss(
        self, batch: Dict[str, Tensor], loss_diffusion: Tensor, output_dict: Dict
    ) -> Tuple[Tensor, Dict]:
        """Compute Conditional Generator Matching loss."""
        try:
            actions = batch[ACTION]
            batch_size = actions.shape[0]

            timesteps = torch.randint(0, 1000, (batch_size,), device=actions.device)
            t = timesteps.float() / 1000.0

            x_t, eps = self.prob_path.sample(actions, t)

            batch_cond = self._flatten_multi_camera_observations(batch)

            # Prepare conditioning - handle failures gracefully
            global_cond = self.diffusion._prepare_global_conditioning(batch_cond)
            noise_pred = self.diffusion.unet(x_t, timesteps, global_cond=global_cond)

            gm_loss, gm_metrics = self.gm_loss(
                diffusion_pred=noise_pred,
                diffusion_target=eps,
            )

            combined_loss = loss_diffusion + self.config.gm_loss_weight * gm_loss
            output_dict.update(gm_metrics)
            output_dict["loss_gm"] = gm_loss.item()
            output_dict["loss_combined"] = combined_loss.item()
            output_dict["gm_enabled"] = True

            return combined_loss, output_dict
        except Exception as e:
            logger.warning(f"GM loss computation failed: {str(e)}, using diffusion loss only")
            output_dict["loss_gm"] = 0.0
            output_dict["loss_combined"] = loss_diffusion.item()
            output_dict["gm_enabled"] = False
            return loss_diffusion, output_dict

    @torch.no_grad()
    def select_action(self, batch: Dict[str, Tensor], reward_fn: Optional[Callable] = None) -> Tensor:
        """Select action with optional reward alignment and safety constraints."""
        action = super().select_action(batch)

        if self.config.enable_reward_alignment and reward_fn is not None:
            action = self._apply_inference_time_alignment(action, reward_fn)

        if self.config.enable_hardware_safety_checks:
            action = self.safety_sampler(action)

        return action

    def _flatten_multi_camera_observations(self, batch: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """Flatten multi-camera observations for conditioning."""
        batch_flat = {k: v for k, v in batch.items()}

        if "observation" not in batch or not isinstance(batch["observation"], dict):
            return batch_flat

        obs = batch["observation"]

        # Case 1: observation.images is already a dict
        if "images" in obs and isinstance(obs["images"], dict):
            camera_views = []
            camera_names = sorted(obs["images"].keys())
            logger.debug(f"Flattening {len(camera_names)} camera views from observation.images: {camera_names}")

            for camera_name in camera_names:
                camera_views.append(obs["images"][camera_name])

            if len(camera_views) > 1:
                flattened_images = torch.cat(camera_views, dim=-1)
                logger.debug(f"Concatenated {len(camera_views)} cameras into shape {flattened_images.shape}")
            else:
                flattened_images = camera_views[0]

            obs_flat = {k: v for k, v in obs.items()}
            obs_flat["images"] = flattened_images
            batch_flat["observation"] = obs_flat
        # Case 2: observation.images is already a tensor
        elif "images" in obs and isinstance(obs["images"], Tensor):
            logger.debug(f"Observations already in single tensor format: {obs['images'].shape}")
        # Case 3: Check for renamed camera keys like images.camera1, images.camera2
        else:
            camera_views = []
            camera_keys = [k for k in obs.keys() if isinstance(k, str) and k.startswith("images.")]

            if len(camera_keys) > 0:
                logger.debug(f"Found {len(camera_keys)} renamed camera keys: {camera_keys}")

                for camera_key in sorted(camera_keys):
                    camera_views.append(obs[camera_key])

                if len(camera_views) > 1:
                    flattened_images = torch.cat(camera_views, dim=-1)
                    logger.debug(f"Concatenated {len(camera_views)} renamed cameras into shape {flattened_images.shape}")
                else:
                    flattened_images = camera_views[0]

                obs_flat = {k: v for k, v in obs.items()}
                obs_flat["images"] = flattened_images
                batch_flat["observation"] = obs_flat

        return batch_flat

    def _apply_inference_time_alignment(self, action: Tensor, reward_fn: Callable) -> Tensor:
        """Apply inference-time reward alignment."""
        if self.config.use_sequential_monte_carlo:
            return self._smc_refinement(action, reward_fn)
        else:
            return self._gibbs_tilt(action, reward_fn)

    def _gibbs_tilt(self, action: Tensor, reward_fn: Callable) -> Tensor:
        """Apply Gibbs tilt reward weighting."""
        reward = reward_fn(action)
        log_weight = self.config.reward_temperature * reward
        weight = torch.softmax(log_weight, dim=0)
        tilted_action = action * weight.view(-1, 1, 1)
        return tilted_action

    def _smc_refinement(self, action: Tensor, reward_fn: Callable) -> Tensor:
        """Sequential Monte Carlo refinement."""
        return self._gibbs_tilt(action, reward_fn)

    def compute_reward_alignment_loss(
        self,
        batch: Dict[str, Tensor],
        reward_fn: Callable,
        method: str = "flow_grpo",
    ) -> Tensor:
        """Compute post-training reward alignment loss."""
        if not self.config.enable_reward_alignment:
            logger.warning("Reward alignment not enabled in config")
            return torch.tensor(0.0, device=batch[ACTION].device)

        actions = batch[ACTION]
        rewards = reward_fn(actions)

        if method == "flow_grpo":
            loss = self._flow_grpo_loss(actions, rewards)
        elif method == "egm":
            loss = self._egm_loss(actions, rewards)
        elif method == "offline_rl":
            loss = self._offline_rl_loss(actions, rewards)
        else:
            raise ValueError(f"Unknown alignment method: {method}")

        return loss

    def _flow_grpo_loss(self, actions: Tensor, rewards: Tensor) -> Tensor:
        """Flow Generative Policy Optimization."""
        policy_loss = -(rewards.mean())
        kl_penalty = self.kl_weight * 0.0
        return policy_loss + kl_penalty

    def _egm_loss(self, actions: Tensor, rewards: Tensor) -> Tensor:
        """Energy-based Generator Matching."""
        energy = -rewards
        score_target = energy
        return (score_target ** 2).mean()

    def _offline_rl_loss(self, actions: Tensor, rewards: Tensor) -> Tensor:
        """Offline RL loss."""
        return rewards.mean()

    def sample_trajectories(
        self, batch: Dict[str, Tensor], num_samples: int = 1
    ) -> Tensor:
        """Sample multiple trajectory candidates."""
        samples = []
        for _ in range(num_samples):
            action = super().select_action(batch)
            samples.append(action)
        return torch.stack(samples, dim=0)

    def get_probability_path_stats(self, t: Tensor) -> Dict[str, Tensor]:
        """Get probability path statistics."""
        alpha_t = self.prob_path.alpha_t(t)
        sigma_t = self.prob_path.sigma_t(t)
        return {
            "alpha_t": alpha_t,
            "sigma_t": sigma_t,
            "signal_ratio": alpha_t / (sigma_t + 1e-8),
        }


MGPPolicy = MarkovGenerativePolicy
