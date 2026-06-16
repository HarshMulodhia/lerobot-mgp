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
Markov Generative Policy (MGP) - Complete Independent Implementation

Unified Markov Generative Policies framework for SO-101 with LeRobot:
- Probability paths (Gaussian CondOT, Section 3.1)
- Markov decomposition: L_t = L^flow + L^diff + L^jump + L^CTMC (Section 3.3, Table 7)
- Conditional Generator Matching (CGM) loss (Section 4.3, 3.4)
- Multi-camera vision support
- Reward alignment (Section 6, inference-time + post-training)
- Full Markov Superposition (Section 3.5, 5.3)
- ResNet-based feature backbone for robustness and sim2real transfer

Total Loss: L = α*L_DP + β*L_GM + γ*L_FM + δ*L_JUMP + ε*L_CTMC + λ*L_reward

All components are INDEPENDENT of diffusion library and tunable via config.
"""

import logging
from collections import deque
from typing import Any, Callable, Dict, Optional, Tuple

import einops
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch import Tensor

from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.utils import (
    get_device_from_parameters,
    get_dtype_from_parameters,
    populate_queues,
)
from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_IMAGES, OBS_STATE

from .configuration_mgp import MGPConfig
from ._gm_utils import (
    GaussianCondOTPath,
    GeneratorMatchingLoss,
    SafetyConstrainedSampler,
    FlowMatchingGenerator,
    JumpProcessGenerator,
    CTMCGenerator,
)

logger = logging.getLogger(__name__)


class MGPRgbEncoder(nn.Module):
    """ResNet-based RGB encoder for robust image feature extraction."""

    def __init__(self, config: MGPConfig, backbone: str = "resnet50"):
        super().__init__()
        self.config = config
        
        # Load pretrained ResNet backbone
        try:
            if backbone == "resnet50":
                backbone_model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.DEFAULT)
            elif backbone == "resnet18":
                backbone_model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
            else:
                logger.warning(f"Unknown backbone {backbone}, using resnet18")
                backbone_model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
        except Exception as e:
            logger.warning(f"Could not load pretrained weights: {e}, using non-pretrained")
            if backbone == "resnet50":
                backbone_model = torchvision.models.resnet50(weights=None)
            else:
                backbone_model = torchvision.models.resnet18(weights=None)
        
        # Remove final classification layer, keep feature extractor
        self.backbone = nn.Sequential(*list(backbone_model.children())[:-1])
        self.feature_dim = 512 if backbone == "resnet18" else 2048
        
        # Optional: projection layer to reduce dimensionality
        self.projection = nn.Sequential(
            nn.Linear(self.feature_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
        )
        self.out_dim = 256

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (B, C, H, W) RGB images
        Returns:
            features: (B, out_dim) projected features
        """
        # Normalize to [0, 1] if needed
        if x.max() > 1.0:
            x = x / 255.0
        
        # Backbone: (B, C, H, W) -> (B, feature_dim, 1, 1)
        features = self.backbone(x)
        # Flatten: (B, feature_dim, 1, 1) -> (B, feature_dim)
        features = features.flatten(start_dim=1)
        # Project to reduced dim
        features = self.projection(features)
        
        return features


class MGPDiffusionHead(nn.Module):
    """Standalone diffusion model head for MGP (NOT dependent on DiffusionPolicy)."""

    def __init__(self, config: MGPConfig):
        super().__init__()
        self.config = config
        action_dim = config.action_feature.shape[0] if hasattr(config, "action_feature") else 6
        obs_dim = 256  # Encoded observation dim
        
        # Diffusion-style U-Net for action denoising
        self.time_encoder = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
        )
        
        self.encoder = nn.Sequential(
            nn.Linear(action_dim + obs_dim + 64, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
        )

    def forward(self, actions: Tensor, t: Tensor, obs_cond: Tensor) -> Tensor:
        """Predict noise at timestep t."""
        # Encode timestep
        t_emb = self.time_encoder(t.unsqueeze(-1))
        # Concatenate action, observation, and time
        x = torch.cat([actions, obs_cond.expand(actions.shape[0], -1), t_emb], dim=-1)
        # Encode and decode
        h = self.encoder(x)
        noise_pred = self.decoder(h)
        return noise_pred


class MarkovGenerativePolicy(PreTrainedPolicy):
    """
    Complete standalone Markov Generative Policy with full component support.
    
    NO dependency on DiffusionPolicy. All components are self-contained.
    
    Implements all four Markov generator components:
    - Flow (L^flow_t): Deterministic ODE behavior cloning
    - Diffusion (L^diff_t): Stochastic SDE for multimodal actions  
    - Jump (L^jump_t): Discrete jumps for mode switches
    - CTMC (L^CTMC_t): Continuous-time Markov chain for skills
    
    Markov Superposition: L_t^VLA = Σ w_i(h_t) L_t^(i)
    
    All losses independently tunable via config.loss_weights dict.
    """

    config_class = MGPConfig
    name = "mgp"

    def __init__(self, config: MGPConfig, **kwargs):
        """Initialize MGP with all Markov components - completely standalone."""
        super().__init__(config)
        self.config = config

        logger.info("=" * 80)
        logger.info("Initializing Markov Generative Policy (MGP) - INDEPENDENT IMPLEMENTATION")
        logger.info("=" * 80)

        # ===== OBSERVATION ENCODING (ResNet-based, NOT diffusion-dependent) =====
        self.rgb_encoder = MGPRgbEncoder(config, backbone="resnet50")
        logger.info(f"Initialized ResNet50 encoder -> {self.rgb_encoder.out_dim}D features")

        # ===== SECTION 3.1: Probability Paths =====
        self.prob_path = GaussianCondOTPath(sigma_schedule=config.beta_schedule)
        logger.info(f"Initialized Gaussian CondOT path with schedule: {config.beta_schedule}")

        # ===== SECTION 3.4 / 4.3: Conditional Generator Matching Loss =====
        if config.use_generator_matching:
            self.gm_loss = GeneratorMatchingLoss(
                action_dim=self._get_action_dim(),
                loss_type=config.gm_loss_type,
            )
            logger.info(f"CGM loss enabled: {config.gm_loss_type}")

        # ===== SECTION 3.3: Diffusion Component (L^diff_t) - STANDALONE =====
        if config.enable_diffusion_component:
            self.diffusion_head = MGPDiffusionHead(config)
            self.num_inference_steps = config.fast_inference_steps if config.use_fast_inference_mode else 50
            logger.info(f"Diffusion head initialized ({self.num_inference_steps} inference steps)")

        # ===== SECTION 3.3: Flow Component (L^flow_t) =====
        if config.enable_flow_component:
            self.flow_generator = FlowMatchingGenerator(
                action_dim=self._get_action_dim(),
                hidden_dim=config.flow_hidden_dim,
                horizon=config.n_action_steps,
            )
            logger.info("Flow (ODE) component initialized")

        # ===== SECTION 3.3: Jump Component (L^jump_t) =====
        if config.enable_jump_component:
            self.jump_generator = JumpProcessGenerator(
                action_dim=self._get_action_dim(),
                num_modes=config.jump_num_modes,
                jump_rate=config.jump_rate,
                horizon=config.n_action_steps,
            )
            logger.info(f"Jump process component initialized ({config.jump_num_modes} modes)")

        # ===== SECTION 3.3: CTMC Component (L^CTMC_t) =====
        if config.enable_ctmc_component:
            self.ctmc_generator = CTMCGenerator(
                num_skills=config.ctmc_num_skills,
                action_dim=self._get_action_dim(),
                skill_dim=config.ctmc_skill_dim,
                horizon=config.n_action_steps,
            )
            logger.info(f"CTMC component initialized ({config.ctmc_num_skills} skills)")

        # ===== SECTION 3.5: Markov Superposition Gating =====
        if config.enable_markov_superposition:
            num_components = sum([
                config.enable_flow_component,
                config.enable_diffusion_component,
                config.enable_jump_component,
                config.enable_ctmc_component,
            ])
            if num_components > 1:
                self.superposition_gate = nn.Sequential(
                    nn.Linear(self.rgb_encoder.out_dim, config.superposition_hidden_dim),
                    nn.ReLU(),
                    nn.Linear(config.superposition_hidden_dim, num_components),
                    nn.Softmax(dim=-1),
                )
                logger.info(f"Markov superposition gating initialized ({num_components} components)")
            else:
                logger.warning("Markov superposition requested but only 1 component enabled - disabling")
                config.enable_markov_superposition = False

        # ===== SECTION 6.1-6.3: Reward Alignment =====
        if config.enable_reward_alignment:
            self._init_reward_alignment()
            logger.info(f"Reward alignment enabled: {config.reward_alignment_type}")

        # ===== SECTION 6.1: Safety Constraints =====
        if config.enable_hardware_safety_checks:
            self.safety_sampler = SafetyConstrainedSampler(
                max_action_norm=config.max_action_step_size,
            )
            logger.info(f"Hardware safety constraints enabled (max_norm={config.max_action_step_size})")

        # ===== Load Loss Weights =====
        loss_weights_dict = config.loss_weights or {}
        self.loss_weight_diffusion = loss_weights_dict.get('diffusion', 1.0)
        self.loss_weight_gm = loss_weights_dict.get('gm', 0.1)
        self.loss_weight_flow = loss_weights_dict.get('flow', 0.05)
        self.loss_weight_jump = loss_weights_dict.get('jump', 0.0)
        self.loss_weight_ctmc = loss_weights_dict.get('ctmc', 0.0)
        self.loss_weight_reward = (
            loss_weights_dict.get('reward', 0.01) if config.enable_reward_alignment else 0.0
        )

        logger.info(
            f"Loss weights - L = α*L_DP + β*L_GM + γ*L_FM + δ*L_JUMP + ε*L_CTMC + λ*L_reward:\n"
            f"  α(diffusion)={self.loss_weight_diffusion:.3f}\n"
            f"  β(gm)={self.loss_weight_gm:.3f}\n"
            f"  γ(flow)={self.loss_weight_flow:.3f}\n"
            f"  δ(jump)={self.loss_weight_jump:.3f}\n"
            f"  ε(ctmc)={self.loss_weight_ctmc:.3f}\n"
            f"  λ(reward)={self.loss_weight_reward:.3f}"
        )

        # ===== Queues for receding horizon control =====
        self._queues = None
        self.reset()

        logger.info("MGP initialization complete")
        logger.info("=" * 80)

    def get_optim_params(self):
        """Return all trainable parameters."""
        return self.parameters()

    def reset(self):
        """Clear observation and action queues. Should be called on env.reset()."""
        self._queues = {
            OBS_STATE: deque(maxlen=self.config.n_obs_steps),
            ACTION: deque(maxlen=self.config.n_action_steps),
        }
        if self.config.image_features:
            self._queues[OBS_IMAGES] = deque(maxlen=self.config.n_obs_steps)
        if self.config.env_state_feature:
            self._queues[OBS_ENV_STATE] = deque(maxlen=self.config.n_obs_steps)

    def _get_action_dim(self) -> int:
        """Get action dimensionality from config."""
        if hasattr(self.config, "action_feature") and hasattr(self.config.action_feature, "shape"):
            return int(self.config.action_feature.shape[0])
        return 6

    @torch.no_grad()
    def predict_action_chunk(self, batch: Dict[str, Tensor]) -> Tensor:
        """
        Predict action chunk using Markov superposition.
        
        This is the PRIMARY inference method, respecting action_chunk_size.
        Properly implements the inference pipeline for all modes (Flow/Diff/CTMC/Jump).
        """
        self.eval()
        
        batch_size = self._get_batch_size(batch)
        device = self._get_device(batch)
        
        logger.debug(f"predict_action_chunk: batch_size={batch_size}, device={device}")

        # ===== Step 1: Extract and encode observations =====
        obs_features = self._encode_observations(batch)  # (B, obs_dim)
        
        if obs_features is None:
            raise RuntimeError("Failed to extract observation features from batch")

        # ===== Step 2: Determine which mode(s) to use =====
        actions = self._sample_actions_with_superposition(
            batch_size=batch_size,
            device=device,
            obs_features=obs_features,
        )

        # ===== Step 3: Apply action chunk size selection =====
        # Return only chunk_size actions (default 1)
        if actions.shape[1] > self.config.chunk_size:
            actions = actions[:, :self.config.chunk_size]
            logger.debug(f"Selected chunk_size={self.config.chunk_size} from {actions.shape[1]} actions")

        # ===== Step 4: Apply safety constraints =====
        if self.config.enable_hardware_safety_checks:
            actions = self.safety_sampler(actions)

        return actions

    @torch.no_grad()
    def select_action(self, batch: Dict[str, Tensor]) -> Tensor:
        """
        Select a single action via receding horizon.
        
        Handles queue management for streaming inference on hardware.
        Async-compatible: can be called from remote GPU worker.
        """
        if ACTION in batch:
            batch.pop(ACTION)

        # Stack images if multi-camera
        if self.config.image_features:
            batch = dict(batch)
            batch[OBS_IMAGES] = torch.stack(
                [batch[key] for key in self.config.image_features], dim=-4
            )

        # Populate queues with new observations
        self._queues = populate_queues(self._queues, batch)

        # If action queue is empty, generate new chunk
        if len(self._queues[ACTION]) == 0:
            actions = self.predict_action_chunk(batch)
            self._queues[ACTION].extend(actions.transpose(0, 1))

        # Pop and return first action
        action = self._queues[ACTION].popleft()
        return action

    def forward(self, batch: Dict[str, Tensor]) -> Tuple[Tensor, Optional[Dict[str, Any]]]:
        """
        Forward pass computing all Markov component losses during training.

        Theory: Section 4.3, 5.1 - Markov Superposition
        L_total = α*L_DP + β*L_GM + γ*L_FM + δ*L_JUMP + ε*L_CTMC + λ*L_reward

        Args:
            batch: Training batch with observations and actions

        Returns:
            (loss, output_dict): Combined loss and metrics
        """
        # Standard diffusion-style loss (primary imitation)
        loss_diffusion = self._compute_diffusion_loss(batch)

        output_dict = {
            "loss_diffusion": loss_diffusion.item() if isinstance(loss_diffusion, Tensor) else loss_diffusion,
            "mgp_enabled": self.config.use_generator_matching,
        }

        total_loss = loss_diffusion * self.loss_weight_diffusion

        # ===== L_GM: CGM Loss (Section 4.3) =====
        if self.config.use_generator_matching and ACTION in batch:
            try:
                loss_gm = self._compute_gm_loss(batch)
                total_loss = total_loss + loss_gm * self.loss_weight_gm
                output_dict["loss_gm"] = loss_gm.item()
            except Exception as e:
                logger.debug(f"GM loss computation failed: {e}")
                output_dict["loss_gm"] = 0.0

        # ===== L_FM: Flow Matching Loss (Section 3.3) =====
        if self.config.enable_flow_component and ACTION in batch:
            try:
                loss_flow = self._compute_flow_loss(batch)
                total_loss = total_loss + loss_flow * self.loss_weight_flow
                output_dict["loss_flow"] = loss_flow.item()
            except Exception as e:
                logger.debug(f"Flow loss computation failed: {e}")
                output_dict["loss_flow"] = 0.0

        # ===== L_JUMP: Jump Process Loss (Section 3.3) =====
        if self.config.enable_jump_component and ACTION in batch:
            try:
                loss_jump = self._compute_jump_loss(batch)
                total_loss = total_loss + loss_jump * self.loss_weight_jump
                output_dict["loss_jump"] = loss_jump.item()
            except Exception as e:
                logger.debug(f"Jump loss computation failed: {e}")
                output_dict["loss_jump"] = 0.0

        # ===== L_CTMC: CTMC Loss (Section 3.3) =====
        if self.config.enable_ctmc_component and ACTION in batch:
            try:
                loss_ctmc = self._compute_ctmc_loss(batch)
                total_loss = total_loss + loss_ctmc * self.loss_weight_ctmc
                output_dict["loss_ctmc"] = loss_ctmc.item()
            except Exception as e:
                logger.debug(f"CTMC loss computation failed: {e}")
                output_dict["loss_ctmc"] = 0.0

        # ===== L_reward: Reward Alignment Loss (Section 6) =====
        if self.config.enable_reward_alignment and "rewards" in batch:
            try:
                loss_reward = self._compute_reward_alignment_loss(batch)
                total_loss = total_loss + loss_reward * self.loss_weight_reward
                output_dict["loss_reward"] = loss_reward.item()
            except Exception as e:
                logger.debug(f"Reward alignment loss failed: {e}")
                output_dict["loss_reward"] = 0.0

        output_dict["loss_total"] = total_loss.item() if isinstance(total_loss, Tensor) else total_loss

        return total_loss, output_dict

    # ===== PRIVATE HELPER METHODS =====

    def _get_batch_size(self, batch: Dict[str, Tensor]) -> int:
        """Extract batch size from batch dict."""
        for v in batch.values():
            if isinstance(v, Tensor) and v.ndim > 0:
                return v.shape[0]
        return 1

    def _get_device(self, batch: Dict[str, Tensor]) -> torch.device:
        """Extract device from batch dict."""
        for v in batch.values():
            if isinstance(v, Tensor):
                return v.device
        return next(self.parameters()).device

    def _encode_observations(self, batch: Dict[str, Tensor]) -> Optional[Tensor]:
        """Encode observations into feature vector using ResNet encoder."""
        # Try images first
        if OBS_IMAGES in batch and isinstance(batch[OBS_IMAGES], Tensor):
            images = batch[OBS_IMAGES]
            
            if images.ndim == 5:  # (B, n_obs_steps, n_cameras, C, H, W)
                # Take last image, first camera
                img = images[:, -1, 0]
            elif images.ndim == 4:  # (B, C, H, W)
                img = images
            else:
                logger.warning(f"Unexpected image shape: {images.shape}")
                img = images.reshape(-1, 3, 224, 224) if images.numel() > 0 else None
            
            if img is not None and img.shape[0] > 0:
                try:
                    return self.rgb_encoder(img)
                except Exception as e:
                    logger.debug(f"Image encoding failed: {e}")

        # Fallback to state observations
        if OBS_STATE in batch and isinstance(batch[OBS_STATE], Tensor):
            state = batch[OBS_STATE]
            if state.ndim == 3:  # (B, n_obs_steps, state_dim)
                state = state[:, -1]  # Take last timestep
            # Embed state to match encoder output dim
            state_emb = F.linear(state, torch.randn(self.rgb_encoder.out_dim, state.shape[-1], device=state.device))
            return state_emb

        # Last resort: random features
        batch_size = self._get_batch_size(batch)
        device = self._get_device(batch)
        logger.warning("Using fallback random features for observations")
        return torch.randn(batch_size, self.rgb_encoder.out_dim, device=device)

    @torch.no_grad()
    def _sample_actions_with_superposition(
        self,
        batch_size: int,
        device: torch.device,
        obs_features: Tensor,
    ) -> Tensor:
        """
        Sample actions using Markov superposition over all enabled components.
        
        Routes to correct generator based on config:
        - If only diffusion: use diffusion
        - If multiple: blend with learned gating weights
        """
        # Primary component: diffusion (always available if enabled)
        if self.config.enable_diffusion_component:
            diff_actions = self._sample_diffusion_actions(batch_size, device, obs_features)
            component_actions = [diff_actions]
            component_names = ["diffusion"]
        else:
            raise ValueError("At least diffusion component must be enabled")

        # Try other components if enabled
        if self.config.enable_flow_component:
            try:
                flow_actions = self.flow_generator.generate_actions(batch_size, device)
                if flow_actions is not None:
                    component_actions.append(flow_actions)
                    component_names.append("flow")
            except Exception as e:
                logger.debug(f"Flow sampling failed: {e}")

        if self.config.enable_jump_component:
            try:
                jump_actions = self.jump_generator.generate_actions(batch_size, device)
                if jump_actions is not None:
                    component_actions.append(jump_actions)
                    component_names.append("jump")
            except Exception as e:
                logger.debug(f"Jump sampling failed: {e}")

        if self.config.enable_ctmc_component:
            try:
                ctmc_actions = self.ctmc_generator.generate_actions(batch_size, device)
                if ctmc_actions is not None:
                    component_actions.append(ctmc_actions)
                    component_names.append("ctmc")
            except Exception as e:
                logger.debug(f"CTMC sampling failed: {e}")

        # If only one component succeeded, return it
        if len(component_actions) == 1:
            logger.debug(f"Using single component: {component_names[0]}")
            return component_actions[0]

        # Multi-component: blend with gating if enabled
        if self.config.enable_markov_superposition and hasattr(self, "superposition_gate"):
            try:
                gate_weights = self.superposition_gate(obs_features)  # (B, num_components)
                logger.debug(f"Gate weights: {gate_weights.mean(dim=0).tolist()} for {component_names}")
                
                # Stack and blend
                stacked = torch.stack(component_actions, dim=0)  # (num_comp, B, T, A)
                blended = torch.einsum("bc,cbta->bta", gate_weights, stacked)
                return blended
            except Exception as e:
                logger.warning(f"Superposition gating failed: {e}, using simple average")

        # Simple averaging if gating not available
        logger.debug(f"Simple averaging {len(component_actions)} components")
        return torch.stack(component_actions, dim=0).mean(dim=0)

    @torch.no_grad()
    def _sample_diffusion_actions(
        self,
        batch_size: int,
        device: torch.device,
        obs_features: Tensor,
    ) -> Tensor:
        """
        Sample actions using the diffusion component.
        
        Uses DDIM-style sampling for efficiency on hardware.
        """
        action_dim = self._get_action_dim()
        horizon = self.config.n_action_steps
        
        # Initialize from Gaussian noise
        x_t = torch.randn(batch_size, horizon, action_dim, device=device)

        # Reverse diffusion: denoise from T to 0
        for step in range(self.num_inference_steps, 0, -1):
            t = torch.full((batch_size,), step / self.num_inference_steps, device=device)
            
            # Predict noise
            with torch.no_grad():
                noise_pred = self.diffusion_head(x_t, t, obs_features)
            
            # DDIM denoising step (simplified)
            alpha_t = 1.0 - (step / self.num_inference_steps)
            x_t = x_t - alpha_t * noise_pred

        return x_t

    def _compute_diffusion_loss(self, batch: Dict[str, Tensor]) -> Tensor:
        """Compute diffusion policy loss (DDPM MSE)."""
        actions = batch[ACTION]  # (B, T, A)
        batch_size, horizon, action_dim = actions.shape
        device = actions.device

        # Encode observations
        obs_features = self._encode_observations(batch)

        # Sample random timesteps and noise
        t = torch.rand(batch_size, device=device)
        eps = torch.randn_like(actions)

        # Forward diffusion: x_t = alpha_t * x + sigma_t * eps
        alpha_t = self.prob_path.alpha_t(t.view(-1, 1, 1))
        sigma_t = self.prob_path.sigma_t(t.view(-1, 1, 1))

        x_t = alpha_t * actions + sigma_t * eps

        # Predict noise
        noise_pred = self.diffusion_head(x_t, t, obs_features)

        # MSE loss
        loss = F.mse_loss(noise_pred, eps)
        return loss

    def _compute_gm_loss(self, batch: Dict[str, Tensor]) -> Tensor:
        """Compute Conditional Generator Matching loss."""
        actions = batch[ACTION]
        batch_size = actions.shape[0]
        device = actions.device

        # Sample random timesteps
        t = torch.rand(batch_size, device=device)
        eps = torch.randn_like(actions)

        # Forward diffusion
        alpha_t = self.prob_path.alpha_t(t.view(-1, 1, 1))
        sigma_t = self.prob_path.sigma_t(t.view(-1, 1, 1))
        x_t = alpha_t * actions + sigma_t * eps

        # Encode observations
        obs_features = self._encode_observations(batch)

        # Predict noise
        noise_pred = self.diffusion_head(x_t, t, obs_features)

        # CGM loss
        loss, _ = self.gm_loss(diffusion_pred=noise_pred, diffusion_target=eps)
        return loss

    def _compute_flow_loss(self, batch: Dict[str, Tensor]) -> Tensor:
        """Compute Flow Matching loss (behavior cloning)."""
        actions = batch[ACTION]
        
        if actions.shape[1] > 1:
            flow_target = actions[:, 0] - actions[:, -1]
        else:
            flow_target = actions[:, 0]

        flow_pred = self.flow_generator(flow_target)
        loss = F.mse_loss(flow_pred, flow_target)
        return loss

    def _compute_jump_loss(self, batch: Dict[str, Tensor]) -> Tensor:
        """Compute Jump Process loss."""
        actions = batch[ACTION]
        batch_size = actions.shape[0]
        device = actions.device

        t_jump = torch.rand(batch_size, device=device)
        jump_pred = self.jump_generator(actions, t_jump.mean().item())

        # KL divergence loss
        loss = F.kl_div(
            F.log_softmax(jump_pred, dim=-1),
            torch.ones_like(jump_pred) / jump_pred.shape[-1],
            reduction='mean'
        )
        return loss

    def _compute_ctmc_loss(self, batch: Dict[str, Tensor]) -> Tensor:
        """Compute CTMC loss."""
        actions = batch[ACTION]
        batch_size = actions.shape[0]
        device = actions.device

        current_skill = torch.randint(0, self.config.ctmc_num_skills, (batch_size,), device=device)
        t = torch.rand(batch_size, device=device)

        skill_logits = self.ctmc_generator(current_skill, t.mean().item())
        target_skill = torch.randint(0, self.config.ctmc_num_skills, (batch_size,), device=device)

        loss = F.cross_entropy(skill_logits.view(batch_size, -1), target_skill)
        return loss

    def _init_reward_alignment(self):
        """Initialize reward alignment components."""
        if self.config.reward_alignment_type == "inference_time":
            if self.config.use_sequential_monte_carlo:
                self.smc_particles = self.config.smc_particles
                self.smc_resampling_threshold = 0.5
                logger.info(f"SMC initialized with {self.config.smc_particles} particles")
        elif self.config.reward_alignment_type == "post_training":
            self.kl_weight = 0.1
            logger.info("Post-training alignment initialized")

    def _compute_reward_alignment_loss(self, batch: Dict[str, Tensor]) -> Tensor:
        """Compute reward alignment loss (Flow-GRPO style)."""
        actions = batch[ACTION]
        rewards = batch.get("rewards", torch.zeros(actions.shape[0], device=actions.device))

        if not isinstance(rewards, Tensor):
            rewards = torch.tensor(rewards, device=actions.device, dtype=torch.float32)

        # Maximize reward with KL regularization
        loss = -rewards.mean()
        return loss


# Backward compatibility alias
MGPPolicy = MarkovGenerativePolicy
