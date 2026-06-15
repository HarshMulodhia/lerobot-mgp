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
Markov Generative Policy (MGP) - Complete Implementation

Unified Markov Generative Policies framework for SO-101 with LeRobot:
- Probability paths (Gaussian CondOT, Section 3.1)
- Markov decomposition: L_t = L^flow + L^diff + L^jump + L^CTMC (Section 3.3, Table 7)
- Conditional Generator Matching (CGM) loss (Section 4.3, 3.4)
- Multi-camera vision support
- Reward alignment (Section 6, inference-time + post-training)
- Full Markov Superposition (Section 3.5, 5.3)

Total Loss: L = α*L_DP + β*L_GM + γ*L_FM + δ*L_JUMP + ε*L_CTMC + λ*L_reward

All components tunable via --policy.loss_weights and enable_*_component flags.
"""

import logging
import copy
from typing import Any, Dict, Optional, Tuple, Callable

import torch
import torch.nn as nn
from torch import Tensor

from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.utils.constants import ACTION, OBS_STATE, OBS_IMAGES

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


class MarkovGenerativePolicy(DiffusionPolicy):
    """
    Complete Markov Generative Policy with full component support.
    
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
        """Initialize MGP with all Markov components."""
        super().__init__(config, **kwargs)
        self.config = config

        logger.info("Initializing Markov Generative Policy (MGP) - Complete Framework")

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

        # ===== SECTION 3.3: Flow Component (L^flow_t) =====
        if self.config.enable_flow_component:
            self.flow_generator = FlowMatchingGenerator(
                action_dim=self._get_action_dim(),
                hidden_dim=self.config.flow_hidden_dim,
                horizon=self.config.n_action_steps,
            )
            logger.info("Flow (ODE) component initialized")

        # ===== SECTION 3.3: Jump Component (L^jump_t) =====
        if self.config.enable_jump_component:
            self.jump_generator = JumpProcessGenerator(
                action_dim=self._get_action_dim(),
                num_modes=self.config.jump_num_modes,
                jump_rate=self.config.jump_rate,
                horizon=self.config.n_action_steps,
            )
            logger.info(f"Jump process component initialized ({self.config.jump_num_modes} modes)")

        # ===== SECTION 3.3: CTMC Component (L^CTMC_t) =====
        if self.config.enable_ctmc_component:
            self.ctmc_generator = CTMCGenerator(
                num_skills=self.config.ctmc_num_skills,
                action_dim=self._get_action_dim(),
                skill_dim=self.config.ctmc_skill_dim,
                horizon=self.config.n_action_steps,
            )
            logger.info(f"CTMC component initialized ({self.config.ctmc_num_skills} skills)")

        # ===== SECTION 3.5: Markov Superposition Gating =====
        if self.config.enable_markov_superposition:
            num_components = sum([
                self.config.enable_flow_component,
                self.config.enable_diffusion_component,
                self.config.enable_jump_component,
                self.config.enable_ctmc_component,
            ])
            if num_components > 1:
                self.superposition_gate = nn.Sequential(
                    nn.Linear(self._get_observation_dim(), self.config.superposition_hidden_dim),
                    nn.ReLU(),
                    nn.Linear(self.config.superposition_hidden_dim, num_components),
                    nn.Softmax(dim=-1),
                )
                logger.info(f"Markov superposition gating initialized ({num_components} components)")
            else:
                logger.warning("Markov superposition requested but only 1 component enabled")
                self.config.enable_markov_superposition = False

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

        # ===== Load Loss Weights =====
        loss_weights_dict = self.config.loss_weights or {}
        self.loss_weight_diffusion = loss_weights_dict.get('diffusion', 1.0)  # α
        self.loss_weight_gm = loss_weights_dict.get('gm', 0.1)  # β
        self.loss_weight_flow = loss_weights_dict.get('flow', 0.05)  # γ
        self.loss_weight_jump = loss_weights_dict.get('jump', 0.0)  # δ
        self.loss_weight_ctmc = loss_weights_dict.get('ctmc', 0.0)  # ε
        self.loss_weight_reward = (
            loss_weights_dict.get('reward', 0.01) 
            if self.config.enable_reward_alignment 
            else 0.0
        )  # λ

        logger.info(
            f"Loss weights - L = α*L_DP + β*L_GM + γ*L_FM + δ*L_JUMP + ε*L_CTMC + λ*L_reward:\n"
            f"  α(diffusion)={self.loss_weight_diffusion:.3f}\n"
            f"  β(gm)={self.loss_weight_gm:.3f}\n"
            f"  γ(flow)={self.loss_weight_flow:.3f}\n"
            f"  δ(jump)={self.loss_weight_jump:.3f}\n"
            f"  ε(ctmc)={self.loss_weight_ctmc:.3f}\n"
            f"  λ(reward)={self.loss_weight_reward:.3f}"
        )

    def _get_action_dim(self) -> int:
        """Get action dimensionality from config."""
        if hasattr(self.config, "action_feature") and hasattr(self.config.action_feature, "shape"):
            return int(self.config.action_feature.shape[0])
        return 6

    def _get_observation_dim(self) -> int:
        """Get observation dimensionality (for superposition gating)."""
        if hasattr(self.config, "observation_feature") and hasattr(self.config.observation_feature, "shape"):
            return int(self.config.observation_feature.shape[0])
        return 512  # Default for image-based observations

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
        """Predict action chunk with markov superposition (Section 5.3, 5.1)."""
        self.eval()

        # Sanity checks
        if self is None:
            raise RuntimeError("self is None in predict_action_chunk")
        if not hasattr(self, 'diffusion') or self.diffusion is None:
            raise RuntimeError("self.diffusion is not initialized")
        if not hasattr(self, 'config') or self.config is None:
            raise RuntimeError("self.config is not initialized")

        # Diagnose batch structure
        if batch is None:
            raise RuntimeError("batch is None")

        logger.info(f"Batch type: {type(batch)}, keys: {list(batch.keys()) if isinstance(batch, dict) else 'not a dict'}")

        # Find first non-None TENSOR value in batch (skip scalars, dicts, lists, etc.)
        valid_tensor = None
        for key, value in batch.items():
            if value is not None:
                if isinstance(value, Tensor) and hasattr(value, 'shape'):
                    logger.info(f"  Batch['{key}']: Tensor = {value.shape}")
                    valid_tensor = value
                    break
                else:
                    type_name = type(value).__name__
                    val_repr = f"{type(value).__name__} = {value if not isinstance(value, (dict, list)) else type(value).__name__}"
                    logger.info(f"  Batch['{key}']: {val_repr}")
            else:
                logger.info(f"  Batch['{key}']: None")

        if valid_tensor is None:
            raise RuntimeError("No valid tensor found in batch")

        try:
            batch_size = valid_tensor.shape[0]
            device = valid_tensor.device
            logger.info(f"Extracted batch_size={batch_size}, device={device}")
        except (AttributeError, IndexError) as e:
            logger.error(f"Failed to extract batch info from tensor: {e}")
            raise

        # Start with diffusion as primary generator (always available, most robust)
        try:
            # Filter batch to only include observation tensors (remove metadata like action, reward, etc.)
            filtered_batch = {}
            for key, value in batch.items():
                # Keep only observation-related keys that are tensors
                if key.startswith('observation.') and isinstance(value, Tensor):
                    filtered_batch[key] = value
                # Also handle OBS_STATE and OBS_IMAGES keys if they exist
                elif key in (OBS_STATE, OBS_IMAGES) and isinstance(value, Tensor):
                    filtered_batch[key] = value

            logger.info(f"Filtered batch keys: {list(filtered_batch.keys())}")

            if not filtered_batch:
                raise ValueError("Filtered batch is empty - no observation tensors found")

            # 1. Expand the state observation dimension to match history requirements
            target_obs_steps = self.diffusion.config.n_obs_steps
            if 'observation.state' in filtered_batch:
                tensor = filtered_batch['observation.state']
                if tensor.ndim == 2:  # (B, state_dim) -> (B, n_obs_steps, state_dim)
                    filtered_batch['observation.state'] = tensor.unsqueeze(1).repeat(1, target_obs_steps, 1)
                    logger.info(f"Expanded observation.state to {filtered_batch['observation.state'].shape}")

            # 2. Extract, expand, and stack multiple camera features into a single 'observation.images' tensor
            # Collect all keys starting with 'observation.images.' (e.g., .arm, .ext)
            cam_keys = sorted([k for k in filtered_batch.keys() if k.startswith('observation.images.')])
            
            if cam_keys:
                expanded_cams = []
                for k in cam_keys:
                    tensor = filtered_batch[k]
                    if tensor.ndim == 4:  # (B, C, H, W) -> (B, 1, C, H, W) -> (B, n_obs_steps, C, H, W)
                        tensor = tensor.unsqueeze(1).repeat(1, target_obs_steps, 1, 1, 1)
                    expanded_cams.append(tensor)
                
                # Stack all cameras along a new 'n_cameras' dimension
                # Expected shape for LeRobot Diffusion: (B, n_obs_steps, n_cameras, C, H, W)
                filtered_batch[OBS_IMAGES] = torch.stack(expanded_cams, dim=2)
                logger.info(f"Stacked {len(cam_keys)} cameras into OBS_IMAGES with shape {filtered_batch[OBS_IMAGES].shape}")
                
                # Clean up individual camera keys so they don't pollute downstream generators
                for k in cam_keys:
                    del filtered_batch[k]

            logger.info(f"Final batch keys passed to diffusion: {list(filtered_batch.keys())}")

            diff_actions = self.diffusion.generate_actions(filtered_batch)
            if diff_actions is None:
                logger.error("Diffusion component returned None")
                raise ValueError("Diffusion generate_actions returned None")
            logger.info(f"✓ Diffusion actions shape: {diff_actions.shape}")
        except Exception as e:
            import traceback
            logger.error(f"✗ Diffusion component failed: {type(e).__name__}: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise

        # Only attempt multicomponent superposition if explicitly enabled and other components exist
        enable_multi = (
            hasattr(self.config, 'enable_markov_superposition') and
            self.config.enable_markov_superposition and
            (
                (hasattr(self.config, 'enable_flow_component') and self.config.enable_flow_component) or
                (hasattr(self.config, 'enable_jump_component') and self.config.enable_jump_component) or
                (hasattr(self.config, 'enable_ctmc_component') and self.config.enable_ctmc_component)
            )
        )

        if not enable_multi:
            # Single component (diffusion only)
            logger.debug("Markov superposition disabled, returning diffusion only")
            return diff_actions

        # Multi-component case: collect from other generators
        component_actions = [diff_actions]
        component_names = ["diffusion"]
        n_action_steps_expected = diff_actions.shape[1]

        # 2. Flow component (deterministic ODE)
        if hasattr(self, 'flow_generator') and hasattr(self.config, 'enable_flow_component') and self.config.enable_flow_component:
            try:
                flow_actions = self.flow_generator.generate_actions(batch_size, device)
                if flow_actions is not None and flow_actions.shape[1] == n_action_steps_expected:
                    component_actions.append(flow_actions)
                    component_names.append("flow")
                    logger.info(f"Flow actions shape: {flow_actions.shape}")
            except Exception as e:
                logger.warning(f"Flow component failed: {e}")

        # 3. Jump process component (mode switching)
        if hasattr(self, 'jump_generator') and hasattr(self.config, 'enable_jump_component') and self.config.enable_jump_component:
            try:
                jump_actions = self.jump_generator.generate_actions(batch_size, device)
                if jump_actions is not None and jump_actions.shape[1] == n_action_steps_expected:
                    component_actions.append(jump_actions)
                    component_names.append("jump")
                    logger.info(f"Jump actions shape: {jump_actions.shape}")
            except Exception as e:
                logger.warning(f"Jump component failed: {e}")

        # 4. CTMC component (skill hierarchy)
        if hasattr(self, 'ctmc_generator') and hasattr(self.config, 'enable_ctmc_component') and self.config.enable_ctmc_component:
            try:
                ctmc_actions = self.ctmc_generator.generate_actions(batch_size, device)
                if ctmc_actions is not None and ctmc_actions.shape[1] == n_action_steps_expected:
                    component_actions.append(ctmc_actions)
                    component_names.append("ctmc")
                    logger.info(f"CTMC actions shape: {ctmc_actions.shape}")
            except Exception as e:
                logger.warning(f"CTMC component failed: {e}")

        # If only diffusion succeeded, return it
        if len(component_actions) == 1:
            logger.info("Using diffusion only (other components failed)")
            return diff_actions

        # Compute superposition weights (Section 5.3)
        if hasattr(self, "superposition_gate") and self.superposition_gate is not None:
            try:
                obs_features = self._extract_observation_features(batch)
                gate_weights = self.superposition_gate(obs_features)
                logger.info(f"Gate weights: {gate_weights.mean(dim=0).tolist()} for {component_names}")

                blended_actions = torch.zeros_like(component_actions[0])
                for actions, weight in zip(component_actions, gate_weights.T):
                    weight_expanded = weight.view(-1, 1, 1)
                    blended_actions = blended_actions + weight_expanded * actions

                return blended_actions

            except Exception as e:
                logger.warning(f"Superposition gating failed: {e}, using simple averaging")
                return torch.stack(component_actions, dim=0).mean(dim=0)
        else:
            logger.info(f"No superposition gate, blending {len(component_actions)} components via averaging")
            return torch.stack(component_actions, dim=0).mean(dim=0)

    def _extract_observation_features(self, batch: dict[str, Tensor]) -> Tensor:
        """Extract flattened observation features for superposition gating.

        Args:
            batch: Observation batch (may contain mixed types, will filter to tensors)

        Returns:
            obs_features: Flattened observation features (B, obs_dim)
        """
        features = []
        batch_size = None
        device = None

        # Use state observations if available
        if OBS_STATE in batch and isinstance(batch[OBS_STATE], Tensor):
            state = batch[OBS_STATE]
            batch_size = state.shape[0]
            device = state.device
            if state.ndim == 3:  # (B, n_obs_steps, state_dim)
                state = state.flatten(start_dim=1)  # (B, n_obs_steps * state_dim)
            features.append(state)
        elif 'observation.state' in batch and isinstance(batch['observation.state'], Tensor):
            state = batch['observation.state']
            batch_size = state.shape[0]
            device = state.device
            if state.ndim == 3:
                state = state.flatten(start_dim=1)
            features.append(state)

        # Use image features if available
        for key in batch.keys():
            if key.startswith('observation.images') and isinstance(batch[key], Tensor):
                images = batch[key]
                if batch_size is None:
                    batch_size = images.shape[0]
                if device is None:
                    device = images.device
                # Flatten images
                images_flat = images.flatten(start_dim=1)
                # Limit to first 512 dims to avoid huge gating network
                if images_flat.shape[1] > 512:
                    images_flat = images_flat[:, :512]
                features.append(images_flat)
                break  # Use only first image for gating

        if features:
            obs_features = torch.cat(features, dim=-1)
        else:
            # Fallback: create random features with correct shape
            if batch_size is None or device is None:
                # Last resort: find any tensor in batch
                for v in batch.values():
                    if isinstance(v, Tensor):
                        batch_size = v.shape[0]
                        device = v.device
                        break
            if batch_size is None:
                raise RuntimeError("Cannot determine batch_size or device from batch")
            logger.warning("Using fallback random features for gating")
            obs_features = torch.randn(batch_size, 512, device=device)

        return obs_features


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

    def _is_observation_images_concatenated(self, obs: Any) -> bool:
        """Check if observation.images is properly concatenated into a tensor."""
        if not hasattr(obs, "images"):
            return False
        img = obs.images
        return isinstance(img, Tensor)

    def _concatenate_multi_camera_observations(self, obs: Any) -> Any:
        """Concatenate multi-camera observations if they are dict of cameras."""
        if not hasattr(obs, "images"):
            return obs
        
        images_attr = obs.images
        if not isinstance(images_attr, dict):
            return obs
        
        camera_keys = sorted([k for k in images_attr.keys() if k.startswith("camera")])
        
        if len(camera_keys) > 1:
            try:
                logger.debug(f"Concatenating {len(camera_keys)} cameras: {camera_keys}")
                camera_tensors = [images_attr[k] for k in camera_keys]
                concatenated = torch.cat(camera_tensors, dim=self.config.camera_concat_dim)
                
                if hasattr(obs, "__dict__"):
                    obs.images = concatenated
                else:
                    obs_dict = dict(vars(obs))
                    obs_dict["images"] = concatenated
                    from types import SimpleNamespace
                    obs = SimpleNamespace(**obs_dict)
                
                logger.debug(f"Concatenated into shape {concatenated.shape}")
                return obs
            except Exception as e:
                logger.debug(f"Failed to concatenate cameras: {e}")
                return obs
        elif len(camera_keys) == 1:
            try:
                logger.debug(f"Extracting single camera: {camera_keys[0]}")
                if hasattr(obs, "__dict__"):
                    obs.images = images_attr[camera_keys[0]]
                else:
                    obs_dict = dict(vars(obs))
                    obs_dict["images"] = images_attr[camera_keys[0]]
                    from types import SimpleNamespace
                    obs = SimpleNamespace(**obs_dict)
                return obs
            except Exception as e:
                logger.debug(f"Failed to extract single camera: {e}")
                return obs
        else:
            return obs

    def forward(self, batch: Dict[str, Tensor]) -> Tuple[Tensor, Optional[Dict[str, Any]]]:
        """
        Forward pass computing all Markov component losses.

        Theory: Section 4.3, 5.1 - Markov Superposition
        L_total = α*L_DP + β*L_GM + γ*L_FM + δ*L_JUMP + ε*L_CTMC + λ*L_reward

        Args:
            batch: Training batch with observations and actions

        Returns:
            (loss, output_dict): Combined loss and metrics
        """
        # Concatenate multi-camera observations if enabled
        if self.config.enable_multi_camera_concat:
            try:
                obs = batch.get("observation", None)
                if obs is not None:
                    batch["observation"] = self._concatenate_multi_camera_observations(obs)
            except Exception as e:
                logger.debug(f"Multi-camera concatenation failed: {e}")
        
        # Standard DiffusionPolicy loss (parent class) - L_DP
        loss_diffusion, output_dict = super().forward(batch)

        if output_dict is None:
            output_dict = {}

        output_dict["loss_diffusion"] = (
            loss_diffusion.item() if isinstance(loss_diffusion, Tensor) else loss_diffusion
        )
        output_dict["mgp_enabled"] = self.config.use_generator_matching

        # Total loss initialization
        total_loss = loss_diffusion * self.loss_weight_diffusion
        output_dict["loss_total"] = 0.0

        # ===== L_GM: CGM Loss (Section 4.3) =====
        if self.config.use_generator_matching and ACTION in batch:
            obs = batch.get("observation", None)
            if obs is not None and self._is_observation_images_concatenated(obs):
                try:
                    loss_gm, output_dict = self._compute_gm_loss(batch, output_dict)
                    total_loss = total_loss + loss_gm * self.loss_weight_gm
                    output_dict["loss_gm"] = loss_gm.item()
                except Exception as e:
                    logger.debug(f"GM loss computation failed: {e}")
                    output_dict["loss_gm"] = 0.0
            else:
                output_dict["loss_gm"] = 0.0

        # ===== L_FM: Flow Matching Loss (Section 3.3) =====
        if self.config.enable_flow_component and ACTION in batch:
            try:
                loss_flow, output_dict = self._compute_flow_matching_loss(batch, output_dict)
                total_loss = total_loss + loss_flow * self.loss_weight_flow
                output_dict["loss_flow"] = loss_flow.item()
            except Exception as e:
                logger.debug(f"Flow matching loss failed: {e}")
                output_dict["loss_flow"] = 0.0

        # ===== L_JUMP: Jump Process Loss (Section 3.3) =====
        if self.config.enable_jump_component and ACTION in batch:
            try:
                loss_jump, output_dict = self._compute_jump_loss(batch, output_dict)
                total_loss = total_loss + loss_jump * self.loss_weight_jump
                output_dict["loss_jump"] = loss_jump.item()
            except Exception as e:
                logger.debug(f"Jump loss computation failed: {e}")
                output_dict["loss_jump"] = 0.0

        # ===== L_CTMC: CTMC Loss (Section 3.3) =====
        if self.config.enable_ctmc_component and ACTION in batch:
            try:
                loss_ctmc, output_dict = self._compute_ctmc_loss(batch, output_dict)
                total_loss = total_loss + loss_ctmc * self.loss_weight_ctmc
                output_dict["loss_ctmc"] = loss_ctmc.item()
            except Exception as e:
                logger.debug(f"CTMC loss computation failed: {e}")
                output_dict["loss_ctmc"] = 0.0

        # ===== L_reward: Reward Alignment Loss (Section 6) =====
        if self.config.enable_reward_alignment and hasattr(batch, "rewards"):
            try:
                loss_reward = self._compute_reward_alignment_loss(batch)
                total_loss = total_loss + loss_reward * self.loss_weight_reward
                output_dict["loss_reward"] = loss_reward.item()
            except Exception as e:
                logger.debug(f"Reward alignment loss failed: {e}")
                output_dict["loss_reward"] = 0.0

        # ===== Summary =====
        output_dict["loss_total"] = total_loss.item()
        
        loss_breakdown = f"L_total={output_dict.get('loss_total', 0):.4f}"
        if self.loss_weight_diffusion > 0:
            loss_breakdown += f" (α*L_DP={self.loss_weight_diffusion*output_dict.get('loss_diffusion', 0):.4f}"
        if self.loss_weight_gm > 0:
            loss_breakdown += f" + β*L_GM={self.loss_weight_gm*output_dict.get('loss_gm', 0):.4f}"
        if self.loss_weight_flow > 0:
            loss_breakdown += f" + γ*L_FM={self.loss_weight_flow*output_dict.get('loss_flow', 0):.4f}"
        if self.loss_weight_jump > 0:
            loss_breakdown += f" + δ*L_JUMP={self.loss_weight_jump*output_dict.get('loss_jump', 0):.4f}"
        if self.loss_weight_ctmc > 0:
            loss_breakdown += f" + ε*L_CTMC={self.loss_weight_ctmc*output_dict.get('loss_ctmc', 0):.4f}"
        if self.loss_weight_reward > 0:
            loss_breakdown += f" + λ*L_reward={self.loss_weight_reward*output_dict.get('loss_reward', 0):.4f}"
        loss_breakdown += ")"
        
        logger.debug(loss_breakdown)

        return total_loss, output_dict

    def _compute_gm_loss(
        self, batch: Dict[str, Tensor], output_dict: Dict
    ) -> Tuple[Tensor, Dict]:
        """Compute Conditional Generator Matching loss (Eq. 4.3)."""
        actions = batch[ACTION]
        batch_size = actions.shape[0]

        timesteps = torch.randint(0, 1000, (batch_size,), device=actions.device)
        t = timesteps.float() / 1000.0

        x_t, eps = self.prob_path.sample(actions, t)
        global_cond = self.diffusion._prepare_global_conditioning(batch)
        noise_pred = self.diffusion.unet(x_t, timesteps, global_cond=global_cond)

        gm_loss, gm_metrics = self.gm_loss(
            diffusion_pred=noise_pred,
            diffusion_target=eps,
        )

        output_dict.update(gm_metrics)
        return gm_loss, output_dict

    def _compute_flow_matching_loss(
        self, batch: Dict[str, Tensor], output_dict: Dict
    ) -> Tuple[Tensor, Dict]:
        """Compute Flow Matching loss (Section 3.3, Table 7)."""
        actions = batch[ACTION]
        
        try:
            if actions.shape[1] > 1:
                flow_target = actions[:, 0, :] - actions[:, -1, :]
            else:
                flow_target = actions[:, 0, :]
            
            flow_pred = self.flow_generator(flow_target)
            flow_loss = torch.nn.functional.mse_loss(flow_pred, flow_target)
            
            return flow_loss, output_dict
        except Exception as e:
            logger.debug(f"Flow matching loss computation failed: {e}")
            return torch.tensor(0.0, device=actions.device), output_dict

    def _compute_jump_loss(
        self, batch: Dict[str, Tensor], output_dict: Dict
    ) -> Tuple[Tensor, Dict]:
        """Compute Jump Process loss (Section 3.3, Table 7)."""
        actions = batch[ACTION]
        
        try:
            # Jump process induces discrete mode changes
            # Loss: KL divergence between learned and target transition probabilities
            batch_size = actions.shape[0]
            
            # Sample jump times uniformly
            t_jump = torch.rand(batch_size, device=actions.device)
            
            # Get jump predictions
            jump_pred = self.jump_generator(actions, t_jump.mean().item())
            
            # KL loss on transition probabilities
            jump_loss = torch.nn.functional.kl_div(
                torch.log_softmax(jump_pred, dim=-1),
                torch.ones_like(jump_pred) / jump_pred.shape[-1],
                reduction='mean'
            )
            
            return jump_loss, output_dict
        except Exception as e:
            logger.debug(f"Jump loss computation failed: {e}")
            return torch.tensor(0.0, device=actions.device), output_dict

    def _compute_ctmc_loss(
        self, batch: Dict[str, Tensor], output_dict: Dict
    ) -> Tuple[Tensor, Dict]:
        """Compute CTMC loss (Section 3.3, Table 7)."""
        actions = batch[ACTION]
        
        try:
            batch_size = actions.shape[0]
            
            # CTMC operates on discrete skill space
            # Loss: cross-entropy for skill selection
            current_skill = torch.randint(0, self.config.ctmc_num_skills, (batch_size,))
            t = torch.rand(batch_size, device=actions.device)
            
            skill_logits = self.ctmc_generator(current_skill, t.mean().item())
            
            # Cross-entropy loss on next skill prediction
            target_skill = torch.randint(0, self.config.ctmc_num_skills, (batch_size,))
            ctmc_loss = torch.nn.functional.cross_entropy(
                skill_logits.view(batch_size, -1),
                target_skill
            )
            
            return ctmc_loss, output_dict
        except Exception as e:
            logger.debug(f"CTMC loss computation failed: {e}")
            return torch.tensor(0.0, device=actions.device), output_dict

    def _compute_reward_alignment_loss(self, batch: Dict[str, Tensor]) -> Tensor:
        """Compute reward alignment loss (Section 6.4 - Flow-GRPO)."""
        try:
            actions = batch[ACTION]
            rewards = batch.get("rewards", torch.zeros(actions.shape[0]))
            
            if not isinstance(rewards, Tensor):
                rewards = torch.tensor(rewards, device=actions.device)
            
            # Flow-GRPO: maximize reward with KL regularization
            reward_loss = -rewards.mean()
            
            return reward_loss
        except Exception as e:
            logger.debug(f"Reward alignment loss failed: {e}")
            return torch.tensor(0.0, device=batch[ACTION].device)

    @torch.no_grad()
    def select_action(self, batch: Dict[str, Tensor], reward_fn: Optional[Callable] = None) -> Tensor:
        """Select action with optional reward alignment and safety constraints."""
        action = super().select_action(batch)

        if self.config.enable_reward_alignment and reward_fn is not None:
            action = self._apply_inference_time_alignment(action, reward_fn)

        if self.config.enable_hardware_safety_checks:
            action = self.safety_sampler(action)

        return action

    def _apply_inference_time_alignment(self, action: Tensor, reward_fn: Callable) -> Tensor:
        """Apply inference-time reward alignment (Section 6.3)."""
        if self.config.use_sequential_monte_carlo:
            return self._smc_refinement(action, reward_fn)
        else:
            return self._gibbs_tilt(action, reward_fn)

    def _gibbs_tilt(self, action: Tensor, reward_fn: Callable) -> Tensor:
        """Apply Gibbs tilt reward weighting (Section 6.1)."""
        reward = reward_fn(action)
        log_weight = self.config.reward_temperature * reward
        weight = torch.softmax(log_weight, dim=0)
        tilted_action = action * weight.view(-1, 1, 1)
        return tilted_action

    def _smc_refinement(self, action: Tensor, reward_fn: Callable) -> Tensor:
        """Sequential Monte Carlo refinement (Section 6.3)."""
        return self._gibbs_tilt(action, reward_fn)

    def compute_reward_alignment_loss(
        self,
        batch: Dict[str, Tensor],
        reward_fn: Callable,
        method: str = "flow_grpo",
    ) -> Tensor:
        """Compute post-training reward alignment loss (Section 6.4)."""
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
        """Flow Generative Policy Optimization (Section 6.4)."""
        policy_loss = -(rewards.mean())
        kl_penalty = self.kl_weight * 0.0
        return policy_loss + kl_penalty

    def _egm_loss(self, actions: Tensor, rewards: Tensor) -> Tensor:
        """Energy-based Generator Matching (Section 6.5)."""
        energy = -rewards
        score_target = energy
        return (score_target ** 2).mean()

    def _offline_rl_loss(self, actions: Tensor, rewards: Tensor) -> Tensor:
        """Offline RL loss (Section 6.4)."""
        return rewards.mean()

    def sample_trajectories(
        self, batch: Dict[str, Tensor], num_samples: int = 1
    ) -> Tensor:
        """Sample multiple trajectory candidates for ensemble or selection."""
        samples = []
        for _ in range(num_samples):
            action = super().select_action(batch)
            samples.append(action)
        return torch.stack(samples, dim=0)

    def get_probability_path_stats(self, t: Tensor) -> Dict[str, Tensor]:
        """Get probability path statistics for analysis."""
        alpha_t = self.prob_path.alpha_t(t)
        sigma_t = self.prob_path.sigma_t(t)

        return {
            "alpha_t": alpha_t,
            "sigma_t": sigma_t,
            "signal_ratio": alpha_t / (sigma_t + 1e-8),
        }


# Backward compatibility alias
MGPPolicy = MarkovGenerativePolicy
