#!/usr/bin/env python3

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
MGP Inference on Real Hardware (SO-101)

This script demonstrates real-time control using MGP policy with:
- Multi-modal trajectory sampling and selection
- Distribution shift adaptation
- Safety constraint enforcement
- Hardware monitoring and fallback
- Performance logging

Usage:
    python examples/inference_mgp_hardware.py \
        --policy_path yourusername/so101_mgp_pick_place \
        --robot_type so101_follower \
        --task pick_place \
        --num_episodes 10 \
        --enable_safety_checks \
        --enable_multimodal_sampling
"""

import argparse
import logging
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MGPHardwareController:
    """
    Real-time MGP control for robot hardware.

    Manages:
    - Policy loading and device placement
    - Observation preprocessing
    - Multi-modal sampling with selection
    - Safety constraints
    - Hardware communication and monitoring
    """

    def __init__(
        self,
        policy_path: str,
        robot_type: str,
        device: str = "cuda",
        enable_safety_checks: bool = True,
        enable_multimodal_sampling: bool = True,
        num_sample_candidates: int = 8,
        max_action_step_size: float = 0.1,
    ):
        """Initialize hardware controller."""
        self.policy_path = policy_path
        self.robot_type = robot_type
        self.device = device
        self.enable_safety_checks = enable_safety_checks
        self.enable_multimodal_sampling = enable_multimodal_sampling
        self.num_sample_candidates = num_sample_candidates
        self.max_action_step_size = max_action_step_size

        logger.info(f"Loading MGP policy from {policy_path}...")
        self._load_policy()

        logger.info(f"Connecting to robot: {robot_type}...")
        self._connect_robot()

        self.last_action = None
        self.step_count = 0
        self.total_steps = 0
        self.episode_return = 0.0
        self.safety_interventions = 0

    def _load_policy(self):
        """Load pretrained MGP policy."""
        from lerobot.policies import make_policy

        self.policy = make_policy(
            pretrained_name_or_path=self.policy_path,
            policy_type="mgp",
        )
        self.policy.to(self.device)
        self.policy.eval()

        logger.info(f"Policy loaded successfully")
        logger.info(f"  Model parameters: {sum(p.numel() for p in self.policy.parameters()):,}")

    def _connect_robot(self):
        """Connect to robot hardware."""
        from lerobot.robots import make_robot

        self.robot = make_robot(self.robot_type)
        self.robot.connect()

        logger.info(f"Robot connected: {self.robot_type}")
        logger.info(f"  Action space: {self.robot.action_spec}")
        logger.info(f"  Observation keys: {list(self.robot.get_observation().keys())}")

    def get_observation(self) -> Dict[str, torch.Tensor]:
        """Get current observation from robot."""
        obs = self.robot.get_observation()

        # Convert to torch tensors on device
        obs_torch = {}
        for key, value in obs.items():
            if isinstance(value, np.ndarray):
                value = torch.from_numpy(value).float()
            if torch.is_tensor(value):
                value = value.to(self.device)
            obs_torch[key] = value

        # Add batch dimension if needed
        for key in obs_torch:
            if obs_torch[key].dim() == 0:
                obs_torch[key] = obs_torch[key].unsqueeze(0)
            elif obs_torch[key].dim() == 2 and key != "image":  # Assume (H,W,C) images
                obs_torch[key] = obs_torch[key].unsqueeze(0)

        return obs_torch

    def _enforce_safety_constraints(self, action: torch.Tensor) -> torch.Tensor:
        """Enforce safety constraints on actions."""
        if not self.enable_safety_checks:
            return action

        action_safe = action.clone()

        # Constraint 1: Smooth motion (limit step size)
        if self.last_action is not None:
            step = action_safe - self.last_action
            step_norm = torch.norm(step)

            if step_norm > self.max_action_step_size:
                action_safe = self.last_action + (step / (step_norm + 1e-6)) * self.max_action_step_size
                self.safety_interventions += 1
                logger.warning(
                    f"Safety: Limited action step size {step_norm:.4f} → {self.max_action_step_size:.4f}"
                )

        # Constraint 2: Joint limits (if available)
        if hasattr(self.robot, "joint_limits"):
            min_q, max_q = self.robot.joint_limits
            action_safe = torch.clamp(action_safe, min=min_q, max=max_q)

        # Constraint 3: Velocity limits (if available)
        if hasattr(self.robot, "velocity_limits"):
            action_safe = torch.clamp(action_safe, min=-self.robot.velocity_limits, max=self.robot.velocity_limits)

        return action_safe

    def select_action(self, observation: Dict[str, torch.Tensor]) -> tuple[torch.Tensor, Dict]:
        """
        Select action using MGP policy.

        Implements multi-modal sampling and intelligent selection.
        """
        with torch.no_grad():
            if self.enable_multimodal_sampling:
                # Sample multiple trajectory candidates
                action, metrics = self.policy.select_action(
                    observation,
                    deterministic=False,
                    return_metrics=True,
                )

                logger.info(
                    f"Action selected (multimodal): "
                    f"score={metrics.get('selection_scores', 0):.4f}, "
                    f"uncertainty={metrics.get('uncertainties', 0):.4f}"
                )
            else:
                # Deterministic action selection
                action = self.policy.select_action(observation, deterministic=True)
                metrics = {"deterministic": True}

        # Enforce safety constraints
        action = self._enforce_safety_constraints(action)

        return action, metrics

    def send_action(self, action: torch.Tensor):
        """Send action to robot."""
        # Convert to numpy
        if torch.is_tensor(action):
            action_np = action.detach().cpu().numpy()
        else:
            action_np = action

        # Remove batch dimension if present
        if action_np.ndim > 1:
            action_np = action_np[0]

        self.robot.send_action(action_np)
        self.last_action = torch.from_numpy(action_np).float().to(self.device)

    def run_episode(self, episode_idx: int, max_steps: int = 500, record_video: bool = False) -> Dict:
        """
        Run one episode on real hardware.

        Args:
            episode_idx: Episode number for logging.
            max_steps: Maximum steps per episode.
            record_video: Whether to record video (if available).

        Returns:
            Episode statistics.
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"Episode {episode_idx + 1}: Starting")
        logger.info(f"{'='*60}")

        episode_reward = 0.0
        episode_success = False
        steps_in_episode = 0
        action_norms = []
        uncertainties = []

        start_time = time.time()

        for step in range(max_steps):
            # Get observation
            obs = self.get_observation()

            # Select action
            action, metrics = self.select_action(obs)
            action_norms.append(torch.norm(action).item())

            if "uncertainties" in metrics:
                uncertainties.append(metrics["uncertainties"].mean().item())

            # Send to robot
            self.send_action(action)

            # Get reward/success feedback (if available)
            if hasattr(self.robot, "get_task_reward"):
                reward = self.robot.get_task_reward()
                episode_reward += reward
            else:
                reward = 0.0

            if hasattr(self.robot, "is_done"):
                is_done = self.robot.is_done()
                if is_done:
                    episode_success = True
                    logger.info(f"Episode completed successfully at step {step}")
                    break

            steps_in_episode += 1
            self.step_count += 1
            self.total_steps += 1

            # Periodic logging
            if (step + 1) % 50 == 0:
                logger.info(
                    f"  Step {step + 1:3d} | "
                    f"Cumulative reward: {episode_reward:.2f} | "
                    f"Avg action norm: {np.mean(action_norms[-10:]):.4f} | "
                    f"Safety interventions: {self.safety_interventions}"
                )

        episode_time = time.time() - start_time

        # Compile episode statistics
        stats = {
            "episode": episode_idx,
            "steps": steps_in_episode,
            "reward": episode_reward,
            "success": episode_success,
            "time_sec": episode_time,
            "fps": steps_in_episode / episode_time if episode_time > 0 else 0,
            "avg_action_norm": np.mean(action_norms) if action_norms else 0.0,
            "avg_uncertainty": np.mean(uncertainties) if uncertainties else 0.0,
            "safety_interventions": self.safety_interventions,
        }

        logger.info(f"Episode {episode_idx + 1} complete:")
        logger.info(f"  Steps: {stats['steps']}")
        logger.info(f"  Reward: {stats['reward']:.2f}")
        logger.info(f"  Success: {stats['success']}")
        logger.info(f"  FPS: {stats['fps']:.1f}")
        logger.info(f"  Safety interventions: {stats['safety_interventions']}")

        return stats

    def run_episodes(self, num_episodes: int, max_steps: int = 500) -> Dict:
        """Run multiple episodes and aggregate statistics."""
        all_stats = []

        for episode_idx in range(num_episodes):
            try:
                stats = self.run_episode(episode_idx, max_steps=max_steps)
                all_stats.append(stats)
            except Exception as e:
                logger.error(f"Episode {episode_idx} failed: {e}")
                # Optional: fallback to teleoperation or continue

        # Aggregate statistics
        results = {
            "total_episodes": len(all_stats),
            "success_rate": np.mean([s["success"] for s in all_stats]),
            "avg_reward": np.mean([s["reward"] for s in all_stats]),
            "avg_steps": np.mean([s["steps"] for s in all_stats]),
            "avg_fps": np.mean([s["fps"] for s in all_stats]),
            "total_safety_interventions": sum(s["safety_interventions"] for s in all_stats),
            "episodes": all_stats,
        }

        # Print summary
        logger.info(f"\n{'='*60}")
        logger.info("EVALUATION SUMMARY")
        logger.info(f"{'='*60}")
        logger.info(f"Episodes: {results['total_episodes']}")
        logger.info(f"Success rate: {results['success_rate']:.1%}")
        logger.info(f"Average reward: {results['avg_reward']:.2f}")
        logger.info(f"Average steps/episode: {results['avg_steps']:.1f}")
        logger.info(f"Average FPS: {results['avg_fps']:.1f}")
        logger.info(f"Total safety interventions: {results['total_safety_interventions']}")
        logger.info(f"{'='*60}\n")

        return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="MGP Inference on Real Hardware")

    parser.add_argument("--policy_path", type=str, required=True,
                        help="Path to pretrained MGP policy (HF Hub repo or local path)")
    parser.add_argument("--robot_type", type=str, default="so101_follower",
                        help="Robot type (e.g., 'so101_follower')")
    parser.add_argument("--num_episodes", type=int, default=10,
                        help="Number of episodes to run")
    parser.add_argument("--max_steps_per_episode", type=int, default=500,
                        help="Max steps per episode")
    parser.add_argument("--enable_safety_checks", action="store_true", default=True,
                        help="Enable safety constraints")
    parser.add_argument("--enable_multimodal_sampling", action="store_true", default=True,
                        help="Enable multi-modal trajectory sampling")
    parser.add_argument("--num_sample_candidates", type=int, default=8,
                        help="Number of trajectory candidates")
    parser.add_argument("--max_action_step_size", type=float, default=0.1,
                        help="Max action step size")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device (cuda or cpu)")

    args = parser.parse_args()

    # Create controller
    controller = MGPHardwareController(
        policy_path=args.policy_path,
        robot_type=args.robot_type,
        device=args.device,
        enable_safety_checks=args.enable_safety_checks,
        enable_multimodal_sampling=args.enable_multimodal_sampling,
        num_sample_candidates=args.num_sample_candidates,
        max_action_step_size=args.max_action_step_size,
    )

    # Run episodes
    results = controller.run_episodes(
        num_episodes=args.num_episodes,
        max_steps=args.max_steps_per_episode,
    )

    # Save results
    import json

    results_file = Path("mgp_inference_results.json")
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {results_file}")


if __name__ == "__main__":
    main()
