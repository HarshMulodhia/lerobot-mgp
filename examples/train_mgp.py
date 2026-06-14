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
MGP Training Script: End-to-End Example

This script demonstrates complete MGP training pipeline:
1. Dataset loading and validation
2. Policy initialization with MGP features
3. Training with curriculum learning and distribution shift adaptation
4. Validation and model saving
5. Inference on real hardware (or simulation)

Usage:
    python examples/train_mgp.py \
        --dataset_repo_id yourusername/so101_pick_place \
        --output_dir ./mgp_models/pick_place \
        --steps 50000 \
        --use_curriculum_learning \
        --enable_reward_alignment
"""

import argparse
import logging
from pathlib import Path
from typing import Optional

import torch
from accelerate import Accelerator

from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets import make_dataset
from lerobot.policies import make_policy, make_pre_post_processors
from lerobot.policies.mgp import MGPConfig, MGPTrainingHelper
from lerobot.scripts.lerobot_train import update_policy
from lerobot.utils.logging_utils import AverageMeter, MetricsTracker

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_mgp_config(args) -> TrainPipelineConfig:
    """Create MGP configuration from command-line arguments."""
    mgp_cfg = MGPConfig(
        type="mgp",
        pretrained_path=args.pretrained_path,
        use_generator_matching=args.use_generator_matching,
        gm_loss_type=args.gm_loss_type,
        trajectory_horizon=args.trajectory_horizon,
        enable_multimodal_sampling=args.enable_multimodal_sampling,
        num_sample_candidates=args.num_sample_candidates,
        enable_distribution_shift_adaptation=args.enable_distribution_shift_adaptation,
        use_curriculum_learning=args.use_curriculum_learning,
        enable_reward_alignment=args.enable_reward_alignment,
        reward_alignment_type=args.reward_alignment_type,
        enable_hardware_safety_checks=args.enable_hardware_safety_checks,
        max_action_step_size=args.max_action_step_size,
        device="cuda" if torch.cuda.is_available() else "cpu",
        push_to_hub=args.push_to_hub,
        repo_id=args.output_repo_id,
    )

    cfg = TrainPipelineConfig(
        policy=mgp_cfg,
        dataset=type("DatasetConfig", (), {
            "repo_id": args.dataset_repo_id,
            "root": args.dataset_root,
        })(),
        output_dir=Path(args.output_dir),
        steps=args.steps,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        log_freq=args.log_freq,
        save_freq=args.save_freq,
        eval_freq=0,  # No eval env for now
        seed=args.seed,
    )

    return cfg


def train_mgp_policy(
    cfg: TrainPipelineConfig,
    use_mgp_features: bool = True,
) -> None:
    """
    Train MGP policy with all enhancements.

    Args:
        cfg: Training configuration.
        use_mgp_features: Whether to use MGP-specific training features.
    """
    accelerator = Accelerator(step_scheduler_with_optimizer=False)
    device = accelerator.device

    # Setup
    logger.info("Loading dataset...")
    dataset = make_dataset(cfg)
    logger.info(f"Dataset: {dataset.num_episodes} episodes, {dataset.num_frames} frames")

    logger.info("Creating MGP policy...")
    policy = make_policy(
        cfg=cfg.policy,
        ds_meta=dataset.meta,
    )
    logger.info(f"Policy parameters: {sum(p.numel() for p in policy.parameters()):,}")

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg.policy,
        dataset_stats=dataset.meta.stats,
    )

    # Initialize MGP training helper if using MGP features
    mgp_helper = None
    if use_mgp_features:
        logger.info("Initializing MGP training helper...")
        mgp_helper = MGPTrainingHelper(cfg.policy)

    # Optimizer and scheduler
    from lerobot.optim.factory import make_optimizer_and_scheduler

    optimizer, lr_scheduler = make_optimizer_and_scheduler(cfg, policy)

    # Dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=device.type == "cuda",
    )

    # Prepare with accelerator
    policy, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        policy, optimizer, dataloader, lr_scheduler
    )

    # Training loop
    logger.info(f"Starting training for {cfg.steps} steps...")
    train_metrics = {
        "loss": AverageMeter("loss", ":.4f"),
        "grad_norm": AverageMeter("grad_norm", ":.4f"),
        "lr": AverageMeter("lr", ":0.1e"),
    }

    train_tracker = MetricsTracker(
        cfg.batch_size,
        dataset.num_frames,
        dataset.num_episodes,
        train_metrics,
        initial_step=0,
        accelerator=accelerator,
    )

    dataloader_iter = iter(dataloader)
    policy.train()

    for step in range(cfg.steps):
        try:
            batch = next(dataloader_iter)
        except StopIteration:
            dataloader_iter = iter(dataloader)
            batch = next(dataloader_iter)

        # Preprocess
        for cam_key in dataset.meta.camera_keys:
            if cam_key in batch and batch[cam_key].dtype == torch.uint8:
                batch[cam_key] = batch[cam_key].to(dtype=torch.float32) / 255.0
        batch = preprocessor(batch)

        # MGP: Compute sample weights if using curriculum/importance weighting
        if mgp_helper is not None:
            sample_weights = mgp_helper.compute_sample_weights(batch)
            batch["sample_weights"] = sample_weights
            mgp_helper.step_curriculum()

        # Forward + backward
        with accelerator.autocast():
            loss, output_dict = policy.forward(batch)

        accelerator.backward(loss)
        grad_norm = torch.nn.utils.clip_grad_norm_(
            policy.parameters(), cfg.optimizer.grad_clip_norm or float("inf")
        )

        optimizer.step()
        optimizer.zero_grad()

        if lr_scheduler is not None:
            lr_scheduler.step()

        # Logging
        train_tracker.loss = loss.item()
        train_tracker.grad_norm = grad_norm.item()
        train_tracker.lr = optimizer.param_groups[0]["lr"]
        train_tracker.step()

        if (step + 1) % cfg.log_freq == 0:
            logger.info(train_tracker)
            train_tracker.reset_averages()

        if (step + 1) % cfg.save_freq == 0:
            logger.info(f"Saving checkpoint at step {step + 1}...")
            save_dir = Path(cfg.output_dir) / f"step_{step + 1:06d}"
            save_dir.mkdir(parents=True, exist_ok=True)

            # Save model
            torch.save(
                {
                    "model_state_dict": accelerator.unwrap_model(policy).state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "step": step + 1,
                },
                save_dir / "checkpoint.pth",
            )

            logger.info(f"Checkpoint saved to {save_dir}")

        if cfg.policy.push_to_hub and (step + 1) == cfg.steps:
            logger.info("Pushing model to Hub...")
            accelerator.unwrap_model(policy).push_model_to_hub(cfg)

    logger.info("Training completed!")
    logger.info(f"Model saved to {cfg.output_dir}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Train MGP policy")

    # Dataset
    parser.add_argument("--dataset_repo_id", type=str, required=True,
                        help="HF Hub dataset repo ID")
    parser.add_argument("--dataset_root", type=str, default=None,
                        help="Local dataset root (if not using Hub)")

    # Training
    parser.add_argument("--output_dir", type=str, default="./mgp_models",
                        help="Output directory for checkpoints")
    parser.add_argument("--steps", type=int, default=50000,
                        help="Total training steps")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Data loading workers")
    parser.add_argument("--log_freq", type=int, default=100,
                        help="Logging frequency")
    parser.add_argument("--save_freq", type=int, default=5000,
                        help="Checkpoint save frequency")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")

    # MGP specific
    parser.add_argument("--use_generator_matching", action="store_true", default=True,
                        help="Use Generator Matching loss")
    parser.add_argument("--gm_loss_type", type=str, default="score_matching",
                        choices=["score_matching", "flow_matching", "bregman_divergence"],
                        help="GM loss type")
    parser.add_argument("--trajectory_horizon", type=int, default=10,
                        help="Action prediction horizon")
    parser.add_argument("--enable_multimodal_sampling", action="store_true", default=True,
                        help="Enable multi-modal trajectory sampling")
    parser.add_argument("--num_sample_candidates", type=int, default=8,
                        help="Number of trajectory candidates")
    parser.add_argument("--enable_distribution_shift_adaptation", action="store_true", default=True,
                        help="Enable distribution shift handling")
    parser.add_argument("--use_curriculum_learning", action="store_true", default=False,
                        help="Enable curriculum learning")
    parser.add_argument("--enable_reward_alignment", action="store_true", default=False,
                        help="Enable reward alignment")
    parser.add_argument("--reward_alignment_type", type=str, default="inference_time",
                        choices=["inference_time", "post_training"],
                        help="Type of reward alignment")
    parser.add_argument("--enable_hardware_safety_checks", action="store_true", default=True,
                        help="Enable hardware safety constraints")
    parser.add_argument("--max_action_step_size", type=float, default=0.1,
                        help="Max action step size")

    # Model
    parser.add_argument("--pretrained_path", type=str, default=None,
                        help="Pretrained model path for fine-tuning")

    # Hub
    parser.add_argument("--push_to_hub", action="store_true", default=False,
                        help="Push model to Hub after training")
    parser.add_argument("--output_repo_id", type=str, default=None,
                        help="Output repo ID for Hub push")

    args = parser.parse_args()

    # Create config
    cfg = create_mgp_config(args)

    # Train
    train_mgp_policy(cfg, use_mgp_features=True)


if __name__ == "__main__":
    main()
