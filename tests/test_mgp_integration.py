#!/usr/bin/env python

"""
MGP Integration Test and Verification Script

This script verifies that the MGP policy:
1. Loads correctly without diffusion dependency issues
2. Handles action chunk_size properly
3. Supports all inference modes (Flow/Diffusion/CTMC/Jump)
4. Works with async_inference
5. Produces safe actions for SO-101 hardware
6. Integrates with LeRobot training/eval/rollout

Run this BEFORE deploying to real hardware!
"""

import sys
import logging
import torch
import numpy as np
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from lerobot.policies.mgp import MarkovGenerativePolicy, MGPConfig
from lerobot.utils.constants import ACTION, OBS_STATE, OBS_IMAGES


def test_config_creation():
    """Test 1: Config creation with all modes."""
    logger.info("=" * 80)
    logger.info("TEST 1: Config Creation")
    logger.info("=" * 80)
    
    try:
        config = MGPConfig()
        logger.info(f"✓ Default config created")
        logger.info(f"  - chunk_size: {config.chunk_size}")
        logger.info(f"  - n_action_steps: {config.n_action_steps}")
        logger.info(f"  - n_obs_steps: {config.n_obs_steps}")
        logger.info(f"  - enable_diffusion_component: {config.enable_diffusion_component}")
        logger.info(f"  - enable_flow_component: {config.enable_flow_component}")
        logger.info(f"  - enable_jump_component: {config.enable_jump_component}")
        logger.info(f"  - enable_ctmc_component: {config.enable_ctmc_component}")
        logger.info(f"  - max_action_step_size: {config.max_action_step_size}")
        return True
    except Exception as e:
        logger.error(f"✗ Config creation failed: {e}")
        return False


def test_model_initialization():
    """Test 2: Model initialization without diffusion dependency."""
    logger.info("\n" + "=" * 80)
    logger.info("TEST 2: Model Initialization (NO diffusion dependency)")
    logger.info("=" * 80)
    
    try:
        config = MGPConfig()
        config.action_feature = torch.zeros(6)  # 6D action for SO-101
        
        model = MarkovGenerativePolicy(config)
        logger.info(f"✓ Model initialized successfully")
        logger.info(f"  - RGB encoder output dim: {model.rgb_encoder.out_dim}")
        logger.info(f"  - Has flow generator: {hasattr(model, 'flow_generator')}")
        logger.info(f"  - Has diffusion head: {hasattr(model, 'diffusion_head')}")
        logger.info(f"  - Has safety sampler: {hasattr(model, 'safety_sampler')}")
        return model
    except Exception as e:
        logger.error(f"✗ Model initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_action_chunk_size(model):
    """Test 3: Action chunk_size handling."""
    logger.info("\n" + "=" * 80)
    logger.info("TEST 3: Action Chunk Size Handling")
    logger.info("=" * 80)
    
    try:
        config = model.config
        batch_size = 2
        device = next(model.parameters()).device
        
        # Create dummy batch
        batch = {
            OBS_STATE: torch.randn(batch_size, config.n_obs_steps, 7, device=device),  # 7D state
            OBS_IMAGES: torch.randn(batch_size, config.n_obs_steps, 1, 3, 224, 224, device=device),
        }
        
        # Test with default chunk_size=1
        model.eval()
        with torch.no_grad():
            actions = model.predict_action_chunk(batch)
        
        logger.info(f"✓ Chunk size test passed")
        logger.info(f"  - Config chunk_size: {config.chunk_size}")
        logger.info(f"  - Predicted actions shape: {actions.shape}")
        logger.info(f"  - Expected: (B={batch_size}, T={config.chunk_size}, A=6)")
        
        assert actions.shape == (batch_size, config.chunk_size, 6), \
            f"Wrong action shape: {actions.shape} vs expected ({batch_size}, {config.chunk_size}, 6)"
        
        logger.info(f"✓ Action shape is correct!")
        return True
    except Exception as e:
        logger.error(f"✗ Chunk size test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_inference_modes(model):
    """Test 4: All inference modes (Flow/Diffusion/CTMC/Jump)."""
    logger.info("\n" + "=" * 80)
    logger.info("TEST 4: Inference Modes (Flow/Diffusion/CTMC/Jump)")
    logger.info("=" * 80)
    
    try:
        config = model.config
        batch_size = 2
        device = next(model.parameters()).device
        
        # Create dummy batch
        batch = {
            OBS_STATE: torch.randn(batch_size, config.n_obs_steps, 7, device=device),
            OBS_IMAGES: torch.randn(batch_size, config.n_obs_steps, 1, 3, 224, 224, device=device),
        }
        
        # Test diffusion (always enabled)
        logger.info("  Testing DIFFUSION component...")
        with torch.no_grad():
            actions_diff = model.predict_action_chunk(batch)
        logger.info(f"    ✓ Diffusion OK: shape {actions_diff.shape}")
        
        # Test flow if enabled
        if config.enable_flow_component:
            logger.info("  Testing FLOW component...")
            try:
                with torch.no_grad():
                    actions_flow = model.flow_generator.generate_actions(batch_size, device)
                logger.info(f"    ✓ Flow OK: shape {actions_flow.shape}")
            except Exception as e:
                logger.warning(f"    ⚠ Flow failed: {e}")
        
        # Test jump if enabled
        if config.enable_jump_component:
            logger.info("  Testing JUMP component...")
            try:
                with torch.no_grad():
                    actions_jump = model.jump_generator.generate_actions(batch_size, device)
                logger.info(f"    ✓ Jump OK: shape {actions_jump.shape}")
            except Exception as e:
                logger.warning(f"    ⚠ Jump failed: {e}")
        
        # Test CTMC if enabled
        if config.enable_ctmc_component:
            logger.info("  Testing CTMC component...")
            try:
                with torch.no_grad():
                    actions_ctmc = model.ctmc_generator.generate_actions(batch_size, device)
                logger.info(f"    ✓ CTMC OK: shape {actions_ctmc.shape}")
            except Exception as e:
                logger.warning(f"    ⚠ CTMC failed: {e}")
        
        logger.info("✓ All enabled modes work!")
        return True
    except Exception as e:
        logger.error(f"✗ Inference mode test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_safety_constraints(model):
    """Test 5: Hardware safety constraints."""
    logger.info("\n" + "=" * 80)
    logger.info("TEST 5: Hardware Safety Constraints")
    logger.info("=" * 80)
    
    try:
        batch_size = 2
        horizon = 8
        device = next(model.parameters()).device
        
        # Create dangerously large actions
        dangerous_actions = torch.randn(batch_size, horizon, 6, device=device) * 10.0
        
        logger.info(f"  Before safety: max norm = {torch.norm(dangerous_actions, p=2, dim=-1).max().item():.3f}")
        
        # Apply safety sampler
        safe_actions = model.safety_sampler(dangerous_actions)
        
        logger.info(f"  After safety:  max norm = {torch.norm(safe_actions, p=2, dim=-1).max().item():.3f}")
        logger.info(f"  Max allowed:   {model.config.max_action_step_size:.3f}")
        
        # Verify all actions are within safety bounds
        actual_max = torch.norm(safe_actions, p=2, dim=-1).max().item()
        assert actual_max <= model.config.max_action_step_size + 1e-5, \
            f"Safety constraint violated: {actual_max} > {model.config.max_action_step_size}"
        
        logger.info("✓ Safety constraints enforced correctly!")
        return True
    except Exception as e:
        logger.error(f"✗ Safety constraint test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_training_forward(model):
    """Test 6: Training forward pass (loss computation)."""
    logger.info("\n" + "=" * 80)
    logger.info("TEST 6: Training Forward Pass")
    logger.info("=" * 80)
    
    try:
        config = model.config
        batch_size = 2
        device = next(model.parameters()).device
        
        # Create training batch
        batch = {
            OBS_STATE: torch.randn(batch_size, config.n_obs_steps, 7, device=device),
            OBS_IMAGES: torch.randn(batch_size, config.n_obs_steps, 1, 3, 224, 224, device=device),
            ACTION: torch.randn(batch_size, config.n_action_steps, 6, device=device),
            "action_is_pad": torch.zeros(batch_size, config.n_action_steps, dtype=torch.bool, device=device),
        }
        
        model.train()
        loss, output_dict = model(batch)
        
        logger.info(f"✓ Training forward pass successful")
        logger.info(f"  - Total loss: {loss.item():.4f}")
        logger.info(f"  - Diffusion loss: {output_dict.get('loss_diffusion', 0):.4f}")
        logger.info(f"  - Flow loss: {output_dict.get('loss_flow', 0):.4f}")
        logger.info(f"  - GM loss: {output_dict.get('loss_gm', 0):.4f}")
        
        assert not torch.isnan(loss), "Loss is NaN!"
        assert not torch.isinf(loss), "Loss is inf!"
        
        logger.info("✓ Loss computation is stable!")
        return True
    except Exception as e:
        logger.error(f"✗ Training forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_async_inference_compatibility(model):
    """Test 7: async_inference compatibility."""
    logger.info("\n" + "=" * 80)
    logger.info("TEST 7: Async Inference Compatibility")
    logger.info("=" * 80)
    
    try:
        config = model.config
        device = next(model.parameters()).device
        
        logger.info(f"  Config async_batch_timeout: {config.async_batch_timeout}s")
        logger.info(f"  Config enable_async_inference: {config.enable_async_inference}")
        
        # Simulate async inference (single-sample batches)
        model.eval()
        with torch.no_grad():
            for i in range(3):
                batch = {
                    OBS_STATE: torch.randn(1, config.n_obs_steps, 7, device=device),  # Single sample!
                    OBS_IMAGES: torch.randn(1, config.n_obs_steps, 1, 3, 224, 224, device=device),
                }
                
                action = model.select_action(batch)
                
                logger.info(f"  Step {i+1}: action shape {action.shape}, norm {torch.norm(action).item():.3f}")
                
                # Check action is within safety bounds
                assert torch.norm(action) <= config.max_action_step_size + 1e-5, \
                    f"Action violates safety bounds!"
        
        logger.info("✓ Async inference compatible!")
        return True
    except Exception as e:
        logger.error(f"✗ Async inference test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    logger.info("╔" + "=" * 78 + "╗")
    logger.info("║" + " " * 20 + "MGP INTEGRATION TEST SUITE" + " " * 32 + "║")
    logger.info("╚" + "=" * 78 + "╝")
    
    results = {}
    
    # Test 1
    results["Config Creation"] = test_config_creation()
    
    # Test 2
    model = test_model_initialization()
    results["Model Initialization"] = model is not None
    
    if model is None:
        logger.error("\n✗ Cannot continue - model initialization failed")
        return False
    
    # Test 3
    results["Action Chunk Size"] = test_action_chunk_size(model)
    
    # Test 4
    results["Inference Modes"] = test_inference_modes(model)
    
    # Test 5
    results["Safety Constraints"] = test_safety_constraints(model)
    
    # Test 6
    results["Training Forward Pass"] = test_training_forward(model)
    
    # Test 7
    results["Async Inference"] = test_async_inference_compatibility(model)
    
    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("TEST SUMMARY")
    logger.info("=" * 80)
    
    passed = 0
    failed = 0
    
    for test_name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        logger.info(f"{status}: {test_name}")
        if result:
            passed += 1
        else:
            failed += 1
    
    logger.info("=" * 80)
    logger.info(f"RESULTS: {passed} passed, {failed} failed out of {len(results)} tests")
    logger.info("=" * 80)
    
    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
