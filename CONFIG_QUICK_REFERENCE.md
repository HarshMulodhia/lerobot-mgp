# MGP Configuration Quick Reference Card

## Most Important Parameters

```
╔═══════════════════════════════════════════════════════════════╗
║ FOR REAL HARDWARE: Use These Settings                        ║
╠═══════════════════════════════════════════════════════════════╣
║ chunk_size: 1                    # Receding horizon control  ║
║ max_action_step_size: 0.05-0.1   # Safety bounds             ║
║ fast_inference_steps: 10         # ~15ms inference           ║
║ loss_weights:                                                ║
║   diffusion: 1.0                 # Primary imitation         ║
║   flow: 0.05-0.1                 # Smooth motion             ║
║   gm: 0.05-0.1                   # Visual grounding          ║
║   others: 0                       # Disable if not needed    ║
╚═══════════════════════════════════════════════════════════════╝
```

## CLI Tuning Examples

### START: Basic Low-Jitter Setup
```bash
lerobot-train policy.type=mgp \
  policy.chunk_size=1 \
  policy.loss_weights='{"diffusion": 1.0, "flow": 0.05}' \
  policy.max_action_step_size=0.05 \
  dataset_repo_id=your_org/so101_dataset
```

### JITTER FIX: Increase Flow Weight
```bash
lerobot-train policy.type=mgp \
  policy.loss_weights='{"diffusion": 1.0, "flow": 0.15}'
```

### MULTI-CAMERA: Different Viewpoints
```bash
lerobot-train policy.type=mgp \
  policy.use_separate_rgb_encoder_per_camera=true
```

### SPEED: Fast Inference
```bash
lerobot-train policy.type=mgp \
  policy.use_fast_inference_mode=true \
  policy.fast_inference_steps=5
```

### MODE SWITCHING: Multiple Grasps
```bash
lerobot-train policy.type=mgp \
  policy.enable_jump_component=true \
  policy.loss_weights='{
    "diffusion": 1.0,
    "flow": 0.05,
    "jump": 0.2
  }'
```

## Parameter Categories

| Category | Parameters | When to Change |
|----------|-----------|-----------------|
| **Control** | chunk_size, n_action_steps, n_obs_steps | If latency issues or planning needs change |
| **Safety** | max_action_step_size, enable_hardware_safety_checks | Based on robot performance, start conservative |
| **Speed** | use_fast_inference_mode, fast_inference_steps | Based on GPU, control frequency requirements |
| **Quality** | loss_weights | Based on motion quality and task requirements |
| **Hardware** | target_hardware, enable_hardware_safety_checks | Based on robot platform |
| **Vision** | vision_backbone, use_separate_rgb_encoder_per_camera, resize_shape | Based on camera setup and compute |

## Theory Reference Cheat Sheet

```
SECTION 3.1 (Probability Paths):
├─ beta_schedule, num_train_timesteps, num_inference_steps
└─ Define noise schedule σ(t) for forward diffusion

SECTION 3.3 (Markov Decomposition):
├─ enable_flow_component:        L^flow_t (deterministic ODE)
├─ enable_diffusion_component:   L^diff_t (stochastic SDE)
├─ enable_jump_component:        L^jump_t (discrete jumps)
└─ enable_ctmc_component:        L^CTMC_t (skill hierarchy)

SECTION 4.2-4.3 (Diffusion Generator):
├─ down_dims: UNet downsampling architecture
├─ diffusion_step_embed_dim: timestep encoding
└─ use_film_scale_modulation: conditioning

SECTION 5.1 (Multi-Camera Vision):
├─ use_separate_rgb_encoder_per_camera: per-camera vs shared
├─ vision_backbone: ResNet backbone
└─ spatial_softmax_num_keypoints: keypoint pooling

SECTION 5.1 (Markov Superposition):
├─ enable_markov_superposition: learned gating weights
├─ loss_weights: L = α*L_DP + β*L_GM + γ*L_FM + δ*L_JUMP + ...
└─ Component blending

SECTION 6.1 (Hardware Safety):
├─ enable_hardware_safety_checks: action norm clipping
├─ max_action_step_size: maximum L2 norm per step
└─ SO-101 constraints
```

## Validation Commands

```bash
# 1. Check configuration is correct
python scripts/validate_mgp_config.py

# 2. Run integration tests
python tests/test_mgp_integration.py

# 3. Quick training verification
lerobot-train policy.type=mgp \
  dataset_repo_id=your_org/so101_dataset \
  num_training_steps=100 \
  checkpoint_save_interval=10

# 4. Evaluate trained model
lerobot-eval checkpoint_path=path/to/model.pt \
  dataset_repo_id=your_org/so101_dataset

# 5. Deploy on hardware
lerobot-rollout checkpoint_path=path/to/model.pt \
  robot_type=so101 \
  num_episodes=5
```

## Troubleshooting Quick Guide

| Problem | Solution | Parameters |
|---------|----------|------------|
| **Jittery motion** | Increase flow weight | `loss_weights["flow"]: 0.15` |
| **Too slow** | Reduce inference steps | `fast_inference_steps: 5` |
| **Unsafe motion** | Reduce action bounds | `max_action_step_size: 0.03` |
| **Low success rate** | Increase diffusion weight | `loss_weights["diffusion"]: 1.5` |
| **Poor visual grounding** | Increase GM weight | `loss_weights["gm"]: 0.2` |
| **Mode switching issues** | Enable jump + increase weight | `enable_jump_component: true` + `loss_weights["jump"]: 0.3` |

## Memory Checklist

✅ **Must have**:
- `chunk_size=1` (real hardware)
- `enable_diffusion_component=true` (primary)
- `enable_hardware_safety_checks=true` (safety)
- `use_fast_inference_mode=true` (speed)

⚠️ **Usually have**:
- `enable_flow_component=true` (smoothness)
- `loss_weights["diffusion"]: 1.0` (imitation)

❌ **Usually disable** (unless needed):
- `enable_jump_component` (unless multi-modal)
- `enable_ctmc_component` (unless hierarchical)
- `enable_markov_superposition` (unless 2+ components)
- `enable_reward_alignment` (unless post-training)

## Configuration Validation Rules

1. **Horizon Divisibility**: `n_action_steps % 2^len(down_dims) == 0`
   - e.g., 8 % 8 == 0 ✓ (for down_dims=(512,1024,2048))

2. **Chunk Size**: `0 < chunk_size <= n_action_steps`
   - For hardware: `chunk_size = 1`

3. **Loss Weight Consistency**:
   - If `loss_weights["jump"] > 0`, must have `enable_jump_component=true`
   - If `loss_weights["ctmc"] > 0`, must have `enable_ctmc_component=true`

4. **Image Consistency** (Multi-camera):
   - All images must have same shape: (C, H, W)
   - Resize/crop before training to match

5. **Safety Bounds**:
   - `0 < max_action_step_size <= 0.3`
   - Typical: 0.05-0.15 for SO-101
   - Start conservative, increase after testing

## Documentation Links

| Document | Purpose |
|----------|---------|
| `configuration_mgp.py` | Complete parameter definitions with theory |
| `CONFIG_THEORY_GUIDE.md` | Parameter tuning guide with examples |
| `CONFIG_UPDATES_SUMMARY.md` | Overview of all changes |
| `scripts/validate_mgp_config.py` | Auto-validation script |
| `MGP_IMPLEMENTATION_GUIDE.md` | Full training/deployment guide |
| `TROUBLESHOOTING.md` | Common issues and solutions |

---

**Theory**: All parameters grounded in Markov Generative Policies and Generator Matching  
**Status**: ✅ Production-ready for SO-101  
**Last Updated**: 2026
