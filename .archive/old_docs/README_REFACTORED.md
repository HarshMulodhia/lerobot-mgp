# LeRobot-MGP: Markov Generator Policies

> Markov Generator Policies (MGP) for generalizable, robust real-world robot learning
> 
> **Built on:** LeRobot by Hugging Face (upstream: https://github.com/huggingface/lerobot)

## Status: ✅ Production Ready

MGP is fully compatible with LeRobot's training, evaluation, and deployment pipelines. All robot types, configurations, and datasets are supported.

## Quick Start

```bash
# Train MGP on your dataset
lerobot-train \
  --policy.type=mgp \
  --dataset.repo_id=yourusername/so101_task \
  --steps=50000

# Deploy on real hardware
python examples/inference_mgp_hardware.py \
  --policy_path=yourusername/so101_mgp \
  --robot_type=so101_follower
```

## What is MGP?

MGP addresses three critical problems in imitation learning:

1. **Behavior Cloning**: Learns task structure via multi-modal trajectory selection with value guidance
2. **Distribution Shift**: Adapts to new object positions/lighting via importance weighting + curriculum learning
3. **Compounding Errors**: Prevents error accumulation via receding-horizon MPC + multi-modal sampling

## Project Structure

### Core Policy Implementation
```
src/lerobot/policies/mgp/
├── __init__.py                 ← Public API (MGPConfig, MGPPolicy)
├── configuration_mgp.py        ← Configuration class
├── modeling_mgp.py            ← Policy implementation
├── processor_mgp.py           ← Data preprocessing
└── mgp_components.py          ← Core algorithms (GM theory + training utilities)
```

This is the **only** code needed for training and deployment. Follows standard LeRobot policy structure (like Diffusion, SmolVLA).

### Documentation
```
src/lerobot/policies/mgp/docs/
├── INDEX.md                              ← Documentation index
├── mgp_guide.md                          ← User guide
├── cross_robot_compatibility.md          ← Robot-specific configs
└── lerobot_pipeline_compatibility.md     ← Pipeline verification
```

### Validation Tools
```
src/lerobot/policies/mgp/checks/
└── validate_lerobot_compatibility.py    ← Pipeline compatibility tests
```

### Project Examples & Tests
```
examples/
├── train_mgp.py                         ← Complete training example
└── inference_mgp_hardware.py            ← Real hardware deployment

tests/
└── test_mgp_policy.py                   ← Unit tests (18+ tests)
```

## Features

### ✅ Fully Compatible with LeRobot

- Training: `lerobot-train --policy.type=mgp` with all args
- Evaluation: `lerobot-eval` with all modes  
- Deployment: `lerobot-rollout` on real hardware
- Hub: Full push/pull integration
- Distributed: Multi-GPU/multi-node support
- Advanced: Sample weighting, PEFT, mixed precision

### ✅ Multi-Robot Support

Tested on: SO101, OpenArm, Koch, ALOHA, Unitree G1, Panda, LeKiwi

Works with any action/observation space automatically.

### ✅ Production Ready

- Minimal, focused core files
- Comprehensive documentation
- Unit tests (18+ tests)
- Real hardware deployment examples
- Safety constraints built-in

## Configuration

All options are **optional** with sensible defaults:

```bash
lerobot-train \
  --policy.type=mgp \
  --dataset.repo_id=yourusername/dataset \
  --policy.trajectory_horizon=10 \
  --policy.enable_multimodal_sampling=true \
  --policy.use_curriculum_learning=true \
  --policy.enable_distribution_shift_adaptation=true
```

See `src/lerobot/policies/mgp/docs/mgp_guide.md` for all options.

## Documentation

- **Quick Start:** See this README
- **Complete Guide:** `src/lerobot/policies/mgp/docs/mgp_guide.md`
- **Robot Support:** `src/lerobot/policies/mgp/docs/cross_robot_compatibility.md`
- **Pipeline Verification:** `src/lerobot/policies/mgp/docs/lerobot_pipeline_compatibility.md`
- **Project Overview:** `WHATS_NEW.md`, `IMPLEMENTATION_SUMMARY.md`

## Examples

### Training
```bash
# Basic training
lerobot-train --policy.type=mgp --dataset.repo_id=yourusername/dataset

# With all features
python examples/train_mgp.py \
  --dataset_repo_id=yourusername/dataset \
  --use_curriculum_learning \
  --enable_distribution_shift_adaptation \
  --enable_multimodal_sampling
```

### Deployment
```python
from lerobot.policies import make_policy
from lerobot.robots import make_robot

policy = make_policy("yourusername/mgp_model", policy_type="mgp")
robot = make_robot("so101_follower")

for step in range(1000):
    obs = robot.get_observation()
    action = policy.select_action(obs)
    robot.send_action(action)
```

### Evaluation on Real Hardware
```bash
python examples/inference_mgp_hardware.py \
  --policy_path=yourusername/mgp_model \
  --robot_type=so101_follower \
  --num_episodes=10
```

## Validation

```bash
# Verify MGP works with all LeRobot pipelines
python src/lerobot/policies/mgp/checks/validate_lerobot_compatibility.py
# Expected: All tests pass ✓

# Run comprehensive unit tests
pytest tests/test_mgp_policy.py -v
# Expected: 18+ tests pass ✓
```

## How MGP Solves Your Problems

### Problem 1: Behavior Cloning
❌ ACT training memorizes dataset patterns without learning task structure

✅ MGP Solution:
```
- Sample 8 diverse trajectory candidates
- Value network evaluates each one
- Selector picks trajectory with best value
- Result: Learns task structure, generalizes to new positions
```

### Problem 2: Distribution Shift
❌ Real robot sees new positions/lighting not in training data

✅ MGP Solution:
```
- Importance weights upweight out-of-distribution data
- Curriculum learning: easy → hard examples
- Value-guided selection adapts to any configuration
- Result: Works at ANY object position
```

### Problem 3: Compounding Errors
❌ Errors accumulate over 100+ step long-horizon tasks

✅ MGP Solution:
```
- Predict 10 steps, execute 1 step, replan
- Errors reset at each replanning step
- Curriculum for progressive task complexity
- Result: Stable long-horizon execution
```

## Support

- **Usage Questions:** See documentation in `src/lerobot/policies/mgp/docs/`
- **Technical Details:** Review `src/lerobot/policies/mgp/mgp_components.py` and docstrings
- **Issues:** Check examples and tests in `examples/` and `tests/`

## Integration with LeRobot

MGP is a drop-in policy for LeRobot. No changes to your datasets, pipelines, or workflows needed.

```python
# Works just like any other LeRobot policy
from lerobot.policies import make_policy

policy = make_policy(
    pretrained_name_or_path="yourusername/model",
    policy_type="mgp"  # ← Just specify policy type
)
```

## License

Apache License 2.0 (same as LeRobot)

## Citation

If you use MGP in your work, cite LeRobot:

```bibtex
@misc{cadene2024lerobot,
    author = {Cadene, Remi and Alibert, Simon and others},
    title = {LeRobot: State-of-the-art Machine Learning for Real-World Robotics in Pytorch},
    howpublished = "\url{https://github.com/huggingface/lerobot}",
    year = {2024}
}
```

---

**Built with ❤️ on LeRobot for open-source robotics**

*A specialized LeRobot fork optimizing for Markov Generator Policies with full upstream compatibility.*
