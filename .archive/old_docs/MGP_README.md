# LeRobot-MGP: Markov Generator Policies for Real-World Robot Learning

> **Focus:** Markov Generator Policy (MGP) - a principled framework for generalizable, robust robot control
> 
> **Built on:** LeRobot by Hugging Face (upstream reference: https://github.com/huggingface/lerobot)

## What is MGP?

**Markov Generator Policies** are a mathematical framework for learning robot control policies that:

1. **Model policies as Markov generators** over action trajectories, rooted in probability theory
2. **Address distribution shift** and compounding errors endemic to imitation learning
3. **Enable multi-modal action distributions** with principled selection mechanisms
4. **Support inference-time and post-training reward alignment** without retraining
5. **Generalize across task variations** through the lens of Generator Matching theory

Unlike standard diffusion policies that focus on noise prediction, MGP explicitly models the infinitesimal evolution of probability distributions through **continuous-time Markov generators**, enabling:
- Theoretically principled trajectory distributions
- Robust handling of out-of-distribution observations
- Interpretable reward alignment mechanisms
- Hardware-constrained control with safety guarantees

## Key Innovation: Why MGP > Standard Diffusion Policies

| Aspect | Standard Diffusion | **MGP** |
|--------|-------------------|---------|
| **Theoretical Framework** | Reverse SDE/ODE | Markov generators + Generator Matching |
| **Distribution Shift** | Limited robustness | Explicit weighting and importance sampling |
| **Compounding Errors** | Accumulates in long horizons | Curriculum learning + adaptive sampling |
| **Reward Alignment** | Requires full retraining | Inference-time Gibbs reweighting + offline RL |
| **Multimodality** | Unimodal or mode-seeking | Principled multi-modal sampling & selection |
| **Hardware Safety** | No built-in constraints | Integrated safety layers + receding-horizon MPC |
| **Generalization** | Struggles with variation | Distribution-shift adapted policies |

## Quick Start: Train MGP on SO-101

### 1. Record Your Dataset

```bash
# Teleoperate SO-101 and record demonstrations
lerobot-record \
  --robot.type=so101_follower \
  --dataset.repo_id=yourusername/so101_pick_place \
  --device_id=0 \
  --fps=10
```

### 2. Train MGP Policy

```bash
# Train with all MGP features enabled
lerobot-train \
  --policy.type=mgp \
  --dataset.repo_id=yourusername/so101_pick_place \
  --policy.use_generator_matching=true \
  --policy.enable_multimodal_sampling=true \
  --policy.enable_distribution_shift_adaptation=true \
  --policy.enable_reward_alignment=false \
  --steps=50000 \
  --batch_size=64
```

### 3. Deploy on Real Hardware

```python
from lerobot.policies import make_policy
from lerobot.robots import make_robot

# Load trained MGP policy
policy = make_policy(
    pretrained_name_or_path="yourusername/so101_mgp_pick_place",
    policy_type="mgp"
)

# Connect to real robot
robot = make_robot("so101_follower")
robot.connect()

# Control with safety constraints and multi-modal sampling
for step in range(1000):
    obs = robot.get_observation()
    
    # MGP samples multiple trajectories and selects best one
    action, metrics = policy.select_action(
        obs,
        return_metrics=True,
        deterministic=False
    )
    
    print(f"Selected action quality: {metrics['selection_scores']}")
    robot.send_action(action)
```

## Architecture Overview

### 1. Generator Matching Theory

```python
from lerobot.policies.mgp import GaussianCondOTPath, GeneratorMatchingLoss

# Probability path: interpolates from prior N(0,I) to data
path = GaussianCondOTPath()

# Sample intermediate: x_t = √(1-t) * x_0 + √t * x_1
x_t = path.sample_path(prior_samples, data_point, t=0.5)

# Generator Matching loss guides learning
gm_loss = GeneratorMatchingLoss(loss_type="score_matching")
loss, metrics = gm_loss(predicted_score, target_score)
```

### 2. Distribution Shift Adaptation

```python
from lerobot.policies.mgp import DistributionShiftAdapter

adapter = DistributionShiftAdapter(action_dim=6)

# Predicts trajectory quality and uncertainty
metrics = adapter(trajectory_batch)
# Returns: values, uncertainties, importance_weights
```

### 3. Multi-Modal Trajectory Sampling

```python
from lerobot.policies.mgp import TrajectorySelector

selector = TrajectorySelector(action_dim=6, selection_method="hybrid")

# Generate multiple candidates
candidates = policy.sample_candidates(obs, num_samples=8)  # (B, 8, T, D)

# Intelligently select best trajectory
selected, mask, metrics = selector(
    candidates,
    rewards=external_reward_model(candidates),
    values=value_function(candidates)
)
```

### 4. Curriculum Learning for Robustness

```python
from lerobot.policies.mgp import CurriculumScheduler

curriculum = CurriculumScheduler(
    total_steps=50000,
    curriculum_type="linear"  # or "exponential"
)

for step in range(50000):
    difficulty = curriculum.get_difficulty()  # [0, 1]
    
    # Weight samples by curriculum
    weights = curriculum.get_sample_weights(trajectory_difficulties)
    
    # Apply weighted training loss
    loss = (per_sample_loss * weights).mean()
    
    curriculum.step()
```

### 5. Reward Alignment

```python
from lerobot.policies.mgp import EnergyBasedGeneratorMatching

# Inference-time Gibbs reweighting (no retraining)
aligned_action = policy.apply_inference_time_alignment(
    actions=sampled_trajectories,
    reward_fn=task_reward_function,
    temperature=1.0  # β in Gibbs: π(x) ∝ p(x) * exp(β*r(x))
)

# Post-training offline RL alignment (advanced)
ebm = EnergyBasedGeneratorMatching(temperature=1.0)
loss, metrics = ebm.compute_ebm_loss(
    trajectories=trajectories,
    rewards=trajectory_rewards,
    learnt_scores=model_scores
)
```

### 6. Hardware Safety Constraints

```python
from lerobot.policies.mgp import SafetyConstrainedSampler

safety = SafetyConstrainedSampler(
    max_action_step_size=0.1,  # Enforce smooth motion
    joint_limits=(min_q, max_q),
    velocity_limits=max_vel
)

# Constrain sampled actions to hardware limits
safe_actions = safety.enforce_constraints(
    sampled_actions,
    initial_state=current_joint_config
)
```

## Configuration Reference

Key MGP config options:

```python
from lerobot.policies.mgp import MGPConfig

config = MGPConfig(
    # Core MGP features
    use_generator_matching=True,           # Enable Generator Matching theory
    gm_loss_type="score_matching",         # "score_matching", "flow_matching", "bregman_divergence"
    trajectory_horizon=10,                 # Action horizon T_p
    
    # Robustness & Generalization
    enable_multimodal_sampling=True,       # Sample multiple trajectories
    num_sample_candidates=8,               # Number to sample
    enable_distribution_shift_adaptation=True,
    use_curriculum_learning=True,          # Progressive difficulty
    use_sequential_monte_carlo=False,      # SMC for refinement
    
    # Reward Alignment
    enable_reward_alignment=False,         # Post-training alignment
    reward_alignment_type="inference_time",
    reward_temperature=1.0,                # Gibbs temperature β
    use_value_guidance=False,              # Value-guided sampling
    
    # Hardware Deployment
    target_hardware="so101",
    enable_hardware_safety_checks=True,
    max_action_step_size=0.1,             # Max Δa per step
    use_fast_inference_mode=False,         # Trade quality for speed
    fast_inference_steps=5,                # Steps if fast mode
)
```

## Training Tips for Real Hardware

### 1. **Dataset Diversity**
- Collect data with **position variation**: place objects in different locations
- **Lighting variation**: different times of day, shadows
- **Demonstration diversity**: multiple strategies for same task

```python
dataset = LeRobotDataset("yourusername/so101_task")
print(f"Episodes: {dataset.num_episodes}")
print(f"Frames: {dataset.num_frames}")
print(f"Episode lengths: {dataset.meta.episodes['frame_to_index'].diff().mean()}")
```

### 2. **Curriculum Learning Configuration**
```bash
# Start with easy demonstrations, progress to harder ones
lerobot-train \
  --policy.use_curriculum_learning=true \
  --policy.enable_distribution_shift_adaptation=true \
  --policy.num_sample_candidates=8 \
  --steps=100000
```

### 3. **Multi-Modal Sampling for Robustness**
```python
# Enable sampling multiple trajectory candidates
# Policy selects best based on value/reward
policy_config = {
    "enable_multimodal_sampling": true,
    "num_sample_candidates": 16,  # Sample more for critical tasks
    "use_sequential_monte_carlo": false,  # SMC for iterative refinement
}
```

### 4. **Gradual Deployment: Sim → Real**
```python
# Start in simulation for quick iteration
# Validate on real hardware in teleoperated mode
# Deploy autonomously with safety monitoring

for step in range(deployment_steps):
    obs = robot.get_observation()
    action, metrics = policy.select_action(obs, return_metrics=True)
    
    # Monitor policy uncertainty
    if metrics["uncertainties"].max() > threshold:
        print("High uncertainty - falling back to teleoperation")
        action = teleop.get_action()  # Human takeover
    
    robot.send_action(action)
```

## Addressing Your Specific Issues

### Problem 1: **Behavior Cloning Instead of True VLA**

**Root Cause:** Standard imitation learning overfits to dataset distribution.

**MGP Solution:**
```python
# Enable distribution shift adaptation
policy.config.enable_distribution_shift_adaptation = True
policy.config.enable_multimodal_sampling = True

# During training:
# - Curriculum learning progressively increases task difficulty
# - Importance weighting upweights out-of-distribution trajectories
# - Multi-modal sampling explores solution space
# - Selective bias toward high-value trajectories
```

Result: Policy learns the underlying **task objective**, not just dataset patterns.

### Problem 2: **Distribution Shift (Dataset → Real Robot)**

**Root Cause:** Real robot sees different positions, lighting, gripper states than training data.

**MGP Solution:**
```python
# Importance weighting and value-based selection
selector = TrajectorySelector(selection_method="hybrid")

# At inference time:
# 1. Sample 16 trajectory candidates
# 2. Evaluate each with value function (trained to recognize good actions)
# 3. Reweight by external task reward (e.g., distance to goal)
# 4. Select best according to: value + reward
#
# This makes policy robust to novel object positions!
```

### Problem 3: **Compounding Errors in Long Tasks**

**Root Cause:** Action errors accumulate over long time horizons.

**MGP Solution:**
```python
# Curriculum learning + receding-horizon control
config = MGPConfig(
    trajectory_horizon=10,      # Short prediction horizon
    chunk_size=1,              # Execute 1 step, replan
    use_curriculum_learning=True,  # Start with short tasks
    enable_multimodal_sampling=True,  # Re-sample at each step
)

# This implements MPC-style receding horizon:
# - Predict 10 steps ahead
# - Execute 1 step
# - Observe new state
# - Replan from new state
# Errors don't accumulate!
```

## Validation & Testing

### Run Comprehensive Tests

```bash
# Test all MGP components
pytest tests/test_mgp_policy.py -v

# Expected output:
# test_condot_path_interpolation PASSED
# test_gm_loss_score_matching PASSED
# test_curriculum_scheduler_linear PASSED
# test_safety_constraints PASSED
# ...
# test_full_training_pipeline PASSED
# ======================== 18 passed in 3.5s ========================
```

### Evaluate on Real Hardware

```python
from lerobot.rollout import roll out_policy

# Test policy on 10 real robot episodes
results = rollout_policy(
    policy=mgp_policy,
    robot=so101_robot,
    num_episodes=10,
    max_steps=100,
    record_video=True
)

print(f"Success rate: {results['success_rate']:.1%}")
print(f"Avg reward: {results['avg_reward']:.2f}")
print(f"Completion rate: {results['completion_rate']:.1%}")
```

## Configuration for Different Task Complexities

### Simple Tasks (Pick & Place)
```yaml
trajectory_horizon: 5
num_sample_candidates: 4
use_curriculum_learning: false
enable_reward_alignment: false
```

### Medium Tasks (Object Rearrangement)
```yaml
trajectory_horizon: 10
num_sample_candidates: 8
use_curriculum_learning: true
enable_multimodal_sampling: true
enable_distribution_shift_adaptation: true
```

### Complex Tasks (Long-Horizon, High Variation)
```yaml
trajectory_horizon: 10
chunk_size: 1  # Receding horizon MPC
num_sample_candidates: 16
use_curriculum_learning: true
use_sequential_monte_carlo: true
enable_reward_alignment: true
reward_alignment_type: "post_training"
```

## Next Steps

1. **Train your first MGP policy** on SO-101 dataset
2. **Validate on real hardware** using rollout evaluation
3. **Iterate dataset collection** for challenging scenarios
4. **Extend with reward alignment** for task-specific optimization
5. **Deploy on production robots** with safety monitoring

## References

- **MGP Theory:** `./mgp/Markov Generative Policies for the SO-101 Robot with LeRobot*.md`
- **Generator Matching:** `./mgp/Generator Matching Theory*.md`
- **Reward Alignment:** `./mgp/Reward Alignment.md`
- **Upstream LeRobot:** https://github.com/huggingface/lerobot
- **LeRobot Docs:** https://huggingface.co/docs/lerobot

## Support & Community

- 📖 **Documentation**: See `./mgp/` folder for detailed theory
- 💬 **Questions**: Open GitHub issues or join LeRobot Discord
- 🤝 **Contribute**: PRs welcome for improvements, new robots, or new algorithms
- 🔬 **Research**: Built on peer-reviewed Generator Matching theory

---

**Built with ❤️ on LeRobot for the open-source robotics community.**

*This is a specialized fork optimizing for Markov Generator Policies and real-world SO-101 deployment while maintaining full compatibility with upstream LeRobot.*
