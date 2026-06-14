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
MGP Training Guide - Complete Tuning & Usage Examples

Implements Markov Generative Policies (Section 3.3, 4.3, 5.1):
Total Loss: L = α*L_DP + β*L_GM + γ*L_FM + δ*L_JUMP + ε*L_CTMC + λ*L_reward

All components independently tunable via command-line arguments.
"""

# ====================================================================
# QUICK START: Basic Training (Diffusion + Generator Matching)
# ====================================================================

def example_1_basic_training():
    """
    Baseline: DiffusionPolicy + Generator Matching (default config)
    Good starting point for most SO-101 tasks.
    """
    cmd = """
lerobot-train --policy.type=mgp \\
  --dataset.repo_id=lerobot/svla_so101_pickplace \\
  --batch_size=2 \\
  --steps=2000
"""
    print("EXAMPLE 1: Basic Training")
    print(cmd)
    print("""
What's enabled:
- α*L_DP (1.0): DDPM noise-prediction MSE - primary imitation
- β*L_GM (0.1): Conditional Generator Matching - multi-camera grounding
- γ*L_FM (0.05): Flow/ODE baseline - smoother actions
- Others: disabled

Expected behavior:
- Stable training, good for imitation learning
- Smooth action trajectories
- Multi-camera observations automatically concatenated
""")


# ====================================================================
# EXAMPLE 2: Custom Loss Weights (Fine-tuning for specific task)
# ====================================================================

def example_2_custom_weights():
    """
    Tune loss weights for your specific task.
    Each component addresses different aspects of behavior.
    """
    cmd = """
lerobot-train --policy.type=mgp \\
  --dataset.repo_id=lerobot/svla_so101_pickplace \\
  --batch_size=4 \\
  --steps=5000 \\
  --policy.loss_weights='{"diffusion": 1.5, "gm": 0.3, "flow": 0.1}'
"""
    print("\nEXAMPLE 2: Custom Loss Weights")
    print(cmd)
    print("""
Key tuning parameters:
- "diffusion" (α): 0.5-2.0
  Lower: faster but less stable
  Higher: slower but more imitation-focused

- "gm" (β): 0.05-0.5
  Higher: stronger visual grounding via multi-camera
  Useful when observability from single camera is poor

- "flow" (γ): 0.0-0.2
  Higher: more deterministic, smoother motions
  Useful for precise manipulation tasks

- "reward": 0.0-0.1
  Only if you have reliable reward signal
  Used for post-training alignment
""")


# ====================================================================
# EXAMPLE 3: Enable Jump Process (Discrete Mode Switching)
# ====================================================================

def example_3_jump_process():
    """
    Jump process for abrupt strategy switches.
    Good for: regrasp, mode switching, discrete behaviors.
    """
    cmd = """
lerobot-train --policy.type=mgp \\
  --dataset.repo_id=lerobot/svla_so101_pickplace \\
  --batch_size=4 \\
  --steps=5000 \\
  --policy.enable_jump_component=true \\
  --policy.jump_num_modes=4 \\
  --policy.jump_rate=0.1 \\
  --policy.loss_weights='{"diffusion": 1.0, "gm": 0.1, "jump": 0.2}'
"""
    print("\nEXAMPLE 3: Jump Process for Mode Switching")
    print(cmd)
    print("""
Jump Process (L^jump_t): Models discrete behavior modes

Configuration:
- jump_num_modes (default=4): Number of distinct strategies
  Examples:
    - 2 modes: grasp vs place
    - 4 modes: approach, grasp, lift, place
    - 8 modes: fine-grained behavior repertoire

- jump_rate (default=0.1): Poisson jump intensity λ_t
  Higher: more frequent mode switches
  Lower: longer dwells in each mode

- loss_weights['jump'] (default=0.2): How much to emphasize jumps
  0.0-0.05: subtle mode changes
  0.1-0.3: strong discrete transitions
  >0.3: heavily quantized behavior

When to use:
- Task has clear behavioral phases (grasp → place → retract)
- Continuous smoothing causes performance degradation
- Want explicit discrete mode modeling
""")


# ====================================================================
# EXAMPLE 4: Enable CTMC (Skill-Based Policies)
# ====================================================================

def example_4_ctmc_skills():
    """
    CTMC for hierarchical skill-based policies.
    Good for: complex multi-stage tasks, skill reuse.
    """
    cmd = """
lerobot-train --policy.type=mgp \\
  --dataset.repo_id=lerobot/svla_so101_pickplace \\
  --batch_size=4 \\
  --steps=5000 \\
  --policy.enable_ctmc_component=true \\
  --policy.ctmc_num_skills=8 \\
  --policy.ctmc_skill_dim=64 \\
  --policy.loss_weights='{"diffusion": 1.0, "ctmc": 0.1}'
"""
    print("\nEXAMPLE 4: CTMC Skill-Based Policy")
    print(cmd)
    print("""
CTMC (L^CTMC_t): Continuous-Time Markov Chain for discrete skills

Configuration:
- ctmc_num_skills (default=8): Number of learned skills
  4-8: for most manipulation tasks
  16+: for very complex behaviors

- ctmc_skill_dim (default=64): Skill embedding dimensionality
  Higher: more expressive per-skill representations
  64-128: typical range

- loss_weights['ctmc']: Weight for skill switching loss
  0.05-0.2: typical range

When to use:
- Need hierarchical multi-stage behaviors
- Want skill reuse across tasks
- Have natural skill decomposition (grasp vs place vs retract)

Output: Discrete skill embeddings s_t, transitions via rate matrix Q
""")


# ====================================================================
# EXAMPLE 5: Full Markov Superposition (All Components)
# ====================================================================

def example_5_full_markov_superposition():
    """
    Combine ALL components with learned gating.
    Most expressive but requires careful tuning.
    """
    cmd = """
lerobot-train --policy.type=mgp \\
  --dataset.repo_id=lerobot/svla_so101_pickplace \\
  --batch_size=8 \\
  --steps=10000 \\
  --policy.enable_flow_component=true \\
  --policy.enable_jump_component=true \\
  --policy.enable_ctmc_component=true \\
  --policy.enable_markov_superposition=true \\
  --policy.superposition_hidden_dim=256 \\
  --policy.superposition_learn_weights=true \\
  --policy.loss_weights='{
    "diffusion": 1.0,
    "gm": 0.15,
    "flow": 0.08,
    "jump": 0.1,
    "ctmc": 0.05
  }'
"""
    print("\nEXAMPLE 5: Full Markov Superposition")
    print(cmd)
    print("""
Markov Superposition (Section 3.5, 5.3):
L_t = w_flow(h_t)*L^flow_t + w_diff(h_t)*L^diff_t + w_jump(h_t)*L^jump_t + w_ctmc(h_t)*L^ctmc_t

All four components enabled with learned gating weights w_i(h_t).

Configuration:
- superposition_hidden_dim (default=128): Gating network hidden size
  128-256: typical
  256+: for very complex observation spaces

- superposition_learn_weights (default=true): Learn weights vs fixed
  true: fully adaptive gating
  false: uniform weighting (ablation)

Components:
1. Flow (γ=0.08): Deterministic ODE baseline
   - Smooth, predictable motions
   
2. Diffusion (α=1.0): Stochastic DDPM
   - Multimodal action distributions
   
3. Jump (δ=0.1): Poisson mode switches
   - Discrete behavioral transitions
   
4. CTMC (ε=0.05): Skill-level transitions
   - Hierarchical behavior organization

Gating learns when to use each:
- Single-camera scene: emphasize diffusion
- Complex observations: emphasize gm + jump
- Hierarchical task: emphasize ctmc

When to use:
- Complex tasks with multiple behavioral modes
- Want to study which components matter
- Have sufficient computational budget
- Large diverse training dataset

Expected training time: 2-4x longer than baseline
""")


# ====================================================================
# EXAMPLE 6: Reward Alignment (Post-Training)
# ====================================================================

def example_6_reward_alignment():
    """
    Fine-tune with reward signal (inference-time or post-training).
    """
    cmd = """
lerobot-train --policy.type=mgp \\
  --dataset.repo_id=lerobot/svla_so101_pickplace \\
  --batch_size=4 \\
  --steps=5000 \\
  --policy.enable_reward_alignment=true \\
  --policy.reward_alignment_type=inference_time \\
  --policy.reward_temperature=1.0 \\
  --policy.use_sequential_monte_carlo=true \\
  --policy.smc_particles=16 \\
  --policy.loss_weights='{"diffusion": 1.0, "reward": 0.05}'
"""
    print("\nEXAMPLE 6: Reward Alignment (Inference-Time)")
    print(cmd)
    print("""
Reward Alignment (Section 6):

Inference-time methods:
- Gibbs tilt: p_ψ(a_t|o_t,r_t) ∝ p_θ(a_t|o_t) exp(β*r(a_t))
- SMC: Sequential Monte Carlo refinement with reward

Post-training methods:
- Flow-GRPO: Generative Policy Optimization
- EGM: Energy-based Generator Matching

Configuration:
- reward_temperature (β, default=1.0): Reward scaling
  Higher: more reward-focused, less diverse
  Lower: more faithful to base policy

- smc_particles (default=16): Number of SMC particles
  More: better reward alignment, slower
  Fewer: faster, less accurate

When to use:
- Have reliable reward function r(a)
- Want to bias policy toward high-reward actions
- Running in post-training phase

Safety notes:
- Always validate rewards on small batch first
- Start with low temperature (0.5-1.0)
- Gradually increase if needed
""")


# ====================================================================
# EXAMPLE 7: Multi-Camera & Hardware Safety
# ====================================================================

def example_7_safety_and_multicamera():
    """
    Multi-camera observation handling and hardware safety.
    """
    cmd = """
lerobot-train --policy.type=mgp \\
  --dataset.repo_id=lerobot/svla_so101_pickplace \\
  --batch_size=4 \\
  --steps=5000 \\
  --policy.enable_multi_camera_concat=true \\
  --policy.camera_concat_dim=-3 \\
  --policy.enable_hardware_safety_checks=true \\
  --policy.max_action_step_size=0.1 \\
  --policy.target_hardware=so101
"""
    print("\nEXAMPLE 7: Multi-Camera & Hardware Safety")
    print(cmd)
    print("""
Multi-Camera Support (Section 4.1):
- Cameras named: camera_0, camera_1, side, up, etc.
- Automatically concatenated along channel dimension (default: -3)
- Multi-camera observations improve visual grounding
- Enhances GM loss effectiveness (β*L_GM)

Hardware Safety (Section 6.1):
- max_action_step_size=0.1: SO-101 constraint
- Projects actions to safe region: ||a_t|| ≤ 0.1
- Preserves likelihood while ensuring hardware constraints

SO-101 specific:
- target_hardware='so101'
- max_action_step_size=0.1 (6-DOF robot, 0.1m steps)
- Multi-camera: side + up + front views

Debugging multi-camera:
Look for logs:
  "Concatenating N cameras: [camera_0, camera_1, ...]"
  "Concatenated into shape torch.Size(...)"
If concatenation fails, falls back to single camera gracefully.
""")


# ====================================================================
# EXAMPLE 8: Fast Inference Mode
# ====================================================================

def example_8_fast_inference():
    """
    Reduced steps for real-time deployment.
    """
    cmd = """
lerobot-train --policy.type=mgp \\
  --dataset.repo_id=lerobot/svla_so101_pickplace \\
  --batch_size=2 \\
  --steps=2000 \\
  --policy.use_fast_inference_mode=true \\
  --policy.fast_inference_steps=5
"""
    print("\nEXAMPLE 8: Fast Inference Mode")
    print(cmd)
    print("""
Fast inference (Section 6.1):
- Reduces diffusion steps from default (50) to fast_inference_steps (5)
- 10x faster inference, slight quality loss
- Good for real-time control

Tradeoff:
- Fewer denoising steps = less action refinement
- Better for smooth, well-trained policies
- Suitable for pick-and-place, not fine manipulation

When to use:
- Real-time robot deployment
- Latency critical (< 100ms per step)
- Policy already well-trained

Typical values:
- fast_inference_steps=3-5: Very fast (~20ms)
- fast_inference_steps=10-15: Balanced (~50ms)
- fast_inference_steps=50+: Full quality (~200ms)
""")


# ====================================================================
# HYPERPARAMETER TUNING GUIDE
# ====================================================================

def tuning_guide():
    """
    Systematic tuning strategy for MGP.
    """
    print("\n" + "="*70)
    print("HYPERPARAMETER TUNING GUIDE")
    print("="*70)
    
    print("""
PHASE 1: Get baseline working
---------------------------------
1. Run EXAMPLE 1 (basic training)
2. Check training curves (loss decreasing?)
3. Run inference on held-out trajectory
4. Ensure multi-camera logs appear

PHASE 2: Tune core loss weights
---------------------------------
Step A: Fix α (diffusion) = 1.0, vary others

  Increase β (gm) if:
  - Multi-camera observations seem underused
  - Single camera makes mistakes
  - Observed: loss plateaus but action quality poor
  Decrease if: Training unstable, loss oscillates

  Increase γ (flow) if:
  - Actions too jittery / high variance
  - Want smoother trajectories
  - Many continuous motions (reaching)
  Decrease if: Actions too stiff / slow

Step B: Add components incrementally
  1. Get baseline + GM working
  2. Add flow if needed (smoother)
  3. Add jump if task has discrete phases
  4. Add CTMC if hierarchical

PHASE 3: Component-specific tuning
---------------------------------

Jump process:
  - Start: jump_num_modes=4, jump_rate=0.1
  - Increase modes if task has >4 distinct phases
  - Increase rate if modes switch too slowly
  - Tune δ in 0.05-0.3 range

CTMC:
  - Start: ctmc_num_skills=8, skill_dim=64
  - Increase if skills seem undifferentiated
  - Use with EXAMPLE 4 config

Superposition:
  - Only if above don't work individually
  - Start with EXAMPLE 5 config
  - May need 2x training time

PHASE 4: Reward alignment (if applicable)
---------------------------------
  1. Train base policy (EXAMPLE 1-2)
  2. Evaluate with reward function
  3. Add reward alignment with low temperature (0.5)
  4. Gradually increase temperature if needed
  5. Re-evaluate after each step

PHASE 5: Deployment
---------------------------------
  - Use fast_inference_mode=true if latency matters
  - Keep safety constraints enabled
  - Test on physical robot with small batch first

DEBUGGING CHECKLIST
---------------------------------
[] Training loss decreasing? (if not: learning rate too high/low)
[] Multi-camera concatenation working? (check logs)
[] GM loss > 0? (if 0: no multi-camera observations)
[] Flow/Jump/CTMC loss sensible magnitude? (should be < diffusion loss)
[] Rewards improving over training? (if enabled)
[] Action magnitudes reasonable? (check safety constraints)
[] Inference latency acceptable? (use fast mode if needed)
""")


# ====================================================================
# MAIN EXECUTION
# ====================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("MARKOV GENERATIVE POLICY (MGP) - COMPLETE TRAINING GUIDE")
    print("="*70)
    
    example_1_basic_training()
    example_2_custom_weights()
    example_3_jump_process()
    example_4_ctmc_skills()
    example_5_full_markov_superposition()
    example_6_reward_alignment()
    example_7_safety_and_multicamera()
    example_8_fast_inference()
    tuning_guide()
    
    print("\n" + "="*70)
    print("For more details, see:")
    print("  - configuration_mgp.py: All config parameters with descriptions")
    print("  - modeling_mgp.py: Loss computation and forward pass")
    print("  - _gm_utils.py: Generator implementations and utilities")
    print("="*70)
