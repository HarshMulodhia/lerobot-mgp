# Markov Generative Policies for the SO-101 Robot with LeRobot: Diffusion Policy, Generator Matching, and Reward Alignment

## 1. Overview

This document extends the "Unified Markov Generative Policies" framework originally written for OpenArm in Isaac Lab to the SO-101 robot controlled via LeRobot, while preserving the same notation and Generator Matching (GM) foundations. The SO‑101 is a 6‑DOF open‑source robotic arm natively supported by Hugging Face's LeRobot library, which provides standardized datasets, teleoperation interfaces, and policy training utilities. The central idea remains to treat the policy as a Markov generative model over action trajectories whose generator is a Markov superposition of flow, diffusion, jump, and CTMC components, instantiated concretely by Diffusion Policy and ACT‑style action chunking, now in the LeRobot stack.[^1][^2][^3][^4]

The document proceeds in four parts: (i) task and environment integration for SO‑101 and LeRobot; (ii) algorithmic theory, focusing on Diffusion Policy as a conditional generator in action‑space through the lens of Generator Matching; (iii) a VLA architecture that couples vision–language encoders with Markov‑superposition action heads; and (iv) a reward‑alignment layer that retargets the Markov generator to task‑specific reward‑tilted laws using inference‑time and post‑training alignment methods built on GM.[^5][^6][^1]

## 2. Task and Environment Integration for SO‑101 + LeRobot

### 2.1 SO‑101 Robot Platform and State Notation

Let the SO‑101 follower arm have $n_q = 6$ actuated joints with configuration vector $q_t \in \mathbb{R}^{n_q}$ at time $t$, joint velocities $\dot q_t$, and gripper opening percentage $g_t$. LeRobot exposes these through robot classes such as `SO101Follower` with observations that include joint positions, velocities, and gripper state logged into the LeRobotDataset format. The low‑level proprioceptive state is written as[^2][^7][^8]
$$
 x_t^{\text{prop}} = (q_t, \dot q_t, g_t) \in \mathbb{R}^{d_\text{prop}}, \quad d_\text{prop} = 2 n_q + 1. \tag{2.1}
$$

For control, two action parameterizations are common in LeRobot for SO‑10x arms:[^7][^3]

- Joint‑space actions: $a_t \in \mathbb{R}^{n_q}$ as target joint positions or position deltas, executed through the Feetech motor bus controllers.
- Cartesian end‑effector actions: $a_t \in \mathbb{R}^7$ for pose deltas plus gripper command, analogous to the 7D OpenVLA action vectors.

The action space is denoted by $S_A \subseteq \mathbb{R}^{d_A}$ with $d_A = n_q$ (joint space) or $d_A = 7$ (end‑effector space), and action sequences of horizon $T_p$ are stacked as $S_A^{T_p} \cong \mathbb{R}^{d_A T_p}.$[^1]

### 2.2 Observations, Language, and VLA Conditioning in LeRobot

The observation at time $t$ in a typical LeRobot SO‑101 dataset consists of multi‑camera RGB images, proprioceptive signals, and optional task metadata:[^8][^2]
$$
 O_t = (I_t^{(1)}, \dots, I_t^{(K)}, x_t^{\text{prop}}, m_t),
$$
where $I_t^{(k)}$ are RGB images from K cameras (e.g., wrist, front, overhead), and $m_t$ may include timestamps or calibration indicators. Language instructions or task descriptions are supplied as a tokenized text sequence $C$, in line with OpenVLA and LeRobot's language‑conditioned policies for manipulation.[^9][^10]

The VLA control problem is to learn a policy
$$
 \pi_\Theta(a_t \mid O_{0:t}, C),
$$
parameterized by $\Theta$, which maps vision–proprioception–language context to actions, potentially by predicting multi‑step action chunks rather than single actions.[^3][^1]

### 2.3 LeRobot Environment, Teleoperation, and Reward APIs

LeRobot provides hardware drivers for SO‑101 (leader/follower arms), teleoperation tooling, and a standardized dataset format (LeRobotDataset v3) combining time‑synchronized sensor streams and actions. A typical SO‑101 setup uses:[^2][^8]

- A follower SO‑101 arm connected via Feetech motor bus and calibrated with `lerobot-calibrate --robot.type=so101_follower`.
- Optionally, a leader arm for kinesthetic teleoperation with `so101_leader`.
- Multi‑camera capture via OpenCV or RealSense wrappers integrated with LeRobot cameras.[^7][^2]

Rewards in LeRobot datasets are usually stored as task‑specific scalars or success flags in metadata fields (e.g., `episode/0/meta/success` or dense rewards attached per timestep). For imitation‑only training, these are used primarily for evaluation, but they play a central role in the reward‑alignment layer when retargeting the Markov generator to task‑specific objectives.[^11][^5][^8]

### 2.4 Demonstration Dataset Schema for SO‑101

LeRobotDataset v3.0 stores multi‑modal time‑series data in Parquet plus video or images, but its logical structure matches the HDF5/RLDS schema used in the OpenArm document. An SO‑101 manipulation dataset has episodes indexed by $e$:[^8][^1]
$$
 \{(O_{t,e}, a_{t,e}, r_{t,e}) : t = 0, \dots, T_e - 1,\; e = 1, \dots, N\},
$$
with a directory layout on the Hub such as:

- `observations/cam_main`: encoded MP4 or image sequence, shape $[T, H, W, 3]$.
- `observations/qpos`: $\mathbb{R}^{T \times n_q}$ joint positions.
- `actions/qpos_target` or `actions/ee_action`: $\mathbb{R}^{T \times d_A}$ action vectors.
- `rewards/r_task` and `meta/success`: scalar reward per step and episodic success label.

This matches the ACT and Diffusion Policy conventions and allows direct reuse of dataset utilities from LeRobot, including automatic frame alignment and streaming from the Hugging Face Hub.[^4][^8]

### 2.5 VLA‑Level Problem Formulation for SO‑101

At the VLA level, the architecture mirrors OpenVLA: a multimodal backbone $F$ fuses visual tokens and text tokens into a sequence of hidden states, and an action head maps these to a Markov generator over actions.[^9][^1]
$$
 h_t = F(v_t, w, x_t^{\text{prop}}), \quad a_t \sim \pi_\Theta(\cdot \mid h_t),
$$
where $v_t$ are image patch tokens, $w$ are instruction tokens, and $x_t^{\text{prop}}$ is optionally projected to tokens and concatenated. The policy $\pi_\Theta$ is realized as a Markov‑superposition generator in action space $S_A$, with component generators for flow, diffusion, and jump/ACT as detailed later.[^6][^1][^9]

## 3. Generator Matching Theory for SO‑101 Policies

### 3.1 Probability Paths in Action Space

Let the state space be the action sequence space $S = S_A^{T_p}$. The data distribution $p_{\text{data}}$ on $S$ is the empirical distribution of demonstration action sequences from LeRobot SO‑101 datasets. A simple prior $p_{\text{simple}}$ (e.g., standard Gaussian over $\mathbb{R}^{d_A T_p}$) and a conditional probability path $(p_t(dx \mid z))_{0 \le t \le 1}$ are chosen such that[^1][^8]
$$
 p_0(dx \mid z) = p_{\text{simple}}(dx), \quad p_1(dx \mid z) = \delta_z(dx),
$$
for each data point $z \in S$. Sampling $z \sim p_{\text{data}}$ and then $x \sim p_t(\cdot \mid z)$ induces a marginal path $(p_t)_{0 \le t \le 1}$ with $p_0 = p_{\text{simple}}$ and $p_1 = p_{\text{data}}$, recovering the standard probabilistic view of diffusion models and flow matching in action space.[^5][^6][^1]

Common path choices include mixture paths and Gaussian CondOT paths, which are directly reused from the OpenArm document and the GM companion notes.

### 3.2 Markov Generators and Kolmogorov Forward Equation

A continuous‑time Markov process $(X_t)_{0 \le t \le 1}$ on $S$ is governed by a generator $L_t$ acting on test functions $f$ via
$$
 [L_t f](x) = \lim_{h \to 0} \frac{ \mathbb{E}[f(X_{t+h}) \mid X_t = x] - f(x)}{h}. \tag{3.1}
$$
The adjoint generator $L_t^*$ drives the time evolution of densities $p_t$ through the Kolmogorov Forward Equation (KFE): $\partial_t p_t = L_t^* p_t$, which unifies the continuity equation, Fokker–Planck equation, and jump equations. In GM, $L_t$ is parameterized linearly in a feature operator $K$ and time‑dependent neural parameters $F_t$, which will later be realized as VLA‑conditioned networks.[^5][^1]

### 3.3 Generator Decomposition and Markov Superposition

On Euclidean state spaces $S = \mathbb{R}^{d_A T_p}$, any valid continuous‑time generator decomposes into flow, diffusion, and jump components:[^1][^5]
$$
 L_t = L^{\text{flow}}_t + L^{\text{diff}}_t + L^{\text{jump}}_t.
$$

- $L^{\text{flow}}_t$ corresponds to an ODE drift field $u_t(x)$, modeling deterministic behavior cloning.
- $L^{\text{diff}}_t$ corresponds to an SDE with drift and diffusion tensor, modeling multimodal action distributions (Diffusion Policy).
- $L^{\text{jump}}_t$ corresponds to jumps or CTMC‑like transitions, modeling abrupt strategy changes or discrete mode switches (e.g., between grasp types or high‑level skills).

Markov superposition states that convex combinations of generators remain valid generators, giving a principled way to ensemble multiple component policies for SO‑101 in a single Markov family.[^5]

### 3.4 Generator Matching Loss and Conditional Generator Matching

Generator Matching defines a Bregman‑divergence loss between the infinitesimal evolution induced by $L_t$ and the target evolution consistent with $p_t$. Conditional Generator Matching (CGM) applies this at the level of conditional paths $p_t(\cdot \mid z)$ and yields scalable objectives that reduce to familiar MSE or KL‑like losses in standard diffusion and flow‑matching settings.[^6][^5]

For SO‑101 policies, the conditional variable $z$ consists of observation and language context $(O_{0:t}, C)$, and the state variable is the action sequence or action chunk to be generated. CGM therefore yields training objectives that directly match the conditional Markov generator of the policy to the demonstration‑derived probability path in action space.

## 4. Diffusion Policy for SO‑101 as Conditional Generator Matching

### 4.1 Action Space as State Space for Diffusion over SO‑101 Actions

Diffusion Policy treats the robot policy as a conditional denoising diffusion model in action space, and has been successfully applied to contact‑rich manipulation on low‑cost arms similar to SO‑101. Let $S = S_A^{T_p}$ be the space of action sequences of length $T_p$. For each observation context $O_t$ (which may stack several recent frames) the model aims to approximate $p_{\text{data},A}(A_0 \mid O_t)$ using a Gaussian forward process and a learned reverse denoising process.[^11][^3][^1]

The forward noising process for a given $O_t$ is
$$
 p_0(A \mid O_t) = \mathcal{N}(0, I), \quad p_1(A \mid O_t) \approx p_{\text{data},A}(A_0 \mid O_t),
$$
realized by discrete steps $A_k = \alpha_k A_0 + \sigma_k \varepsilon$ with $k \in \{1, \dots, K\}.$[^11][^1]

### 4.2 Reverse Process and Diffusion Generator

The reverse process is a discretized Langevin‑like update
$$
 A_{k-1} = \alpha_k (A_k - \gamma_k \varepsilon_\theta(O_t, A_k, k)) + \eta_k, \quad \eta_k \sim \mathcal{N}(0, \sigma_k^2 I), \tag{4.1}
$$
where $\varepsilon_\theta$ is the learned noise‑prediction network conditioned on $O_t$. In the continuous‑time limit, this defines a diffusion generator $L_t^{\text{diff}}$ whose drift depends on the learned score $\nabla_A \log p_t(A \mid O_t)$, making Diffusion Policy a special case of CGM with quadratic Bregman divergence.[^6][^11][^1][^5]

### 4.3 Training Objective as Conditional Generator Matching

The standard Diffusion Policy loss on LeRobot SO‑101 datasets is the noise‑prediction objective
$$
 L_{\text{DP}}(\theta) = \mathbb{E}_{k, A_0, \varepsilon, O_t} \big[ \lVert \varepsilon_\theta(O_t, A_k, k) - \varepsilon \rVert_2^2 \big], \tag{4.2}
$$
which is equivalent to conditional score matching for the Gaussian CondOT path. In GM language, this is precisely CGM with state space $S = S_A^{T_p}$, forward kernel given by the diffusion scheduler, and Bregman divergence defined by squared Euclidean distance.[^6][^11][^1][^5]

### 4.4 Network Architecture for SO‑101 and Receding‑Horizon Control

On SO‑101, two architecture variants are practical:

- **Image‑centric Diffusion Policy**: A CNN or ViT‑based encoder processes stacked RGB observations from SO‑101 cameras into a latent feature map, which is fused with proprioceptive features and optional language embeddings before being fed into a temporal U‑Net over action sequences.[^3][^11]
- **Transformer‑based Diffusion Policy**: A Transformer over time operates jointly on image tokens, proprio tokens, and action tokens, similar to OpenVLA backbones, enabling longer horizons and richer temporal dependencies.[^9][^1]

Receding‑horizon control samples an action sequence $A^0_t$ via $K$ denoising steps but executes only the first $H \le T_p$ actions on the physical SO‑101 arm before re‑planning with new observations, implementing an MPC‑like Markov policy under the learned $L_t^{\text{diff}}$.[^11][^1]

### 4.5 Integration into the LeRobot Execution Stack

An SO‑101 Diffusion Policy controller built on LeRobot operates as:

1. Query LeRobot's observation API to construct $O_t$ from camera frames and proprioceptive state.
2. Condition the diffusion model on $O_t$ and sample $A^K_t \sim \mathcal{N}(0, I)$.
3. Run $K$ denoising steps using Equation (4.1) to obtain $A^0_t$.
4. Send the first $H$ elements of $A^0_t$ as joint or end‑effector commands to the SO‑101 controller via LeRobot's motor interface.
5. Repeat until the task terminates, forming a Markov process on $S_A^{T_p}$ governed by $L_t^{\text{diff}}$.[^2][^1]

The difference from the original OpenArm/Isaac pipeline is purely at the environment interface: observations and actions now flow through LeRobot's drivers and datasets instead of Isaac Lab simulation APIs.[^4][^2]

## 5. VLA Architecture for SO‑101 with Markov Superposition

### 5.1 Vision and Language Encoders in the LeRobot Context

The VLA backbone for SO‑101 may reuse OpenVLA's dual‑stream vision encoders (e.g., SigLIP plus DINOv2) and LLM backbone, or lighter variants distilled for embedded deployment. Images from SO‑101 cameras are tokenized via ViT patch embeddings, concatenated across views, and optionally projected into the LLM embedding space through a small MLP projector. Text instructions are tokenized and embedded using the same LLM, then fused with visual tokens in a Transformer to produce hidden states $h_t$ that condition the action generators.[^10][^9]

### 5.2 Action Tokenization vs Continuous Actions on SO‑101

Two strategies for actions are considered:

1. **Tokenized actions**: Map joint or end‑effector actions to discretized bins and treat them as action tokens; a CTMC‑style generator in token space models transition probabilities, connecting to Discrete Flow Maps and DRIFT for reward‑aligned discrete control.[^12][^5]
2. **Continuous actions**: Use the VLA backbone purely as a feature extractor, feeding $h_t$ into continuous‑action generators like Diffusion Policy and ACT; this is the default for high‑precision manipulation with SO‑101.[^1][^11]

The Markov‑superposition viewpoint accommodates both: continuous components $L^{\text{flow}}_t$ and $L^{\text{diff}}_t$ operate on $\mathbb{R}^{d_A T_p}$, while discrete/CTMC components handle tokenized high‑level modes (e.g., "reach", "grasp", "place").[^12][^5]

### 5.3 Action Heads: Diffusion, ACT, Flow, and CTMC Components

Given $h_t$ from the VLA backbone, the action generator on SO‑101 is a Markov superposition
$$
 L_t^{\text{VLA}} = w_{\text{diff}}(h_t) L_t^{\text{diff}} + w_{\text{ACT}}(h_t) L_t^{\text{ACT}} + w_{\text{flow}}(h_t) L_t^{\text{flow}} + w_{\text{CTMC}}(h_t) L_t^{\text{CTMC}}, \tag{5.1}
$$
where a gating network $g(h_t)$ outputs convex weights $w_\cdot(h_t)$. Each component is:

- **Diffusion head**: Implements $L_t^{\text{diff}}$ via a Diffusion Policy noise‑predictor $\varepsilon_\theta(h_t, A_k, k)$ trained with Equation (4.2).[^11][^1]
- **ACT (Action Chunking Transformer) head**: Implements $L_t^{\text{ACT}}$ as a jump/diffusion‑like generator over action chunks $A_{t:t+H-1}$ conditioned on short observation histories; it provides structured, temporally coherent multi‑step predictions.[^12][^1]
- **Flow head**: Implements $L_t^{\text{flow}}$ as a deterministic mapping $f_{\text{flow}}(h_t)$ trained with behavior cloning, acting as a stabilizing baseline for simple segments of a task.
- **CTMC head**: Implements $L_t^{\text{CTMC}}$ on a discrete state space of skill or behavior modes, using rate matrices or Discrete Flow Maps; this supports mode switching and reward‑aligned high‑level policies.[^12][^5]

In practice, Diffusion and ACT heads will be implemented as LeRobot policies (`DiffusionPolicySO101`, `ACTPolicySO101`) that share early layers with the VLA backbone.

### 5.4 Training Strategies: Pretraining, Fine‑Tuning, and Markov Superposition

A practical training pipeline for SO‑101 VLA Markov generative policies is:

1. **Backbone pretraining**: Pretrain the VLA backbone on large multi‑robot datasets (e.g., Open X‑Embodiment, LeRobot Hub datasets including SO‑101 manipulations), using masked‑action prediction and contrastive cross‑modal objectives.[^10][^9]
2. **Component policy training**: Train Diffusion and ACT heads on SO‑101 LeRobot datasets using $L_{\text{DP}}$ and $L_{\text{ACT}}$ respectively; optionally train flow and CTMC components on the same data.
3. **Superposition weight learning**: Freeze or slowly adapt component heads and train the gating network $g(h_t)$ to output $w(h_t)$ that minimize task‑level imitation losses or RL objectives on rollouts, implementing Markov superposition at the generator level.[^5][^6]

This modular training makes it possible to reuse strong generic components (e.g., a general Diffusion Policy) while specializing superposition weights and ACT chunks for particular SO‑101 tasks.

## 6. Reward Alignment for SO‑101 Under Generator Matching

### 6.1 Reward‑Tilted Terminal and Path‑Space Laws

Reward alignment introduces a reward function $r : S \to \mathbb{R}$ or a trajectory reward functional $R(X_{0:1})$ and seeks a new law that concentrates more probability mass on high‑reward states while remaining close to the pretrained $p_1$ or path law. The canonical terminal reward‑tilted distribution is the Gibbs tilt[^5][^11]
$$
 \pi_1(x) = \frac{1}{Z} p_1(x) e^{\beta r(x)}, \quad Z = \int p_1(x) e^{\beta r(x)} dx, \tag{6.1}
$$
where $\beta > 0$ controls alignment strength. In the path‑space setting, the aligned law over trajectories is[^11]
$$
 d\mathbb{P}_{\text{align}}(X_{0:1}) = d\mathbb{P}_{\text{base}}(X_{0:1}) e^{R(X_{0:1})},
$$
which corresponds to a Feynman–Kac reweighting of the base Markov process.[^5][^11]

On SO‑101, rewards are defined at different levels:

- Terminal success (e.g., object in goal region, binary success flag).
- Dense shaping from environment metrics (e.g., distance to goal, gripper closure metrics).
- Task‑specific utilities (e.g., smoothness penalties, energy costs, safety margins).

### 6.2 Alignment Strategies: Inference‑Time vs Post‑Training

Reward alignment methods are organized into two broad classes:[^6][^11]

1. **Inference‑time alignment**: Modify sampling from the pretrained generator without updating its parameters. This includes SMC/beam search with value functions, gradient‑based guidance, and sampling schemes built on GLASS and Diamond Maps.
2. **Post‑training alignment**: Update the generator parameters using reward feedback, either via offline importance‑weighted objectives (EGM) or online RL (Flow‑GRPO, DRIFT), potentially followed by distillation into a cheaper map.

For SO‑101, inference‑time methods are attractive initially because they preserve the pretrained imitation behavior and exploit external reward models (e.g., safety classifiers, task success estimators) without additional training of the generative model. Post‑training methods are used once reliable reward signals are available and computational budgets allow fine‑tuning.[^6][^11]

### 6.3 Inference‑Time Alignment for SO‑101 Diffusion Policies

Inference‑time alignment in diffusion models can be cast as approximating the soft optimal denoising policy, which reweights the pretrained reverse transitions according to a soft value function $v_t^r$. For SO‑101 action diffusion, the reverse kernel at step $k$ is modified as[^6]
$$
 \tilde p_{k-1}(A_{k-1} \mid A_k) \propto p_{k-1}^{\text{pre}}(A_{k-1} \mid A_k) v_{k-1}^r(A_{k-1}), \tag{6.2}
$$
where $v_{k-1}^r(A_{k-1})$ estimates expected task reward at the end of the trajectory when starting from $A_{k-1}$.[^6]

Practical methods for SO‑101 include:

- **Sequential Monte Carlo (SMC) over action trajectories**: Maintain a particle set of candidate action sequences, propagate each backward diffusion step via the pretrained reverse kernel, reweight by estimated reward or value, and resample.[^6]
- **Beam search over trajectories**: At each diffusion step, generate multiple candidate predecessors per particle, evaluate approximate value $v_{k-1}^r$, and keep the top‑K partial trajectories, trading compute at inference time for higher reward.[^6]
- **Gradient‑based guidance using differentiable value functions**: Train a value network $v(A_k, k, O_t)$ that predicts reward from noisy actions and add its gradient to the reverse drift, analogous to classifier guidance but now in action space.[^6]

The GLASS and Diamond Maps frameworks provide efficient stochastic transition samplers and posterior estimators that can be used to implement these procedures without reverting to full SDE sampling, keeping inference efficient on SO‑101 hardware.[^5]

### 6.4 Post‑Training Alignment via Generator Retargeting

Post‑training alignment views reward‑aligned control as learning a new generator $L_t^{\text{r}}$ whose path law approximates the reward‑tilted target while staying close to the base generator. For continuous SO‑101 actions, Flow‑GRPO exemplifies this: the deterministic flow corresponding to a rectified flow model is converted into an SDE with matching marginals; then an RL objective with KL regularization against the base flow is used to update the velocity field.[^11][^5]

The generic structure of such objectives is
$$
 \max_\Theta \; \mathbb{E}_{X_{0:1} \sim L_\Theta} [R(X_{0:1})] - \lambda \, \text{Reg}(L_\Theta, L_{\text{base}}),
$$
where $\text{Reg}$ is typically a KL or Wasserstein divergence on transitions or path laws. In discrete spaces (e.g., CTMC skill modes), DRIFT performs a similar offline‑to‑online generator retargeting with advantage‑weighted discrete flow matching.[^12][^5]

For SO‑101, these methods can be instantiated at different levels:

- Continuous generator retargeting for joint‑space diffusion and flow components.
- Discrete generator retargeting for CTMC skill‑mode policies using discrete flow matching and DRIFT.

### 6.5 Energy‑Based Generator Matching and Reward as Energy

Energy‑based Generator Matching (EGM) considers reward as energy, defining an unnormalized target law $\pi(x) \propto e^{-E(x)}$ with $E(x) = - r(x)$, and trains a generator to sample from this law using importance‑weighted estimates of expectations under $\pi$. This is especially useful when the aligned target is known only up to a normalizing constant, which is common when SO‑101 rewards combine multiple objectives and safety constraints.[^11]

EGM extends the GM framework to general state spaces and directly implements reward alignment without requiring explicit data distributions, making it naturally compatible with SO‑101 path‑space rewards defined in simulation or on real hardware.[^11]

## 7. Synthesis Matrix for SO‑101 Markov Generators

The following synthesis matrix parallels the OpenArm table but is tailored to SO‑101 with LeRobot:

| Component | State Space $S$ | Generator / Path | Objective | Role in SO‑101 VLA |
|----------|--------------------|-------------------|-----------|---------------------|
| Flow / ODE | $\mathbb{R}^{d_A}$ or $\mathbb{R}^{d_A H}$ | $[L_t f](x) = \nabla f \cdot u_t(x)$ (CondOT path) | Behavior cloning, GM loss | Smooth baseline actions; stabilizing low‑variance behavior, e.g., simple reaching motions. |
| Diffusion | $\mathbb{R}^{d_A T_p}$ | SDE generator with drift from score $\nabla_A \log p_t$ | DDPM MSE (score matching) | Handles multimodal grasping and placement trajectories; robust to observation noise and dataset multimodality. |
| Jump Process | $\mathbb{R}^{d_A H}$ | $\int (f(y) - f(x)) Q_t(dy \mid x)$ | GM jump losses | Models abrupt strategy shifts, such as regrasp attempts or switching approach angles. |
| CTMC | Discrete skills / tokens | $\sum_y Q_t(y \mid x)(f(y) - f(x))$ | CTMC/DFM loss | High‑level mode switching (reach, grasp, place, retract); connects to discrete flow matching and DRIFT. |
| ACT Policy | $\mathbb{R}^{d_A H}$ | Near‑deterministic within chunk, jump‑like at boundaries | CVAE loss (KL + reconstruction) | Precise chunked motion, especially for fine‑grained, temporally extended manipulations; complements diffusion. |

This matrix clarifies how the Markov‑superposition VLA architecture combines the strengths of different generator types for SO‑101 tasks.

## 8. Execution Pipeline for SO‑101 with LeRobot

### 8.1 Phase I: Hardware, Environment, and Data

1. **Provision SO‑101 hardware and LeRobot**: Assemble the SO‑101 arm, configure motors and calibration using `lerobot-setup-motors` and `lerobot-calibrate`, and install the LeRobot Python package with Feetech extras.[^4][^2]
2. **Define tasks**: Specify manipulation tasks (reaching, pick‑and‑place, drawer opening, tool use) and set up multi‑camera capture (e.g., wrist and overhead views) consistent with the VLA encoder input resolution.[^3]
3. **Design action space**: Choose joint vs end‑effector parameterization and decide on action horizon $T_p$, chunk length $H$, and history length $L$ to match Diffusion and ACT architectures.
4. **Collect demonstrations**: Use teleoperation (leader arm or gamepad) via LeRobot's recording utilities to log successful episodes into LeRobotDataset v3 format on the Hub (e.g., `yourname/so101_task_dataset`).[^8][^2]

### 8.2 Phase II: Training Component Policies

1. **Train Diffusion Policy**: Implement a dataset loader that returns stacked observations and action sequences; train the conditional diffusion network using $L_{\text{DP}}$ with standard schedulers, monitoring task‑level success on validation episodes.[^3][^11]
2. **Train ACT**: Adapt the dataset into ACT's expected chunked format, configure chunk length $H$, history $L$, and latent dimension, and train the CVAE‑Transformer with $L_{\text{ACT}}$; integrate it into LeRobot as a policy class for deployment.[^3][^12]
3. **Optional Flow/CTMC heads**: Train a behavior‑cloning baseline for deterministic actions and, if using discrete skills, train CTMC/DFM‑style policies with KL‑based consistency losses for non‑autoregressive sequence generation.[^12][^5]

### 8.3 Phase III: VLA Backbone and Markov Superposition

1. **Backbone pretraining**: Initialize from OpenVLA or a similar model and adapt it to SO‑101 image domains via self‑supervised or imitation‑based fine‑tuning on LeRobot datasets.[^10][^9]
2. **Conditioned component training**: Replace the standalone encoders of Diffusion and ACT with the shared VLA backbone, so $h_t$ conditions all heads; fine‑tune with joint losses $L_{\text{DP}} + L_{\text{ACT}}$ plus GM consistency regularizers.[^1][^5]
3. **Superposition weight learning**: Train the gating network $g(h_t)$ end‑to‑end on rollouts, either by minimizing imitation losses or by maximizing reward with KL regularization to preserve base behavior; this yields the final Markov generator $L_t^{\text{VLA}}$ in Equation (5.1).[^5][^11]

### 8.4 Phase IV: Reward Alignment, Evaluation, and Sim‑to‑Real

1. **Reward design and estimation**: Define task‑level rewards, including safety and smoothness terms; train reward models if necessary (e.g., preference‑based or classifier‑based metrics on trajectories).[^11]
2. **Inference‑time alignment experiments**: Apply SMC, beam search, or gradient‑guided sampling on top of the pretrained SO‑101 diffusion generator to explore reward‑aligned behavior without retraining the base generator.[^6]
3. **Post‑training alignment and distillation**: Once aligned samplers are understood, perform Flow‑GRPO or EGM‑style generator retargeting to learn reward‑aligned velocity fields or generators and, if needed, distill them into cheaper flow maps compatible with SO‑101 real‑time constraints.[^5][^11]
4. **Sim‑to‑real and hardware deployment**: Validate the aligned Markov policy on real SO‑101 hardware using LeRobot's runtime, monitoring safety and stability; iteratively collect on‑robot data and refine alignment using GM‑based objectives.[^2][^3]

## 9. Conclusion

This document transposes the Markov Generative Policies framework from OpenArm/Isaac to the SO‑101/LeRobot ecosystem while preserving the unified Generator Matching notation and emphasizing Diffusion Policy as the primary continuous‑action generator. The resulting VLA architecture combines a shared vision–language backbone with Markov‑superposition action heads (Diffusion, ACT, Flow, CTMC) and a GM‑based reward alignment layer capable of both inference‑time and post‑training generator retargeting. This provides a principled, extensible blueprint for building and aligning high‑capacity VLA policies on SO‑101 that can generalize across tasks, reward functions, and deployment regimes while retaining sample quality and safety.[^4][^1][^5][^11]

---

## References

1. [Markov-Generative-Policies-for-OpenArm.pdf](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/88949578/49a9c295-47e5-4922-9b0b-06e5739f0cfd/Markov-Generative-Policies-for-OpenArm.pdf?AWSAccessKeyId=ASIA2F3EMEYE3JVGB4J4&Signature=Ldw%2BRdMirUzwsrFqKVi%2BT4sBdMI%3D&x-amz-security-token=IQoJb3JpZ2luX2VjEIT%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEaCXVzLWVhc3QtMSJHMEUCIEmi19xY4%2Fmb7Twj%2FInFoxX%2BiwEa1TTbvtu3BAcw83CuAiEA1Ue4h67zC%2F3KcRxqT7xzcjIBlHZnW6m2EElchhMGv1cq8wQITRABGgw2OTk3NTMzMDk3MDUiDNMemJ0Hwl7qeTmDyirQBHY%2FSW2JCFFz6psikQhdZ34C34M8ojUMN33NkTsDelOl35KgrQprEGZ8geBAYIqkeFfFWlTESRqveMQm0u3G1owGgaGmmn%2FM6Y3YHU3aqjog%2B%2BbulgPUMLwOUiozuWdW%2FU8mE20xv8XdRmHJru0n5z75%2BjJAlzMmalptV3JZmI5rTvdiS7RKuyvOqR0WgJFuOn8d0E%2Fnyn8iJNSRVI%2BoztpiIP1U%2FTp0Ysy3qpFnxX4osprCmenkeTKuXJuIXcNFskfsw%2FH5I9BIx%2FuNEJNIiQ6hry3JHO1TJUpwPn70xGIGUGew9c%2BXG6aDPBen9cm9SuXUvD0xSNc4Kcx2ys3gAGMtQ%2BEFeli71ijyYjYN%2FTzd12zGyjvok1iofJtzBkudnDM077OyPHSb2dvT2RXywrsW1Z78a5fe%2BCFWkSlY1xehKlWnlJCP39NOgEW79b2%2FlmrdmapNbSftQvGI0ffwmozAtorC7IkfCVhpmkZTyU01L5g0HU2W6081LUqAg5rRr0hx1lJoQFYs%2BK1Z6GcGm7RvbpZ7C39B5V6UrhS8qIt6XqKMalkxwF7MXGf3SvefUTjUJbqyvB4mkad2LNPyT6DR1oc75Oni6lkyk%2Bj2kac7OJuCrUUcuRLVdD0qy8Kf89aIsrKLykWLaIons4ux9tDe3AKPY44UxPYAgi1WRwhJN6uUqa2Uiu92cjFASAnMo0kAMTjDyTmLlqcCm9HXLJPbHoANdWBQaPODnR53yY7kwNgCcvmXSnWB%2BIrnMQPIe3Tceq4p0sjYDqmckacNB8owj%2B6D0QY6mAHubySTAsLg%2BV%2BOsS3L58x%2FXif58Ctm0ckypZrZKYTnKqRVitbV363MYwyQ6tnkBRyHvFT2HS48wx5wdulJRnpJTqcq%2BP89H64spoae5mMkp7KtuEat%2BvEitNIMzylEqNasP89oiyjZL%2FdVQ68%2F9hlFc4u4lGDBOm1FV6nwzmv263YbFoKVax0Ylf0Cfv%2FFMr062ldUM4KtlA%3D%3D&Expires=1780548834) - page-1 UniedMarkovGenerativePoliciesAVision-Language-ActionFrameworkforOpenArmGeneratorMatching,Dius...

2. [SO-101](https://huggingface.co/docs/lerobot/so101) - In the steps below, we explain how to assemble our flagship robot, the SO-101. Source the parts. Fol...

3. [SO-101 Robot Arm — Hardware Hub | SVRC](https://www.roboticscenter.ai/hardware/so-101/) - The SO-101 is a community-driven, fully open-source 6-DOF arm used in academic labs worldwide. It is...

4. [LeRobot: Making AI for Robotics more accessible with end- ...](https://github.com/huggingface/lerobot) - A standardized, scalable LeRobotDataset format (Parquet + MP4 or images) hosted on the Hugging Face ...

5. [Reward-Alignment-Raw.md](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/88949578/331c1c04-9eea-4fa6-9204-84e02568f6fc/Reward-Alignment-Raw.md?AWSAccessKeyId=ASIA2F3EMEYE3JVGB4J4&Signature=njOOIEIxkmoTSvVpfdwQG8%2BU62g%3D&x-amz-security-token=IQoJb3JpZ2luX2VjEIT%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEaCXVzLWVhc3QtMSJHMEUCIEmi19xY4%2Fmb7Twj%2FInFoxX%2BiwEa1TTbvtu3BAcw83CuAiEA1Ue4h67zC%2F3KcRxqT7xzcjIBlHZnW6m2EElchhMGv1cq8wQITRABGgw2OTk3NTMzMDk3MDUiDNMemJ0Hwl7qeTmDyirQBHY%2FSW2JCFFz6psikQhdZ34C34M8ojUMN33NkTsDelOl35KgrQprEGZ8geBAYIqkeFfFWlTESRqveMQm0u3G1owGgaGmmn%2FM6Y3YHU3aqjog%2B%2BbulgPUMLwOUiozuWdW%2FU8mE20xv8XdRmHJru0n5z75%2BjJAlzMmalptV3JZmI5rTvdiS7RKuyvOqR0WgJFuOn8d0E%2Fnyn8iJNSRVI%2BoztpiIP1U%2FTp0Ysy3qpFnxX4osprCmenkeTKuXJuIXcNFskfsw%2FH5I9BIx%2FuNEJNIiQ6hry3JHO1TJUpwPn70xGIGUGew9c%2BXG6aDPBen9cm9SuXUvD0xSNc4Kcx2ys3gAGMtQ%2BEFeli71ijyYjYN%2FTzd12zGyjvok1iofJtzBkudnDM077OyPHSb2dvT2RXywrsW1Z78a5fe%2BCFWkSlY1xehKlWnlJCP39NOgEW79b2%2FlmrdmapNbSftQvGI0ffwmozAtorC7IkfCVhpmkZTyU01L5g0HU2W6081LUqAg5rRr0hx1lJoQFYs%2BK1Z6GcGm7RvbpZ7C39B5V6UrhS8qIt6XqKMalkxwF7MXGf3SvefUTjUJbqyvB4mkad2LNPyT6DR1oc75Oni6lkyk%2Bj2kac7OJuCrUUcuRLVdD0qy8Kf89aIsrKLykWLaIons4ux9tDe3AKPY44UxPYAgi1WRwhJN6uUqa2Uiu92cjFASAnMo0kAMTjDyTmLlqcCm9HXLJPbHoANdWBQaPODnR53yY7kwNgCcvmXSnWB%2BIrnMQPIe3Tceq4p0sjYDqmckacNB8owj%2B6D0QY6mAHubySTAsLg%2BV%2BOsS3L58x%2FXif58Ctm0ckypZrZKYTnKqRVitbV363MYwyQ6tnkBRyHvFT2HS48wx5wdulJRnpJTqcq%2BP89H64spoae5mMkp7KtuEat%2BvEitNIMzylEqNasP89oiyjZL%2FdVQ68%2F9hlFc4u4lGDBOm1FV6nwzmv263YbFoKVax0Ylf0Cfv%2FFMr062ldUM4KtlA%3D%3D&Expires=1780548834) - This document is written in the same notation and conceptual language as the MIT flowdiffusion lectu...

6. [Inference_Time_Alignment_Tutorial.md](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/88949578/563090ef-3342-4bf3-a8b0-7b4e1ef9d596/Inference_Time_Alignment_Tutorial.md?AWSAccessKeyId=ASIA2F3EMEYE3JVGB4J4&Signature=6LrcSKTmr1uyMTeBhck6dGUbkqQ%3D&x-amz-security-token=IQoJb3JpZ2luX2VjEIT%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEaCXVzLWVhc3QtMSJHMEUCIEmi19xY4%2Fmb7Twj%2FInFoxX%2BiwEa1TTbvtu3BAcw83CuAiEA1Ue4h67zC%2F3KcRxqT7xzcjIBlHZnW6m2EElchhMGv1cq8wQITRABGgw2OTk3NTMzMDk3MDUiDNMemJ0Hwl7qeTmDyirQBHY%2FSW2JCFFz6psikQhdZ34C34M8ojUMN33NkTsDelOl35KgrQprEGZ8geBAYIqkeFfFWlTESRqveMQm0u3G1owGgaGmmn%2FM6Y3YHU3aqjog%2B%2BbulgPUMLwOUiozuWdW%2FU8mE20xv8XdRmHJru0n5z75%2BjJAlzMmalptV3JZmI5rTvdiS7RKuyvOqR0WgJFuOn8d0E%2Fnyn8iJNSRVI%2BoztpiIP1U%2FTp0Ysy3qpFnxX4osprCmenkeTKuXJuIXcNFskfsw%2FH5I9BIx%2FuNEJNIiQ6hry3JHO1TJUpwPn70xGIGUGew9c%2BXG6aDPBen9cm9SuXUvD0xSNc4Kcx2ys3gAGMtQ%2BEFeli71ijyYjYN%2FTzd12zGyjvok1iofJtzBkudnDM077OyPHSb2dvT2RXywrsW1Z78a5fe%2BCFWkSlY1xehKlWnlJCP39NOgEW79b2%2FlmrdmapNbSftQvGI0ffwmozAtorC7IkfCVhpmkZTyU01L5g0HU2W6081LUqAg5rRr0hx1lJoQFYs%2BK1Z6GcGm7RvbpZ7C39B5V6UrhS8qIt6XqKMalkxwF7MXGf3SvefUTjUJbqyvB4mkad2LNPyT6DR1oc75Oni6lkyk%2Bj2kac7OJuCrUUcuRLVdD0qy8Kf89aIsrKLykWLaIons4ux9tDe3AKPY44UxPYAgi1WRwhJN6uUqa2Uiu92cjFASAnMo0kAMTjDyTmLlqcCm9HXLJPbHoANdWBQaPODnR53yY7kwNgCcvmXSnWB%2BIrnMQPIe3Tceq4p0sjYDqmckacNB8owj%2B6D0QY6mAHubySTAsLg%2BV%2BOsS3L58x%2FXif58Ctm0ckypZrZKYTnKqRVitbV363MYwyQ6tnkBRyHvFT2HS48wx5wdulJRnpJTqcq%2BP89H64spoae5mMkp7KtuEat%2BvEitNIMzylEqNasP89oiyjZL%2FdVQ68%2F9hlFc4u4lGDBOm1FV6nwzmv263YbFoKVax0Ylf0Cfv%2FFMr062ldUM4KtlA%3D%3D&Expires=1780548834) - Building on the MIT FlowDiffusion Lecture Notes Part I, Generator Matching Theory Part II, and Rewar...

7. [Visualizing LeRobot (SO-100) using Foxglove](https://foxglove.dev/blog/visualizing-lerobot-so-100-using-foxglove) - SO-100 and SO-101 by RobotStudio are 3D-printable robot arms that are great entry-point hardware for...

8. [LeRobotDataset v3.0](https://huggingface.co/docs/lerobot/en/lerobot-dataset-v3) - LeRobotDataset v3.0 is a standardized format for robot learning data. It provides unified access to ...

9. [Module 0 – Context for Vision-Language-Action Models (With Deep RL & Autonomous Systems Background).md](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/collection_9840a175-2f98-4ee4-a759-5e7c9bfca8e0/00376252-b16a-493e-b8cf-675f2037c77a/Module-0-Context-for-Vision-Language-Action-Models-With-Deep-RL-Autonomous-Systems-Background.md?AWSAccessKeyId=ASIA2F3EMEYE3JVGB4J4&Signature=AhIw9UShS%2FJnZzZ6AyBqHFT7%2BOE%3D&x-amz-security-token=IQoJb3JpZ2luX2VjEIT%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEaCXVzLWVhc3QtMSJHMEUCIEmi19xY4%2Fmb7Twj%2FInFoxX%2BiwEa1TTbvtu3BAcw83CuAiEA1Ue4h67zC%2F3KcRxqT7xzcjIBlHZnW6m2EElchhMGv1cq8wQITRABGgw2OTk3NTMzMDk3MDUiDNMemJ0Hwl7qeTmDyirQBHY%2FSW2JCFFz6psikQhdZ34C34M8ojUMN33NkTsDelOl35KgrQprEGZ8geBAYIqkeFfFWlTESRqveMQm0u3G1owGgaGmmn%2FM6Y3YHU3aqjog%2B%2BbulgPUMLwOUiozuWdW%2FU8mE20xv8XdRmHJru0n5z75%2BjJAlzMmalptV3JZmI5rTvdiS7RKuyvOqR0WgJFuOn8d0E%2Fnyn8iJNSRVI%2BoztpiIP1U%2FTp0Ysy3qpFnxX4osprCmenkeTKuXJuIXcNFskfsw%2FH5I9BIx%2FuNEJNIiQ6hry3JHO1TJUpwPn70xGIGUGew9c%2BXG6aDPBen9cm9SuXUvD0xSNc4Kcx2ys3gAGMtQ%2BEFeli71ijyYjYN%2FTzd12zGyjvok1iofJtzBkudnDM077OyPHSb2dvT2RXywrsW1Z78a5fe%2BCFWkSlY1xehKlWnlJCP39NOgEW79b2%2FlmrdmapNbSftQvGI0ffwmozAtorC7IkfCVhpmkZTyU01L5g0HU2W6081LUqAg5rRr0hx1lJoQFYs%2BK1Z6GcGm7RvbpZ7C39B5V6UrhS8qIt6XqKMalkxwF7MXGf3SvefUTjUJbqyvB4mkad2LNPyT6DR1oc75Oni6lkyk%2Bj2kac7OJuCrUUcuRLVdD0qy8Kf89aIsrKLykWLaIons4ux9tDe3AKPY44UxPYAgi1WRwhJN6uUqa2Uiu92cjFASAnMo0kAMTjDyTmLlqcCm9HXLJPbHoANdWBQaPODnR53yY7kwNgCcvmXSnWB%2BIrnMQPIe3Tceq4p0sjYDqmckacNB8owj%2B6D0QY6mAHubySTAsLg%2BV%2BOsS3L58x%2FXif58Ctm0ckypZrZKYTnKqRVitbV363MYwyQ6tnkBRyHvFT2HS48wx5wdulJRnpJTqcq%2BP89H64spoae5mMkp7KtuEat%2BvEitNIMzylEqNasP89oiyjZL%2FdVQ68%2F9hlFc4u4lGDBOm1FV6nwzmv263YbFoKVax0Ylf0Cfv%2FFMr062ldUM4KtlA%3D%3D&Expires=1780548834)

10. [LeRobot v0.4.0: Supercharging OSS Robot Learning](https://huggingface.co/blog/lerobot-release-v040) - The Hugging Face Robot Learning Course. We're launching a comprehensive, self-paced, and entirely op...

11. [Lecture-Notes-Reward-Alignment-as-an-Application-of-Generator-Matching-Theory.md](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/88949578/7a686b69-642a-4fbd-9519-97715ff3ed27/Lecture-Notes-Reward-Alignment-as-an-Application-of-Generator-Matching-Theory.md?AWSAccessKeyId=ASIA2F3EMEYE3JVGB4J4&Signature=l7L09bTBKkkhQL9kKS9bP0LAZPo%3D&x-amz-security-token=IQoJb3JpZ2luX2VjEIT%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEaCXVzLWVhc3QtMSJHMEUCIEmi19xY4%2Fmb7Twj%2FInFoxX%2BiwEa1TTbvtu3BAcw83CuAiEA1Ue4h67zC%2F3KcRxqT7xzcjIBlHZnW6m2EElchhMGv1cq8wQITRABGgw2OTk3NTMzMDk3MDUiDNMemJ0Hwl7qeTmDyirQBHY%2FSW2JCFFz6psikQhdZ34C34M8ojUMN33NkTsDelOl35KgrQprEGZ8geBAYIqkeFfFWlTESRqveMQm0u3G1owGgaGmmn%2FM6Y3YHU3aqjog%2B%2BbulgPUMLwOUiozuWdW%2FU8mE20xv8XdRmHJru0n5z75%2BjJAlzMmalptV3JZmI5rTvdiS7RKuyvOqR0WgJFuOn8d0E%2Fnyn8iJNSRVI%2BoztpiIP1U%2FTp0Ysy3qpFnxX4osprCmenkeTKuXJuIXcNFskfsw%2FH5I9BIx%2FuNEJNIiQ6hry3JHO1TJUpwPn70xGIGUGew9c%2BXG6aDPBen9cm9SuXUvD0xSNc4Kcx2ys3gAGMtQ%2BEFeli71ijyYjYN%2FTzd12zGyjvok1iofJtzBkudnDM077OyPHSb2dvT2RXywrsW1Z78a5fe%2BCFWkSlY1xehKlWnlJCP39NOgEW79b2%2FlmrdmapNbSftQvGI0ffwmozAtorC7IkfCVhpmkZTyU01L5g0HU2W6081LUqAg5rRr0hx1lJoQFYs%2BK1Z6GcGm7RvbpZ7C39B5V6UrhS8qIt6XqKMalkxwF7MXGf3SvefUTjUJbqyvB4mkad2LNPyT6DR1oc75Oni6lkyk%2Bj2kac7OJuCrUUcuRLVdD0qy8Kf89aIsrKLykWLaIons4ux9tDe3AKPY44UxPYAgi1WRwhJN6uUqa2Uiu92cjFASAnMo0kAMTjDyTmLlqcCm9HXLJPbHoANdWBQaPODnR53yY7kwNgCcvmXSnWB%2BIrnMQPIe3Tceq4p0sjYDqmckacNB8owj%2B6D0QY6mAHubySTAsLg%2BV%2BOsS3L58x%2FXif58Ctm0ckypZrZKYTnKqRVitbV363MYwyQ6tnkBRyHvFT2HS48wx5wdulJRnpJTqcq%2BP89H64spoae5mMkp7KtuEat%2BvEitNIMzylEqNasP89oiyjZL%2FdVQ68%2F9hlFc4u4lGDBOm1FV6nwzmv263YbFoKVax0Ylf0Cfv%2FFMr062ldUM4KtlA%3D%3D&Expires=1780548834) - These notes are written as lecture notes for the reward-alignment layer that sits on top of the prob...

12. [Structured Learning Guide for Vision-Language-Action (VLA) Models.md](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/collection_9840a175-2f98-4ee4-a759-5e7c9bfca8e0/aab38293-fb33-4b75-baed-501b425437b3/Structured-Learning-Guide-for-Vision-Language-Action-VLA-Models.md?AWSAccessKeyId=ASIA2F3EMEYE3JVGB4J4&Signature=gyhASBSCZ%2FR%2FNnnw57FlFNOuoEE%3D&x-amz-security-token=IQoJb3JpZ2luX2VjEIT%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEaCXVzLWVhc3QtMSJHMEUCIEmi19xY4%2Fmb7Twj%2FInFoxX%2BiwEa1TTbvtu3BAcw83CuAiEA1Ue4h67zC%2F3KcRxqT7xzcjIBlHZnW6m2EElchhMGv1cq8wQITRABGgw2OTk3NTMzMDk3MDUiDNMemJ0Hwl7qeTmDyirQBHY%2FSW2JCFFz6psikQhdZ34C34M8ojUMN33NkTsDelOl35KgrQprEGZ8geBAYIqkeFfFWlTESRqveMQm0u3G1owGgaGmmn%2FM6Y3YHU3aqjog%2B%2BbulgPUMLwOUiozuWdW%2FU8mE20xv8XdRmHJru0n5z75%2BjJAlzMmalptV3JZmI5rTvdiS7RKuyvOqR0WgJFuOn8d0E%2Fnyn8iJNSRVI%2BoztpiIP1U%2FTp0Ysy3qpFnxX4osprCmenkeTKuXJuIXcNFskfsw%2FH5I9BIx%2FuNEJNIiQ6hry3JHO1TJUpwPn70xGIGUGew9c%2BXG6aDPBen9cm9SuXUvD0xSNc4Kcx2ys3gAGMtQ%2BEFeli71ijyYjYN%2FTzd12zGyjvok1iofJtzBkudnDM077OyPHSb2dvT2RXywrsW1Z78a5fe%2BCFWkSlY1xehKlWnlJCP39NOgEW79b2%2FlmrdmapNbSftQvGI0ffwmozAtorC7IkfCVhpmkZTyU01L5g0HU2W6081LUqAg5rRr0hx1lJoQFYs%2BK1Z6GcGm7RvbpZ7C39B5V6UrhS8qIt6XqKMalkxwF7MXGf3SvefUTjUJbqyvB4mkad2LNPyT6DR1oc75Oni6lkyk%2Bj2kac7OJuCrUUcuRLVdD0qy8Kf89aIsrKLykWLaIons4ux9tDe3AKPY44UxPYAgi1WRwhJN6uUqa2Uiu92cjFASAnMo0kAMTjDyTmLlqcCm9HXLJPbHoANdWBQaPODnR53yY7kwNgCcvmXSnWB%2BIrnMQPIe3Tceq4p0sjYDqmckacNB8owj%2B6D0QY6mAHubySTAsLg%2BV%2BOsS3L58x%2FXif58Ctm0ckypZrZKYTnKqRVitbV363MYwyQ6tnkBRyHvFT2HS48wx5wdulJRnpJTqcq%2BP89H64spoae5mMkp7KtuEat%2BvEitNIMzylEqNasP89oiyjZL%2FdVQ68%2F9hlFc4u4lGDBOm1FV6nwzmv263YbFoKVax0Ylf0Cfv%2FFMr062ldUM4KtlA%3D%3D&Expires=1780548834)

