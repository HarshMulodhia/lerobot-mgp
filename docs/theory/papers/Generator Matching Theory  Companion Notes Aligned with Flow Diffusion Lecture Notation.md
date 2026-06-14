# Generator Matching Theory: Companion Notes Aligned with Flow/Diffusion Notation

## 1. Goal and High-Level Picture

These notes are a companion to the paper **“Generator Matching: A General Framework For Markov Generative Models” (GM)** and the lecture notes on **flow matching and diffusion models**. The aim is to explain Generator Matching (GM) using the notation and concepts from the lecture notes: probability paths, vector fields, score functions, SDEs, and CTMCs.

At a high level, GM answers the question:

> Given a desired probability path $(p_t)_{0 \le t \le 1}$ that interpolates between a simple prior and the data distribution, how can we **systematically construct and train a Markov process** whose marginals follow $p_t$?

GM does this by:
- Working at the level of **generators** $L_t$ of Markov processes rather than direct transition kernels.
- Using the **Kolmogorov Forward Equation (KFE)** (a unified view of continuity and Fokker–Planck equations) to match a generator to a probability path.
- Providing a **scalable training objective** called **Generator Matching (GM) loss**, implemented in practice as **Conditional Generator Matching (CGM)**, built from **Bregman divergences**.

Flows, diffusion models, flow matching, discrete diffusion models, and jump processes all appear as special cases of this general Markov-generator view.

***

## 2. Generative Modeling via Probability Paths (Recap from Lecture Notes)

The lecture notes model generation as sampling from a **data distribution** $p_{\text{data}}$ over a state space $S\subseteq \mathbb{R}^d$ (e.g., images, videos, molecules). One typically chooses a simple prior $p_{\text{init}}$ (e.g., standard Gaussian) and learns a transformation that maps $p_{\text{init}}$ to $p_{\text{data}}$.

### 2.1 Conditional and Marginal Probability Paths

A **conditional probability path** $(p_t(\mathrm{d}x\mid z))_{0\le t\le 1}$ is a family of distributions that interpolate between a simple base and a deterministic point mass at data point $z$:
- $p_0(\mathrm{d}x\mid z) = p_{\text{simple}}(\mathrm{d}x)$ (e.g., a Gaussian).
- $p_1(\mathrm{d}x\mid z) = \delta_z$.

Given a data distribution $p_{\text{data}}$, this induces a **marginal probability path** via hierarchical sampling:
$$
 z \sim p_{\text{data}}, \quad x \sim p_t(\mathrm{d}x\mid z) \Rightarrow x \sim p_t(\mathrm{d}x).\tag{1}
$$

The marginal path satisfies the boundary conditions:
$$
 p_0 = p_{\text{simple}}, \quad p_1 = p_{\text{data}}.\tag{2}
$$

This is exactly the “conditional/marginal path” picture from the flow matching notes, but written in measure notation $p_t(\mathrm{d}x)$ instead of densities when needed.

### 2.2 Example Paths: Mixtures and Geometric Averages

GM lists two standard constructions (for state space $S$ and data point $z$):

1. **Mixture path (works on arbitrary $S$)**
   $$
   p_t(\mathrm{d}x\mid z)
   = (1-\kappa_t) p_{\text{simple}}(\mathrm{d}x)
     + \kappa_t \, \delta_z(\mathrm{d}x),\tag{3}
   $$
   where $\kappa_t$ is differentiable with $\kappa_0=0,\kappa_1=1$.

2. **Geometric average path (CondOT path on $\mathbb{R}^d$)**
   $$
   p_t(\mathrm{d}x\mid z) = \mathbb{E}_{x_0\sim p_{\text{simple}}} \big[\delta_{\sigma_t x_0 + \alpha_t z}(\mathrm{d}x)\big]
   \iff x_t = \sigma_t x_0 + \alpha_t z,\tag{4}
   $$
   where $\alpha_t,\sigma_t$ satisfy
   $\alpha_0 = 0, \ \alpha_1 = 1, \ \sigma_0 = 1, \ \sigma_1 = 0$.

The Gaussian CondOT path $p_t(x\mid z) = \mathcal{N}(\alpha_t z, \sigma_t^2 I)$ used in the lecture notes is a special case of (4).

**GM Principle 1 (Probability path design).** Choose a prior $p_{\text{simple}}$ and conditional path $p_t(\cdot \mid z)$ such that the induced marginal path $p_t$ satisfies $p_0 = p_{\text{simple}}$ and $p_1 = p_{\text{data}}$.

***

## 3. Markov Processes and Generators

### 3.1 Time-Continuous Markov Processes

Let $S$ be a Polish state space (e.g., $\mathbb{R}^d$ or a discrete set), and $(X_t)_{0\le t\le 1}$ a stochastic process with values in $S$. The process is **Markov** if for all times
$0 \le t_1 < \dots < t_n < t_{n+1} \le 1$ and measurable sets $A \subseteq S$,
$$
\mathbb{P}[X_{t_{n+1}} \in A \mid X_{t_1},\dots,X_{t_n}]
= \mathbb{P}[X_{t_{n+1}} \in A \mid X_{t_n}].\tag{5}
$$

The evolution is described by a **transition kernel** $k_{t+h\mid t}$:
$$
\mathbb{P}[X_{t+h} \in A \mid X_t = x] = k_{t+h\mid t}(A \mid x).\tag{6}
$$

This parallels the CTMC and SDE viewpoints in the lecture notes, but GM treats all state spaces in a unified way.

### 3.2 Generators via Test Functions

Directly parameterizing the full kernel $k_{t+h\mid t}$ is difficult; GM instead works with its **infinitesimal generator**. To define it rigorously, GM uses **test functions**:

- A set $\mathcal{T}$ of bounded, sufficiently regular functions $f : S \to \mathbb{R}$ that uniquely determine probability measures via expectations.

GM defines two actions:
- **Marginal action**:
  $$
  \langle p_t, f \rangle := \int f(x) p_t(\mathrm{d}x) = \mathbb{E}_{X_t\sim p_t}[f(X_t)].\tag{7}
  $$
- **Transition action**:
  $$
  \langle k_{t+h\mid t}, f \rangle(x)
  := \int f(y) k_{t+h\mid t}(\mathrm{d}y \mid x)
  = \mathbb{E}[f(X_{t+h}) \mid X_t = x].\tag{8}
  $$

The **generator** $L_t$ is the first-order derivative of the transition action:
$$
[L_t f](x)
:= \lim_{h\to 0} \frac{\langle k_{t+h\mid t}, f \rangle(x) - f(x)}{h}.\tag{9}
$$

Intuitively, $L_t f$ describes the instantaneous rate of change of the expectation of test function $f$ under the Markov dynamics. This is exactly the level at which the continuity equation and Fokker–Planck equation are defined in the lecture notes.

### 3.3 Examples: Flows, Diffusions, Jumps, CTMCs

GM provides a table of standard Markov model classes and their generators, which match objects you saw in the lecture notes:

Let $f\in C_c^\infty(\mathbb{R}^d)$ (smooth, compact support) and $u_t : \mathbb{R}^d \to \mathbb{R}^d$ a vector field.

1. **Flow (ODE)** $dX_t = u_t(X_t)\,dt$
   $$
   [L_t f](x) = \nabla f(x)^\top u_t(x).\tag{10}
   $$
   The adjoint generator gives the **continuity equation** used in flow matching:
   $$
   \partial_t p_t(x) = -\nabla \cdot (u_t(x) p_t(x)).\tag{11}
   $$

2. **Diffusion (SDE)** $dX_t = u_t(X_t)\,dt + \sigma_t(X_t)\,dW_t$
   with diffusion matrix $\sigma_t(x)\in\mathbb{R}^{d\times d}$, covariance $\Sigma_t(x)=\sigma_t(x)\sigma_t(x)^\top$:
   $$
   [L_t f](x) = \nabla f(x)^\top u_t(x)
   + \tfrac12 \operatorname{Tr}\big(\Sigma_t(x) \, \nabla^2 f(x)\big).\tag{12}
   $$
   The adjoint yields the **Fokker–Planck equation**:
   $$
   \partial_t p_t(x)
   = -\nabla \cdot (u_t p_t)(x)
     + \tfrac12 \nabla^2 : (\Sigma_t p_t)(x).
   \tag{13}
   $$

2. **Jump process on $\mathbb{R}^d$** (non-local generator)
   with **jump measure** $Q_t(\mathrm{d}y; x)$:
   $$
   [L_t f](x) = \int \big(f(y) - f(x)\big) Q_t(\mathrm{d}y; x).\tag{14}
   $$

3. **Continuous-time Markov chain (CTMC) on discrete $S$**
   with rate matrix $Q_t(y; x)$:
   $$
   [L_t f](x) = \sum_{y\in S} Q_t(y; x) f(y), \quad
   Q_t(x; x) = -\sum_{y\ne x} Q_t(y; x).\tag{15}
   $$
   This matches the CTMC construction and rate matrix conditions in Section 7 of the lecture notes.

GM’s key insight: **all these models are just different parameterizations of $L_t$.**

***

## 4. Kolmogorov Forward Equation (KFE)

### 4.1 KFE in Test-Function Form

Using the generator, the marginal path $(p_t)$ of a Markov process satisfies the **Kolmogorov Forward Equation** (KFE):
$$
\partial_t \langle p_t, f \rangle
= \langle p_t, L_t f \rangle,\quad \forall f\in \mathcal{T}.\tag{16}
$$

Equivalently:
$$
\partial_t \mathbb{E}_{X_t\sim p_t}[f(X_t)]
= \mathbb{E}_{X_t\sim p_t}[L_t f(X_t)].\tag{17}
$$

This is the abstract form underlying the continuity equation and Fokker–Planck equation you saw for flows and SDEs.

### 4.2 Adjoint Form: Continuity and Fokker–Planck as Special Cases

When $p_t$ has a density $p_t(x)$ w.r.t. reference measure $\nu$, one defines the **adjoint generator** $L_t^*$ by
$$
\int p_t(x) L_t f(x)\,\nu(\mathrm{d}x)
= \int (L_t^* p_t)(x) f(x)\,\nu(\mathrm{d}x), \quad \forall f.\tag{18}
$$

Then the KFE becomes the **adjoint KFE**:
$$
\partial_t p_t(x) = (L_t^* p_t)(x).\tag{19}
$$

Plugging the concrete generators from Section 3.3 recovers exactly:
- **Continuity equation** for flows: Equation (11).
- **Fokker–Planck equation** for SDEs: Equation (13).

GM adopts the KFE (and its adjoint) as the unifying equation that connects a generator $L_t$ to the desired marginal path $p_t$.

***

## 5. Generator Matching: Principles 1–4

GM introduces four core principles that define the framework.

### 5.1 Principle 1 – Choose a Probability Path

Already discussed: specify $p_{\text{simple}}$ and conditional path $p_t(\cdot \mid z)$ such that the marginal path $p_t$ connects $p_{\text{simple}}$ to $p_{\text{data}}$.

This is exactly what the lecture notes do when defining Gaussian CondOT paths for flow matching and diffusion models.

### 5.2 Principle 2 – Parameterize a Markov Process via Its Generator

Given the KFE, if we can parameterize a generator family $L_t$ such that the corresponding Markov process has marginals $p_t$, we have a valid generative model. GM states:

> **Principle 2.** Parameterize the Markov process via a parameterized generator $L_t^\theta$.

This is the common theme behind:
- Flow matching: parameterize **vector fields** $u_t(x)$ determining $L_t f = \nabla f^\top u_t$.
- Diffusion models: parameterize score or drift, corresponding to specific generator components.
- Discrete diffusion: parameterize a **rate matrix** $Q_t$ for CTMCs.

### 5.3 Theorem 1 – Universal Characterization of Generators

GM’s main structural result describes all possible generators on discrete spaces and on $\mathbb{R}^d$.

> **Theorem 1 (Universal characterization of generators).** Under mild regularity:
> 1. On a finite discrete space $|S|<\infty$, the generator is a rate matrix $Q_t$ and the Markov process is a CTMC.
> 2. On Euclidean space $S=\mathbb{R}^d$, any generator can be written as
> $$
 [L_t f](x) = \underbrace{\nabla f(x)^\top u_t(x)}_{\text{flow}}
 + \underbrace{\tfrac12 \, \nabla^2 f(x) : \Sigma_t(x)}_{\text{diffusion}}
 + \underbrace{\int (f(y)-f(x)) Q_t(\mathrm{d}y; x)}_{\text{jump}},\tag{20}
 $$
 where $u_t$ is a vector field, $\Sigma_t(x)$ a positive semidefinite diffusion matrix, and $Q_t(\mathrm{d}y; x)$ a jump measure.

Thus, **flows, diffusions, and jump processes span the entire design space on $\mathbb{R}^d$**. Most existing generative models only use part of this space (e.g., flows or diffusions with fixed $\Sigma_t$). GM exposes the full space, including **jump models on $\mathbb{R}^d$** which had been largely unexplored.

### 5.4 Principle 3 – Solve the KFE for the Probability Path

The KFE gives the condition for a generator to realize a given probability path:
$$
\partial_t \langle p_t, f \rangle = \langle p_t, L_t f \rangle, \quad \forall f\in \mathcal{T}.\tag{21}
$$

> **Principle 3.** Given a marginal path $(p_t)$, find a generator $L_t$ that satisfies the KFE.

Directly solving for $L_t$ at the marginal level is hard, so GM introduces **conditional generators** instead.

#### 5.4.1 Conditional and Marginal Generators

Assume for each data point $z$ we can find a **conditional generator** $L_t^z$ that realizes the conditional path $p_t(\cdot \mid z)$. That is, if $X_t^z$ follows $L_t^z$ with initial distribution $p_0(\cdot \mid z)$, then $X_t^z \sim p_t(\cdot \mid z)$ for all $t$.

GM shows how to obtain the marginal generator from conditional ones.

> **Proposition 1 (Marginal generator).** Let $(p_t)$ be the marginal path induced by conditional paths $p_t(\cdot \mid z)$. If $L_t^z$ is a generator for the conditional path, then the marginal path is generated by
> $$
> [L_t f](x) = \mathbb{E}_{z\sim p_{1\mid t}(\mathrm{d}z\mid x)} \big[ L_t^z f(x) \big],\tag{22}
> $$
> where $p_{1\mid t}(\mathrm{d}z\mid x)$ is the **posterior** over data points given $x$.

For $S=\mathbb{R}^d$ and decomposition (20), this yields marginal components:
$$
\begin{aligned}
 u_t(x) &= \mathbb{E}[u_t(x\mid z) \mid x],\\
 \Sigma_t(x) &= \mathbb{E}[\Sigma_t(x\mid z) \mid x],\\
 Q_t(\mathrm{d}y; x) &= \mathbb{E}[Q_t(\mathrm{d}y; x\mid z) \mid x].
\end{aligned}\tag{23}
$$

This parallels the **“marginalization trick”** in the lecture notes: marginal vector fields and scores are posterior expectations over conditional objects.

> **Practical consequence.** To get a generator for the marginal path, it suffices to find **conditional generators** that solve the KFE at the conditional level.

### 5.5 Principle 4 – Train by Generator Matching (GM/CGM Losses)

So far, $L_t$ (and thus $F_t$, the parameterization) is unknown. GM proposes a training objective to fit a **parameterized generator** $L_t^\theta$.

#### 5.5.1 Linear Parameterization of Generators

GM formalizes **linear parameterizations** of generators as:
$$
 [L_t f](x) = \langle K f(x), F_t(x) \rangle_V,\tag{24}
$$
where
- $K f(x)$ is a fixed linear map from test functions to a feature space $V$,
- $F_t : S\to \Omega\subset V$ is a parameterized function (e.g., neural network),
- $\langle \cdot, \cdot \rangle_V$ is an inner product.

Examples (matching Section A.6 in GM and the lecture notes):
- Flows: $K f = \nabla f$, $F_t(x) = u_t(x)\in\mathbb{R}^d$.
- Diffusions: $K f = \nabla^2 f$, $F_t(x)=\Sigma_t(x)\in S_d^{++}$.
- Jumps: $K f(x) = y \mapsto f(y)-f(x)$, $F_t(x) = Q_t(\cdot; x)$ a non-negative function.

Similarly, conditional generators are parameterized by $F_t^z(x)$ and the marginal generator by $F_t(x) = \mathbb{E}[F_t^z(x)\mid x]$.

#### 5.5.2 GM Loss and Conditional GM Loss

Define a **Bregman divergence** $D(a,b)$ on $\Omega$ via a convex function $\varphi: \Omega\to \mathbb{R}$:
$$
D(a,b) = \varphi(a) - \big( \varphi(b) + \langle a-b, \nabla \varphi(b) \rangle_V \big).\tag{25}
$$

Examples:
- Squared Euclidean distance: $D(a,b)=\tfrac12\|a-b\|_2^2$.
- KL divergence (for probability vectors) as a Bregman divergence w.r.t. negative entropy.

The **Generator Matching (GM) loss** compares the true marginal parameterization $F_t$ to the learned one $F_t^\theta$:
$$
\mathcal{L}_{\text{GM}}(\theta)
:= \mathbb{E}_{t\sim \text{Unif}[0,1],\, x\sim p_t}\big[ D(F_t(x), F_t^\theta(x)) \big].\tag{26}
$$

However, $F_t(x)$ is unknown in practice. Using conditional parameterizations $F_t^z$ and the induced conditional path $p_t(\cdot\mid z)$, GM defines the **Conditional GM (CGM) loss**:
$$
\mathcal{L}_{\text{CGM}}(\theta)
:= \mathbb{E}_{t\sim \text{Unif}[0,1],\, z\sim p_{\text{data}},\, x\sim p_t(\cdot\mid z)}
\big[ D(F_t^z(x), F_t^\theta(x)) \big].\tag{27}
$$

> **Proposition 2 (Gradient equivalence).** For any Bregman divergence, the GM and CGM losses satisfy
> $$
> \nabla_\theta \mathcal{L}_{\text{GM}}(\theta)
> = \nabla_\theta \mathcal{L}_{\text{CGM}}(\theta).
> \tag{28}
> $$
> Thus, minimizing $\mathcal{L}_{\text{CGM}}$ by SGD implicitly minimizes $\mathcal{L}_{\text{GM}}$.

This is a direct analogue of the **conditional flow matching** and **denoising score matching** results in the lecture notes, where marginal losses equal conditional losses up to constants.

> **Principle 4.** Train the parameterized generator $L_t^\theta$ by minimizing the CGM loss with a Bregman divergence.

Examples of CGM losses for different model classes appear in Table 1 of GM; in particular:
- Flows: MSE between true and learned velocity fields.
- Diffusions: MSE between true and learned diffusion coefficients.
- Jumps and CTMCs: KL-type terms involving rate measures.

***

## 6. Concrete Generative Model Classes Under GM

GM shows how classical and new generative models are instances of Generator Matching.

### 6.1 Flow Matching and Rectified Flows

If we restrict to **pure flows** (no diffusion, no jumps):
$$
[L_t f](x) = \nabla f(x)^\top u_t(x),\tag{29}
$$
then the adjoint KFE becomes the **continuity equation** used in the lecture notes’ Section 3:
$$
\partial_t p_t(x) = -\nabla \cdot (u_t(x) p_t(x)).\tag{30}
$$

For Gaussian CondOT paths, GM recovers **flow matching** and **rectified flows** as direct instances of generator matching with a flow-specific KFE (the continuity equation) and MSE loss on velocities.

### 6.2 Denoising Diffusion Models and Score-Based SDEs

From the GM perspective, a “denoising diffusion model” is a **flow model trained with GM loss**, and stochastic sampling is obtained by adding a divergence-free component via a Langevin-type SDE (predictor–corrector), as in Proposition 3 (see Section 7.1).

GM also notes that one could in principle learn state-dependent diffusion coefficients $\Sigma_t(x)$, which is rarely done in standard diffusion models where $\Sigma_t$ is fixed.

### 6.3 New Model Class: Jump Models on $\mathbb{R}^d$

A major new contribution of GM is to point out **jump processes on Euclidean space** as a legitimate, expressive generative model class, complementing flows and diffusions.

Given a conditional path (e.g., CondOT Gaussian in 1D), GM derives a **pure jump solution** to the KFE. For the CondOT path $p_t(\cdot\mid z)=\mathcal{N}(t z, (1-t)^2)$, the paper constructs a conditional jump rate kernel $Q_t(x' ; x \mid z)$ that also realizes this path:
$$
Q_t(x' ; x \mid z) = \frac{[k_t(x)]_+ [ -k_t(x')]_+ \, p_t(x' \mid z)}{(1-t)^3 \int [ -k_t(\tilde x)]_+ \, p_t(\tilde x \mid z)\,\mathrm{d}\tilde x},\tag{31}
$$
with
$$
 k_t(x) = x^2 - (t+1) x z - (1-t)^2 + t z^2, \quad [a]_+ = \max(a,0).
\tag{32}
$$

Sampling trajectories from this jump model produces very different **sample paths** compared to flows or SDEs, but the marginal histograms over time still match the same probability path.

### 6.4 Pure Diffusion Solution to a Mixture Path

GM also constructs a **pure diffusion solution** (drift-free SDE) to a mixture path of the form
$$
 p_t(\mathrm{d}x\mid z) = \kappa_t \delta_z(\mathrm{d}x) + (1-\kappa_t)\,\text{Unif}[a_1,a_2](\mathrm{d}x).
\tag{33}
$$

The SDE has zero drift and state-dependent diffusion coefficient
$$
\sigma_t^2(x\mid z)
= \frac{2 \dot \kappa_t (a_2-a_1)}{1-\kappa_t} \Bigg[ \tfrac12 \frac{(z-a_1)^2}{a_2-a_1}
+ [x-z]_+ - \tfrac12 \frac{(x-a_1)^2}{a_2-a_1} \Bigg],\tag{34}
$$
plus reflecting boundary conditions at $a_1,a_2$.

This model **only specifies how much noise to add** as a function of $(x,t,z)$ and yet can generate the target distribution, showing that generative modeling is not limited to “denoising” processes with fixed $\sigma_t$.

***

## 7. Combining Generators and Predictor–Corrector Schemes

Because both $L_t$ and the KFE are linear, we can **combine** multiple generators for the same probability path.

> **Proposition 3 (Combining models).** Let $(p_t)$ be a marginal path, and $L_t, L'_t$ two generators that both satisfy the KFE for $p_t$. Then the following are also valid generators:
> 1. **Markov superposition:** $\tilde L_t = \alpha_{1,t} L_t + \alpha_{2,t} L'_t$ with non-negative weights $\alpha_{1,t}+\alpha_{2,t}=1$.
> 2. **Adding divergence-free components:** $\tilde L_t = L_t + \beta_t L^{\text{div}}_t$ where $L^{\text{div}}_t$ satisfies $\langle p_t, L^{\text{div}}_t f \rangle = 0$ for all $f$.
> 3. **Predictor–corrector:** $\tilde L_t = \alpha_{1,t} L_t + \alpha_{2,t} \bar L_t$, where $L_t$ solves KFE forward in time and $\bar L_t$ solves it backward, with $\alpha_{1,t}-\alpha_{2,t}=1$.

Applications:
- **Markov superposition** allows constructing ensembles of generative models (e.g., flow + jump) that share the same marginals but different sample paths.
- **Divergence-free components** include Langevin dynamics and MCMC kernels, which enrich sampling without changing the target distribution.
- **Predictor–corrector** schemes generalize familiar predictor–corrector samplers in diffusion models and discrete diffusion.

GM’s experiments show that **Markov superposition of flows and jump models** can improve FID for image generation compared to flows alone.

***

## 8. Multimodal and High-Dimensional Modeling

GM extends the framework to **product spaces** $S = S_1 \times S_2$ (and higher), which is important for multimodal generative models (e.g., image–text, protein structure + sequence).

> **Proposition 4 (Informal multimodal construction).** Let $q_t^1(\cdot\mid z_1)$ and $q_t^2(\cdot\mid z_2)$ be conditional paths on $S_1$ and $S_2$. Define the factorized conditional path on $S_1\times S_2$ as
> $$
> p_t(\mathrm{d}x_1,\mathrm{d}x_2 \mid z_1,z_2) = q_t^1(\mathrm{d}x_1 \mid z_1)\, q_t^2(\mathrm{d}x_2 \mid z_2).
> \tag{35}
> $$
> Then, to solve the KFE on $S_1\times S_2$, it suffices to solve it individually on each component and combine the resulting generators component-wise.

Consequences:
- We can build **multimodal generative models** (e.g., joint image–text, joint protein structure–sequence) by combining generators on each modality.
- In high dimensions, we can often **reduce KFE solving to 1D** building blocks and then apply this product construction.

Concrete example from GM:
- **Protein generation** on state space $S = \mathbb{R}^d \times SO(3)^d \times \{1,\dots,20\}^d$, representing translations, rotations, and amino-acid types.
- GM derives a **jump solution on $SO(3)$** and combines it with an existing flow model (MultiFlow) in a multimodal space, improving diversity and novelty metrics on protein design benchmarks without retraining the underlying model.

***

## 9. Applications and Experiments in GM

GM’s experiments focus on three aspects:
1. Jump models as a new class in $\mathbb{R}^d$.
2. Combining models via Markov superposition.
3. Multimodal, high-dimensional generative modeling.

### 9.1 Image Generation with Jump and Flow Models

GM implements jump models and flow models on CIFAR-10 and ImageNet-32 using U-Net backbones, then evaluates FID.

From Table 2 in GM:

- Flow model (Euler sampler): FID $\approx 2.94$ on CIFAR-10, $4.58$ on ImageNet-32.
- Jump model (Euler): FID $\approx 4.23$ on CIFAR-10, $7.66$ on ImageNet-32.
- Flow + jump Markov superposition improves over each individually, and a “mixed” sampler combining 2nd-order ODE sampling for flow and Euler for jumps achieves FID improvements beyond previous state-of-the-art flow models.

This demonstrates that the **jump component is useful and complementary** rather than just a theoretical curiosity.

### 9.2 Protein Generation with SO(3) Jumps

For protein structure and sequence generation, GM augments the MultiFlow model on $\mathbb{R}^d\times\text{discrete}$ with an $SO(3)$ jump model for rotations, using GM’s multimodal construction.

Using diversity (share of unique proteins passing designability) and novelty (average inverse similarity) metrics:
- Adding SO(3) jumps improves diversity significantly in both unimodal (structure-only) and multimodal (structure + sequence) settings, while remaining competitive or better on novelty.

This illustrates how GM can **plug in new components into existing models without retraining them end-to-end**, leveraging the Markov generator abstraction.

***

## 10. Discrete Diffusion and CTMC Models as GM Instances

Section 7 of the lecture notes introduces **continuous-time Markov chain (CTMC) models** for discrete diffusion, with rate matrices $Q_t$, mixture paths, and discrete flow/diffusion training objectives.

GM shows that these discrete models are precisely instances of Generator Matching on discrete spaces.

### 10.1 CTMCs and Generators

On a finite state space $S$, a CTMC has generator (rate matrix) $Q_t\in\mathbb{R}^{S\times S}$ with conditions:
- $Q_t(y; x) \ge 0$ for $y\ne x$.
- $Q_t(x; x) = -\sum_{y\ne x} Q_t(y; x)$.

The KFE becomes a system of linear ODEs over state probabilities, matching the discrete diffusion equations in the lecture notes.

### 10.2 Factorized Mixture Paths and Marginal Rate Matrices

For language-like sequences $x=(x_1,\dots,x_d)\in V^d$, both GM and the lecture notes consider a **factorized mixture path** where each position is either kept or replaced with noise according to a Bernoulli with parameter $\kappa_t$.

GM shows that the **marginal rate matrix** has a simple classifier-like form:
$$
Q_t(v_i, j \mid x)
= \frac{\dot \kappa_t}{1-\kappa_t} \big( p_{1\mid t}(z_j=v_i \mid x) - \delta_{x_j}(v_i) \big),\tag{36}
$$
where $p_{1\mid t}(z_j=v_i\mid x)$ is the posterior probability that the clean token at position $j$ equals $v_i$ given noisy sequence $x$.

This implies that learning the rate matrix $Q_t$ is equivalent to learning position-wise denoising classifiers.

### 10.3 Discrete Flow Matching Loss

The lecture notes derive a **Discrete Flow Matching loss** using cross-entropy over tokens, by training a denoising classifier that predicts the original token $z_j$ from noisy $x$.

GM shows that the same loss is an instance of **CGM with KL/Bregman divergence** for the rate matrix parameterization, and connects it to ELBO-style lower bounds on log-likelihood.

Thus, **discrete diffusion, discrete flow matching, and masking-based language diffusion models** all fall under Generator Matching with CTMC generators.

***

## 11. Conceptual Summary and How to Read GM with These Notes

### 11.1 Conceptual Summary

- GM is a **unified theory of Markov generative models** across continuous, discrete, and manifold state spaces.
- It operates at the level of **generators $L_t$** and the **KFE**, which subsumes the continuity and Fokker–Planck equations from the lecture notes.
- It provides a **universal decomposition** of generators on $\mathbb{R}^d$ into flow, diffusion, and jump components (Theorem 1), exposing new design space such as **jump models in Euclidean space**.
- It shows how to **construct generators from probability paths** via conditional generators and posterior averaging (Proposition 1).
- It introduces a **scalable training recipe** (GM/CGM loss) based on linear parameterizations and Bregman divergences, generalizing flow matching and score matching.
- It explains how to **combine models** (Markov superposition, divergence-free components, predictor–corrector) and how to build **multimodal models** on product spaces.
- It demonstrates practical benefits on **image** and **protein generation**, including new jump models and hybrid flow+jump models.

### 11.2 How to Read the GM Paper with These Notes

When reading GM.pdf, you can map its sections to these notes and the lecture notation as follows:

- **Section 2 (Generative modeling via probability paths)** → Review Section 2 here, plus Section 3.1 of the lectures on conditional/marginal paths.
- **Section 3 (Markov processes)** → Review Section 3 and Appendix A of GM, plus Section 2.2 of the lectures for SDEs and Section 7.1 for CTMCs.
- **Section 4 (Generators)** → Study Section 3.3 and Theorem 1 here; compare with the continuity/Fokker–Planck derivations in the lecture notes’ Appendix B.
- **Section 5 (KFE and marginal generator)** → Use Section 4 and 5.4 here, and recall the continuity and Fokker–Planck equations in the lectures.
- **Section 6 (Generator Matching)** → Use Section 5.5 here and the lecture’s Theorem 12/22 analogues on conditional vs marginal losses.
- **Section 7 (Applications)** → See Sections 6–9 here for the concrete jump and multimodal models.
- **Section 8 (Related work) and beyond** → Use Section 10 here to connect to discrete diffusion and CTMC-based language models.

With this mapping, you can read GM.pdf line-by-line and interpret every equation in terms of familiar objects from the lecture notes: probability paths, vector fields, score functions, SDEs, CTMCs, and their generators.

---

## References

1. [lecture_notes.pdf](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/88949578/c224d6ce-4f45-45c1-9946-927f5014fead/lecture_notes.pdf?AWSAccessKeyId=ASIA2F3EMEYER3KXC2ZV&Signature=MqVIxemN4iQFgU0KrKXR0kDHud0%3D&x-amz-security-token=IQoJb3JpZ2luX2VjEJr%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEaCXVzLWVhc3QtMSJHMEUCIQC622sYlhQwm3SIES%2FdMveADbOK6m2ORN2ZAGpOVLT8LAIgdyFcrkKw1tq5Gua%2BhNyWFrpfsljZ%2BJuaI1vZ66recoIq8wQIYxABGgw2OTk3NTMzMDk3MDUiDL4DlfLSFARhRypbEyrQBId0%2FIUDKp001iSsQruS%2F0yE9onpjgxZREtW59YqVx%2BkUTwIlmA08dO3nqI65Vbt81szwtup20egSkgh3u0v4nlbztx7dGrAVghITnHI%2BCPcwOgMr7HsurniLLgzgC7JgR0y93MWSteOTip5tONX4cK3m58aZNZ3vquaeiQXF7Soo5tJchx0nVN2xAVEIdjWg%2BJ9riSAwi3BNemgPEhtMn2jwF%2B%2B8b9So%2Fs0VkqE0D1VWynsbYMuxJ%2FO9sOo2FrbPgcur3Cm46nl31ff16G7%2BX4hwgsBRZqSkPr%2FZ9i7aZx5bP2sbrx1172TjB%2BTHDgvaO%2F%2B50%2Bjh%2FDk%2B2zHbbgqCi3MbE9Fz5yIavfDMOH4bv5aT%2BaEU93G24ilPbHKXOctWtbvxQ31%2FyOqDtSw9QjleNWv6DAD%2B0qj6mwf%2BRmmPaNk8lAOJVCWQhVN5OdqbYo7GQS2n9lm4M68ociSL2QkrznmMBBLcYIz1ra03HsS8NlXkdTTigZX9eCxGE4S%2BWEwrzbe35Ff3c2Mbh%2BvX5pDU69Ab%2FkGYRipY3OQK8IOpdkdDRS%2Fjs3Xz2lvk5kzrXQyGIYkFhhvWcuIzpe%2F5%2Fx4HvMi5VetoOPWyL5EdLJgKoPtrJTZl8cgyMLURmKyIHqdr%2BsRvkNVd6d1C0GIv8jnBCZbXz6iloOTuCd%2F0az%2Fab7jNvQVImZKhJBmHA70Q6tI4H2Tqd8fJinqv%2BP3yuHvmyzY9Ad8nHmQKBPRn1gLG0rDOL1K7PU%2Btz01hQArbC0xZawDIrTpat8SSSJKKYTNUYEwvbvQ0AY6mAG0pnWFCE2agS4jwuVsKGrTITjN3s7pIZWEAq%2BXwyCalZs5%2Fy3Fi8SYfBx3%2BesnJaEplg%2FdcFz0S1GSlAtDcmdh3%2FcrHHgqhiUBmMjzpoSwE8zkYlZ10asQXU%2FpTUUovJLsrXY%2FBJjHHzUmvv%2BcSdFNraWQHsAR4NL%2B6K4vl3Hr2J6KlqsXUTEVLJTyhlysZ1dvvRDj%2FJk5SA%3D%3D&Expires=1779706768) - **page-1**
MITClass6.S184:GenerativeAIWithStochasticDiﬀerentialEquations,2026AnIntroductiontoFlowMat...

2. [GM.pdf](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/88949578/11f9f9dd-6860-456f-95ea-7f7807ed65e8/GM.pdf?AWSAccessKeyId=ASIA2F3EMEYER3KXC2ZV&Signature=KeWIzjidJVFTwuUzLSvdFlwK50I%3D&x-amz-security-token=IQoJb3JpZ2luX2VjEJr%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEaCXVzLWVhc3QtMSJHMEUCIQC622sYlhQwm3SIES%2FdMveADbOK6m2ORN2ZAGpOVLT8LAIgdyFcrkKw1tq5Gua%2BhNyWFrpfsljZ%2BJuaI1vZ66recoIq8wQIYxABGgw2OTk3NTMzMDk3MDUiDL4DlfLSFARhRypbEyrQBId0%2FIUDKp001iSsQruS%2F0yE9onpjgxZREtW59YqVx%2BkUTwIlmA08dO3nqI65Vbt81szwtup20egSkgh3u0v4nlbztx7dGrAVghITnHI%2BCPcwOgMr7HsurniLLgzgC7JgR0y93MWSteOTip5tONX4cK3m58aZNZ3vquaeiQXF7Soo5tJchx0nVN2xAVEIdjWg%2BJ9riSAwi3BNemgPEhtMn2jwF%2B%2B8b9So%2Fs0VkqE0D1VWynsbYMuxJ%2FO9sOo2FrbPgcur3Cm46nl31ff16G7%2BX4hwgsBRZqSkPr%2FZ9i7aZx5bP2sbrx1172TjB%2BTHDgvaO%2F%2B50%2Bjh%2FDk%2B2zHbbgqCi3MbE9Fz5yIavfDMOH4bv5aT%2BaEU93G24ilPbHKXOctWtbvxQ31%2FyOqDtSw9QjleNWv6DAD%2B0qj6mwf%2BRmmPaNk8lAOJVCWQhVN5OdqbYo7GQS2n9lm4M68ociSL2QkrznmMBBLcYIz1ra03HsS8NlXkdTTigZX9eCxGE4S%2BWEwrzbe35Ff3c2Mbh%2BvX5pDU69Ab%2FkGYRipY3OQK8IOpdkdDRS%2Fjs3Xz2lvk5kzrXQyGIYkFhhvWcuIzpe%2F5%2Fx4HvMi5VetoOPWyL5EdLJgKoPtrJTZl8cgyMLURmKyIHqdr%2BsRvkNVd6d1C0GIv8jnBCZbXz6iloOTuCd%2F0az%2Fab7jNvQVImZKhJBmHA70Q6tI4H2Tqd8fJinqv%2BP3yuHvmyzY9Ad8nHmQKBPRn1gLG0rDOL1K7PU%2Btz01hQArbC0xZawDIrTpat8SSSJKKYTNUYEwvbvQ0AY6mAG0pnWFCE2agS4jwuVsKGrTITjN3s7pIZWEAq%2BXwyCalZs5%2Fy3Fi8SYfBx3%2BesnJaEplg%2FdcFz0S1GSlAtDcmdh3%2FcrHHgqhiUBmMjzpoSwE8zkYlZ10asQXU%2FpTUUovJLsrXY%2FBJjHHzUmvv%2BcSdFNraWQHsAR4NL%2B6K4vl3Hr2J6KlqsXUTEVLJTyhlysZ1dvvRDj%2FJk5SA%3D%3D&Expires=1779706768) - **page-2**
PublishedasaconferencepaperatICLR2025Figure1:OverviewoftheGeneratorMatching(GM)frameworkt...

