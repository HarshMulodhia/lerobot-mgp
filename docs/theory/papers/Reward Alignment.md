# Reward Alignment and Transition Sampling for Flow/Diffusion Models

_This document is written in the same notation and conceptual language as the MIT flow/diffusion lecture notes and the Generator Matching (GM) companion notes. It focuses on **new concepts** introduced in recent reward-alignment and transition-sampling papers (GLASS Flows, Flow‑GRPO, Diamond Maps, Discrete Flow Maps), and only recalls previously-covered concepts when needed to state new results._

---

## 1. Setting and Notation

We work on a state space $S\subseteq \mathbb{R}^d$ (e.g., images, latent codes), with a **data distribution** $p_{\mathrm{data}}$ and a simple **prior** $p_{\mathrm{simple}}$, typically a standard Gaussian. As in the lecture notes, generation is cast as sampling along a **probability path** $(p_t)_{0\le t\le 1}$ that interpolates between prior and data.

### 1.1 Conditional and marginal probability paths

From the GM notes, a **conditional probability path** is a family
$$
\bigl(p_t(\mathrm{d}x\mid z)\bigr)_{0\le t\le 1}, \quad z\sim p_{\mathrm{data}},
$$
with
$$
 p_0(\mathrm{d}x\mid z) = p_{\mathrm{simple}}(\mathrm{d}x),\qquad p_1(\mathrm{d}x\mid z) = \delta_z(\mathrm{d}x).
$$
This induces a **marginal path** via hierarchical sampling
$$
 z \sim p_{\mathrm{data}},\quad x\sim p_t(\mathrm{d}x\mid z) \;\Rightarrow\; x\sim p_t(\mathrm{d}x),
$$
with boundary conditions $p_0=p_{\mathrm{simple}},\ p_1=p_{\mathrm{data}}$.[file:1][file:3]

In the **Gaussian CondOT path** used throughout the lecture notes and in the GLASS/Diamond papers, we work with
$$
 x_t = \alpha_t z + \sigma_t \epsilon,\quad \epsilon\sim \mathcal{N}(0,I_d),
$$
so that
$$
 p_t(x_t\mid z) = \mathcal{N}(x_t;\alpha_t z,\sigma_t^2 I_d),
$$
with schedulers $\alpha_t,\sigma_t\ge 0$ satisfying
$$
 \alpha_0=0,\alpha_1=1,\qquad \sigma_0=1,\ \sigma_1=0,\qquad \alpha_t \text{ increasing}, \sigma_t \text{ decreasing}.
$$

### 1.2 Marginal vector field, score, and flow/diffusion sampling

The marginal path $(p_t)$ can be realized either as a **flow** (probability-flow ODE) or as a **diffusion** (time-reversal SDE), using the notation of the lecture notes:

- **Flow (ODE) representation**: a velocity field $u_t: \mathbb{R}^d\to\mathbb{R}^d$ such that
  $$
   X_0 \sim p_0,\qquad \frac{\mathrm{d}}{\mathrm{d}t}X_t = u_t(X_t)\;\Rightarrow\; X_t\sim p_t.
  $$
- **Score-based SDE representation**: a reverse-time SDE
  $$
   \mathrm{d}X_t = \bigl[u_t(X_t) + \tfrac{\nu_t^2}{2}\nabla\log p_t(X_t)\bigr] \mathrm{d}t + \nu_t\,\mathrm{d}W_t,
  $$
  where $\nu_t$ is determined by the schedulers $\alpha_t,\sigma_t$.

The **posterior** of the probability path plays a central role:
$$
 p_{1\mid t}(z\mid x_t) = \frac{p_t(x_t\mid z)p_{\mathrm{data}}(z)}{p_t(x_t)},
$$
with associated **denoiser** (conditional mean)
$$
 D_t(x_t) = \int z\, p_{1\mid t}(z\mid x_t)\,\mathrm{d}z.
$$
The denoiser reparameterizes the velocity field $u_t$ and the score $\nabla\log p_t$ as in the lecture notes.[file:2]

In what follows, we treat the pretrained flow/diffusion model (vector field $u_t$, denoiser $D_t$, or score) as **fixed**, and focus on **reward alignment** and **transition sampling** on top of this model.

---

## 2. Reward Alignment: Reward-Tilted Distributions and Value Functions

### 2.1 Reward-tilted target distribution

In reward alignment, the data distribution $p_{\mathrm{data}}$ is used only as a **prior**; the true target is the **reward-tilted distribution**
$$
 p^r(z) = \frac{1}{Z^r} p_{\mathrm{data}}(z)\exp(r(z)),\qquad Z^r>0,
$$
where $r: S\to\mathbb{R}$ is a user-specified reward (e.g., human preference scores, CLIP, task success metrics).

We typically do not train a new model for $p^r$ from scratch. Instead, we reuse the pretrained probability path $(p_t)$ as a **transport prior**, and steer sampling at inference (or fine-tuning) time to approximate $p^r$.

### 2.2 Noised reward-tilted path and value function

Define the **noised reward-tilted path**
$$
 p_t^r(x_t) := \mathbb{E}_{z\sim p^r}\bigl[p_t(x_t\mid z)\bigr].
$$
Using Bayes’ rule and the conditional path, one can show (Diamond Maps) that the **value function**
$$
 V_t^r(x_t) := \log \mathbb{E}_{z\sim p_{1\mid t}(\cdot\mid x_t)}[\exp(r(z))]
$$
obeys the fundamental identity
$$
 \exp(V_t^r(x_t)) \propto \frac{p_t^r(x_t)}{p_t(x_t)}.
$$
Thus $V_t^r(x_t)$ measures how much more likely the noisy state $x_t$ is under the reward-tilted path than under the original path.

This value function is the central object for many inference-time alignment methods (guidance, SMC, search), because
- its **gradient** w.r.t. $x_t$ gives an **exact guidance update** for the drift, and
- its **value** gives **importance weights** for particles in Sequential Monte Carlo.

### 2.3 Guidance via value function gradient

Replacing $p_{\mathrm{data}}$ by $p^r$ in the flow-matching derivation yields a **reward-tilted drift**
$$
 u_t^r(x_t) = u_t(x_t) + b_t\,\nabla_{x_t}V_t^r(x_t),
$$
where the scalar factor $b_t$ depends only on the schedulers:
$$
 b_t = \sigma_t^2\frac{\dot{\alpha}_t}{\alpha_t} - \dot{\sigma}_t\sigma_t.
$$
Thus
- original drift $u_t$ steers samples from $p_0$ to $p_1=p_{\mathrm{data}}$, while
- the additional term $b_t\nabla V_t^r$ steers samples toward the reward-tilted path $p_t^r$.

In practice, $V_t^r$ is intractable to evaluate exactly, because it requires sampling from the posterior $p_{1\mid t}(z\mid x_t)$. The papers below propose **efficient approximations** based on stochastic transitions and flow maps.

---

## 3. Transition Sampling and GLASS Flows

Reward alignment algorithms (search, SMC, value-estimation-guided guidance) operate not just on endpoints $X_0, X_1$, but on **Markov transitions** along the probability path.

### 3.1 Transition kernels $p_{t'\mid t}$

For times $0\le t < t'\le 1$, define the **transition kernel**
$$
 p_{t'\mid t}(x_{t'}\mid x_t) := \mathbb{P}[X_{t'}=x_{t'}\mid X_t=x_t].
$$

- **ODE sampling** (flow matching / probability-flow ODE) is deterministic: $X_{t'}$ is a deterministic function of $X_t$. This gives **no transition stochasticity**.
- **SDE (DDPM) sampling** has stochastic transitions: $X_{t'}$ is random given $X_t$, with a law $p^{\mathrm{DDPM}}_{t'\mid t}(\cdot\mid x_t)$.

Many alignment algorithms require drawing samples from $p_{t'\mid t}$ (e.g., for branching search trees or SMC proposals), so **transition stochasticity** is essential.

However, SDE sampling is much less efficient than ODE sampling in practice. GLASS Flows resolve this tension.

### 3.2 GLASS transitions: Gaussian latent sufficient statistic

GLASS (Gaussian Latent Sufficient Statistic) Flows define a family of **Gaussian transitions** between $t$ and $t'$ induced by the CondOT path $x_t = \alpha_t z + \sigma_t \epsilon$.

Fix $z\in\mathbb{R}^d$. Consider the joint two-time vector
$$
 \mathbf{X} = (X_t,X_{t'})^T \in \mathbb{R}^{2d}.
$$
Conditional on $z$, each coordinate pair $(X_t^j,X_{t'}^j)$ is bivariate Gaussian with mean
$$
 \mu = \begin{pmatrix}\alpha_t\\ \alpha_{t'}\end{pmatrix}
$$
and covariance
$$
 \Sigma = \begin{pmatrix}\sigma_t^2 & \rho\sigma_t\sigma_{t'}\\ \rho\sigma_t\sigma_{t'} & \sigma_{t'}^2\end{pmatrix},
$$
where **correlation parameter** $\rho\in[-1,1]$ is a free degree of freedom.

This defines the joint law
$$
 p_{t,t'}(X_t,X_{t'}\mid z) = \prod_{j=1}^d \mathcal{N}((X_t^j,X_{t'}^j); z^j\mu,\Sigma),
$$
which induces a **transition kernel**
$$
 p_{t'\mid t}(X_{t'}\mid X_t) = \frac{p_{t,t'}(X_t,X_{t'})}{p_t(X_t)}.
$$
This **GLASS transition** family preserves the marginal path $p_t$ for all $\rho$.

**Special case: DDPM transitions.** For
$$
 \rho_* = \frac{\alpha_t\sigma_{t'}}{\sigma_t\alpha_{t'}},
$$
one recovers the standard DDPM transition kernel:
$$
 p^{\mathrm{DDPM}}_{t'\mid t}(X_{t'}\mid X_t) = p_{t'\mid t}(X_{t'}\mid X_t; \rho_*).
$$
Thus DDPM transitions are a **special case** of GLASS transitions.

### 3.3 GLASS denoiser via sufficient statistic

To implement GLASS transitions with a **pretrained flow model**, we must express them in terms of objects we already know: the denoiser $D_t$ or vector field $u_t$.

Define the **GLASS denoiser** as the posterior mean
$$
 D_{\mu,\Sigma}(\mathbf{x}) = \int z\, p(Z=z\mid \mathbf{X}=\mathbf{x})\,\mathrm{d}z,\qquad \mathbf{x} = (x_t,x_{t'}),
$$
where $p(Z\mid \mathbf{X})$ is induced by $p_{t,t'}(X_t,X_{t'}\mid z)p_{\mathrm{data}}(z)$.

The key observation is that the pair $\mathbf{x}=(x_t,x_{t'})$ can be compressed into a **sufficient statistic**
$$
 S(\mathbf{x}) = \frac{\mu^T\Sigma^{-1}\mathbf{x}}{\mu^T\Sigma^{-1}\mu},
$$
which is itself a CondOT-like noisy observation of $z$ at some **effective time** $t^*$.

Define
$$
 g(t) = \frac{\sigma_t^2}{\alpha_t^2},\qquad t^* = g^{-1}\Bigl((\mu^T\Sigma^{-1}\mu)^{-1}\Bigr).
$$
Then one can show (Proposition 2 in GLASS)
$$
 D_{\mu,\Sigma}(\mathbf{x}) = D_{t^*}\bigl(\alpha_{t^*} S(\mathbf{x})\bigr).
$$
Thus the **GLASS denoiser** can be implemented by **one call** to the pretrained denoiser $D_t$ at suitably transformed inputs and time.

### 3.4 GLASS velocity field and inner ODE

We now construct an **inner flow-matching model** $u_s(\bar{x}_s\mid x_t,t)$ on an auxiliary time $s\in[0,1]$ such that the ODE
$$
 \bar{X}_0 \sim \mathcal{N}(\bar{\gamma}x_t,\bar{\sigma}_0^2 I_d),\qquad \frac{\mathrm{d}}{\mathrm{d}s}\bar{X}_s = u_s(\bar{X}_s\mid x_t,t)
$$
yields
$$
 \bar{X}_1 \sim p_{t'\mid t}(\cdot\mid x_t)
$$
for chosen $t,t'$ and $\rho$.

The construction proceeds by defining an **inner conditional path** between an initial Gaussian around $x_t$ and the GLASS transition endpoint. For schedulers $\bar{\alpha}_s,\bar{\sigma}_s$ with $\bar{\alpha}_0=0,\bar{\alpha}_1=\bar{\alpha},\bar{\sigma}_1=\bar{\sigma}$, one can verify that
$$
 p_s(\bar{x}_s\mid x_t,z) = \mathcal{N}(\bar{x}_s;\bar{\alpha}_s z + \bar{\gamma}x_t,\bar{\sigma}_s^2 I_d)
$$
interpolates between
$$
 p_0(\bar{x}_0\mid x_t) = \mathcal{N}(\bar{\gamma}x_t,\bar{\sigma}_0^2 I_d),\qquad p_1(\bar{x}_1\mid x_t) = p_{t'\mid t}(X_{t'}\mid X_t=x_t).
$$

Theorem 1 (GLASS) shows that the **GLASS velocity field** can be written as a linear combination
$$
 u_s(\bar{x}_s\mid x_t,t) = w_1(s)\bar{x}_s + w_2(s) D_{\mu(s),\Sigma(s)}(x_t,\bar{x}_s) + w_3(s) x_t,
$$
where
$$
 \mu(s) = \begin{pmatrix}\alpha_t\\ \bar{\alpha}_s + \bar{\gamma}\alpha_t\end{pmatrix},\qquad
 \Sigma(s) = \begin{pmatrix}\sigma_t^2 & \sigma_t^2\bar{\gamma}\\ \sigma_t^2\bar{\gamma} & \bar{\sigma}_s^2 + \bar{\gamma}^2\sigma_t^2\end{pmatrix},
$$
and
$$
 w_1(s) = \frac{\partial_s \bar{\sigma}_s}{\bar{\sigma}_s},\quad
 w_2(s) = \partial_s \bar{\alpha}_s - \bar{\alpha}_s w_1(s),\quad
 w_3(s) = -\bar{\gamma} w_1(s).
$$
Here $D_{\mu(s),\Sigma(s)}$ is the GLASS denoiser, implemented via the sufficient statistic and $D_t$.

The resulting ODE defines a **stochastic transition** (because of the random initial $\bar{X}_0$), yet it is **implemented entirely with ODE integration and calls to the original denoiser**.

### 3.5 Using GLASS for reward alignment

GLASS is a **transition sampler**; it does not prescribe a specific reward algorithm, but it gives an efficient way to plug stochastic transitions into many methods.

1. **Sequential Monte Carlo (SMC)** and Feynman–Kac Steering
   - Use GLASS transitions as proposals: $x_{t'}^k\sim p_{t'\mid t}(\cdot\mid x_t^k)$.
   - Weights are updated using potentials $G(x_t,x_{t'})$, e.g. $\exp(r(x_{t'})-r(x_t))$.
1. **Search methods**
   - Use GLASS transitions to sample branches of a search tree, while value-function estimators guide node selection.
3. **Reward guidance**
   - Modify the GLASS velocity field analogously to guidance, by adding a gradient term $\propto \nabla_{\bar{x}_s} r_t(\bar{x}_s)$. Appendix B of GLASS details this construction.

Compared to SDE-based DDPM sampling, GLASS yields
- **similar or better sample quality** at a given NFE, and
- **stochastic transitions compatible with ODE-level efficiency**, eliminating the ODE–SDE efficiency/stochasticity tradeoff.

---

## 4. Flow‑GRPO: Online RL for Flow Matching Models

Flow‑GRPO integrates **online reinforcement learning** (GRPO) with **flow matching** text-to-image models, using an ODE-to-SDE conversion and a denoising-reduction trick.

### 4.1 Denoising as an MDP

Following the lecture notes and Flow‑GRPO, the discrete-time denoising trajectory
$$
 (x_T, x_{T-1},\dots,x_0)
$$
for a conditional flow model (e.g., rectified flow) can be cast as a **Markov Decision Process** $(\mathcal{S},\mathcal{A},\rho_0,P,R)$:

- **State**: $s_t = (c,t,x_t)$, where $c$ is the prompt.
- **Action**: $a_t = x_{t-1}$, the predicted next (less noisy) state.
- **Policy**:
  $$
   \pi_\theta(a_t\mid s_t) = p_\theta(x_{t-1}\mid x_t,c).
  $$
- **Transition** (deterministic):
  $$
   P(s_{t+1}\mid s_t,a_t) = (\delta_c, \delta_{t-1}, \delta_{x_{t-1}}).
  $$
- **Initial state**:
  $$
   \rho_0(s_0) = (p(c), \delta_T, \mathcal{N}(0,I_d)).
  $$
- **Reward**: concentrated at the final step
  $$
   R(s_t,a_t) = \begin{cases}r(x_0,c), & t=0,\\ 0, & t>0.\end{cases}
  $$

This is a **generator-matching view in discrete time**: the flow model defines a Markov chain over $x_t$ whose terminal output receives a scalar reward.[file:1]

### 4.2 GRPO objective for flows

GRPO is a **value-free policy-gradient** method that estimates advantages via group-level normalization. Flow‑GRPO adopts a **KL-regularized objective**
$$
 \max_{\theta} \mathbb{E}\Biggl[\sum_{t=0}^T \bigl( R(s_t,a_t) - \beta D_{\mathrm{KL}}(\pi_\theta(\cdot\mid s_t)\Vert \pi_{\mathrm{ref}}(\cdot\mid s_t))\bigr)\Biggr],
$$
where the reference $\pi_{\mathrm{ref}}$ is typically the pretrained flow model.

Given a prompt $c$, the flow model samples a **group** of $G$ trajectories (images and their denoising paths)
$$
 \{(x_T^i, x_{T-1}^i,\dots,x_0^i)\}_{i=1}^G.
$$
The **group-relative advantage** for image $i$ and step $t$ is
$$
 \hat{A}_t^i = \frac{R(x_0^i,c)-\mathrm{mean}_j R(x_0^j,c)}{\mathrm{std}_j R(x_0^j,c)}.
$$
The **importance ratio** is
$$
 r_t^i(\theta) = \frac{p_\theta(x_{t-1}^i\mid x_t^i,c)}{p_{\theta_{\mathrm{old}}}(x_{t-1}^i\mid x_t^i,c)},
$$
leading to the clipped GRPO-style objective
$$
 \mathcal{J}_{\mathrm{Flow\mbox{-}GRPO}}(\theta) = \mathbb{E}[f(r,\hat{A},\theta,\varepsilon,\beta)],
$$
with
$$
 f = \frac{1}{G}\sum_{i=1}^G \frac{1}{T}\sum_{t=0}^{T-1}\Bigl(\min\bigl(r_t^i(\theta)\hat{A}_t^i,\, \mathrm{clip}(r_t^i(\theta),1-\varepsilon,1+\varepsilon)\hat{A}_t^i\bigr) - \beta D_{\mathrm{KL}}(\pi_\theta\Vert \pi_{\mathrm{ref}})\Bigr).
$$

### 4.3 ODE-to-SDE conversion for exploration

Flow matching models are **deterministic** ODE samplers:
$$
 \mathrm{d}x_t = v_t(x_t)\,\mathrm{d}t,
$$
so state transitions are deterministic functions of the initial noise seed. This causes two problems for RL:

1. The likelihood $p_\theta(x_{t-1}\mid x_t,c)$ is complicated to evaluate.
2. Lack of transition stochasticity harms exploration.

Flow‑GRPO constructs a **stochastic counterpart** SDE whose marginals still match the ODE path. Starting from a generic SDE
$$
 \mathrm{d}x_t = f_{\mathrm{SDE}}(x_t,t)\,\mathrm{d}t + \sigma_t\,\mathrm{d}w_t,
$$
its density $p_t$ obeys the Fokker–Planck equation
$$
 \partial_t p_t = -\nabla\cdot(f_{\mathrm{SDE}} p_t) + \tfrac{1}{2}\nabla^2(\sigma_t^2 p_t).
$$

The ODE density $p_t$ for $\mathrm{d}x_t = v_t(x_t)\,\mathrm{d}t$ satisfies
$$
 \partial_t p_t = -\nabla\cdot(v_t p_t).
$$
Matching the two yields
$$
 f_{\mathrm{SDE}}(x_t,t) = v_t(x_t) + \tfrac{\sigma_t^2}{2}\nabla\log p_t(x_t),
$$
so the **forward SDE** with the same marginals is
$$
 \mathrm{d}x_t = \bigl(v_t(x_t) + \tfrac{\sigma_t^2}{2}\nabla\log p_t(x_t)\bigr)\,\mathrm{d}t + \sigma_t\,\mathrm{d}w_t.
$$

Using the standard time-reversal result, the **reverse SDE** becomes
$$
 \mathrm{d}x_t = \bigl(v_t(x_t) - \tfrac{\sigma_t^2}{2}\nabla\log p_t(x_t)\bigr)\,\mathrm{d}t + \sigma_t\,\mathrm{d}w_t.
$$
For the rectified flow parameterization (linear interpolant), the marginal score relates to the velocity as
$$
 \nabla\log p_t(x) = -\frac{x}{t} - \frac{1-t}{t} v_t(x),
$$
so the final SDE can be rewritten purely in terms of the learned velocity $v_t$.

Discretizing with Euler–Maruyama yields the update
$$
 x_{t+\Delta t} = x_t + \Bigl[v_\theta(x_t,t) + \tfrac{\sigma_t^2}{2t}(x_t + (1-t)v_\theta(x_t,t))\Bigr]\Delta t + \sigma_t\sqrt{\Delta t}\,\epsilon,
$$
where $\epsilon\sim\mathcal{N}(0,I_d)$.

Thus
- $\pi_\theta(x_{t+\Delta t}\mid x_t,c)$ is an **isotropic Gaussian**, making KL terms tractable.
- The noise level $\sigma_t$ (often parameterized as $\sigma_t = a\sqrt{t/(1-t)}$) controls exploration.

### 4.4 Closed-form KL and KL regularization

Because $\pi_\theta(\cdot\mid x_t)$ is Gaussian with mean $\bar{x}_{t+\Delta t,\theta}$, the KL against a reference policy is
$$
 D_{\mathrm{KL}}(\pi_\theta\Vert \pi_{\mathrm{ref}}) = \frac{\|\bar{x}_{t+\Delta t,\theta} - \bar{x}_{t+\Delta t,\mathrm{ref}}\|^2}{2\sigma_t^2\Delta t}.
$$
This can be expressed in terms of the velocity fields as
$$
 D_{\mathrm{KL}}(\pi_\theta\Vert \pi_{\mathrm{ref}}) = \frac{\Delta t}{2}\Bigl(\frac{\sigma_t(1-t)}{2t} + \frac{1}{\sigma_t}\Bigr)^2 \|v_\theta(x_t,t) - v_{\mathrm{ref}}(x_t,t)\|^2.
$$

KL regularization
- keeps the RL-tuned model close to the pretrained base,
- empirically mitigates **reward hacking** (e.g., loss of image quality/diversity), and
- provides a knob to trade off reward maximization vs. fidelity.

### 4.5 Denoising Reduction

Full-resolution flow models may require many denoising steps (e.g., 40) for high-quality generation. For online RL, this makes sampling excessively expensive.

Flow‑GRPO introduces **Denoising Reduction**:
- Use a **small number of denoising steps** $T_{\mathrm{train}}$ (e.g., 10) when generating rollouts for RL.
- Retain the **original, larger number of steps** $T_{\mathrm{test}}$ (e.g., 40) for inference.

Empirically, reducing $T$ from 40 to 10 yields
- $\gtrsim 4\times$ speedup in data collection,
- similar final reward and image quality when using full inference steps.

---

## 5. Diamond Maps: Stochastic Flow Maps for Reward Alignment

Diamond Maps redesign generative models to **natively support efficient reward alignment** by combining
- **flow maps** (single-step mappings that amortize ODE integration), and
- **stochasticity** (needed for value function estimation and SMC/search).

### 5.1 Flow maps recap

A **flow map** $X_{t,r}(x_t)$ is the solution operator of the probability-flow ODE:
$$
 x_0\sim p_0,\quad \frac{\mathrm{d}}{\mathrm{d}t}x_t = u_t(x_t)\;\Rightarrow\; X_{t,r}(x_t) = x_r.
$$

Standard flow maps are **deterministic**: given $x_t$, the output $x_r$ is fixed. Diamond Maps introduce **stochastic flow maps** that still respect the same marginals $p_t$.

### 5.2 Posterior Diamond Maps

A **Posterior Diamond Map** is a stochastic flow map
$$
 X_{s,r}(\bar{x}_s\mid x_t,t),\quad 0\le s\le r\le 1,
$$
that, for fixed $x_t$, samples from the posterior
$$
 \bar{x}_0\sim \mathcal{N}(0,I_d)\;\Rightarrow\; X_{0,1}(\bar{x}_0\mid x_t,t) \sim p_{1\mid t}(\cdot\mid x_t).
$$

Given a batch of posterior samples $z^k = X_{0,1}(\bar{x}_0^k\mid x_t,t)$, one can estimate
$$
 V_t^r(x_t) \approx \log\frac{1}{N}\sum_{k=1}^N \exp(r(z^k)),
$$
$$
 \nabla_{x_t} V_t^r(x_t) \approx \sum_{k=1}^N \frac{\exp(r(z^k))}{\sum_j \exp(r(z^j))} \nabla_{x_t} r(z^k),
$$
using the reparameterization trick (differentiating through $X_{0,1}$).

Thus Posterior Diamond Maps give **consistent and efficient** estimators of both $V_t^r$ and $\nabla V_t^r$, enabling **exact guidance**.

#### Training via GLASS distillation

Posterior Diamond Maps are trained by **distilling GLASS Flows** into a flow map. GLASS provides a velocity field
$$
 \bar{u}_s(\bar{x}_s\mid x_t,t)
$$
for an inner ODE whose endpoint $X_1$ is distributed as $p_{1\mid t}(\cdot\mid x_t)$.

For the CondOT path, the GLASS posterior velocity admits a closed form
$$
 \bar{u}_s(\bar{x}_s\mid x_t,t) = a_{s,t} \bar{x}_s + b_{s,t} D_{t^*}(\alpha_{t^*} S_{s,t}(\bar{x}_s,x_t)),
$$
where the **sufficient statistic**
$$
 S_{s,t}(\bar{x}_s,x_t) = \frac{\alpha_s\sigma_t^2 \bar{x}_s + \alpha_t\sigma_s^2 x_t}{\sigma_t^2\alpha_s^2 + \alpha_t^2\sigma_s^2},
$$
and the **effective time**
$$
 t^*(s,t) = g^{-1}\left(\frac{\sigma_t^2\sigma_s^2}{\sigma_t^2\alpha_s^2 + \alpha_t^2\sigma_s^2}\right),\qquad g(t)=\frac{\sigma_t^2}{\alpha_t^2}.
$$
The coefficients $a_{s,t}, b_{s,t}$ are scalar functions derived in the appendix.

To train a Diamond Map $X_{s,r}^\theta$, one minimizes a Lagrangian distillation loss
$$
 \mathbb{E}\bigl[\|\partial_r X_{s,r}^\theta(\bar{x}_s\mid x_t,t) - \bar{u}_r(X_{s,r}^\theta(\bar{x}_s\mid x_t,t)\mid x_t,t)\|^2\bigr],
$$
with expectations over $z\sim p_{\mathrm{data}}, x_t\sim p_t(\cdot\mid z), \bar{x}_s\sim p_s(\cdot\mid z)$.

### 5.3 Diamond DDPM sampling: one-step transitions

To use Posterior Diamond Maps for **iterative sampling**, we need transitions $x_t\to x_{t'}$ for $t'<1$, not just $t\to 1$. The Diamond paper shows that **DDPM transition kernels are embedded inside the Posterior Diamond Map**.

Define **Diamond Early Stop DDPM sampling**:
1. Choose $t' > t$ and compute a corresponding inner time $r^*(t,t')$ via
   $$
    r^*(t,t') = g^{-1}\Bigl(\frac{g(t)g(t')}{g(t)-g(t')}\Bigr),
   $$
   which satisfies $t^*(r^*,t)=t'$.
2. Sample $\bar{x}_0\sim \mathcal{N}(0,I_d)$ and flow to $r^*$:
   $$
    \bar{x}_{r^*} = X_{0,r^*}^\theta(\bar{x}_0\mid x_t,t).
   $$
3. Map back to outer time $t'$ using the sufficient statistic:
   $$
    x_{t'} = \alpha_{t'} S_{r^*,t}(\bar{x}_{r^*},x_t).
   $$

Proposition 4.3: **Diamond DDPM sampling** produces an exact DDPM transition
$$
 x_{t'} \sim p^{\mathrm{DDPM}}_{t'\mid t}(\cdot\mid x_t).
$$

Thus Posterior Diamond Maps provide
- **one-step posterior samplers** $p_{1\mid t}$ for value estimation and guidance, and
- **one-step DDPM transition samplers** for SMC and search, all within a flow-map architecture.

### 5.4 Weighted Diamond Maps

Weighted Diamond Maps show how to make a **standard deterministic flow map** $X_{t,r}$ into a **stochastic estimator** of the value function, **without retraining** for GLASS.

#### Renoising and mapping to data

Given state $x_t$ and flow map $X_{t,r}$, choose $t'< t$ and define the **renoising map**
$$
 x_{t'}(x_t,\epsilon) = \frac{\alpha_{t'}}{\alpha_t}x_t + \sqrt{\sigma_{t'}^2 - \frac{\alpha_{t'}^2}{\alpha_t^2}\sigma_t^2}\,\epsilon,\qquad \epsilon\sim\mathcal{N}(0,I_d).
$$
This corresponds to applying the **forward diffusion** from $t$ to $t'$.

Then, push forward via the flow map to time $1$:
$$
 z(x_t,\epsilon) = X_{t',1}(x_{t'}(x_t,\epsilon)).
$$
This defines a **proposal distribution** $q_{1\mid t}(\cdot\mid x_t)$, which is generally **not equal** to the posterior $p_{1\mid t}(\cdot\mid x_t)$.

#### Local reward and recovery term

Define the **local reward**
$$
 r_{\mathrm{local}}(x_t,\epsilon) = r(z(x_t,\epsilon)) - \frac{\|x_t - \alpha_t z(x_t,\epsilon)\|^2}{2\sigma_t^2}.
$$
The second term is a **recovery reward** (negative log-likelihood of $x_t$ under the CondOT kernel given $z$), encouraging consistency of $z$ with the noisy observation $x_t$.

#### Weighted value-gradient estimator

For $N$ noise samples $\epsilon_i$, define weights
$$
 v_i = r_{\mathrm{local}}(x_t,\epsilon_i) + \gamma_i + \tfrac{1}{2}\|\epsilon_i\|^2,
$$
$$
 w_i = \mathrm{softmax}(v_1,\dots,v_N)_i.
$$
The correction term $\gamma_i$ accounts for the difference between the true posterior and the proposal; in practice one uses a **score-based line integral** approximation
$$
 \gamma_i \approx \tfrac{1}{2}\bigl(\nabla\log p_{t'}(x_t) + \nabla\log p_{t'}(x_{t'}^i)\bigr)^T (x_{t'}^i - x_t),
$$
where $x_{t'}^i = x_{t'}(x_t,\epsilon_i)$.

The **Weighted Diamond estimator** for the value gradient is
$$
 \nabla_{x_t} V_t^r(x_t) \approx \sum_{i=1}^N w_i\bigl[\nabla_{x_t} r_{\mathrm{local}}(x_t,\epsilon_i) + \delta_{\mathrm{score}}^i\bigr],
$$
with score correction
$$
 \delta_{\mathrm{score}}^i = \nabla_{x_{t'}}\log p_{t'}(x_{t'}^i)\frac{\alpha_{t'}}{\alpha_t} - \nabla_{x_t}\log p_t(x_t).
$$
The scores $\nabla\log p_t$ are obtained from the pretrained model (reparameterization of $u_t$ or $D_t$).

Proposition 5.1 asserts this is a **consistent estimator** of $\nabla V_t^r$. Thus Weighted Diamond Maps provide a **training-free** pathway to stochastic value-estimation using existing distilled flow maps.

---

## 6. Discrete Flow Maps (for Sequence Models)

While most of the above is in Euclidean space, **Discrete Flow Maps (DFMs)** extend flow maps to the **probability simplex**, enabling non-autoregressive sequence generation with flow-style guidance.[web:4]

### 6.1 Mean denoiser parameterization

Tokens are one-hot vectors $e_k\in\mathbb{R}^K$, and distributions lie on the simplex
$$
 \Delta^{K-1} = \{x\in\mathbb{R}^K: x\ge 0, \langle \mathbf{1}, x\rangle = 1\}.
$$

Instead of parameterizing the average velocity $v_{s,t}:\mathbb{R}^K\to\mathbb{R}^K$ directly, DFM parameterizes the **mean denoiser**
$$
 \psi_{s,t}: \mathbb{R}^K\to\Delta^{K-1},\qquad v_{s,t}(x) = \frac{\psi_{s,t}(x) - x}{1-s}.
$$
Then the flow map becomes the convex combination
$$
 X_{s,t}(x) = \frac{1-t}{1-s} x + \frac{t-s}{1-s} \psi_{s,t}(x).
$$
By construction, $\psi_{s,t}(x)$ always lies on the simplex (implemented via softmax over logits), ensuring that the flow map is geometrically aligned with discrete probabilities.[web:4]

### 6.2 Diagonal and consistency losses on the simplex

- **Diagonal loss (denoiser training)**:
  $$
   \mathcal{L}_{\mathrm{diag}}(\hat{\psi}) = \int_0^1 \mathbb{E}\bigl[-\sum_k I_1^{(k)}\log \hat{\psi}_{t,t}^{(k)}(I_t)\bigr] \mathrm{d}t,
  $$
  where $I_t$ is the interpolant and $I_1$ the final token.[web:4]

- **Consistency losses** enforce semigroup/Lagrangian/Eulerian identities using **KL divergences** between distributions instead of $L^2$ losses in $\mathbb{R}^K$. For example, the **semigroup loss** uses the identity
  $$
   \psi_{s,t}(x) = \alpha_{s,u,t} \psi_{s,u}(x) + \beta_{s,u,t} \psi_{u,t}(X_{s,u}(x)),
  $$
  and trains $\hat{\psi}_{s,t}$ to match the RHS via
  $$
   \mathcal{L}_{\mathrm{PSD}}(\hat{\psi}) = \mathbb{E}\left[D_{\mathrm{KL}}\left(\mathrm{sg}\bigl[\alpha\hat{\psi}_{s,u} + \beta\hat{\psi}_{u,t}\circ \hat{X}_{s,u}\bigr]\,\Vert\, \hat{\psi}_{s,t}\right)\right].
  $$

This yields discrete flow maps that support
- **few-step non-autoregressive generation** (large blocks of tokens at once), and
- **classifier-free guidance** via drift modification
  $$
   b_t^{\mathrm{CFG}}(x; c) = b_t(x) + \omega\bigl(b_t(x; c) - b_t(x)\bigr),
  $$
  analogously to continuous flows.[web:4]

From the reward-alignment viewpoint, DFMs provide a **discrete counterpart** to flow maps that can be combined with GLASS/Diamond ideas to build **reward-aligned sequence generators**.

---

## 7. Conceptual Summary in the Generator-Matching Language

Using the Generator Matching (GM) viewpoint, we can summarize these developments as follows.[file:1][file:3]

1. **Base generator (flow/diffusion)**
   - A pretrained model defines a generator $L_t$ (flow ODE, diffusion SDE, CTMC) whose marginals follow a probability path $(p_t)$ between $p_0=p_{\mathrm{simple}}$ and $p_1=p_{\mathrm{data}}$.

2. **Reward-tilted generator**
   - Reward alignment targets a new generator $L_t^r$ whose marginals follow the reward-tilted path $p_t^r$ induced by $p^r(z)\propto p_{\mathrm{data}}(z)\exp(r(z))$.
   - In continuous flows, this corresponds to shifting the drift by $b_t\nabla V_t^r(x_t)$.

3. **Transition samplers for alignment**
   - GLASS Flows: construct an **inner ODE-based generator** whose transitions reproduce a family of Markov kernels $p_{t'\mid t}$ while reusing the original denoiser.
   - Diamond Maps: distill this inner generator into **stochastic flow maps** for posteriors and transitions, enabling efficient value estimation and search/SMC.
   - Weighted Diamond Maps: obtain a **training-free**, generator-matching importance estimator using renoising and flow maps.

4. **RL-based generator adaptation**
   - Flow‑GRPO: treats the discrete denoising process as an MDP and applies online RL (GRPO) to update the generator parameters, using an ODE-to-SDE conversion and denoising reduction for efficient exploration.

5. **Discrete-state generators**
   - Discrete Flow Maps: implement generator matching on the simplex via mean denoisers and KL-consistency losses, giving non-autoregressive sequence generators with flow-like guidance.

In all cases, the core **Generator Matching picture** remains:
- Start from a probability path $(p_t)$.
- Design or adapt a Markov generator $L_t$ whose marginals follow $(p_t)$ (or its reward-tilted variant).
- Implement $L_t$ and its transitions using reparameterizations of existing neural components (velocity fields, denoisers, scores) so that **reward alignment and transition sampling are efficient and scalable**.
