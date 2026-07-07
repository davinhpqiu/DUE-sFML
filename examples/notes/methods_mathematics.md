# Mathematics of the sFML pipeline components

Notation: state $x_n\in\mathbb R^d$, time step $\Delta$, latent noise $z\sim\mathcal N(0,I_{n_s})$.
The one‑step (stochastic) flow map is $G_\Delta$, with $x_{n+1}\stackrel{d}{=}G_\Delta(x_n)$.

---

## 0. The sFML decomposition (background)

Any one‑step transition splits into its conditional mean and a zero‑mean fluctuation:

$$
x_{n+1}=\underbrace{\mathbb E[x_{n+1}\mid x_n]}_{D_\Delta(x_n)}+\underbrace{\big(x_{n+1}-\mathbb E[x_{n+1}\mid x_n]\big)}_{S_\Delta(x_n)},\qquad \mathbb E\!\left[S_\Delta(x_n)\mid x_n\right]=0 .
$$

We learn approximations $\tilde D_\Delta,\tilde S_\Delta$ and predict

$$
x_{n+1}=\tilde D_\Delta(x_n)+\tilde S_\Delta(x_n,z),\qquad z\sim\mathcal N(0,I),
$$

with the deterministic sub‑map in ResNet form $\tilde D_\Delta=\mathbf I+\mathbf N_\theta$.
Phase 1 fits $\tilde D_\Delta$ by (multi‑step) MSE; Phase 2 fits $\tilde S_\Delta$ (the generator)
adversarially. The **effective drift/diffusion** recovered from the model are

$$
\hat a(x)=\frac{\mathbb E_z\!\left[\tilde G_\Delta(x,z)-x\right]}{\Delta},\qquad
\hat b(x)=\frac{\operatorname{Std}_z\!\left[\tilde G_\Delta(x,z)\right]}{\sqrt\Delta}.
$$

The $1/\Delta$ and $1/\sqrt\Delta$ factors are the source of the sensitivity analysed in §2.

---

## 1. WGAN‑GP objective (the base learner for $\tilde S_\Delta$)

The critic $C_\psi$ approximates the Wasserstein‑1 distance via its Kantorovich dual,

$$
W_1(\mathbb P_r,\mathbb P_g)=\sup_{\|C\|_{\mathrm{Lip}}\le 1}\ \mathbb E_{y\sim\mathbb P_r}[C(y)]-\mathbb E_{\hat y\sim\mathbb P_g}[C(\hat y)] .
$$

We match **whole increment sequences** $y_{1:L}$ conditioned on $x_0$ (Algorithm 4.1). Writing
$\hat y_{1:L}$ for the generated sequence,

$$
\mathcal L_C=\underbrace{\mathbb E\big[C(x_0,\hat y_{1:L})\big]-\mathbb E\big[C(x_0,y_{1:L})\big]}_{-\,(\text{Wasserstein gap})}
+\;\lambda\,\mathbb E_{\tilde y}\Big[\big(\|\nabla_{(x_0,\tilde y)}C\|_2-1\big)^2\Big],
$$

$$
\mathcal L_G=-\,\mathbb E\big[C(x_0,\hat y_{1:L})\big],
$$

with the gradient penalty (GP) evaluated at interpolations
$\tilde y=\epsilon\,y_{1:L}+(1-\epsilon)\hat y_{1:L}$, $\epsilon\sim\mathcal U(0,1)$, and
$\lambda=10$. Critic updated every batch, generator every $n_{ct}=5$ th batch.

---

## 2. The $1/\Delta$ hypersensitivity (why the mean is the hard part)

Linearise the true map near steady state. For OU, $\tilde D_\Delta(x)=x+\theta\Delta(\mu-x)$, i.e.

$$
x_{n+1}=a\,x_n+b,\qquad a=1-\theta\Delta,\quad b=\theta\mu\Delta,\qquad x^\ast=\frac{b}{1-a}=\mu .
$$

Suppose the **generator carries a small spurious conditional mean** $\varepsilon=\mathbb E_z[\tilde S_\Delta(x,z)]\neq0$.
The effective map becomes $x_{n+1}=a x_n+b+\varepsilon$, whose fixed point moves to

$$
\boxed{\,x^\ast_\varepsilon=\frac{b+\varepsilon}{1-a}=\mu+\frac{\varepsilon}{\theta\Delta}\,}.
$$

With $\theta=1,\ \Delta=0.01$ the amplification is $1/(\theta\Delta)=100$. A sub‑noise error
$\varepsilon\approx0.004$ (vs. per‑step noise $\sigma\sqrt\Delta=0.03$) shifts the long‑run mean by
$\approx0.4$. **The mean must be accurate to $\sim10^{-4}$**, far below the noise floor of
adversarial training ($\sim10^{-2}$). This is why the mean cannot be left to the GAN.

**Why the WGAN can't supply that precision.** For a pure translation by $\delta$,
$W_1(\mathbb P,\mathbb P+\delta)=|\delta|$ — the objective moves by only $\sim\delta$, and this
tiny signal shares the critic's single unit‑Lipschitz budget with the (larger) variance/shape
mismatch. In the minimax (gradient descent–ascent) dynamics the mean is a nearly‑flat,
weakly‑damped direction that orbits/diverges rather than converging (confirmed empirically:
70k un‑centred epochs still give the wrong mean).

---

## 3. Hard‑centering of $\tilde S_\Delta$ (the core fix)

**Idea.** Enforce the decomposition's own property $\mathbb E_z[\tilde S_\Delta(x,z)]=0$ *by
construction*, at every state, so the mean is owned entirely by $\tilde D_\Delta$ (fit by MSE,
which targets the mean directly and is unaffected by the $1/\Delta$ dilution).

**Estimator.** At a state $x$, draw $K$ i.i.d. latents $z_1,\dots,z_K$ and set

$$
\boxed{\ \tilde S^{c}_\Delta(x,z_1)=\tilde S_\Delta(x,z_1)-\frac1K\sum_{k=1}^{K}\tilde S_\Delta(x,z_k)\ }
$$

(the used sample $z_1$ is included in the mean). This is applied **in training and evaluation**.

**Exact zero mean.** With $S_k:=\tilde S_\Delta(x,z_k)$ i.i.d., mean $m$, variance $\sigma_S^2$,

$$
\mathbb E\!\left[S_1-\bar S\right]=m-m=0,\qquad \bar S=\tfrac1K\textstyle\sum_k S_k .
$$

**Variance shrinkage (derivation).** Write $S_1-\bar S=\big(1-\tfrac1K\big)S_1-\tfrac1K\sum_{k\ge2}S_k$. By independence,

$$
\operatorname{Var}\!\big(S_1-\bar S\big)=\Big(1-\tfrac1K\Big)^2\sigma_S^2+(K-1)\tfrac1{K^2}\sigma_S^2
=\sigma_S^2\,\frac{(K-1)K}{K^2}=\boxed{\ \sigma_S^2\Big(1-\tfrac1K\Big)\ }.
$$

So the standard deviation is scaled by $\sqrt{1-1/K}$: for $K=16$ that is $0.968$ — a **3.2 %**
reduction (negligible; the diffusion problem was the weak critic, not this). If desired, multiply
$\tilde S^{c}_\Delta$ by $\sqrt{K/(K-1)}$ to make it exactly variance‑neutral.
Cost: $K\times$ generator evaluations per rollout step. $K$ is a Monte‑Carlo accuracy setting,
**not** a tuned hyperparameter.

**Effect on the fixed point.** After centering, $\mathbb E_z[x_{n+1}\mid x_n]=\tilde D_\Delta(x_n)$
exactly, so the long‑run mean equals $\tilde D_\Delta$'s fixed point $x^\ast:\ \tilde D_\Delta(x^\ast)=x^\ast$.
The mean is now a regression problem (well‑posed), not an adversarial one (ill‑posed at $1/\Delta$).

**Faithfulness.** This enforces Remark 4.1 of the paper ("the purpose of $\tilde D_\Delta$ is to
ensure the stochastic learning part has mean $\approx 0$"), which the plain WGAN leaves emergent.
Necessity (not just convenience) is demonstrated in Part 13 of the log.

---

## 4. Maximum Mean Discrepancy (MMD) stabiliser

For distributions $\mathbb P,\mathbb Q$ and a positive‑definite kernel $k$,

$$
\mathrm{MMD}^2(\mathbb P,\mathbb Q)=\mathbb E_{y,y'\sim\mathbb P}k(y,y')+\mathbb E_{u,u'\sim\mathbb Q}k(u,u')-2\,\mathbb E_{y\sim\mathbb P,u\sim\mathbb Q}k(y,u).
$$

With a **characteristic** kernel this is $0$ iff $\mathbb P=\mathbb Q$; it is sensitive to *all*
moments. We use the Gaussian kernel and the biased empirical estimator over a batch:

$$
k(a,b)=\exp\!\Big(-\frac{\|a-b\|_2^2}{2h^2}\Big),\qquad
\widehat{\mathrm{MMD}}^2=\overline{k_{\hat y\hat y}}+\overline{k_{yy}}-2\,\overline{k_{\hat y y}} ,
$$

on the flattened increment sequences. Bandwidth by the **median heuristic** (detached from the graph):

$$
h^2=\operatorname{median}\big\{\|a-b\|_2^2\big\}.
$$

It is added to the generator loss: $\ \mathcal L_G\leftarrow\mathcal L_G+\lambda_{\mathrm{MMD}}\,\widehat{\mathrm{MMD}}^2$, $\lambda_{\mathrm{MMD}}=1$.

**What it does here.** The median bandwidth is set by the *typical* pairwise distance, dominated
by the noise scale $\sigma\sqrt\Delta$; hence MMD is strong on variance/shape but weak on a
sub‑noise mean (same blind spot as the critic). Empirically it damps the increment‑std
oscillation to near zero and makes checkpoint selection unnecessary — the CR‑GAN "consistency"
role. It does **not** fix the state‑dependent mean (that is centering's job) — MMD matches the
*pooled* increment marginal, which can be right on average while the conditional mean at $x^\ast$
is wrong.

---

## 5. Cosine learning‑rate decay

Per‑epoch schedule ($t$ = epoch index, $T$ = total epochs):

$$
\eta(t)=\eta_{\min}+\tfrac12\big(\eta_0-\eta_{\min}\big)\Big(1+\cos\tfrac{\pi t}{T}\Big),\qquad \eta_0=5\!\times\!10^{-5},\ \eta_{\min}=10^{-5}.
$$

**Why it helps minimax training.** GAN training seeks a *saddle* of a two‑player game; gradient
descent–ascent with a *constant* step can cycle or diverge along weakly‑damped directions (the
classic $\min_x\max_y xy$ cycles forever at fixed step). Robbins–Monro stochastic approximation
converges when $\sum_t\eta_t=\infty,\ \sum_t\eta_t^2<\infty$, i.e. a **shrinking** step damps the
orbit. Decay therefore stabilises the un‑centred mean's oscillation — but only partially, and
only late in a long schedule (see Part 13): it does not overcome the $1/\Delta$ ill‑conditioning,
so it complements centering rather than replacing it.

---

## 6. Normalization

The flow map is learned in a normalised coordinate $u=T(x)$ applied identically at every step
(so autonomy is preserved); predictions are inverted, $x=T^{-1}(u)$. $T$ must be monotone and
invertible, fit **from data** (no equation knowledge).

### 6.1 Min–max (affine; DUE default)
$$
u=T(x)=\frac{2\big(x-c\big)}{v_{\max}-v_{\min}},\qquad c=\tfrac12(v_{\max}+v_{\min}),\qquad
x=T^{-1}(u)=\tfrac12 u\,(v_{\max}-v_{\min})+c .
$$
Because it is affine, a network with an identity skip absorbs it — **immaterial for OU**. But for
GBM (range $[0,172]$) it crushes $\sim$94 % of the data below $-0.9$: representation failure.

### 6.2 Yeo–Johnson power transform (the GBM fix)
Per coordinate, with parameter $\lambda$:
$$
\psi(x;\lambda)=
\begin{cases}
\dfrac{(x+1)^{\lambda}-1}{\lambda}, & x\ge 0,\ \lambda\neq0,\\[4pt]
\log(x+1), & x\ge 0,\ \lambda=0,\\[4pt]
-\dfrac{(-x+1)^{2-\lambda}-1}{2-\lambda}, & x<0,\ \lambda\neq2,\\[4pt]
-\log(-x+1), & x<0,\ \lambda=2,
\end{cases}
$$
followed by an affine rescale of $\psi(x)$ to $[-1,1]$ (§6.1 on the transformed data).
$\lambda$ is chosen by **maximum likelihood** — maximise the Gaussian log‑likelihood of
$\psi(x;\lambda)$ including the Jacobian term $(\lambda-1)\sum_i\operatorname{sign}(x_i)\log(|x_i|+1)$.
Limits: $\lambda=1\Rightarrow$ identity, $\lambda=0\Rightarrow\log$, $\lambda<0\Rightarrow$ stronger‑than‑log
compression. Fitted values here: **OU $\lambda=0.60$** (near‑affine, corr. $0.999$ with identity ⇒ safe
default), **GBM $\lambda=-0.41$** (log‑like, *discovered*, never assuming geometric).

**Why it variance‑stabilises (delta method).** For a transform $\psi$ and small step,
$$
\operatorname{Var}\!\big(\psi(x_{n+1})\mid x_n=x\big)\approx \psi'(x)^2\,\operatorname{Var}(x_{n+1}\mid x_n)=\psi'(x)^2\,g(x)^2\Delta .
$$
This is **constant in $x$** iff $\psi'(x)\propto 1/g(x)$. For GBM, $g(x)=\sigma x\Rightarrow \psi'\propto 1/x\Rightarrow\psi=\log$.
Yeo–Johnson discovers $\lambda\approx0$, turning multiplicative noise into (approximately)
**additive** noise in $u$‑space — i.e. GBM becomes an OU‑type problem the pipeline already solves.
Measured: raw increment std grows $16\times$ across the range; in YJ space it is flat ($0.033$–$0.042$).

**Jensen bias from nonlinear normalization.** Centering enforces $\mathbb E[\tilde S\mid x]=0$
*in $u$‑space*. The physical mean is $\mathbb E[x_{n+1}]=\mathbb E\big[T^{-1}(u_{n+1})\big]$.
Since $T^{-1}$ is convex for a log‑like $T$, Jensen gives
$$
\mathbb E\big[T^{-1}(u_{n+1})\big]\ \ge\ T^{-1}\big(\mathbb E[u_{n+1}]\big)=\tilde D_\Delta\text{'s (deterministic) rollout},
$$
so the deterministic rollout tracks the **geometric** mean and *underestimates* the arithmetic
mean $\mathbb E[x]$; the zero‑mean noise lifts the ensemble mean partway back but not fully. This is
the observed $\sim$5 % GBM mean undershoot — a price of the nonlinear transform, present for any
multiplicative system.

---

## 7. Summary of what each piece controls

| Component | Math role | Controls |
|---|---|---|
| $\tilde D_\Delta=I+N$, multi‑step MSE | regress conditional mean $\mathbb E[x_{n+1}\mid x_n]$ | the **mean / fixed point** |
| **Hard‑centering** | project $\mathbb E_z[\tilde S]=0$ per state ($K$‑sample MC) | removes the ill‑conditioned mean direction from the GAN |
| WGAN‑GP + **4×128 critic** | Wasserstein‑1 matching of $(x_0,y_{1:L})$ | the **variance / shape** |
| **MMD** ($\lambda{=}1$) | all‑moment kernel matching of the pooled increment marginal | **stabilises** (damps std oscillation) |
| **Cosine decay** | Robbins–Monro step shrinkage on a saddle | **stabilises** minimax dynamics |
| **Yeo–Johnson** ($\lambda$ by MLE) | variance‑stabilising, invertible reparametrisation | **representation** for multiplicative/wide‑range data |

Key inequalities to remember: fixed‑point sensitivity $x^\ast_\varepsilon-\mu=\varepsilon/(\theta\Delta)$
(amplification $1/\Delta$); centering variance $\sigma_S^2(1-1/K)$; variance‑stabilisation
$\psi'\propto 1/g$; Jensen undershoot $\mathbb E[T^{-1}(u)]\ge T^{-1}(\mathbb E[u])$.
