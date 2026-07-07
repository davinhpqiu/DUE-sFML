# Model choices: mathematical justification and supporting evidence

The base method (Chen & Xiu 2024) decomposes each step into a deterministic mean map and a
stochastic residual, $x_{n+1}=\tilde D(x_n)+\tilde S(x_n,z)$, with $\tilde D$ fitted by MSE
(Phase 1) and the generator $\tilde S$ fitted adversarially by a WGAN (Phase 2). The choices
below are the modifications required for a faithful reproduction at our compute budget; each is
stated as *formula → justification → the experiment that motivated it*. The guiding principle was that every modification or hyperparameter
must be derivable from the data or the method's own mathematics, never from the identity of the
SDE. Full derivations are in `methods_mathematics.md`; the chronological record is `progress_log.md`.

*Baseline correction (to match the paper):* the original code used single-step
Phase-1 MSE and a critic scoring single-step $(x,r)$ pairs. We restored the multi-step rollout loss
(eq. 4.11) and the sequence-level critic $(x_0,y_{1:L})$ of Algorithm 4.1. With this in place, running
longer made the mean *worse* (stuck at 1.51 at 10k epochs), indicating WGAN instability plus a
spurious generator mean rather than a bug which motivated everything below.

---

### 1. Hard-centering of the generator  *(our addition — enforces Remark 4.1 of Chen & Xiu 2024)*
$$
\tilde S_c(x,z)=\tilde S(x,z)-\tfrac1K\textstyle\sum_{k=1}^{K}\tilde S(x,z_k),
$$
enforcing $\mathbb E_z[\tilde S_c(x,z)]=0$ per state by construction.

**Justification.** The long-run mean is the fixed point of the effective map. A residual mean
$\varepsilon$ in the generator displaces it by $\varepsilon/(\theta\Delta)$; for $\Delta=10^{-2}$
this is a $100\times$ amplification, so the mean must be accurate to $\sim10^{-4}$—below the noise
floor of adversarial training. Centering removes the mean from the generator and assigns it to
$\tilde D$, for which MSE regression attains that accuracy. (It reduces the residual variance only
by $1-1/K$, i.e. ~3% at $K=16$.)

**Evidence.** An *oracle ablation* (D̃ replaced by the exact map, GAN alone trained) settled the
attribution: with a flawless D̃ the mean *still* collapsed—to 0.82 (with the MMD/selection extras)
and 0.245 (plain WGAN)—so the generator, not D̃, carried the bias; across all un-centred runs the
mean landed differently each time (0.245, 0.82, 1.0, 1.57), the signature of an unconstrained
direction. Adding centering (oracle D̃ still on) made the mean **track 1.2 exactly**
(`mean_abs_err ≈ 0.002`), and revealed that the apparent "over-dispersion" had been an artifact of
the wrong mean ($C(0)=\mathrm{Var}+(m-\mu)^2$). This is the primary correction, and it enforces the
paper's own Remark 4.1. It is *necessary*, not merely convenient: the un-centred design given a
strong critic, LR decay, and **70,000 epochs** still produced the wrong mean (1.57) with a diverging
loss—budget is not the missing ingredient.

### 2. Higher-capacity critic (4×128, vs. the paper's 3×20)  *(our change — 128-unit precedent from the CR-GAN paper, Yeo, Li & Gifford)*
**Justification.** The Wasserstein critic must be an adequate witness; too little capacity yields
false convergence (its estimated gap vanishes while the distributions still differ).

**Evidence.** After centering isolated the variance defect (diffusion ≈0.13 vs 0.30, std ≈0.098 vs
0.21), the 3×20 critic's Wasserstein gap sat at $\sim7\times10^{-6}$ while the real/fake increment
std differed by a factor of five. Enlarging to 4×128 (width ≈ 4× input dimension; the CR-GAN paper
uses 128-unit networks) brought the gap alive ($\sim2\times10^{-3}$, now tracking the mismatch), the
fake std stopped collapsing, and **diffusion recovered to ≈0.30, std to ≈0.22** (best-checkpoint
`std_rel_err` 2.2%). This is what fixed the variance channel.

### 3. Maximum Mean Discrepancy regulariser  *(from the CR-GAN paper, Yeo, Li & Gifford, SIAM J. Sci. Comput.)*
$$
\mathcal L_G \mathrel{+}= \lambda\,\widehat{\mathrm{MMD}}^2(\hat y, y),\qquad \lambda=1,
$$
a Gaussian-kernel two-sample statistic sensitive to all moments of the increment distribution.

**Justification / evidence.** With the strong critic alone the generated noise level *oscillated*
across epochs (fake-std swing 0.020 ↔ 0.069), leaving the result dependent on checkpoint selection.
Adding the MMD term collapsed the swing (**0.049 → 0.0006**; best `std_rel_err` 2.2% → 0%) and made
every checkpoint good—stable, reproducible training (the consistency role of CR-GAN). It does *not*
alter the state-dependent tail over-dispersion, since it matches only the pooled increment marginal.

### 4. Cosine learning-rate decay  *(generic cosine annealing; in this lineage from the predecessor GAN paper, Chen & Xiu 2022, SIAM J. Sci. Comput.)*
$$
\eta(t)=\eta_{\min}+\tfrac12(\eta_0-\eta_{\min})\big(1+\cos\tfrac{\pi t}{T}\big),\qquad 5\times10^{-5}\!\to\!10^{-5}.
$$

**Justification.** WGAN training is a minimax problem whose gradient dynamics may cycle under a fixed
step size; a vanishing step (Robbins–Monro) damps the oscillation.

**Evidence.** Adding decay to the un-centred run damped its growing oscillation (late-epoch mean swing
~0.04 → ~0.005; critic bounded rather than diverging), confirming the mechanism. The 70k-epoch test
then showed decay stabilises but does *not* overcome the $1/\Delta$ ill-conditioning. Retained as
cheap stabilisation folded into the centered pipeline.

### 5. Yeo-Johnson normalization (data-derived, applied only where required)  *(our addition — standard statistical transform, Yeo & Johnson 2000; not from any of the SDE papers)*
A per-coordinate power transform with parameter $\lambda$ fitted by maximum likelihood
($\lambda\!\approx\!1$: identity; $\lambda\!\approx\!0$: logarithmic).

**Why normalize.** Neural-network initialization, gradient magnitudes, and the WGAN
gradient penalty are all calibrated for inputs of order 1 with comparable per-feature scales. Raw
physical data need not be in that regime: (i) init (Glorot/He) assumes O(1) inputs—large values
saturate the activations and explode the first-layer gradients; (ii) the loss landscape becomes
ill-conditioned when inputs span orders of magnitude, so one learning rate cannot serve all
samples; (iii) the critic sees $(x_0,y_{1:L})$, whose state and increments can differ by orders of
magnitude, and the large-scale features dominate; (iv) the gradient penalty $\|\nabla_{\text{input}}C\|\approx1$
is scale-dependent and only well-calibrated for O(1) inputs. Normalization maps the data into that
standard regime. Empirically it is only *needed* when the raw data is far from it: for OU (states
already O(1)) raw ≈ min-max ≈ Yeo-Johnson (all ~1.157), so it is nearly innocent; for GBM
(multiplicative, range $[0,172]$) raw training breaks.

**Why Yeo-Johnson over min–max.** The paper applies no normalization (verified); the library's
min–max scaling only fixes the overall *range*—it compresses the GBM range $[0,172]$ into a
negligible interval (skewed data crushed into a sliver) and leaves the noise multiplicative, so the
model diverges.

**Evidence.** Min–max maps 94% of GBM data below $-0.9$ (representation failure). For OU, `minmax`
(1.157) ≈ raw (1.148)—normalization is immaterial. GBM *raw* fails (trajectories go negative; a
constant diffusion ~7 is learned instead of $\sigma x$; std ~15). Yeo-Johnson infers $\lambda=-0.41$
from the data (log-like, with no assumption that the process is geometric), which distributes the
values and converts multiplicative noise to additive—reducing GBM to the OU-type problem the pipeline
already solves: **diffusion tracks $\sigma x$, std ≈4.8, trajectories positive, mean ≈3.5**. For OU it
returns $\lambda\approx0.6$ (near-identity), so it is safe as a default.

---

### Alternatives tested and rejected (part of the justification)

| Tried | Result | Verdict |
|---|---|---|
| More Phase-1 epochs (→3000) | D̃ fixed point drifted 1.15 → 1.10 (worse) | Rejected — bias is over-flexibility, not under-training |
| Single-step Phase-1 (eq. 4.12) | fixed point 1.13, worse than multi-step 1.15 | Rejected — OLS's *linearity* helped, not the objective |
| latent_dim 1 vs 8 | fixed diffusion tilt but worsened the mean (0.94→0.81) | Kept 1 (convention); not the mean fix |
| Mean-aware checkpoint selection | moved mean 0.245 → 0.82; insufficient | Superseded by centering (structural) |
| Larger batch (1024) for over-dispersion | worse | Rejected |
| Log-transform for GBM | would work | Rejected — uses knowledge the process is geometric |
| Un-centred + 70k epochs + decay | mean 1.57, diverging | Rejected — proves centering is necessary |

**Summary.** $\tilde D$ carries the conditional mean (hard-centering); the higher-capacity critic and
the MMD term govern the variance; learning-rate decay stabilises the minimax dynamics; and a
data-derived transform accommodates disparate scales. The choices were reached by single-variable
tests and elimination, and the two residuals (OU mean 1.157, GBM ~5% under) trace to one cause—D̃'s
Phase-1 accuracy—whose fix (complexity control on D̃) is deferred to a single global pass.
