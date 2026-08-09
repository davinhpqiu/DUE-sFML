# Reproducing Stochastic Flow Map Learning: a diagnostic study

A reproduction and stress-test of **sFML** (stochastic flow map learning), the WGAN-based method for
learning unknown stochastic differential equations from trajectory data, introduced in:

> Y. Chen and D. Xiu (2024). *Learning stochastic dynamical system via flow map operator.*
> Journal of Computational Physics **508**, 112984.

**Summary.** Working from the paper alone, the reported results proved difficult to recover. Rather than
stop there, this study characterises *where* the training dynamics depart from the expected behaviour
and *why*. Two mechanisms are identified: one is resolved with a small equation-agnostic modification,
and the other is shown to be a property of the training objective rather than of the model, the
optimiser, or the training budget.

Several hyperparameters are not specified in the paper (batch size, the gradient-penalty constant
$\lambda$, the activation function, the noise dimension, and whether "epochs" counts data passes or
iterations). Any of these could account for part of the gap, and the analysis below identifies which of
them the evidence points to. The intent is diagnostic, not a claim that the method does not work.

---

## Attribution

This repository is built **on top of the [DUE library](https://github.com/aiforsciencelab/DUE)** (Chen,
Wu & Xiu), which is not my work and is included under its original LGPL-2.1 licence. See `README.md`
for the upstream project.

**My contribution is the SDE / sFML component and everything used to evaluate it:**

| Path | What it is |
|---|---|
| `due/models/sde.py` | Phase-2 WGAN-GP trainer implementing Algorithm 4.1, plus all method toggles |
| `due/networks/gan.py` | generator and critic networks |
| `examples/OU`, `GBM`, `NonlinearDiffusion`, `TrigDrift`, `DoubleWell` | five benchmark SDEs: data generation, training drivers, evaluation and diagnostics |
| `examples/OU/sweep_checkpoints.py` | checkpoint sweep used for the stability analysis |
| `examples/notes/progress_log.md` | full research log, ~36 parts, including negative results |

Upstream modules (`fcn.py`, `fno.py`, `transformer.py`, the ODE/PDE models, packaging) are DUE's.

---

## The method in one paragraph

Given only trajectories from an unknown stochastic system, sFML learns the one-step *stochastic flow
map* by splitting it into a deterministic and a stochastic part,

```
x_{n+1} = D̃(x_n) + S̃(x_n, z),        z ~ N(0, I)
```

`D̃` is a ResNet trained by multi-step rollout MSE and carries the conditional mean; `S̃` is trained
adversarially (WGAN-GP) against a critic that scores an entire increment sequence conditioned on the
initial state. No distributional assumption is made about the noise, which is why a GAN is used at all.

---

## What was found

### 1. Training does not settle: a metastable window, then an escape

Run at the paper's own budget (100,000 epochs, 3×20 networks, `n_ct=5`, `lr=5e-5`), the long-run
behaviour of the model is **metastable, not convergent**. Sweeping all 1,000 saved checkpoints and
rolling each one out:

| phase | epochs | rollout mean (truth 1.205) |
|---|---|---|
| early chaos | 0 – 9k | 0.10 → 6494 |
| **metastable plateau** | **9k – 55k** | 1.05 – 1.79 |
| irreversible escape | 56k – 90k | up to 57.6 |
| partial recovery | 90k – 100k | 2.2 – 11.2 |

In this run the difficulty is therefore **over-training rather than under-training**: the stated budget of
100k epochs lands past the escape. Since the batch size and the meaning of "epochs" are both unspecified in the paper, an effective
budget inside the plateau would produce results that look correct at figure resolution.

Notably, the per-step training statistics never reveal this. At 100k the increment-std error is a
healthy 7×10⁻³ while the rollout has diverged by 364%, because the instability lives at high `x`,
outside the training support, and every training state sits at low `x`.

### 2. The conditional mean is hard to pin down adversarially (fixable)

The effective drift is a conditional mean divided by `Δ`, so a residual generator bias `δ` enters as
`δ/Δ` and displaces the model's fixed point by `δ/(θΔ)`. Measured: `δ = 3.8×10⁻⁴` predicts a fixed point
of 1.145 against 1.12 observed.

That bias is **1.3% of the noise amplitude**, and **0.4× the standard error of a batch mean** at
`B=1000`, so no single batch resolves it. Meanwhile W₁ penalises a mean shift only *linearly*, with none
of the `1/Δ` weighting that makes it matter downstream.

**Resolution:** enforce `E[S̃|x] = 0` structurally, so the mean is owned by `D̃` and estimated by
regression instead of adversarially. This is the property Remark 4.1 describes; in our runs it does not
emerge from training on its own.

### 3. The state-dependent diffusion shape appears not to be identifiable

`b(x)` is a property of the **conditional** law `p(Δx | x)`, but the WGAN and MMD objectives are sample
means over the batch and therefore constrain the **marginal** `p(Δx)`, which is dominated ~20:1 by
densely-visited states. Measured directly on the data:

```
W₁(true bell, flat b)  =  1.3×10⁻⁴        sampling-noise floor  =  1.1×10⁻⁴
```

The discriminating signal sits at the noise floor, so the objective cannot see it. The learned `b(x)`
even breaks a symmetry that is provably present in the data.

**Ruled out:** training budget (5k / 70k / 100k), critic capacity, sequence vs single-step critic, both
gradient-penalty forms, increment rescaling, MMD, the Muon optimiser, difficulty weighting, and
density weighting (verified to shift 43% of the critic's loss onto the sparse regions). Within this setup, the only interventions that recovered the shape required either more data in the
sparse regions or a structural assumption about the noise.

---

## Results by benchmark

| Example | Drift | Diffusion | Moments |
|---|---|---|---|
| Ornstein–Uhlenbeck (§5.1.1) | ✔ | ✔ flat 0.30 | ✔ |
| Geometric Brownian motion (§5.1.2) | ✔ | ✔ tracks `σx` over two decades | ✔ |
| Double well (§5.2.3) | ✔ | ✔ | ✔ bimodal stationary density recovered |
| Nonlinear diffusion (§5.2.1) | ✔ | ✘ bell comes out flat | ✔ |
| Trigonometric drift (§5.2.2) | ✔ | ✘ cosine comes out flat | ✔ |

Multiplicative noise and bimodality are recovered, so the distribution-free part of the method does
work. Only the state-dependent diffusion *shape* in sparsely-visited regions is not recovered.

---

## Modifications, and why each was needed

Every change is derivable from the data or from the method's own mathematics, never from knowing the
governing equation. All are config toggles that default to the paper's behaviour.

| Change | Paper | Why |
|---|---|---|
| Zero-mean centering of `S̃` | left to emerge (Rem. 4.1) | deletes the generator's mean degree of freedom |
| Gated ResNet for `D̃` | `D̃ = I + N` (4.10) | a learned affine path makes an affine conditional mean exact |
| Critic 4×128 | 3×20 | the small critic false-converges: W-gap → 10⁻⁴ with std 5× off |
| Increment rescaling | none | increments are ~0.03, so variance errors sit below the GP floor |
| MMD consistency term | none | damps a ±50% oscillation in generated variance |
| Data-derived normalisation | none | min–max crushes 94% of GBM's range; Yeo–Johnson `λ` is fitted from data |

---

## Reproducing

```sh
pip install .
cd examples/OU
python generate_data.py          # simulate the benchmark trajectories
python OU.py                     # Phase 1 + Phase 2 + evaluation figures
python sweep_checkpoints.py      # stability analysis over saved checkpoints
```

Each example is driven by a single `config.yaml`. Setting `eval_only: true` regenerates all figures from
saved models without retraining.

---

## Notes on rigour

- The implementation was audited line-by-line against Algorithm 4.1 and equations 4.6–4.23, and then
  **independently audited by a second reviewer** working only from the paper and the source. One real
  bug was found by that process (the gradient penalty), and is documented in the log.
- Residual errors are compared against the **statistical resolution of the dataset**, not against zero.
  For OU the fixed point is an ill-conditioned functional (a 0.1% slope error moves it 11%); bootstrap
  gives 1.2063 ± 0.0145, and the learned values fall inside that interval.
- Negative results are recorded in full, including the ones that cost the most time.

---

## References

- Chen & Xiu (2024), *Learning stochastic dynamical system via flow map operator*, JCP 508, 112984.
- Chen, Wu & Xiu (2025), *DUE: A Deep Learning Framework and Library for Modeling Unknown Equations*, SIAM Review 67(4).
- Yeo, Li & Gifford (2022), *Generative adversarial network for probabilistic forecast of random dynamical systems*, SISC 44(4).
- Gulrajani et al. (2017), *Improved training of Wasserstein GANs*, NeurIPS 30.
