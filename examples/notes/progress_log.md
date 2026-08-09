# sFML reproduction — progress log (Chen & Xiu 2024)

---

# ★ CURRENT BEST MODELS & PIPELINE (quick reference, updated latest)

**One equation-agnostic pipeline reproduces both OU and GBM.** The only per-example
difference is the data-derived normalization; nothing assumes the governing equation.

### The winning recipe (Phase 2 / GAN)
- **Hard-centering ON** (`center_generator: true`, `center_K: 16`) — S̃ zero-mean per
  state ⇒ D̃ owns the mean. *The single most important fix* (see Part 13 for why it's
  necessary, not optional).
- **Stronger critic 4×128** (`critic_depth: 4`, `critic_width: 128`) — witnesses the
  increment variance (the 3×20 critic could not).
- **MMD ON** (`mmd_lambda: 1.0`) — CR-GAN consistency stabiliser (kills the std
  oscillation; checkpoint-independent).
- **Cosine LR decay ON** (`lr_decay: true`, 5e-5→1e-5) — damps minimax oscillation.
- Learned multi-step D̃ (`use_oracle_det: false`, `phase1_single_step: false`), Phase-1
  batch 256; Phase-2 batch 1000, n_critic 5, gp_lambda 10, 1000 epochs, checkpoint select.
- **Normalization:** `minmax`/`none` for OU (immaterial), **`yeojohnson` for GBM**.

### Best OU model  — `examples/OU/config.yaml` (minmax or none)
- Mean **~1.157** (= D̃'s fixed point; true 1.2 — the ~0.04 gap is D̃'s Phase-1 bias:
  ~0.018 data floor + ~0.025 recoverable over-flexibility), std ~0.21, drift on the
  line, diffusion ~0.30 (mild high-x tail tilt = data coverage), stable.
- Files: `examples/OU/{det_model,gan_model}/` (generator_best, critic_best, model).

### Best GBM model  — `examples/GBM/config.yaml` (**`normalization: yeojohnson`**)
- Diffusion tracks **σx**, std **~4.8** (matches), drift **μx**, positive trajectories,
  mean **~3.5** (true 3.69; ~5% under = D̃ bias + Jensen effect of the nonlinear
  transform), conditional at x=6 excellent, stable.
- Files: `examples/GBM/{det_model,gan_model}/`.

### Known residuals (both small, understood, equation-agnostic to chase)
1. **OU mean 1.157 vs 1.2** — D̃ Phase-1 over-flexibility; fix = complexity control on D̃.
2. **GBM mean ~5% under** — D̃ bias + Jensen (centering is zero-mean in YJ space).
3. **High-x diffusion tilt (OU)** — sparse-tail extrapolation; a data-coverage limit.

### Settled questions
- **Centering is necessary** (Part 13): un-centered + decay + **70k epochs** still gives
  the wrong mean (1.57) with diverging loss. Budget is NOT the missing ingredient.
- **Paper does not normalize** (Part 12): min-max is a DUE artifact; it breaks GBM.
- **Normalization is immaterial for OU, essential for GBM** (Part 12).

### Toggles (all in config; defaults = original behaviour)
`center_generator`, `center_K`, `mmd_lambda`, `lr_decay`/`lr_min`, `critic_depth`/
`critic_width`, `use_oracle_det`, `phase1_single_step`, `network.det_arch`
(resnet | gated_resnet), `data.normalization` (minmax | none | yeojohnson).
`OU.py`/`GBM.py`/`check_phase1.py` take an optional config path arg.
(Removed: the `phase1_loss_space` J-reweighting experiment — dead end, see Part 15.)

### Next
**Decouple tried and REJECTED (Part 16):** raw Phase-1 wrapped into YJ transfers the
deterministic backbone (det_rollout 3.82) but blows up the full model (mean +68%, std ~7×) —
coupled YJ (mean 3.5) stays best; the "undershooting" YJ D̃ is a *feature* (pre-compensates the
noise Jensen lift). Decouple is OFF but kept (`phase1_normalization` commented; `WrappedDet`).
**Jensen correction (Part 17): NEGATIVE RESULT.** Delta-method correction: one-step gap ~0 in training region (estimation noise > signal); tail artifact at high-u (−0.07) catastrophically drops mean from 3.5→3.0. The 5% gap is multi-step/compounding, not a one-step bias. Correction OFF. The ~5% GBM mean gap is accepted as a documented residual.
**Double-well (Part 18):** GAN learned bimodal behavior ✓; drift recovery excellent ✓;
Phase 1 symmetry breaking found (D_theta→-0.7 from x0=0) — a saddle-point pathology not
visible in the paper's x0=1.5 test protocol. Paper uses IC U(-2.5,2.5), test x0=1.5, T=300
transition plot, PDF at T=0.5/10/30/100. Our DoubleWell.py used x0=0 — update needed.
**TrigDrift §5.2.2 (Part 19):** Mean/std near-perfect; drift good within IC range; diffusion
flat (~0.40 constant) instead of cosine-shaped 0.5|cos(2πx)| — same NLD pooled-loss pathology.
Best epoch 900 (std_rel_err 0.43%). Phase 1 clean (no saddle, stable IC range).
**Open issues (as of Part 19):** (1) Diffusion-shape miss: NLD, TrigDrift both show pooled
loss → spatial average, not state-dependent shape. Needs state-aware objective or heteroscedastic
generator. (2) generator_width/depth key missing from gan.py (only critic has separate width).
(3) DoubleWell.py update needed (x0=1.5, T=300 transitions). (4) 2D examples not started.
**Next to investigate:** architecture fix for diffusion miss — heteroscedastic generator S(x,z)=σ_net(x)·z or generator_width increase.

---

## Part 1 — Earlier work (setup & first fixes)

**Started with:** a partially-working sFML in DUE. Phase 1 used single-step MSE
on ~400k shuffled pairs; Phase 2 critic scored single-step (x, r) pairs (dim 2).

**Fixed:**
1. **Phase 1 → multi-step rollout loss (eq. 4.11).** Feed N=10,000 initial
   conditions as trainX and full L=40 trajectories as trainY; D_θ rolls 40 steps,
   penalised at every step.
2. **Phase 2 critic → full sequence.** Rewrote `models/sde.py` to sequence-level
   WGAN-GP per Algorithm 4.1: L-step rollout, critic scores (x_0, y_{1:L}) (dim 41
   for OU), GP in joint (x_0, y_{1:L}) space.

**Verified:** both phases checked line-by-line against the paper; Phase 2 matches
Algorithm 4.1.

**Ran:** 1k epochs (~44 min) and 10k (~7 h). 10k was *worse* — mean stuck flat at
1.51 instead of decaying to μ=1.2.

**Diagnosis:** WGAN instability. Generator carries a small spurious mean
(~+0.031 normalised), amplified ~100× by 1/Δ, shifting the fixed point. The 3×20
critic is too weak to police it; training is non-monotonic (more epochs → worse
final checkpoint).

**Plan set:** merge the professor's checkpoint selection, add a mean/drift term to
its score, set `latent_dim=1`, drop Phase-2 cosine LR.

---

## Part 2 — This session (attribution: is it D̃ or the generator?)

**Built the Phase-1 oracle ablation.** Added `OracleDet` (exact OU map
D(x)=x+θ(μ−x)Δt in normalised space) and a `use_oracle_det` toggle, so D̃ is
*perfect by construction* and only the GAN trains. Single-variable test: same GAN,
D̃ swapped learned → oracle.

**Three runs, all with the oracle D̃** (`det_rollout` confirms it sits exactly on
the analytical mean, settling at 1.2):

| Run | MMD | mean-selection | Long-run mean (true = 1.2) |
|---|---|---|---|
| A | λ=1 | on | **~0.82** |
| B (control) | off | off | **~0.245** |

- **Run A:** even with a flawless D̃, the mean still collapses to 0.82 → **Phase 1
  is exonerated; the generator is the source.**
- **Run B (plain WGAN-GP):** *worse* — mean 0.245, one-step conditional at x=0.8
  shifts from correct 0.80 → 0.73, effective drift over-reverts to −13, dispersion
  collapses. The bare generator learns a **negative** conditional mean (~−0.01 per
  step), the opposite of the zero-mean it should be.

**Key reads:**
- The generator, not D̃, carries the mean bias; with oracle D̃ a zero-mean
  generator would give the right answer, but the trained one doesn't.
- **MMD + mean-selection were substantially helping** (0.245 → 0.82) — corrects an
  earlier claim that MMD was too weak to matter on the mean.
- **Critic collapses:** loss / Wasserstein gap → 0 by epoch ~100 and stay flat,
  while distributions clearly still differ. Classic weak-critic *false*
  convergence — it can't witness the residual mismatch, so gradient vanishes and
  the un-witnessed direction is exactly the conditional mean.
- Data floor is minor: OLS on the raw data gives fixed point ~1.18 (bias ~0.02),
  nowhere near the 0.245/0.82 gap. Generator dominates.

**Professor (陈峻峰):** agrees the oracle idea is good; notes that *in theory* a
converged WGAN with oracle D̃ should yield a zero-mean generator. Reconciled: the
theory is right, but training never reaches that optimum — the 3×20 critic is too
weak to converge, so "under-trained" here means "critic too weak," not just "too
few epochs."

---

## Where we are — open issues

1. **Generator mean bias (dominant, universal).** Sub-noise mean error ×100 by
   1/Δ relocates the fixed point. Affects *every* small-Δ SDE, not just OU.
2. **Weak critic (3×20).** Collapses to false convergence; can't police mean or
   x-dependent variance. Root enabler of (1) and (3).
3. **Over-dispersion / wrong covariance.** Generator noise too high and
   x-dependent; not fixed by more epochs or bigger batch.
4. **Training instability & budget.** Non-monotonic; paper's 100k epochs
   infeasible on CPU. Checkpoint selection mitigates but only along its scored axes.
5. **GBM (separate defect).** `[-1,1]` min-max normalisation crushes low-value
   dynamics for multiplicative/exponential systems → divergence. A log-transform
   would fix it but uses knowledge of the true equation (not allowed). Needs a
   data-derived rescaling.

## How to progress (ordered)

1. **Zero-mean centering of S̃** — subtract the generator's per-state z-average so
   D̃ owns the mean and S̃ only fluctuates. Uses *only* the model's own assumption
   (Remark 4.1), no equation knowledge. Test rollout-only first (5-line change),
   then as a training term. With oracle D̃ this should send the mean back to 1.2.
2. **Stronger critic + more n_critic + longer train** — let the WGAN actually
   reach its optimum (the professor's "train to convergence" path).
3. **Keep / strengthen an MMD consistency term** — explicit moment signal to cover
   the weak critic (already shown to move the mean 0.245 → 0.82).
4. **Then** tackle over-dispersion (3), **then** the GBM dynamic-range problem (5).

**Litmus for every fix:** each knob must be derivable from the training data or the
method's own math — never from the identity of the SDE. If it passes, it
generalises to the other examples; if it needs to "know the answer" (e.g. log for
GBM), it's out. The goal is one equation-agnostic pipeline that runs unchanged on a
black-box dataset.

---

## Part 3 — Line-by-line comparison with the paper

Motivated by: if the paper is correct, we are mis-implementing or under-running
something — the fix is to find that, not to stack MMD/selection on top. Checked the
current code against Chen & Xiu (2024) §4 and Algorithm 4.1.

### What MATCHES the paper (verified)
- **Decomposition** x_{n+1}=D̃(x_n)+S̃(x_n,z); increment form ŷ=D̃(x)−x+S̃
  (Alg. 4.1 lines 6–7) — matches `generate_increment_sequence`.
- **Phase-1 loss** (eq. 4.11): D̃=I+N (ResNet), multi-step deterministic
  composition, MSE to noisy trajectories. Oracle test confirms 4.11's minimizer =
  the perfect map.
- **Critic** scores (x_0, y_{1:L}); **GP** over (x_0, ỹ) with per-sample ε
  (lines 10–12) — matches.
- **Hyperparameters**: n_ct=5, β=(0.5,0.999), lr=5e-5, GELU, 3×20 — matches §5.
- **Data generation**: IC~U(0,0.25), 100 EM steps, random L=40 window, test from
  x0=1.5 — matches §5.1.1 exactly (`generate_data.py` verified).

Conclusion: **not a math/architecture bug.** The gap is scale/training.

### Where we DEVIATE (ranked by likely impact)
1. **Training budget — 1,000 epochs vs paper's 100,000.** And it's really
   *generator updates*: with B=1000, n_B=N/B=10 batches/epoch, generator updates
   once per 5 batches → **~2 gen updates/epoch** → ~2,000 total. Paper-scale is
   ~200,000. We are ~100× short.
   - **Evidence (CSV, run A):** the generator increment mean never converges, it
     *oscillates* ±0.05 epoch-to-epoch (target rmean=+0.008):
     `ep1..1000 fmean: -.060 -.049 -.058 -.036 -.009 +.029 -.011 +.025 -.047`.
     GP collapsed to ~0, W-gap wobbles around 0 with no trend. Under-converged /
     oscillating WGAN, not settled-but-biased. Fixed point needs mean accurate to
     ~1e-4; generator is bouncing at ±5e-2. Professor's "训练没到位" is literally
     right, by ~2 orders of magnitude.
2. **Batch size B=1000 not specified in the paper.** Professor's speed choice; it
   is exactly what starves the generator (2 updates/epoch). Smaller B raises n_B →
   more generator updates/epoch at similar wall-cost.
3. **Add-ons not in the paper:** checkpoint selection (professor's), our MMD, our
   mean-selection. MMD + mean-selection currently OFF — keep off. Checkpoint
   selection just picks one lucky snapshot out of the oscillation.
4. **[-1,1] min-max normalization** (DUE library) — paper never mentions
   normalizing. Undocumented; probably minor.

### Genuine AMBIGUITIES in the paper
- **Batch size B** for OU — unspecified.
- **"100,000 epochs" vs iterations** — the 2022 predecessor GAN paper counted
  40,000 *iterations*. If "epochs" = data passes, that's ~1M critic / 200k gen
  steps at B=1000; if iterations, far fewer. 1–2 orders of magnitude of compute.
- **Normalization** — whether they scaled the data at all.
- **n_s (latent dim)** — stated only as "≥1, user's choice."
- **Phase-1 optimizer/schedule** — not given (DUE uses Adam + cosine).

### Runtime estimate for a faithful long run (CPU, from measured 0.49 s/epoch @ B=1000)
| Generator updates | Epochs (B=1000) | Est. time |
|---|---|---|
| ~20k (convergence check) | 10,000 | ~1.4 hr |
| ~40k (2022 iteration count) | 20,000 | ~2.7 hr |
| ~200k (paper 100k-epoch scale) | 100,000 | ~13–14 hr (overnight) |

B=250 (~8 gen/epoch) reaches the same update count in ~4× fewer epochs, net ~2–2.5×
faster in gen-update terms (~5–6 hr for 200k) — but verify per-epoch cost.
Caveats: **GPU likely won't help** (3×20 net rolled 40 steps is latency/overhead-
bound, not matrix-bound — matches the earlier "Colab was slower"). **Measure, don't
guess:** run 200 epochs, read seconds/epoch, multiply.

### Decisive next test (no add-ons)
Keep **oracle D̃**, turn **MMD + selection off**, optionally drop batch to buy more
generator updates/epoch, and train long. Watch whether `fmean` **trends to ~0**
instead of oscillating. If it converges → it was pure budget (professor is right),
no MMD needed. If it still oscillates at 50–100k generator updates → a real
instability worth addressing. Start with the ~1.4 hr / 20k-update check before
committing to the overnight run.

---

## Part 4 — Pinpointing the failure inside the GAN, and the decisions

**Update to Part 3's advice:** the "just train longer" path is likely wrong. The
CSV shows the generator mean got *better* mid-run (ep 301–701, |err|~0.017) then
*worse* again (ep 1000, |err|~0.055) — a random walk with no restoring force, the
"10k worse than 1k" effect inside a single run. So more iterations will not pin the
mean.

### Pinpoint: generator vs critic
- **The mean is a free parameter of the generator.** Nothing forces E_z[S̃(x,z)]=0,
  so the generator's mean settles wherever the critic's gradient vanishes — i.e.
  the mean is set *entirely by the critic*.
- **The critic cannot supply that gradient at the needed precision.** In
  Wasserstein-1 a mean shift δ changes the distance by only ~δ (≈0.004), while the
  variance/shape mismatch (the over-dispersion) is far larger and eats the critic's
  1-Lipschitz budget → the mean gets a diluted share. With ~2 gen / ~10 critic
  updates per epoch the critic never reaches its optimum, so even that signal is
  noisy with no consistent sign.
- **Precision mismatch is the root.** ÷Δ (Δ=0.01) means the fixed point needs the
  residual mean correct to ~1e-4. The noise floor of adversarial training (SGD
  jitter + GP + oscillating critic) is ~1e-2. ×100 turns that floor into a mean
  error of order 1 → the 0.245 / 0.82 collapse.
- **So it is NOT capacity** (a linear critic detects a mean fine) and **NOT epochs**
  (more iterations wander). **The mean is being routed through a channel —
  adversarial training — that physically cannot carry it.** The critic *can* still
  help the variance/shape, which W-1 weights meaningfully.

### How zero-mean construction helps
It deletes the mean DOF from the generator: subtracting the per-state z-average
forces S̃ zero-mean by construction. The increment mean is then owned by D̃, trained
by **MSE — which targets the mean directly, at the right scale, with no ÷Δ dilution
and no adversarial noise.** Net re-routing: **mean → regression (D̃), residual shape
→ GAN.** With oracle D̃ this provably lands the mean at μ; the only open question
becomes whether the variance is right (now the critic's isolated job).

### How it deviates from the paper
Conceptually it doesn't — it enforces exactly what Remark 4.1 *wants* ("the purpose
of D̃ is to ensure the stochastic learning part has mean value close to zero"). The
paper leaves that **emergent** from 100k-epoch training of a free DNN S̃ (eq. 4.18);
we make it **architectural**. Narrow but real deviation: our generator can no longer
represent a nonzero conditional mean. Since the target is zero-mean anyway, we only
remove "wrong" capacity. Caveat: the paper's bar is visual agreement (Fig. 5) — its
mean may also be slightly off, just not scrutinised at the ÷Δ level.

### Decisions taken
- **Normalization:** min-max is a dead end for GBM. Switch to a data-derived,
  invertible, per-coordinate transform (**Yeo-Johnson** default — discovers a
  log-like map from skewed data without assuming the equation). But **not now** —
  keep min-max for the OU mean test; change it only when moving to GBM (avoid
  confounded runs). Note Yeo-Johnson is nonlinear → drift/diffusion (4.23) then live
  in transformed space.
- **Iterations vs structure:** do **not** spend 14 h on a long run to fix the mean;
  fix it structurally (hard-centering). Reserve longer training / stronger critic
  for the **variance/over-dispersion**, where W-1 actually has signal.
- **One change at a time.** Reject the "do all five at once" plan (center + residual
  matching + bigger critic + normalization + diagnostics). Isolate each.

### One-line summary
The generator has an unconstrained mean DOF; the critic cannot pin it to the ~1e-4
precision the flow map needs; zero-mean construction removes that DOF and hands the
mean to D̃, which can.

---

## Next model to train (step 1 — single change)
**Hard-centered generator, oracle D̃, everything else untouched.**
- `use_oracle_det: true` (perfect mean map).
- Generator wrapped so its output is centered per state:
  S̃_c(x,z_0) = S̃(x,z_0) − (1/K) Σ_{k=1}^K S̃(x,z_k), fixed **K=16**, applied in
  **both training and evaluation** (not post-hoc rollout only).
- `mmd_lambda: 0`, `selection_mean_weight: 0`, 3×20 critic, min-max, B=1000 — all
  unchanged.
- **Success = mean → 1.2** (provable if centering works) and we then read off
  whether the **diffusion/std** is right.
  - Mean 1.2 + std ~0.3 → decomposition fixed; move to learned D̃, then variance,
    then GBM.
  - Mean 1.2 + std wrong → mean solved structurally; over-dispersion is the next
    isolated target (stronger critic / residual-matching).
- Short run (~8 min at 1000 epochs) is enough to see the mean.

---

## Part 5 — Step-1 result: mean FIXED, and "over-dispersion" was a mean artifact

Ran the hard-centered generator + oracle D̃ (K=16, MMD off, mean-sel off, 3×20
critic, min-max, B=1000, 1000 epochs). Runtime ~20 s/epoch (16× generator calls in
the rollout for the MC mean estimate — expected; lower `center_K` to trade back).

### The mean is solved
- Mean tracks 1.2 exactly; `mean_abs_err ≈ 0.002`; residual check `mean = 0.0001`
  (zero by construction). Binned **drift** now sits right on the analytical line.
- Confirms the identifiability diagnosis operationally: with the mean removed from
  the generator and owned by D̃, the fixed point is correct. Centering works.

### Centering did NOT break the diffusion — it UNMASKED under-dispersion
Two independent proofs the low diffusion is not caused by centering:

1. **Mathematically it can't be.** Subtracting the mean of K samples (sample
   included) scales variance by (1 − 1/K); at K=16 that is a **3%** reduction
   (√(15/16)=0.968). Cannot turn std 0.0355 → 0.007.
2. **The earlier "over-dispersion" was an artifact of the wrong mean.** The
   covariance plot uses the true μ=1.2 as reference, so
   **C(0) = Var(x) + (mean − 1.2)²**. Against past runs:
   - plain, mean 0.245 → (0.955)² = **0.91**, plot showed C(0) ≈ 0.91 ✓
   - MMD, mean 0.82 → (0.38)² = **0.144**, plot showed C(0) ≈ 0.156 ✓
   The "over-dispersion" was ~entirely the squared mean error. With the mean now
   correct, the artifact is gone and C(0) ≈ 0.01 vs 0.045 — the model was
   **under-dispersed all along** (the mean error inflated the variance metric 20–90×).

### The real, now-isolated defect: variance collapse from a weak critic
- Diffusion ~0.13 vs 0.30 (~2.3× too low); std settles ~0.098 vs 0.212;
  conditional at x=0.8 is too narrow.
- Generator variance **collapses over training**: `dy std fake` 0.025 (ep10) →
  0.007 (ep1000), real fixed at 0.0355. Variance was *best* early (ep10–20) before
  it drifted down.
- Root cause = same weak critic, now in the variance channel: **W-gap ≈ 0.0002
  while real/fake std differ 5×.** The critic cannot witness the variance mismatch,
  so nothing holds the variance up. Checkpoint selection (interval 100) also missed
  the better early epochs; its score is dominated by std_rel_err (~0.6–0.85).

### Status
Staged plan is working: **mean isolated and fixed structurally; variance is now the
sole remaining defect, and it is the same under-powered-critic problem flagged from
the start — finally visible in isolation.** No add-ons needed for the mean.

## Next model to train (step 2 — single change)
**Stronger critic**, everything from step 1 kept (oracle D̃, hard-centering on, MMD
off, mean-sel off, min-max). Goal: give the critic capacity to witness the variance.
- Scale critic capacity by a fixed rule (e.g. width ≈ 4× input dim, 3–5 layers) —
  capacity scaling with problem dimension, not per-example tuning.
- Optional companion: **residual-matching** — critic sees x_{n+1}−D̃(x_n) instead of
  raw increments, removing the drift ramp from its input.
- Success = (a) W-gap becomes nonzero when std differs, (b) variance stops
  collapsing, (c) diffusion → ~0.3 / std → ~0.21.
- Then: learned D̃ (mean error should equal D̃'s Phase-1 bias), then GBM with
  Yeo-Johnson normalization.

---

## Part 6 — Step-2 result: stronger critic FIXED the variance channel

Ran 4×128 critic (55,041 params vs 1,701 at 3×20), everything else from step 1 kept
(oracle D̃, centering K=16, MMD off, mean-sel off, min-max, B=1000, 1000 ep). Runtime
~2 s/epoch (bigger critic + GP double-backward on top of K=16 centering).

### The critic came alive
- **W-gap:** dead 7e-6 (ep80) → responsive ~2e-3 that tracks the std mismatch;
  critic loss goes **negative** late (−0.002) — it now scores real above fake, i.e.
  it discriminates. (The training-history *plot* shows W-gap/GP as flat lines only
  because the critic-loss spike to ~10 sets the y-scale; the log confirms they move.)
- **Fake std no longer collapses.** `dy std fake` oscillates around the real 0.0355
  (0.020 ↔ 0.069) instead of decaying to 0.007.
- **Residual check std = 0.0349 vs 0.0354** (was 0.0137). One-step noise amplitude
  now correct.
- Best checkpoint ep500, **std_rel_err 2.2%**.

### OU now reproduces on BOTH moments (evaluated at ep500)
- Mean locked at 1.2 (centering); rollout std ~0.22 vs 0.212; diffusion ~0.30;
  drift on the analytical line; covariance C(0) 0.051 vs 0.045. A real reproduction
  vs the morning's 0.13 diffusion / 0.098 std.

### Residual defects (second-order)
1. **Mild, x-dependent over-dispersion.** Diffusion tilts up 0.27 (low x) → 0.35
   (high x); since the test IC is x0=1.5, this inflates rollout std ~6% and C(0)
   ~13%. Lives in the **sparse high-x tail** — the known coverage/identifiability
   limit.
2. **Under-damped training.** Fake std oscillates ±50% around real; the good result
   relies on checkpoint selection catching a crossing (ep500). Final model (ep1000,
   std 0.044) is over-dispersed — that's the oscillation, not a trend.
Both are exactly what the CR-GAN consistency (MMD) term was built to damp — and we
now have the strong critic that the CR-GAN recipe pairs it with.

### Status
Step 1 (diffusion) **done** via the stronger critic. OU reproduces on mean + std +
drift + diffusion + covariance. Division of labor: **oracle D̃ + centering own the
mean; 4×128 critic owns the variance.** Remaining is polish (tail over-dispersion,
oscillation).

## Next model to train (MMD run)
Add a **light MMD term** on top of the current working setup (oracle D̃, centering
K=16, 4×128 critic) — everything else unchanged. Goal: damp the std oscillation and
pull in the mild tail over-dispersion (CR-GAN's designated stabilizer, now that the
critic is strong). `mmd_lambda: 1.0` to start; lower if it over-damps the variance,
raise if no effect. **This is an exploratory single run — record the result whatever
it is, then move to Phase 1 (learned D̃).**

---

## Part 7 — MMD result + the diffusion tilt is PROVEN to be data coverage

### MMD (λ=1) stabilised training but didn't move the eval
- **Training:** oscillation gone. `dy std fake` swing (ep100+) **0.049 → 0.0006**,
  locked on real 0.0355; best std_rel_err **2.2% → 0.0%** (ep439); final-epoch
  std_rel_err **~25% → 0.1%**. No longer depends on catching a lucky checkpoint.
- **Eval:** basically unchanged (std ~0.23, C(0) ~0.053, diffusion tilt 0.28→0.36).
- **Why:** MMD matches the **pooled** increment distribution (flattened over all
  states), so it locks the *aggregate* std but not the *state-dependent* variance.
  Our residual defect is x-dependent, so MMD can't touch it.
- **Decision: keep MMD** — it buys checkpoint-independent stability (matters more
  for learned D̃ and harder examples), even though it didn't raise the peak.

### The high-x diffusion tilt = data coverage (measured, not asserted)
Binned the training data's own coverage + empirical diffusion:
```
data x-range: [-0.258, 1.437]   <- data NEVER exceeds 1.44
 x-bin      count    frac   emp.diffusion (true 0.30)
 0.45-0.60 102579  25.6%   0.300
 0.75-0.90  46001  11.5%   0.300
 0.90-1.05  17033   4.3%   0.298
 1.05-1.20   3799   0.95%  0.302
 1.20-1.35    495   0.12%  0.288
 1.35-1.50     25   0.01%  0.300
 1.50+          0   0%     --
```
- Data's own diffusion is **flat 0.30 everywhere sampled** — not misleading.
- Model matches **0.30 exactly on-domain** (x<~1.0, >99% of data).
- Model tilts up **exactly where data vanishes** (>1.05 is <1%, >1.44 is zero).
- **Test IC x0=1.5 is entirely OUTSIDE the training range.** The tilt is the
  generator **extrapolating** the diffusion off-domain — honest uncertainty, not a
  GAN bug. Drift stays exact there only because the oracle D̃ is exactly linear.
- **Must not** force it flat: "constant diffusion" is OU-specific knowledge that
  would break GBM (which has genuine x-dependent diffusion).

### On "more data"
- More **same-distribution** trajectories (N↑): only marginal — the process from
  IC~U(0,0.25) essentially never reaches x=1.5 (>3σ event), so the test region
  stays empty. Tilt shrinks slightly, not gone.
- Data **covering the test region** (wider ICs / longer runs): would fix it, but
  converts the paper's *extrapolation* test into *interpolation* (different experiment).
- More **epochs**: no — the information isn't in the data.

### Three defects, three different causes (don't over-attribute to data)
- **Mean** → not data; unconstrained generator DOF → fixed by centering.
- **On-domain diffusion** → not data; weak critic → fixed by 4×128.
- **Off-domain diffusion tilt** → data coverage; the paper's extrapolation setup.

**OU is reproduced on-domain (mean, drift, one-step std all exact).** Moving to Phase 1.

---

## Part 8 — Phase-1 correction (learned D̃)
Flip `use_oracle_det: false`. With hard-centering, the model's mean error now equals
D̃'s Phase-1 bias **exactly and attributably** (the GAN can't compensate it).
- **Prior evidence:** learned D̃ fixed point ~1.15–1.16 (multi-step), OLS ~1.18; the
  data floor is ~1.18–1.19 for seed 0 (irreducible without more data).
- **Diagnostic first:** run **Phase 1 only** (`phase1_only.py`) — trains the ResNet
  D̃ and plots its rollout vs analytical + reports its fixed point and its drift
  extrapolation to x=1.5. Fast (no GAN).
- **Candidate correction:** with centering, D̃'s only job is the one-step conditional
  mean, for which **single-step MSE (L=1, eq. 4.12) is the direct estimator** — and
  it measured *better* on the fixed point (1.18) than the multi-step loss (1.16).
  Also check D̃'s **extrapolation** to x=1.5 (a NN may not extrapolate the linear
  drift as cleanly as the oracle). Decide after seeing the diagnostic.

---

## Part 9 — Phase-1 diagnostic: single-step did NOT help; root cause is coverage + over-flexibility

Ran `phase1_only.py`. Numbers:
```
multi-step D̃   1.153     (paper's eq 4.11)
single-step NN 1.130     (WORSE than multi-step)
single-step OLS 1.1825   (best LINEAR conditional-mean fit)
true μ         1.200
```
- Single-step made the fixed point *worse*, not better — my 1.18 prediction was
  wrong. **The linearity is what gave OLS 1.18, not the single-step objective.**
- **Extrapolation table** showed the learned drift is *curved*: right in the dense
  region [0.3,1.0], then it steepens above x=1.0 (crosses zero at 1.13) and
  **saturates** at high x (f≈−0.29 at x=1.5 vs true −0.30... but reaches only −0.29
  at 1.6 vs true −0.40). It's a density-weighted curve, not the true line.

### Root cause (thought about generally, NOT OU-specific)
**A mismatch between where the training loss puts information and where the
quantities we care about live.** ICs~U(0,0.25) ⇒ 99% of data in x∈[0,1.0]; but the
fixed point (~1.2) and the test IC (1.5) sit in the sparse tail / beyond it. MSE is
dominated by the dense region, so D̃ is unconstrained exactly where the fixed point
lives → a flexible NN extrapolates it however its inductive bias dictates. Same root
cause as the diffusion tilt. Two layers:
- **Irreducible (data):** even OLS only reaches 1.18 on this seed (finite-sample).
- **Reducible (model):** the NN reaches only 1.13 < 1.18 — it extracts *less* than
  the data supports because it overfits the dense region.
- **General fix = complexity control, NOT a fixed form:** prefer the simplest D̃
  consistent with the data (weight decay / curvature penalty / capacity selection by
  validation). For OU that auto-lands near-linear; for GBM/double-well it lands on
  the minimal curvature the data supports. Occam, data-driven — never "assume linear."
- Plus honest **out-of-distribution** posture: x0=1.5 is outside training support
  (max 1.44) — the model should flag extrapolation there, not confidently predict.

### Flow-map framing (why the paper tests OOD)
G_Δ is a **local, autonomous operator over state space** — IC-agnostic by design. So
the natural, sharpest test is to march it from a **fresh IC** (1.5) in a **fresh
direction** (down from above μ; training trajectories rose up from below). That
demonstrates it learned the *operator*, not the training ensemble — the whole selling
point of FML over methods that fit p(trajectory). The catch: 1.5 is also just past
the **state coverage**, and a flow map is only trustworthy on states its data
visited. So our residual errors are the model at the frontier of learned coverage —
a property of learning an operator from finite coverage, not an OU quirk.

---

## Part 10 — Un-centered test: the GAN CANNOT carry the mean (centering is necessary)

Tested the paper's design directly: **learned multi-step D̃, centering OFF, MMD OFF,
4×128 critic** (≈ professor's clean setup + strong critic).
- **D̃ alone → 1.157. Full model → ~1.00** — the un-centered GAN dragged the mean
  0.16 *below* D̃'s own fixed point (injected a spurious negative mean). Worse than
  doing nothing.
- **Training destabilised late:** mean-abs-err oscillated with *growing* amplitude
  (0.18→0.055→0.34→…→0.73 by ep900), C(real) 0.07→0.5, W-gap swings ±0.01→±0.09.
- Across *every* un-centered run the mean lands wrong and different (0.245, 0.82,
  1.00). Only **centered** runs hit 1.2. **⇒ hard-centering is genuinely necessary
  at our budget, not a weak-critic crutch.**

### Why un-centered fails (and why the paper may not) — the mechanism
- WGAN is a **minimax game** → gradient dynamics seek a *saddle*, which can orbit /
  diverge rather than converge (classic constant-step min-max cycling).
- The **mean is a nearly-flat, weakly-damped direction** (W-1 penalises a mean shift
  δ by only ~δ). Under constant LR that direction's orbit *grows* → the generator's
  mean runs away, ×100 by 1/Δ. Centering **deletes that direction**, leaving only
  well-damped (variance/shape) directions the strong critic controls → stable & fast.
- **Why the paper might get away with un-centered (hypotheses, not certain):**
  (a) **decaying LR** damps the orbit (Robbins-Monro) — the 2022 paper uses cosine
  decay; we were on the constant-LR (professor) branch; (b) scale/averaging at 100k
  epochs; (c) checkpointing a *bounded* orbit; (d) a more accurate D̃. So "more
  training" only helps *with damping* — more constant-LR training just grows the
  spiral (exactly what we saw).

### Current run (test of the mechanism)
Added Phase-2 **cosine LR decay** back as a toggle (`lr_decay: true`, 5e-5→1e-5).
Kept everything else from the un-centered run. Watching whether the late-epoch
oscillation *shrinks* (mean spirals toward ~1.15–1.2) as LR decays. Caveat: 1000
epochs is a short schedule (5× shrink) — expect partial damping at best; the paper's
version plays out over ~100k. If it clearly damps → mechanism confirmed, centering is
just the cheaper route. If not → centering stays the practical answer.

---

## Part 11 — Results confirmed + normalization solved for GBM

### Learned-D̃ full pipeline (centered + MMD + decay + 4x128 critic) — OU baseline
Ran and matches the prediction exactly:
- **Mean settles at ~1.157 = D̃'s fixed point** (centering hands the mean to D̃).
- std ~0.22 (vs 0.21), diffusion ~0.30 on-domain (tail tilt = coverage), C(0) 0.049
  (vs 0.045), drift on the line, conditional good. **Training rock-stable.**
- The *only* residual is the mean 1.157 vs 1.2 = D̃'s Phase-1 bias: ~0.018 data floor
  (OLS 1.1825) + ~0.025 recoverable over-flexibility. OU is essentially reproduced
  with a *learned* D̃; the remaining lever is D̃ complexity control (optional polish).

### Un-centered + cosine decay run (mechanism test)
- Decay **worked as predicted**: oscillation damped — `dy mean fake` late swing
  collapsed ~0.04 -> ~0.005, C(real) stayed bounded ~0.23 (vs blowup to 0.5), no
  divergence. Confirms the minimax-oscillation / weakly-damped-mean story.
- **But the mean still converged to ~1.0 (wrong).** Deeper finding: the un-centered
  *equilibrium itself* is wrong — the GAN matches the **pooled** increment mean
  (fake 0.006 ~ real 0.008) while getting the **state-dependent** mean wrong at
  x*~1.2 (sparse, W-1-blind). Centering enforces E[S|x]=0 **per state**, which is what
  pins the fixed point. => centering is necessary for *correctness*, not just stability.
- Overnight scale test queued: un-centered + decay + 70k epochs (`config_overnight.yaml`).

### Normalization for GBM — SOLVED and verified (equation-agnostic)
GBM state range [0.0002, 172.7], skew 6.1. Min-max **crushes 94% of data below -0.9**
(median -> -0.98) — that is why GBM diverged (representation failure, not the GAN).
**Yeo-Johnson** (per-coordinate power transform, lambda fit by MLE):
- lambda = **-0.41 discovered from data** (log-like) — *no* knowledge that GBM is geometric.
- skew 6.1 -> 0.06; round-trip error ~1e-13 (invertible for prediction).
- **Variance-stabilises:** raw increment std grows 16x across the range; in YJ space
  it's ~flat (0.033-0.042). Multiplicative noise -> additive => GBM becomes OU-like for
  the GAN in transformed space.
- **Safe as default:** on OU, lambda=0.60 with 0.999 correlation to identity, so
  OU/existing runs are unchanged.

**Implemented:** `due/datasets/normalizer.py` (minmax | yeojohnson, per-coordinate,
invertible, no clip so the OOD test IC still extrapolates). Wired into `sde_dataset`
as config toggle `normalization` (**default minmax** -> OU byte-identical; stores
`data_loader.normalizer` for eval to invert). Verified numerically; not yet run
end-to-end (needs torch = the user's Mac).

**Still needed to run GBM:** the GBM scripts were deleted earlier (only the .mat data +
model dirs remain). Need `examples/GBM/config.yaml` (full pipeline + `normalization:
yeojohnson`) and `GBM.py` (eval inverting via `data_loader.normalizer.inverse`; note
GBM has NO mean reversion / no fixed point — compare to GBM's analytical lognormal
moments, not an OU-style fixed point).

---

## Part 12 — Normalization proven end-to-end + a `none` (raw) mode

Added a `none` (raw, no transform) mode and switched OU.py eval to route through
`data_loader.normalizer` (byte-identical to the old min-max formula for `minmax`).
Rebuilt the GBM scripts (μ=2, σ=1, ICs U(0,2), test x0=0.5, T=1.0, conditional at x=6).
**Confirmed: the paper does NOT normalize** (searched the whole PDF — only "normal
distribution" for z, "normalizing flow" as an alternative, "lognormal" noise). Min-max
is a DUE-library artifact, and it's what breaks GBM.

### OU three-way (all ~equivalent — normalization is immaterial for OU)
- `minmax` mean 1.157 | `none` (raw) mean 1.148 | (predicted `yeojohnson` ~same, λ=0.60
  near-affine). Differ only ~0.01 from D̃-optimisation scale-sensitivity. Raw's variance
  match was excellent (`dy std` pinned 0.030 for 1000 ep). ⇒ **normalization was innocent
  for OU all along; the pipeline runs paper-faithfully on raw data.**

### GBM raw (paper-faithful) — FAILS, exactly as predicted
- Model trajectories go **negative** (down to −8) — impossible for GBM (strictly positive).
- Root cause: learned **constant diffusion ~7** instead of σx (flat, not the line). Raw
  states 0–172 + increments 0–32 are badly conditioned; the network fits the median-scale
  noise and misses the x-dependence. → too much noise at small x → crosses zero → D̃'s
  linear map runs it away negative.
- Consequences: std ~15 (vs 4.8), mean 2.9 (vs 3.69, corrupted by runaways).

### GBM Yeo-Johnson — FIXED (λ=−0.41 discovered from data)
- **Diffusion now tracks σx** (multiplicative captured); **std ~4.8** (matches);
  **drift μx, positive x only**; **mean ~3.5** (vs 3.69, ~5% under); conditional at x=6
  excellent; **training rock-stable**. In YJ space the increments are ~0.034 — **OU-like**
  (`dy std 0.034/0.0337` matched all run) — confirming "GBM → OU in transformed space".
- Residual mean undershoot (3.5 vs 3.69) = D̃ Phase-1 bias **+ a Jensen effect**: hard-
  centering enforces zero-mean in the **nonlinear YJ space**, so D̃ ≈ geometric mean;
  inverting underestimates the arithmetic mean E[x]. Noise lifts it partway (3.5 > D̃'s
  2.55) but not fully. *Keep in mind for all multiplicative examples.*

**MILESTONE: one equation-agnostic pipeline (centering + 4×128 critic + MMD + decay +
data-derived normalization) now reproduces BOTH OU and GBM** — only the normalization
differs, and it's *discovered* from data (λ), never assuming the equation.

---

## Part 13 — Overnight scale test: budget is NOT the answer (decisive)

Ran the un-centered design at scale: **un-centered, MMD off, cosine decay on, 4×128
critic, learned multi-step D̃, 70,000 epochs** (`config_overnight.yaml`, ~13 h).
- **Mean landed at ~1.57** — wrong, and *above* both 1.2 and D̃'s 1.157 (this run the
  un-centered generator added a *positive* mean; earlier runs went the other way to
  1.0/0.82/0.245). The mean is an **unconstrained direction** — lands somewhere different
  every run, never homes to 1.2.
- **Generator loss diverged** — climbed 0 → ~12 over 70k while W-gap/GP stayed ~0 (WGAN
  score-level drift): no stable equilibrium even at 70k epochs.
- Covariance C(0) 0.19 (dominated by (1.57−1.2)²); one-step conditional still fine.

**Conclusion:** gave the paper's un-centered design its best shot — strong critic, LR
decay (Robbins-Monro damping), 70× the epochs — and the mean *still* doesn't converge;
training doesn't even reach equilibrium. ⇒ **The mean is a structurally ill-conditioned
direction of the WGAN objective; hard-centering is genuinely necessary at any reachable
budget, not optional.** Caveat: cosine over 70k decays LR very slowly (near-constant for
most of the run), but even the low-LR tail shows no convergence.

### Side-by-side (the story for the professor)
| | mean | training |
|---|---|---|
| Centered (our design) | 1.157 (= D̃ fixed pt) | stable, reproducible |
| Un-centered, 70k + decay (paper's) | 1.57 | diverging loss, no convergence |

Full result: **one equation-agnostic pipeline reproduces OU and GBM**, plus a direct
demonstration of *why* the structural centering is required.

---

## Part 14 — Nonlinear diffusion (§5.2.1): moments + drift excellent, bell-shaped diffusion under-resolved

Built `examples/NonlinearDiffusion/` (data + config + eval). SDE
$dx=-\mu x\,dt+\sigma e^{-x^2}dW$, μ=5, σ=0.5, ICs U(−1,1), test x0=−0.4. Full pipeline,
min-max (states bounded [−1,1]).

**What reproduced well:**
- **Mean** reverts to 0, sits on the analytical $x_0e^{-\mu t}$. (Strong reversion θ=5 ⇒
  easy mean, no stiff fixed point.)
- **Std** ~0.15 vs ground-truth (EM) ~0.157 (~4% under).
- **Effective drift a(x)=−μx** — perfect across [−1,1], edges included (D̃ owns it).
- Training rock-stable; pooled `std_rel_err` → **0.0001** (pooled increment std matched
  essentially exactly).

**The miss — the bell diffusion $b(x)=\sigma e^{-x^2}$:**
- True bell peaks 0.5 at x=0, decays to ~0.18 at x=±1. Model got the **center right**
  (~0.48) but learned a roughly **constant/tilted ~0.45–0.53** — missed the edge decay.
- **Root cause (coverage again):** θ=5 makes trajectories rush to 0, so states near x=±1
  are visited only briefly (first ~14 steps of early-start windows, a few % of samples).
  The bell's informative structure (the *drop* at the edges) lives exactly where the
  dynamics spend no time → weak signal → generator defaults to the central variance.
- **Pooled-vs-conditional, again:** critic + MMD matched the **pooled** increment std
  *perfectly* (1e-4), yet the **state-dependent** diffusion at the sparse edges is wrong.
  Matching the aggregate ≠ pinning b(x) where data is thin. Moments still come out great
  because the test trajectories live near 0, where the diffusion *is* right — only the
  effective-diffusion *figure* (the example's showcase) is partial.

**Interpretation.** Drift and moments generalise to the nonlinear case; what's
under-resolved is the *fine structure of a strongly state-dependent diffusion in a
sparsely-visited region* — same family as OU's high-x tail tilt. Not catastrophic (unlike
raw GBM); a coverage/resolution limit.

### Suggested fix (deferred global pass — analogous to the D̃ complexity pass for the mean)
The loss matches the **pooled** increment distribution, under-weighting sparse states.

1. **More epochs — TESTED (5000 ep) → did NOT fix the bell.** Diffusion plot identical to
   the 1000-ep run; training converged by ~epoch 200 and stayed flat to 5000. ⇒ **not
   under-training** — the model fully fit what this data + pooled loss allow, and the edges
   stay wrong. Rules out the budget lever; confirms a coverage / loss-weighting limit.

### Why GBM's b(x)=σx WAS captured but NLD's b(x)=σe^{−x²} is NOT (the general statement)
The pooled loss captures state-dependent diffusion **only when the informative structure sits
in the dense / large-variance region**:
- **GBM** (σx): the structure is at large x — big increments that *dominate* the pooled
  marginal, so the critic/MMD are automatically driven to get it right. **Aligned** with the
  pooled weighting.
- **NLD** (σe^{−x²}): the structure is the *drop* at the edges — a *small* variance in a
  *sparsely-visited* region, so it is **doubly under-weighted** (few samples AND small
  contribution). The generator has no incentive; it defaults to the central value.

### Remaining fix (now the recommended experiment, since epochs is ruled out)
2. **State-aware objective — importance weighting.** Weight training samples by $1/\rho(x)$
   ($\rho$ = local state density estimated from data) so sparse edges get proper weight.
   Data-derived, equation-agnostic. **This is the deferred global "resolution" pass** (pairs
   with the D̃ complexity-control pass for the mean).
3. **Or more DATA (same ICs).** More trajectories densify the edges proportionally; helps
   this class (resolution/coverage) but not extrapolation cases (double-well) or model-limits.
4. **Complementary — one-step conditional / stratified-by-state matching** (the L=40 sequence
   loss dilutes state-conditioning: edge-start sequences revert to centre within the window).

**Status: still a success overall** — mean, std, drift all excellent and stable; only the
effective-diffusion *figure* at the sparse edges is partial. Headline reproduction holds.

---

## Part 15 — GBM Phase-1 undershoot SOLVED: raw units + gated_resnet (normalization destroys linearity)

**Problem.** GBM Phase-1 rollout undershot badly: D̃ from x₀=0.5 reached only ~2.54 at
T=1.0 vs analytical 3.6945 (**−31%**), despite near-perfect single-step accuracy. A ~0.38%
per-step multiplicative shortfall compounded over 100 steps. (`check_phase1.py`.)

**Root cause (general, not GBM-specific).** GBM's conditional mean E[x₁|x₀]=x₀·e^{μΔ} is
**linear** in x₀. But Yeo-Johnson is nonlinear, so *in YJ coordinates the linear map becomes a
curve*. The network must approximate that curve with its MLP, leaving sub-percent ripples in
the per-step growth that compound. **The normalization that Phase 2 needs (variance
stabilisation for the multiplicative diffusion) destroys the linearity Phase 1 depends on.**
Same lesson applies to any linear-/affine-mean SDE (OU included).

**Dead end — loss-coordinate reweighting (removed).** Hypothesis: YJ compresses the high-x
tail, so the normalized MSE down-weights it. Tried a `phase1_loss_space` toggle
(normalized|physical|relative) that reweights the Phase-1 error by the transform's local
stretch J = d(physical)/d(normalized) (delta-method, evaluated at the in-range target so it
can't explode; an earlier version that denormalised the *prediction* blew up to 1e22 and
collapsed to a contracting map). Result: **`relative` made it WORSE** (−60%, 1.49). It treats
a symptom; the real cause is the coordinate, not the loss weighting. **All of it reverted/
removed** — `ode.py` back to plain MSE, no `normalizer` arg, no `phase1_loss_space` key.

**The fix — raw units.** Set `data.normalization: "none"` (raw). The mean is linear again,
the net fits and extrapolates it cleanly. Ladder (T=1.0 endpoint, x₀=0.5, analytical 3.6945):

| Phase-1 setup | endpoint | error | per-step bias |
|---|---|---|---|
| YJ, resnet, plain MSE (baseline) | 2.54 | −31% | −0.38% |
| YJ, resnet, `relative` (J-reweight) | 1.49 | −60% | −0.90% |
| raw, resnet | 3.98 | +8% | +0.07% |
| **raw, gated_resnet** | **3.83** | **+3.8%** | **+0.036%** |

**Winner: raw + gated_resnet.** The gated_resnet's learned affine term (mDMD) represents the
linear drift *exactly* with the MLP gated off (α≈−10), so growth rate and extrapolation are
near-perfect. resnet-vs-gated on YJ were identical (~2.54) — the gate only helps once raw
restores linearity for the affine to capture.

**New toggle: `network.det_arch`** = `resnet` | `gated_resnet` (wired in OU, GBM, NLD;
`getattr(due.networks.fcn, det_arch)`). Default `resnet`.

**OU corollary.** OU was *already* raw (`normalization: none`) — which is exactly why its
Phase 1 was already good (fixed point ~1.21). OU's mean is also affine, so raw + gated_resnet
is being tested to tighten 1.21→~1.20. OU has *no* coordinate tension (constant diffusion →
raw serves both phases).

**Open tension (the real remaining issue).** Raw fixes GBM's *mean* but Phase 2's *diffusion*
needs YJ to tame the multiplicative (∝x²) noise. A single coordinate can't obviously do both.
Decisive next test: run the **full `GBM.py` in raw** and inspect effective drift/diffusion —
- diffusion survives raw ⇒ raw is simply GBM's setting, done;
- diffusion degrades at high x ⇒ must **decouple coordinates** (D̃ raw for the linear mean,
  S̃ in YJ for the variance) — a real change to the SDE model, only if this run forces it.

**Status.** GBM Phase-1 mean solved (raw + gated_resnet, per-step bias ~0.036%). Phase-2
raw-vs-YJ coordinate question is the open item. `phase1_loss_space`/J-reweighting removed as a
confirmed dead end.

---

## Part 16 — Decoupled Phase-1 (raw D̃ wrapped into YJ): built, ran, and it makes the full model WORSE (key negative result)

**Built (modular, toggle-gated, core trainers untouched).** New `due/networks/wrapped_det.py`
`WrappedDet`: sandwiches a raw-trained D̃ between the Phase-2 transforms,
`D_wrapped(u) = T(D_raw(T^{-1}(u)))`, so it drops into the (unchanged) SDE model as a
YJ-coordinate det-net. Config key `data.phase1_normalization` (absent/==normalization ⇒
coupled; else decouple). `GBM.py` guarded branch loads raw for Phase 1, wraps into YJ for
Phase 2. Verified: transforms match `Normalizer` (round-trip 8e-16); a perfect raw-linear
D̃ wrapped + rolled deterministically hits 3.694 vs 3.6945.

**Ran (GBM, raw Phase-1 → YJ Phase-2, 250 + 1000 ep, stable training).**
- Deterministic backbone TRANSFERRED: `det_rollout` 3.82 (vs coupled 2.54) — the raw linear
  mean shows through the wrap. GAN stable, increment std matched ~0.3%.
- **But the full stochastic model got much WORSE:** mean overshoots to **~6.2 (+68%)** vs 3.69,
  std blows up to **~35 (~7×)** vs ~5, both diverging after t≈0.6. Drift/diffusion overshoot too.
  Far worse than coupled (mean 3.5, std 4.8).

**Why (the real lesson — centering coordinate vs noise Jensen lift):**
- Coupled YJ: D̃ = YJ-space conditional mean ⇒ inverts to ~**geometric** mean (~2.55, an
  undershoot *alone*). The noise, through convex YJ⁻¹, gets a Jensen **lift** to ~3.5
  (≈arithmetic). Undershoot + lift **cancel** — self-consistent.
- Decoupled: D̃_raw = **arithmetic** conditional mean (correct alone, hence the clean 3.82
  deterministic rollout). Now the noise lift **stacks on top** instead of compensating ⇒
  mean floats up; and the distribution sits in the steeper part of convex YJ⁻¹ ⇒ the same
  YJ-space noise maps to a much wider physical spread ⇒ variance explodes. Both **compound**
  over 100 steps (fine early, blow up late).

**Conclusion.** The coupled model's "undershooting" D̃ is a **feature**, not a bug — the
geometric center pre-compensates the noise's Jensen lift. Decoupling gives a perfect
deterministic backbone but destroys the mean/noise self-consistency; the raw D̃'s accuracy
actively *hurts* the full model. **Raw-Phase-1 (Part 15) is a Phase-1-diagnostic truth, it
does NOT transfer to the full YJ model.** Coupled YJ (mean 3.5, ~5% under) stays the best GBM.
Decouple turned **OFF** (`phase1_normalization` commented) but the code is kept as a documented
negative result.

### Next — the Jensen correction (the actual lever for the residual ~5%)
The residual undershoot is a lognormal/Jensen bias: matching increments in YJ space does not
pin the *physical* mean. Delta-method: for a step `u' = D̃(u) + S̃` (Var S̃ = σ²(u)),
`E[x'|x] = E[YJ⁻¹(u')] ≈ YJ⁻¹(D̃(u)) + ½ (YJ⁻¹)''(D̃(u)) σ²(u)`. All three ingredients are
**equation-agnostic**: `YJ⁻¹` and its 2nd derivative from the fitted normaliser; `σ²(u)` the
generator's per-state increment variance (measurable); the data's physical conditional mean
`m(u)` estimated from training pairs. Fix = a per-state **center correction** Δ(u) so the
model's physical conditional mean matches `m(u)` (one Newton/delta step), applied either at
prediction (correct D̃(u) before inversion) or as a small correction head. This targets the
Jensen gap directly WITHOUT breaking the variance channel (unlike raw D̃). **→ Implemented as Part 17.**

---

## Part 17 — Jensen center-correction: implemented (awaiting results)

**Implementation (eval-only, config-toggled).** Added `compute_jensen_delta` function to `GBM.py`.
After loading the trained models, if `jensen_correction: true`:
1. **m_x(u)**: bin training pairs (u, physical x') to get the data-estimated physical conditional mean.
2. **D̃(u)**: evaluate det_net at bin centers.
3. **σ²_S̃(u)**: sample 2000 generator draws per bin center; compute empirical variance (equation-agnostic: never uses the SDE form).
4. **δ(u)**: delta-method correction in normalised space:
   `δ(u) = (m_x(u) − T⁻¹(D̃(u)) − ½(T⁻¹)''(D̃(u))·σ²) / (T⁻¹)'(D̃(u))`
   where `(T⁻¹)'` and `(T⁻¹)''` are computed analytically from the fitted normaliser (NZ.lam, NZ.smin, NZ.smax).
   Derivative formulas (x≥0, λ≠0): `(T⁻¹)'(u) = (scale/2)·(λy+1)^(1/λ−1)` and
   `(T⁻¹)''(u) = (scale/2)²·(1/λ−1)·λ·(λy+1)^(1/λ−2)` where `y = u·scale/2 + (smax+smin)/2`.
5. **Interpolant**: scipy linear interp, clamped at boundary.
6. **Applied at eval**: `d = det_net(u) + δ(u)` before each rollout step.
   Saves `jensen_delta.png` (δ vs u diagnostic).

**Derivative verification** (numpy, synthetic GBM data): analytic (T⁻¹)' and (T⁻¹)'' match
numerical finite differences to 5 decimal places — formulas confirmed.

**Config**: `gan.jensen_correction: false` (default, no change to existing behaviour); flip to `true` to enable.

**To run:**
```bash
# In GBM/config.yaml: set jensen_correction: true
caffeinate -i python GBM.py   # trains fresh + applies correction in eval
```

**Expected outcome**: physical mean corrected from ~3.5 toward ~3.69 (~5%→~0%).
δ(u) should be small (~0.01–0.05 in normalised space), positive (shifting D̃ up slightly).
If δ is large or has oscillations, the training data binning is too coarse — increase n_bins.
Std should NOT change materially (δ shifts the center, not the variance channel).

**Result: NEGATIVE. Correction made things much WORSE (mean 3.5→3.0, std ~4.8→~2.3).**

**Why it failed:**

1. **Tail corruption.** At u > 0.3 (high-x end of training data), δ spikes to **−0.07** — 14× the main-body amplitude. This is a data artifact: at high x there are very few training pairs → m_x(u) estimated from few samples, divided by dT⁻¹/du which is LARGE (grows as x^(1/λ−1) for λ = −0.414) → tiny numerator noise ÷ large denominator → enormous δ. The clamped boundary fill carries this −0.07 to all u > 0.57, hammering every trajectory downward at late time.

2. **One-step gap is already ~0 in the training region.** In the main body (u ∈ [−1.0, 0.3]) δ oscillates ±0.005 around zero — pure estimation noise, no systematic signal. This means: the model's per-step physical mean is ALREADY accurate in the training data distribution. The 5% rollout gap is NOT a one-step bias patchable by a per-step correction.

3. **The 5% gap is a multi-step compounding effect**, not a one-step mean bias. D̃ in YJ space gives det_rollout ~2.54; noise Jensen-lifts it to ~3.5 over 100 steps. The remaining gap to 3.69 is a cumulative nonlinear interaction between compounding geometric steps and the nonlinear YJ inverse — not addressable by shifting D̃(u) at each step.

**Conclusion.** Jensen delta-method correction: theoretically well-motivated but empirically wrong tool for this gap. The signal is below the estimation noise floor (σ(δ) ~ 0.025 >> true δ ~ 0.003), and the tail corruption is catastrophic. `jensen_correction` set back to `false`. Code kept as documented negative result.

**The 5% GBM mean gap is a known, understood residual.** Fix requires either: (a) better D̃ Phase-1 coverage in the x∈[0.5,4] test-trajectory range, or (b) more training data densifying the high-x region. Both are data/architecture passes, not post-hoc corrections. Accepting the ~5% gap and moving to the next paper examples.

---

## Part 18 — Double-well (§5.2.3): bimodal generator works; Phase 1 saddle pathology found

### Paper setup (from PDF, §5.2.3)
- SDE: dx = (x−x³)dt + 0.5 dW. IC: **U(−2.5, 2.5)**. dt=0.01, 100 steps, L=40 window, N=10,000.
- Test IC: **x0=1.5** (stable well), simulated to **T=300** to show inter-well transitions.
- Key claim: "trajectories in training data do not contain transitions" (T_window=0.4 too short).
  sFML reproduces transitions at O(10) timescale from training data with none.
- Eval plots: PDF evolution at T=0.5, 10, 30, 100 from x0=1.5; drift/diffusion recovery.
- Diffusion error: **"about 2%, rather acceptable"** (their words).
- Paper runs 100,000 epochs, 3×20 nets (no hard-centering, no large critic in paper).

### Our setup (diff from paper)
- IC: U(−2, 2) instead of U(−2.5, 2.5) — minor.
- Test IC: **x0=0** (saddle top) instead of x0=1.5 — this exposed the Phase 1 pathology.
- 3,000 Phase-2 epochs (vs paper's 100k — but our improved pipeline converged faster).

### What worked ✓
- **Bimodal conditional at T=0.50 from x0=0**: perfect match to EM ground truth. Generator
  successfully learned to spread trajectories into both wells. This is the hardest test.
- **Drift recovery f(x)=x−x³**: excellent across [−1.8, 1.8] including the cubic nonlinearity.
- **GAN convergence**: losses → ~0 within ~100 epochs; selection score 0.000716 at epoch 800.
  One-step std_rel_err = 0.022% — essentially perfect at the per-step level.
- **Stationary distribution**: bimodal histogram with peaks at ±1 reproduced.
- **Training speed**: 240s/100 epochs on CPU — consistent with GBM/NLD (center_K=16 + MMD dominate).

### Phase 1 symmetry breaking (the main finding, x0=0 specific)
From x0=0, D_theta rolls out to **−0.7 by T=5** while EM mean stays at 0.
- **Mechanism**: x=0 is an unstable fixed point (f'(0)=1>0). Under L=40-step unrolling with
  stochastic gradient batches, finite-sample noise breaks the saddle's symmetry. Random init
  of gated_resnet's affine/MLP causes D_theta to slide to the negative well.
- **Not a GAN issue**: the full model (D_theta + generator) mean only reaches +0.06 (not −0.7).
  The generator partially compensates D_theta's negative bias but slightly overcorrects.
- **Diffusion asymmetry**: g(x) slopes 0.53→0.47 (left→right) instead of flat 0.5. This is a
  downstream consequence of Phase 1 asymmetry — generator adds more noise on left to compensate.
- **Invisible in paper's protocol**: paper tests from x0=1.5 (stable well), where D_theta
  correctly stays near x=1 initially. The saddle pathology only shows from x0=0.
- **Std underestimate**: ~10% (0.84 vs 0.93 EM) — connected to asymmetric diffusion.

### Paper result vs ours
| Metric | Paper | Ours (x0=0) |
|---|---|---|
| Drift recovery | Good | Excellent ✓ |
| Diffusion error | ~2% | ~6% asymmetric (Phase 1 artifact) |
| Bimodal conditional | ✓ | ✓ (T=0.5 from x0=0) |
| D_theta from saddle | not tested | drifts to −0.7 (pathology) |
| Mean rollout | (x0=1.5, good) | +0.06 residual (from x0=0) |

### DoubleWell.py update needed (to match paper)
Add x0=1.5 test path: run N_STEPS=30000 (T=300), show 2 sample trajectories with transitions;
add PDF evolution subplots at T=0.5, 10, 30, 100 (compare sFML vs EM). This is the paper's
headline result — demonstrating sFML learns transition behavior it never saw in training data.
Currently DoubleWell.py only tests from x0=0. **Deferred; note in config `eval_only: true`.**

### Conclusion
The GAN (Phase 2) is NOT the bottleneck for double-well — it learned bimodal behavior correctly
and fast. Phase 1 multi-step MSE at an unstable fixed point is a structural limitation: the saddle
at x=0 is genuinely unstable under L-step rollout. This is a known failure mode of FML-style
approaches at saddle points, not double-well specific. Document as a known limitation.

---

## Paper reference (Chen & Xiu 2024, J. Comput. Phys. 508, 112984)

### Confirmed hyperparameters (from §5, p.9)
- All 1D examples: N=10,000, L=40, dt=0.01, 100 EM steps, random window.
- Network: **3 layers × 20 nodes** for ALL nets (generator, discriminator, D_theta).
  Exception: 2D examples use 40 nodes per layer.
- Training: n_ct=5, β₁=0.5, β₂=0.999, lr=5×10⁻⁵. **100,000 epochs** for all examples.
- No normalization mentioned. No hard-centering. No MMD. No LR decay mentioned.
  (Paper likely succeeds at 100k epochs via the Robbins-Monro / long-run convergence path
  we ruled out in Part 13. Our centered + 4×128 pipeline reaches comparable quality at
  ~1,000–3,000 epochs.)

### Example-by-example paper setup
| Example | IC | Test IC | T_test | Key eval |
|---|---|---|---|---|
| OU §5.1.1 | U(0, 0.25) | x0=1.5 | T=4 | mean/std, drift/diff, G(0.8), covariance |
| GBM §5.1.2 | U(0, 2) | x0=0.5 | T=1 | mean/std, drift/diff, G(6) |
| NLD §5.2.1 | U(−1, 1) | x0=−0.4 | T=10 | mean/std, drift/diff, G(−0.3) |
| TrigDrift §5.2.2 | U(0.35, 0.7) | x0=0.6 | T=10 | mean/std, drift/diff, G(0.5) |
| DoubleWell §5.2.3 | U(−2.5, 2.5) | x0=1.5 | T=300 | PDF at T=0.5/10/30/100, transitions |

---

## Part 19 — TrigDrift (§5.2.2): mean/std excellent, diffusion-shape miss (same NLD pathology)

### Setup
- SDE: dx = sin(2πx)dt + 0.5cos(2πx)dW, k=1, σ=0.5, dt=0.01. IC: U(0.35, 0.7), test x0=0.6.
- Config: gated_resnet (near x=0.5, f(x)≈−2π(x−0.5) linear → affine captures it), minmax norm.
- Phase 1: 500 epochs. Phase 2: 1200 epochs run (Ctrl-C at 1400), best epoch 900.
- Best epoch 900: selection score 0.01242, std_rel_err 0.43%.

### Results

**Phase 1** — perfect. D_theta rollout from x0=0.6 tracks EM mean exactly to T=10; convergence
to stable point x=0.5 is clean. Loss plateaued at 0.1584 within ~50 epochs. No symmetry breaking
(IC range is entirely within the stable basin; no saddle exposed).

**Mean & Std** — near-perfect. Both track EM across T=10. Stationary std ~0.12 matched exactly.
This is the best Phase-2 result so far in terms of aggregate moment tracking.

**Effective drift** — good within IC range [0.35, 0.70]. Sin(2πx) shape recovered; degrades
outside support as expected (extrapolation). The zero-crossing at x=0.5 is correctly placed.

**Effective diffusion — the miss:**
- Analytical: 0.5|cos(2πx)| — peaks 0.5 at x=0.5, falls to ~0.15 near IC boundaries at
  x=0.35/0.70, and zero at x=0.25/0.75 (outside training range).
- sFML: nearly flat ~0.40 across entire range. Misses the cosine shape; outputs spatial average.
- This is the identical NLD pooled-loss pathology: generator trained on pooled increment
  distribution, which is dominated by the dense/central states (near x=0.5). Edge states
  (near x=0.35/0.70) have both fewer samples AND smaller true diffusion → doubly under-weighted.
  Generator outputs a constant near the density-weighted mean.

**Conditional at x=0.5** — sFML too narrow. Analytical: N(0.5, 0.05²) (g(0.5)=0.5, std=0.05).
sFML: visibly narrower. Consistent with g_pred(0.5)≈0.40 → predicted std≈0.04. ~20% underestimate
at the peak of the diffusion curve.

**Despite the diffusion miss, mean/std rollout is excellent.** The generator compensates in
aggregate: even with wrong per-state diffusion shape, the ensemble variance integrates correctly
over the T=10 trajectory (test IC x0=0.6 spends most time near x=0.5 where diffusion is
over-estimated, which compensates). The per-state figure is wrong; the aggregate statistic is right.

### Comparison with NLD
| Aspect | NLD (§5.2.1) | TrigDrift (§5.2.2) |
|---|---|---|
| Diffusion shape | Bell σe^{−x²}: max 0.5 at 0, drops to 0.18 at ±1 | Cosine 0.5|cos(2πx)|: max 0.5 at x=0.5, falls to 0 at edges |
| Variation ratio | ~3× (0.18–0.5) | ~3× (0.15–0.5) |
| sFML diffusion | Flat ~0.48–0.50 (overestimates edges) | Flat ~0.40 (underestimates peak) |
| Mean/std rollout | Excellent | Excellent |
| Phase 1 | Clean | Clean |
| Root cause | Pooled loss, sparse edges | Pooled loss, sparse edges |

Both are the same structural issue. Different direction of miss (NLD overestimates edges,
TrigDrift underestimates the peak) because the training density is different relative to the
diffusion's shape, but same pooled-loss mechanism.

### Open issues confirmed by TrigDrift
1. **Diffusion-shape miss is systematic**, not example-specific. Affects any SDE where the
   diffusion's spatial structure is in a poorly-covered region. NLD + TrigDrift both show it.
2. **Fix options (unchanged from Part 14 analysis):**
   - Heteroscedastic generator S(x,z) = σ_net(x)·z — regression for diffusion shape, adversarial
     for residuals. Principled; equation-agnostic.
   - Importance weighting by 1/ρ(x) — up-weights sparse edges. Equation-agnostic but high
     weight-variance instability risk.
   - Separate generator_width: wider generator sees stronger gradient signal for diffusion.
     Currently generator uses config["width"]=20 (same as det_net); critic has separate 4×128.
     Easy to add; worth testing as a cheap fix before the architectural change.
3. **generator_width/depth key missing** — generator uses config["width"], critic has
   critic_depth/critic_width. Add generator_depth/generator_width to gan.py before testing wider
   generators.

### Files
- `examples/TrigDrift/generate_data.py` — SDE simulation, TD_train.mat / TD_test.mat
- `examples/TrigDrift/config.yaml` — Phase 1: 500ep, Phase 2: 1000ep (updated from 3000)
- `examples/TrigDrift/TrigDrift.py` — full pipeline + 4 eval plots
- `examples/TrigDrift/check_phase1.py` — Phase 1 diagnostic (rollout + drift recovery grid)
| Exp noise §5.3.1 | — | x0=0.4 | T=5 | mean/std, G(0.34), drift/diff |
| Lognormal §5.3.2 | U(0.1, 2) | x0=1.5 | T=5 | mean/std, G(0.4), drift/diff |
| 2D-OU §5.4.1 | U([−4,4]×[−3,3]) | x0=(0.3,0.4) | T=5 | mean/std, joint G(0,0) |
| Oscillator §5.4.2 | U([−1.5,1.5]²) | x0=(0.3,0.4) | T=6.5 | mean/std, marginals G(−0.5,−0.5) |

---

## Part 20 — OU Phase 2 diffusion tilt: grid-based diagnostic + single-step critic attempt

### Context
Professor wants OU "perfect." Known residual: ~8% std overshoot and state-dependent diffusion tilt
(upward slope in b̂(x) beyond x≈1.3). Root cause: sequence critic (L=40) converges W-gap→0 before
per-step variance error is resolved — GP regularization floor > per-step variance correction signal.

### Diagnostic improvement: grid-based drift/diffusion (eq. 4.23)
Old OU.py used trajectory binning for drift/diffusion — (x_t, Δx_t) pairs pulled from rollout
trajectories and binned by x value. This caused sparse/empty bins at tail states (OOD zeros and
scatter). Replaced with the paper's direct method (eq. 4.23):
- Fixed grid of 100 points x ∈ [0.5, 2.0] (in-distribution range, stationary ~N(1.2, 0.21²))
- N_Z=20,000 fresh i.i.d. z draws per grid point
- â(x) = mean(G̃(x,z) − x)/Δ, b̂(x) = std(G̃(x,z))/√Δ

Bug fixed: `normalize(scalar)` returned 0-d array → `torch.tensor(0-d).expand(N_Z, -1)` overflowed.
Fix: `normalize(np.array([[xg]]))` gives (1,1) shape → expands correctly to (N_Z, 1).

DoubleWell already used this method. OU/DoubleWell both show state-dependent diffusion tilt
(b̂(x) not flat), confirming this is a real Phase 2 issue, not a binning artifact.

**Key finding**: DoubleWell shows the SAME diffusion non-flatness (0.46–0.53 vs σ=0.5, downward slope
left-to-right) even with uniform IC U(-2,2). The root cause is not just OOD extrapolation — the
pooled sequence critic fundamentally cannot enforce constant per-step variance across all states.

### DoubleWell: x0=1.5 update (matching paper protocol)
Changed from x0=0 (saddle, Phase 1 pathology) to x0=1.5 (stable well, paper protocol):
- N_STEPS=2000 (T=20) to show inter-well transitions (mean escape time ~7)
- Inline EM ground truth generated on-the-fly (DW_test.mat was from x0=0)
- Plot 5: bimodal conditional at T=1 and T=20 — paper's headline result ✓
- config.yaml: `eval_only: true`

### Normalization: OU switched to minmax
Changed `normalization: "none"` → `"minmax"` for OU (matching DoubleWell). Rationale:
standardizes generator inputs; OU with none showed upward diffusion tilt partly because
x=2.0 is raw-OOD (physical value the generator never saw in training at that raw scale).
With minmax, x=2.0 → u≈1.02 (barely outside [0,1]) — far better extrapolation posture.
Requires full retrain.

### Single-step critic — ATTEMPTED (status: pending run)
**⚠️ IMPORTANT: single-step was the ORIGINAL implementation (Part 1) and was deliberately
replaced with the sequence critic to match Algorithm 4.1.** Now re-introducing it as an
experiment to fix the diffusion tilt, with key differences from Part 1:
- Part 1: single-step with **weak 3×20 critic** (couldn't witness variance → variance collapsed)
- This attempt: single-step with **strong 4×128 critic** + centering + MMD

**Rationale:** 8% per-step variance error → Wasserstein distance over L=40 steps is below the GP
floor. Single-step critic sees (x_t, Δx_t) directly: 8% variance mismatch is immediately detectable
with no gradient attenuation through 40 steps.

**Changes made:**
- `due/models/sde.py`: added `single_step_critic` flag (default False — fully backward compatible).
  When True: flattens N×L pairs into N*L=400k one-step dataset; fake generation = D(x_t)-x_t+S(x_t,z)
  (one forward pass, no rollout). Generator update: det_inc in no_grad (frozen), grad flows through S only.
- `examples/OU/OU.py`: sets `conf_net["sequence_length"] = 1` before Critic construction when flag on
  (critic input dim = 2d instead of d*(1+L)=41d).
- `examples/OU/config.yaml`: `single_step_critic: true`, `eval_only: false`, `use_oracle_det: false`.

**To revert to sequence critic:** set `single_step_critic: false` (or remove the key) in config.yaml.
The sde.py change is backward compatible — no other configs affected.

**Expected outcome:** flatter b̂(x) across the evaluation grid. Watch `dy std real/fake` in training
logs — should converge tighter than before (was ~8% gap). If variance still tilts, the issue is
data coverage not the critic architecture.

---

## Part 21 — Muon optimizer toggle (Phase 1 + Phase 2) — targeting the GAN diffusion channel

**Motivation.** Professor's suggestion, and a direct follow-on to **Part 20** (the sequence critic
converges W-gap→0 before the per-step variance/diffusion is resolved — GP floor > variance signal).
The residual defect across examples is the **GAN not fully learning the diffusion** (the variance
channel the critic must witness — cf. the 4×128-critic fix in Part 6, NLD's under-resolved bell in
Part 14, the OU/DoubleWell diffusion tilt + single-step-critic attempt in Part 20). Try **Muon**
(orthogonalising optimizer for hidden-layer weight matrices) on the WGAN-GP, especially the
**critic**, to see if it learns the diffusion better than Adam. Trial on **OU first** (like the
single-step-critic experiment), available to all examples.

**Why Muon needs a wrapper (the core complication).** Muon is a *matrix-level* optimiser: its step
**orthogonalises** the momentum matrix (≈ nearest matrix with unit singular values, via a few
Newton–Schulz iterations) so every singular direction of a weight matrix gets a comparable update.
That operation is only defined for **2D** params — biases (1D), scalars (the gated_resnet gate),
embeddings/conv have no orthogonalisation. So Muon *cannot* be `Muon(model.parameters())`; you must
run **two optimisers** (Muon on 2D weights + AdamW on the rest). Its natural step size also differs
from Adam (orthogonalised update ≈ unit-scale regardless of grad magnitude), so the LR must be
retuned. This is intrinsic to the algorithm, not a PyTorch quirk — it's why all the plumbing below.

**Added (config-toggled, default `adam` ⇒ byte-identical to before):**
- `due/utils.py` — `MuonAdamW`: routes params by `ndim` (Muon 2D / AdamW rest), drives both
  (step/zero_grad), exposes concatenated `param_groups`. `get_optimizer` gains a `muon` branch
  (Phase-1 via `training.optimizer`).
- `due/models/sde.py` — Phase-2 `opt_G`/`opt_C` switch on `gan.optimizer` (`adam` default / `muon`);
  prints `Phase-2 optimizer: <name>`.
- `examples/{OU,GBM,NLD,TrigDrift,DoubleWell}/config.yaml` — surfaced `optimizer` in the `gan:`
  section (`training:` already had one for Phase 1) + LR-guidance comments (raise to ~1e-3 for muon).

**Runtime fix — the scheduler.** First run errored `TypeError: MuonAdamW is not an Optimizer` —
`CosineAnnealingLR` does an `isinstance(optimizer, Optimizer)` check and the wrapper isn't a real
Optimizer. Fix: schedule the **real sub-optimisers** (Muon + AdamW *are* Optimizers) and step them
together via a new `MultiScheduler` (utils). `get_schedule` now detects `MuonAdamW` and wraps each
sub-optimiser (covers Phase-1 too); `sde.py` builds a `MultiScheduler` for `sched_G`/`sched_C` when
muon. The loop's `if sched_G is not None: sched_G.step()` is unchanged. Both files compile.

**Environment (resolved).** torch upgraded **2.12.1 → 2.13.0** (`hasattr(torch.optim,'Muon')` = True;
the venv was already on 2.12.1, not the pinned 2.0.1). `setup.py` pin loosened `torch==2.0.1` →
`torch>=2.0.1` (comment: Muon needs ≥2.13) to silence the pip conflict — re-`pip install -e .` to
clear the stale installed metadata. Only a minor bump (2.12.1→2.13.0), low breakage risk, but re-run
OU/GBM once to confirm the base pipeline before trusting Muon numbers.

**Caveats before trusting a result:**
- **LR is the #1 knob.** Muon's own default LR is 1e-3 (~20× the GAN's 5e-5); at 5e-5 the critic
  barely moves. OU config now set to `gan.optimizer: muon`, `learning_rate: 1e-3`. If C-loss
  explodes/NaNs, drop toward 3e-4.
- **Confound:** OU also carries the Part-20 `single_step_critic` experiment — run muon with
  `single_step_critic: false` (sequence critic) for a clean Adam-vs-Muon comparison.
- Critic (bigger, ~128×128 matrices) is where Muon is designed to help; generator (~900 params) is
  ~moot. Overall scale mismatch: Muon is built for large (LLM) matrices, our nets are tiny — payoff
  genuinely uncertain. WGAN-GP is optimiser-sensitive; treat as an experiment. Muon weight-decay 0.1
  (default) is aggressive for tiny nets — tunable in `MuonAdamW`.

**Result (OU, tested — Muon 1000 ep, lr 1e-3, sequence critic).** **Muon ≈ Adam — no effect on the
diffusion tilt.** Mean tracks (1.5→~1.21), std ~8% over (0.23 vs 0.21), **effective diffusion still
tilts up 0.30→0.35 across x∈[0.5,2.0]**, covariance ~15% high — a carbon-copy of the Adam baseline,
and the critic-loss curve collapses to ~0 by ep~40 identically. Pooled metrics excellent (best ep
100, std rel err 8e-4, `dy std real/fake` 0.0358/0.0358).

**Why it couldn't help (the useful finding).** The tilt is **not an optimisation failure**: under
Muon the critic converges perfectly (W-gap→0, GP→0, pooled increment std matched to 4 digits) —
identical to Adam. Once the pooled increment marginal is matched, there is **no gradient left for
the state-dependent variance**; a better optimiser just optimises the same blind objective. This
narrows the cause definitively: **not critic size** (Part 6), **not budget** (Part 13), **not the
optimiser** (this run) → it is the **objective/critic formulation** — a pooled sequence critic
structurally cannot enforce constant per-state variance (DoubleWell shows the same tilt with uniform
ICs, Part 20). The lever is what the critic *sees*: the **single-step critic** (Part 20) or a
state-conditional / stratified variance term — NOT the optimiser. More Muon tuning won't move it
(the critic already converges).

**Status:** Muon fully wired (Phase 1 + 2, all SDE examples) + scheduler fix (`MultiScheduler`),
torch 2.13 ready, runs cleanly. Tested on OU → **no improvement; optimiser RULED OUT as the
diffusion-tilt lever.** Kept as a toggle (`gan.optimizer`, default adam; OU reverted to adam after
the test). Uncommitted: `due/utils.py`, `due/models/sde.py`, `setup.py`, the 5 example gan configs.

---

## Part 22 — Increment scaling (professor's ×10): RESOLVES the OU diffusion tilt + std overshoot (the win)

**Idea (professor).** The critic compares increment distributions, but raw increments are tiny
(dy std ≈ 0.030). A per-state variance mismatch of ~6% is then ~0.0018 — **below the GP /
estimation-noise floor**, so the critic literally cannot witness it (the Part-20 / Part-17 root
cause, "GP floor > variance signal"). Multiply the increments the critic sees by ~10 so the same
mismatch becomes ~0.018 — an order of magnitude above the floor — and the critic can finally
resolve and correct the per-state variance.

**Implemented (`increment_scale`, config toggle, default 1 = off).** In `sde.py`: the critic and
gradient penalty see `s·y` (real and fake); the **generator, MMD, logged statistics, checkpoint
selection, and prediction all stay in physical units**. So it's physics-neutral — only the critic's
yardstick is rescaled. Modes: a number (fixed factor), or `"auto"` = data-derived `1/std(Δx)`
(scales increments to ~unit std, equation-agnostic; ≈33 for OU).

**Result — OU, single-step critic + `increment_scale: 10`, 1000 ep (best ep 500).** Best OU run to
date; directly fixes the long-standing diffusion channel:

| metric | prior (single-step, no scale) | ×10 increment scale |
|---|---|---|
| effective diffusion b̂(x) | **tilts UP** 0.30 → 0.32 (+6%) | **flat 0.30, slight DOWN to 0.28 at x=2** — matches paper Fig. 6 |
| rollout std | ~8% **over** (0.225 vs 0.213) | ~4% slight **under** (~0.205) — tracks the curve |
| covariance spectra | ~15% over | excellent match |
| best increment std rel err | ~2e-4 | **6.5e-5** (~10× tighter); dy std pinned 0.0300/0.0300 per-state |
| mean / drift / conditional | excellent | excellent (unchanged) |

**Mechanism confirmed exactly as predicted.** The upward tilt — untouched by critic size (Part 6),
budget (Part 13), or optimizer/Muon (Part 21) — vanished the instant the critic could *see* the
variance signal. It was never architecture, duration, or optimizer: it was the **signal-to-floor
ratio the critic operates at.** This closes the Part-20 diffusion-tilt investigation.

**Notes.**
- Now *slightly* under-shoots std (~4%) — ×10 may marginally over-correct; `increment_scale: "auto"`
  or ~×5 could re-center it. Small; the result is already strong.
- Best checkpoint was **epoch 500**; 500–1000 just oscillated ⇒ ~500–600 epochs suffice (not 1000).
- The residual downward drift of b̂ past x≈1.5 is the data-free extrapolation region (training max
  1.47) — now under-estimating instead of over, same direction as the paper's Fig. 6.
- The mean's slight tail overshoot (1.22 vs 1.205) is the *separate* D̃ Phase-1 fixed-point residual,
  not a diffusion issue.
- **Not yet propagated** to GBM/NLD/TrigDrift/DoubleWell configs (they lack the `increment_scale`
  key ⇒ default 1 = off). Candidate to try there (their diffusion misses are the same class).

---

## Part 23 — Scaling propagated; OU crisp figures; NLD diffusion isolated to density-domination (not eval grid)

**Propagation.** `increment_scale` + `single_step_critic` surfaced in ALL five SDE example configs
(default `increment_scale: "auto"` = data-derived 1/std(dx); `single_step_critic` off by default),
and the single-step critic-sizing guard (`if single_step: sequence_length=1`) added to GBM.py /
NonlinearDiffusion.py / TrigDrift.py / DoubleWell.py (only OU.py had it). So single-step is now
available everywhere.

**GBM (auto scaling, sequence critic).** Diffusion tracks σx cleanly, mean/std good — auto-scaling
neutral-to-fine (GBM's diffusion was already fine; its residual is the mean/Jensen, Part 16).

**OU crisp figures (professor's request).** Effective drift/diffusion MC bumped 20k → **1e6** draws,
grid 100 → **200**, chunked accumulation (running sum/sum-sq) so the 1e6 × K=16 centering draws don't
OOM. Note: hard-centering makes this ~3e9 gen-evals/plot on CPU (slow, ~min); an exact 16× speedup is
available (drift is deterministic since E[S|x]=0; diffusion = raw-std × √(1−1/K)) — noted, not yet done.

**NLD — the "diffusion shape not learned" fully diagnosed (two layers):**
1. **Eval-grid artifact (partial red herring).** We plotted b̂ on [−1,1]; the paper's Fig. 15 uses
   **[−0.6, 0.6]** = the data support. Fixed grid to [−0.6,0.6], ylim [0.34,0.52]. BUT:
2. **Real miss = density domination.** Even on [−0.6,0.6], b̂ is **flat ~0.48** vs the bell (0.50→0.35).
   Flat 0.48 ≈ the **density-weighted average** of the bell. Histogram: states 21% at x≈0, <1% each at
   |x|≈0.5, **5 samples at x=±1** (95% within ±0.38, strong −5x reversion). The WGAN loss is a sum over
   samples ⇒ dominated ~20:1 by the dense core ⇒ generator outputs the core variance everywhere. Even
   single-step gets ~6 edge-samples/batch — too weak. This is Part 14's diagnosis, now nailed.
   - **MMD-off test: FAILED.** Turned `mmd_lambda: 0` (hypothesis: MMD flattens by matching the pooled
     marginal). Still flat AND destabilised training (generator loss diverged). MMD ruled out as cause;
     reverted to 1.0 (it's a stabiliser).
   - Same class as OU's residual tilt (data-free/sparse extrapolation): OU b̂ correct in-data, tilts only
     at x>1.47 (data max); NLD b̂ correct at the dense core, flat at the sparse edges.

**State of play.** OU essentially done (increment_scale fixed in-data; residual = data-free tail,
consistent with the paper's own [−0.6,0.6]/[0.7,2.0] grid choices). The open problem is the
**state-dependent diffusion SHAPE in sparse regions** (NLD bell, and by extension TrigDrift/DoubleWell)
— an upstream *sample-weighting/coverage* problem, not a critic/optimizer/budget one.

**Literature (searched).** The weighting idea is a known method — **DenseWeight/DenseLoss** (Steininger
et al. 2021): KDE density, 1/ρ weighting with a single α to soften it. Its risk is documented —
**Byrd & Lipton (ICML'19)**: importance weighting's effect *decays to nothing* over training in
over-parameterised nets, + variance inflation from large weights on rare samples. The disease itself is
**heteroscedastic variance collapse** (Seitzer et al. ICLR'22 "Pitfalls…", fix = β-NLL, a *variance*-based
reweighting; and neural-SDE folklore: diffusion → 0 under NLL). So density weighting is applicable but
its benefit is conditional and can vanish/invert. Candidate fixes below (Part 24 when tried).

---

## Part 24 — The core diagnosis (marginal vs conditional), Fix 1 & Fix 2 built, decoupled "safe Fix 2" test

### The decisive measurement (why the diffusion SHAPE never learns)
Measured directly on NLD data: how much does the **pooled** increment distribution move between the
true bell b(x) and a completely FLAT variance? **W₁ = 1.3e-4**, sitting on the **sampling-noise floor
of 1.1e-4**. So the shape signal is essentially absent from the quantity the WGAN optimises.
- Root cause restated cleanly: **b(x) is a property of the CONDITIONAL p(Δx|x); the WGAN critic + MMD
  both match the MARGINAL/pooled p(Δx), a sample-mean dominated ~20:1 by the dense core.** Getting the
  bell vs flat right perturbs the marginal below the GP/sampling floor ⇒ no gradient toward the shape.
- This UNIFIES every prior negative: more epochs, MMD on/off, stronger critic, Muon, difficulty-weighting
  — all act on/through the same blind marginal objective. Moments/std stay great because they only need
  the pooled variance (flat model already matches it).
- Contrast GBM (b=σx): its variance contrast is huge, so the shape lives in the dense/large-variance
  region and lifts the marginal above the floor ⇒ learnable. Pooled loss captures shape ONLY when the
  structure sits where the data/variance is large.

### Paper cross-check (important negatives)
- Data protocol is **identical** to ours (paper §5: 100 EM steps, random L=40 window, N=10k) — coverage
  is not the differentiator.
- Paper trains **100k epochs**; our vanilla-at-scale run was tried (Part 13: un-centered, MMD off, decay,
  70k) and **diverged** (mean→1.57, generator loss 0→12). So "just run the paper's vanilla config at
  budget" is falsified for us — centering is required for convergence, not optional.

### Fix 1 — state-density (1/ρ) importance weighting  [BUILT, toggle: `density_weight_alpha`]
`sde.py`: weight critic + generator + GP samples by ρ(x0)^(-α) (histogram KDE over x0, α-softened,
clipped). Data-derived ⇒ equation-agnostic. Default 0 = off. Rationale: re-level the sparse edges.
Caveat (why it may fizzle): it reweights the FEW edge samples we have but can't create signal the critic
is structurally blind to — it's downstream of the same marginal objective. Not yet run to conclusion.

### Fix 2 — heteroscedastic generator  S(x,z)=σ_θ(x)·z  [BUILT, toggles: `heteroscedastic`, `hetero_var_lambda`]
`gan.py`: generator becomes an explicit state-dependent scale (softplus MLP) × standard noise
(requires latent_dim==problem_dim). `sde.py`: `_hetero_var_loss` regresses σ_θ(x)² onto the real
residual² — variance by REGRESSION, not adversarially, so it sees the sparse-edge signal (validated in
numpy: recovers the NLD bell to ~few % from the same data the WGAN read as flat; edge/center 0.67 vs true
0.70 vs WGAN ~1.0). Structural zero-mean E[S|x]=0. It's the form of any Itô SDE ⇒ one config should fit
all (OU: σ≈const; NLD: σ bows down).

**Fix 2 result: helps NLD, FAILS terribly on OU.** On OU the full-model mean collapsed 1.22→0.78 (while
Phase-1 D̃ alone reverts correctly to 1.22), effective drift ~3× too steep, std ~half, best selection
score 0.19 (vs ~0.05 for good OU). Diagnosis:
1. OU doesn't NEED Fix 2 — its b(x) is flat; its delicate quantity is the MEAN/drift in the sparse high-x
   tail. Fix 2 aims at a problem OU doesn't have.
2. The collapse can't come from clean σ(x)·z (exactly zero-mean) ⇒ it's an interaction: (a) noise-induced
   drift in the extrapolation tail (x>1.47) where σ_θ runs free and tilts up, the ½σ²D̃'' term biasing the
   ensemble mean down over the rollout; (b) the σ(x)·z straitjacket destabilising training while
   adversarial+MMD+centering still fought — a genuinely bad run, per the selection score.
- **Takeaway: Fix 2 helps when the DIFFUSION shape is the bottleneck (NLD/Trig/DoubleWell), hurts when
  the MEAN is the delicate part (OU). Not yet uniform-safe** — it can regress a case it should leave alone.

### Decoupled "safe Fix 2" (GAN-off) — considered and REJECTED on principle
Idea was: for Gaussian-noise SDEs the model is fully specified by D̃ (MSE, mean) + σ_θ (regression,
variance), so drop the GAN entirely (`adv_lambda=0`). This would make OU safe (nothing pushes the mean)
and still fit NLD's bell. **Rejected:** hard-coding `S=σ(x)·z` (Gaussian z) bakes in a Gaussian noise
model, which defeats the entire purpose of the WGAN approach — the method must make NO distributional
assumption (it has to extend to the §5.3 non-Gaussian cases). Just because our current examples are
Gaussian does not license assuming Gaussian in the model. So this path is a dead end.

### DECISION: Fix 1, Fix 2, and adv_lambda all REVERTED — code back to the Part-22/23 stable form (HEAD)
`due/models/sde.py` and `due/networks/gan.py` restored to commit `c86c7b67` (concat-MLP generator, plain
WGAN-GP; keeps increment_scale, single_step_critic, MMD, centering, Muon). All 5 example configs restored
to their committed stable values (OU = Part-22 best: single_step=true, increment_scale=10, mmd=1.0,
center=true, epochs=1000). All experimental keys removed (`heteroscedastic`, `hetero_var_lambda`,
`density_weight_alpha`, `adv_lambda`). `config_hetero.yaml` files abandoned (tombstoned; delete manually).
Difficulty-weighting (earlier option B) had already been removed the same way.

**What survives as RESULTS (the value of this round), even though the code was reverted:**
1. The **diagnosis** above: b(x) lives in the CONDITIONAL, the WGAN/MMD optimise the MARGINAL, and the
   bell-vs-flat signal in the marginal sits at the sampling-noise floor (W₁≈1.3e-4 vs floor 1.1e-4). This
   is measured, not hand-waved, and it unifies every failed attempt (epochs, MMD, critic size, Muon,
   difficulty-weighting, density-weighting all act on/through the blind marginal).
2. Fix 2 (heteroscedastic regression) **would** recover the shape (numpy-validated) but only by assuming
   location-scale/Gaussian structure — a modelling assumption the project's "assume nothing about the
   noise" principle rejects. Documented as: *the shape IS recoverable by regression, but not without
   assuming the noise form.*
3. The remaining honest options for the diffusion-shape-in-sparse-regions problem, WITHOUT assuming the
   noise form: (a) **density weighting 1/ρ(x)** — reweights DATA, not the noise model, keeps the concat
   GAN fully general (built this round, reverted, not yet run to conclusion — re-add if pursued);
   (b) **accept it as a characterised limitation** of the distribution-free method (arguably the strongest
   scientific framing — the marginal-vs-conditional diagnosis is the contribution).

**State of play (unchanged, clean baseline):** OU + GBM reproduced and stable. NLD/TrigDrift/DoubleWell
diffusion SHAPE in sparse regions remains open, now precisely diagnosed. No Gaussian assumption in the code.

---

## Part 25 — Paper-faithful reset + 100k OU run: diffusion SOLVED the paper's way; mean fails un-centered

> **⚠ CORRECTED BY PARTS 27-28 — read those first.** The headline "the paper's way" is WRONG: this run used a
> **y-only gradient penalty**, which is NOT the paper's Alg 4.1 L12 (that is grad wrt the pair (x0, y~)).
> The flat diffusion achieved here is attributable to that deviation, not to the paper's config. With the
> corrected `pair` GP the tilt returns (Part 28). Conclusions below about "increment_scale was only
> compensating for budget" are therefore NOT established.

**Reset.** All 5 configs reset to match sFML paper as closely as possible: normalization none, plain ResNet
(D̃=I+N), critic 3×20 (= generator, not 4×128), increment_scale 1 (OFF), sequence critic L=40 (single-step
OFF), MMD 0 (OFF), centering OFF, lr_decay OFF, fixed lr 5e-5, n_ct 5, GP λ=10, batch 1000, 100k epochs,
Phase-1 5000 ep. Two code changes: **gradient penalty fixed to y-only** (was penalising ∇ wrt x0 too, which
over-smooths the critic ACROSS states — the direction it must stay sharp in); dtype kept float32 (float64 is
DUE's default but only buys stability, doesn't touch the statistical floor). Left model-selection + activation.

**OU 100k result — the two channels cleanly separate:**
- **Diffusion: EXCELLENT, achieved the PAPER'S WAY.** b̂(x) flat ~0.30 across the data region (tiny up-tilt
  only at x>1.7 extrapolation). Best-ckpt (ep 23,900) increment_std_rel_err = **2.4e-4**; rollout std tracks
  0.21. This used **no increment_scale, no single-step** — just the sequence critic + y-only GP + 100k budget.
  ⇒ **the increment_scale/single-step "win" (Part 22) was compensating for insufficient BUDGET**, not
  fundamentally necessary; at the paper's 100k the sequence critic resolves the per-state variance on its own
  (the y-only GP fix likely also helped by un-smoothing the state direction).
- **Mean: FAILED (un-centered), even at 100k.** Full-model mean collapses to ~1.07 (vs 1.205). Generator loss
  drifts steadily 0 → −1.4 over 100k (WGAN score-level drift, no equilibrium); per-step fake-mean wanders
  (−0.07 → 0.007 → 0.006 across epochs). **Confirms Parts 10/13 at the full paper budget: budget does NOT fix
  the un-centered mean; centering is necessary.**
- **Covariance over-estimate** (C(0) 0.057 vs 0.045) is entirely the mean error leaking in — the variance
  itself is fine.
- **Model selection** ignored the mean (selection_mean_weight=0), so it picked a good-variance/bad-mean
  checkpoint (ep 23,900, mean_abs_err 0.049), compounding the collapse.

**Takeaway.** The two channels have different needs, now cleanly established at full budget:
VARIANCE/diffusion → paper-faithful config works at 100k with no hacks ✓ ; MEAN → un-centered fails even at
100k, centering required ✗ . **Next: paper-faithful diffusion (sequence critic + y-only GP + budget) + centering
for the mean** = the combination that should reproduce OU fully, with centering the one justified deviation.
This also reframes the shape problem: re-run NLD/Trig/DoubleWell paper-faithful at 100k (the diffusion channel
now demonstrably works the paper's way) before concluding the shape is unrecoverable.

---

## Part 26 — Paper-faithful 100k OU, dual eval (best + final): final DIVERGES; add-ons validated as real improvements

Config: paper-faithful (un-centered, sequence critic, 3×20, no increment_scale/MMD/centering, y-only GP,
100k) + two tweaks: `selection_mean_weight=1.0` and `eval_model: both` (OU.py now wraps the eval in
`run_eval(save_path)` and loops over generator_best + generator_final → `gan_model/eval_best`, `eval_final`).

- **eval_final (paper's recipe = last-epoch model): DIVERGES.** Rollout mean blows up to ~4.5, std ~9.6.
  Effective drift turns POSITIVE past x≈1.7 ⇒ the un-centered generator has an **extrapolation instability**
  at high x by 100k, and OU's test (x0=1.5) lives beyond the data max (1.47), so trajectories run away.
  Per-step training stats look fine (fake_mean 0.0062 ≈ real 0.0067) because they sit at low x and MISS the
  high-x instability. Generator-loss drift to −1.4 = this instability. **Corrects Part 25's optimism: the
  final model is NOT good; the paper's "train to 100k, use final" recipe yields an unusable model for us.**
- **eval_best (mean-aware selection, epoch 11,600): ok-ish.** Flat 0.30 diffusion, std tracks 0.21, mean
  ~1.17 (slight undershoot). std_rel_err 4.2e-3, mean_abs_err 1.8e-4. Mean-selection worked (caught an early
  good-mean checkpoint, not a drifted late one).
- **Worse than the add-on config (Part 22):** add-on best (single-step + increment_scale=10 + centering) had
  std_rel_err **6.5e-5 (~65× tighter)** and mean ~1.22 (closer than 1.17), and was stable.

**⚠ CAVEAT ADDED IN PART 27:** these two 100k runs used a **non-paper gradient penalty** (y-only), so their
comparison against the paper is compromised. The "vanilla final diverges" conclusion may be partly an artifact.

**Conclusion — the add-ons are GENUINE improvements, not just budget compensation.** Even at the paper's full
100k: increment_scale lifts the variance signal off the GP floor (~65× tighter variance); centering pins the
mean structurally (no drift/blowup); the vanilla final model is outright unstable. **The paper's vanilla
recipe does not reproduce cleanly in our hands (final diverges, best is looser); our modified pipeline is the
working reproduction.** Part-22 add-on config remains the best OU result. Model-selection is also non-optional
for us: without it the un-centered final blows up.

---

## Part 27 — Line-by-line audit against Algorithm 4.1: ONE REAL BUG FOUND (gradient penalty), now fixed

Audited every line of the paper's Algorithm 4.1 + eqs 4.7-4.21 against the code.

### BUG (mine, introduced in the Part-25 "reset"): gradient penalty was y-only
Paper **line 12** is explicit: `P = (|| grad_{(x_0, y~)} C(x_0, y~) ||_2 - 1)^2` — the gradient is wrt the
**CONCATENATED PAIR (x0, y~)**, i.e. BOTH arguments. In Part 25 I "fixed" the GP to y-only, reasoning from
standard conditional-WGAN-GP convention instead of reading the paper's subscript. **The ORIGINAL code was
correct.** Reverted; the verbatim equation is now quoted in the `gradient_penalty` docstring so it can't drift.
⇒ **The Part 25/26 100k runs used a non-paper GP** — their paper comparison is compromised (caveat added there).

### Verified CORRECT, line by line
| Paper | Code |
|---|---|
| L6 `y_{j+1} = D(x_j) - x_j + S(x_j,z)` | `increment = det_next - current_state + stochastic_increment` |
| L7 `x_{j+1} = x_j + y_{j+1}` | `next_state = current_state + increment` |
| L5 fresh `z~N(0,I)` EACH step j | `_stochastic_increment` draws fresh z per step |
| L4-8 recurrent rollout from real x0 using its OWN states | `x_window` fed forward in `generate_increment_sequence` |
| L11 `y~ = eps*y + (1-eps)*y_hat`, eps~U(0,1) PER-SAMPLE | `alpha_shape=[batch]+[1]*(ndim-1)` |
| L13 `L = C(x0,y_hat) - C(x0,y) + lam*P` | `loss_C = score_fake - score_real + gp_lambda*gp` |
| L15 critic Adam every batch, mean over B | `opt_C.step()` per batch, `.mean()` |
| L16 generator every n_ct | `if critic_steps % n_critic == 0` |
| L18 `L_S = -C(x0,y_hat)` | `loss_G = -score_fake_for_g` |
| Eq 4.10 `D = I + N` | `resnet.forward: mlp(x) + x[...,-output_dim:]` |
| Eq 4.7-4.9 multi-step rollout MSE for D | `ode.py` rolls `multi_steps`, MSE over trajectory |
| Eq 4.21 `x_{n+1} = D(x_n) + S(x_n,z_n)` | eval `x = det_net(x) + r_fake` |
| §5 n_ct=5, betas (0.5,0.999), lr 5e-5, lam=10, 3x20 all nets, 100k ep | configs match |
| §5 data: 100 EM steps, random L=40 window, N=10k | `generate_data.py` matches |

### Remaining deviations — all minor or genuine paper ambiguities
1. **Fake regenerated for the generator step.** Paper generates y_hat ONCE per batch (L4-8) and reuses it for
   both critic (L13) and generator (L18); we regenerate with fresh z (we detach for the critic). Statistically
   equivalent, different z draw.
2. **Activation GELU** — paper says only "fully connected feedforward DNN". Unspecified ⇒ ambiguity.
3. **Batch size 1000** — paper lists B as a parameter but never prints its value. Our assumption (n_B=10).
4. **Paper L10 typo**: says sample `n_B` numbers but indexes k=1..B; we sample B (per-sample) — only sensible read.
5. **Model selection / checkpointing** — ours, not in the paper (paper uses the FINAL model). Now explicit and
   comparable via `eval_model: both` (writes `gan_model/eval_best` + `eval_final`).
6. **Phase-1 epochs 5000** — paper states this only for the 2D case; 1D unspecified.
7. **float32, fixed seed** — unspecified in the paper.

**Status: the algorithm is now a faithful implementation of Algorithm 4.1.** Any 100k paper-comparison runs
should be REDONE with the corrected (x0,y) gradient penalty before drawing conclusions about the paper's recipe.

---

## Part 28 — GP mode is a REAL lever on the diffusion tilt (paper's `pair` GP is worse for us)

Re-ran paper-faithful OU 100k with the **corrected (paper-faithful) gradient penalty** — grad wrt the
concatenated pair (x0, y~), Alg 4.1 line 12, verified verbatim twice.

| GP mode | effective diffusion b(x) | rollout |
|---|---|---|
| `y_only` (Part 26, my accidental deviation) | **FLAT ~0.30** (correct) | best ok-ish (mean ~1.17, std 0.21) |
| `pair` (PAPER, this run) | **TILTS UP 0.30 → 0.33** | best: mean rises to ~1.47, std blows to 1.4; final diverges (mean 6, std 17) |

**Mechanism (now evidenced, not just hypothesised).** In `pair` mode the penalty includes grad wrt x0, which
forces the critic to be **smooth ACROSS states** — exactly the direction it must stay SHARP in to witness
state-dependent variance. Result: the critic under-resolves per-state variance and b̂(x) tilts. `y_only`
removes that constraint and b̂(x) comes out flat. Note x0 is NOT interpolated in either mode (only y~ is), so
`pair` penalises the x0-gradient AT the real x0.
- This is the SAME failure channel `increment_scale` addresses (Part 22 tilt) — two independent levers on the
  critic's ability to resolve per-state variance.
- **Now a toggle:** `gp_mode: "pair"` (default, paper-faithful) | `"y_only"` (documented deviation). Added to
  all 5 configs + summary print.

**Caveat on the mean comparison.** Both runs are UN-CENTERED, and the un-centered mean is a structurally
ill-conditioned direction that lands somewhere different every run (documented Parts 10/13: 0.245, 0.82, 1.0,
1.57, 1.07, now 1.47). So the mean difference between these two runs CANNOT be cleanly attributed to gp_mode;
only the diffusion comparison is systematic. What IS robust: **eval_final diverges in BOTH GP modes** ⇒ the
paper's "use the final model" recipe fails for us regardless of GP mode (Part 26 conclusion survives the fix).

**Net:** the audit's correction stands (we are now paper-faithful by default), and it produced a genuine
finding — the paper's own GP formulation is a contributor to the diffusion tilt we have been chasing since
Part 20. Faithful ≠ best: `pair` is what the paper says, `y_only` is what works.

---

## Part 29 — Consolidated state of play (after the paper-faithful reset + audit)

### Code state (`due/models/sde.py`, `due/networks/gan.py`) — audited faithful to Algorithm 4.1
Verified line-by-line in Part 27. Generator = concat-MLP `S(concat(x,z))` (no structural noise assumption).
All experimental branches removed and NOT present: heteroscedastic `sigma(x)*z`, density weighting,
difficulty weighting, `adv_lambda`. Toggles that remain (all default to paper behaviour unless noted):

| toggle | default | paper? | effect |
|---|---|---|---|
| `gp_mode` | `"pair"` | ✔ paper (Alg 4.1 L12) | `"y_only"` = deviation; gives FLAT b(x) (Part 28) |
| `increment_scale` | `1` | ✔ paper (off) | `10`/`auto` lifts variance signal off GP floor (Part 22) |
| `single_step_critic` | `false` | ✔ paper (sequence L=40) | `true` = (x_t,dx_t) pairs |
| `mmd_lambda` | `0.0` | ✔ paper (none) | CR-GAN MMD stabiliser |
| `center_generator` | `false` | ✔ paper (emergent) | hard zero-mean S; needed for the mean (Parts 10/13) |
| `lr_decay` | `false` | ✔ paper (fixed lr) | cosine decay |
| `selection_mean_weight` | `1.0` | ✘ ours | selection ignoring the mean picks bad-mean ckpts (Part 26) |
| `eval_model` | `"both"` | ✘ ours | evals `generator_best` AND `generator_final` → `gan_model/eval_best`, `eval_final` |
| `det_arch` | `"resnet"` | ✔ paper (eq 4.10) | `gated_resnet` = ours |
| `optimizer` | `"adam"` | ✔ paper | `muon` = ours (no effect, Part 21) |

Configs (all 5): dtype single, norm none, ResNet, critic 3×20, n_ct 5, lr 5e-5, betas (0.5,0.999), GP λ=10,
batch 1000, Phase-1 5000 ep multi-step, Phase-2 100k ep. `OU.py` now wraps eval in `run_eval(save_path)` and
loops over both model tags.

### The two channels (the organising picture)
1. **MEAN / drift.** Un-centered fails at ANY budget — lands somewhere different every run (0.245, 0.82, 1.0,
   1.57, 1.07, 1.47) and `eval_final` diverges outright (mean→6, std→17) in BOTH gp_modes. The paper leaves
   zero-mean emergent (Remark 4.1); for us it is not emergent. **Centering (or at minimum mean-aware model
   selection) is required.** This is the single most robust negative result in the project.
2. **VARIANCE / diffusion shape.** Governed by whether the critic can RESOLVE per-state variance. Two
   independent levers found: `increment_scale` (lifts signal off the GP/noise floor, Part 22) and `gp_mode`
   (`y_only` stops the GP smoothing the critic across states, Part 28). The paper's own `pair` GP is a
   CONTRIBUTOR to the tilt. Underlying limit (Part 24, measured): b(x) is a property of the CONDITIONAL
   p(dx|x) while the WGAN/MMD objective matches the MARGINAL — bell-vs-flat separation in the marginal is
   W1≈1.3e-4 against a sampling floor of 1.1e-4.

### What is settled
- Paper-faithful implementation verified line-by-line; one real bug (GP) found and fixed.
- OU + GBM reproduce with our modified pipeline; OU best-ever = Part 22 (std_rel_err 6.5e-5).
- The paper's vanilla recipe does NOT reproduce for us: `final` model diverges regardless of gp_mode; the
  3×20 critic under-witnesses variance; un-centered mean never homes.
- Assumption-free fixes for the diffusion SHAPE are exhausted: budget, MMD, critic size, Muon,
  difficulty weighting, density weighting (1/rho, 43% of critic loss moved to the edges) — all fail.
  Only a noise-form assumption (heteroscedastic sigma(x)) recovers the shape, and that was rejected on
  principle (defeats the distribution-free purpose of the GAN).

### Open / next
1. **Re-run OU 100k with `gp_mode: "y_only"`** to confirm the flat-b(x) result under the corrected code
   (Part 26's run predates the toggle and conflates the deviation with the fix).
2. **Isolate gp_mode cleanly** by running WITH centering, so the chaotic un-centered mean stops confounding
   the comparison (only the diffusion channel is currently attributable).
3. NLD/TrigDrift/DoubleWell diffusion shape: re-test at 100k with `gp_mode: y_only` + `increment_scale`
   before treating the Part-24 "unrecoverable" verdict as final.
4. Optional faithfulness item: paper reuses ONE fake rollout per batch for both critic and generator (we
   regenerate with fresh z) — Part 27 deviation #1, judged harmless.

---

## Part 30 — The OU mean is NOT recoverable beyond ~±1.2%: the fixed point is ill-conditioned (measured)

**Question:** with centering ON the full-model mean equals D̃'s fixed point, and we keep landing 1.17-1.22 vs
true 1.205. Is that a model failure, or a limit of the data?

**Measurement (numpy, on `OU_train.mat`, no training needed).** For EM-OU the one-step conditional mean is
EXACTLY linear: `E[x_{n+1}|x_n] = (1-θΔ)x_n + θμΔ = 0.99x + 0.012`, fixed point `c/(1-a) = 1.2`.
- **OLS on all 400,000 pairs:** a=0.990096, c=0.011962 → **fixed point 1.2078** (not 1.2000, with 400k samples).
- **Conditioning:** fp = c/(1-a) is brutally ill-conditioned in the slope —
  `a` error +1e-4 → fp +1.0% ; +3e-4 → +3.1% ; +1e-3 → **+11.1%**.
  A 0.1% error in the learned per-step slope moves the fixed point by 11%.
- **Bootstrap over trajectories (200×):** fixed point = **1.2063 ± 0.0145**, 95% CI **[1.179, 1.234]**.

**Implication — the OU mean is already at the data's resolution limit.**
Our learned D̃ fixed points: **resnet/5000ep → 1.183**, **gated_resnet/250ep → 1.22**. **BOTH LIE INSIDE the
95% CI [1.179, 1.234].** So the residual "poor mean" (1.17-1.22 vs 1.205, ~1-3%) is **NOT a model failure —
it is the statistical uncertainty this dataset supports.** Chasing the mean below ~±1.2% is chasing noise.
- Why: the training data lives at LOW x (median 0.52, 99th pct 1.06, max 1.47) while the fixed point is at
  1.2 — reading it off requires extrapolating a near-unit slope into a sparse region, and the fixed point
  amplifies slope error ~100×. Same coverage story as the diffusion, in the mean channel.
- Consistent with Part 9's old puzzle: single-step **OLS** gave 1.1825 — that was never a network failure
  either, it is where this data's linear conditional mean actually sits.

**This cleanly separates two things previously conflated:**
| regime | mean lands | verdict |
|---|---|---|
| **Un-centered** (paper's Remark 4.1) | 0.245 / 0.82 / 1.0 / 1.07 / 1.47 / 1.57, `final` → 6.0 | **REAL failure** — far outside CI; GAN instability |
| **Centered** (ours) | 1.17-1.22 (= D̃'s fixed point) | **NOT a failure** — inside CI, at data resolution |

⇒ **OU's mean channel is CLOSED:** centering is required (un-centered is genuinely unstable), and with
centering the mean is as accurate as the data permits. The remaining OU question is purely the diffusion
channel (gp_mode / increment_scale), which the un-centered mean chaos has been confounding.

**Next-step consequence:** test `gp_mode` on **NLD**, not OU — NLD's mean reverts to 0 with strong (θ=5)
reversion in the DENSE part of its data, so its mean channel is benign and the diffusion effect can be read
cleanly without the ill-conditioned-mean confound.

---

## Part 31 — NLD at 100k, paper-faithful: DECISIVE. Pooled variance perfect, conditional shape qualitatively wrong

Ran NLD with the fully-audited paper-faithful config (Alg 4.1 exact, `gp_mode: pair`, sequence critic L=40,
3×20 critic, no increment_scale / MMD / centering, fixed lr, 100k epochs). Best ckpt ep 22,500.

**Everything EXCEPT the diffusion shape is good:**
- `increment_std_rel_error` = **1.7e-4** — the POOLED increment std is matched essentially perfectly.
- Phase-1 D̃ rollout tracks the analytical mean; effective DRIFT a(x)=−μx recovered across [−0.6,0.6].
- Rollout std 0.152 vs 0.157 GT (~3% under); conditional distribution at x=−0.3 matches well.
- Training stable to 100k (no divergence, unlike OU un-centered).

**The diffusion shape is not merely under-resolved — it is QUALITATIVELY WRONG:**
- Truth `b(x)=σe^{−x²}` is **EVEN**: 0.349 at −0.6, 0.500 at 0, 0.349 at +0.6.
- Learned b̂(x) is **MONOTONE DECREASING**: 0.510 at −0.6 → 0.456 at +0.6. No peak, no symmetry.
- **The model breaks a symmetry that IS present in the data.** Verified: the SDE is equivariant under x→−x
  and IC U(−1,1) is symmetric, and the empirical per-state stds confirm it (x≈−0.55: 0.0369, x≈+0.55: 0.0374;
  x≈−0.45: 0.0410, x≈+0.45: 0.0408). The learned asymmetry (−12%) is ~an order of magnitude larger than the
  data's (+1%), and in the OPPOSITE direction.

**Why this is decisive.** If the objective exerted ANY constraint on the conditional shape, the learned b̂
would at minimum inherit the data's symmetry. It does not: it lands on an essentially arbitrary near-flat
function whose *pooled* average matches (to 1.7e-4). This is exactly the Part-24 measurement realised —
the WGAN objective constrains the MARGINAL and leaves the CONDITIONAL shape unidentified — now demonstrated
at the paper's full budget with the paper's exact algorithm, on the paper's own example.

**The fix space is now exhausted (all tested, all fail to recover the shape):**
budget (5k, 70k, 100k) · critic size (3×20, 4×128) · sequence vs single-step critic · `gp_mode` pair vs
y_only · `increment_scale` (1, 10, auto) · MMD on/off · centering on/off · Muon vs Adam · difficulty
weighting · density weighting 1/ρ (43% of critic loss moved onto the sparse edges) · normalization variants.
The ONLY thing that recovers the shape is assuming the noise form (heteroscedastic S=σ_θ(x)·z fitted by
regression, Part 24) — rejected on principle, as it defeats the distribution-free purpose of the GAN.

**CONCLUSION — the project's headline result.** Within the sFML/WGAN framework as published, on the paper's
own data, the *state-dependent diffusion shape in sparsely-visited regions is not identifiable*: the training
objective is a marginal (pooled) two-sample distance, while b(x) is a property of the conditional p(dx|x),
and the discriminating signal there sits at the sampling-noise floor (W1≈1.3e-4 vs floor 1.1e-4, Part 24).
Moments, drift, pooled variance and one-step conditionals all reproduce; the effective-diffusion FIGURE for
the state-dependent-noise examples (NLD §5.2.1, TrigDrift §5.2.2) does not. Recovering it requires either
more data in the sparse regions or a structural assumption on the noise.

---

## Part 32 — DEFINITIVE paper-faithful OU 100k (post-audit, all fixes): `final` DIVERGES, `best` mean wrong

First run with the **fully audited, fully paper-faithful** implementation: `reuse_fake: true` (one fake
rollout per batch, Alg 4.1 L4-8 reused by L13 + L18), `gp_mode: pair` (Alg 4.1 L12), 3×20 critic,
no increment_scale / MMD / centering, fixed lr, 100k epochs, N_SAMPLES=100,000, plus the corrected Fig-8
diagnostic (covariance-matrix eigenvalue spectra, not lag-covariance).

**Result — qualitatively IDENTICAL to Parts 25/26.**
- **`eval_final` (the paper's recipe): DIVERGES.** mean → 5.4, std → 6.9, effective drift turns POSITIVE
  past x≈1.75 (runaway), covariance spectrum inflated ~3 orders at low k.
- **`eval_best` (mean-aware selection): mean = 1.12**, below D̃'s own fixed point (1.183) — the un-centered
  generator dragged it DOWN by ~0.06. std 0.19 vs 0.212. Effective diffusion 0.30 → 0.283 (mild tilt).
- **1.12 lies OUTSIDE the data's 95% CI [1.179, 1.234]** (Part 30) ⇒ a genuine failure, not a resolution limit.

**★ This LIFTS the Part-26 caveat.** Those runs used a non-paper y-only GP and regenerated the fake sequence,
so their "vanilla diverges" conclusion was compromised. With BOTH of those now corrected to the paper's exact
prescription, **the divergence and the wrong mean persist unchanged.** Therefore:
> The failure of the paper's un-centered design is NOT an artifact of our implementation deviations.
> It reproduces under the literal Algorithm 4.1, at the paper's full 100,000-epoch budget.
This is now the strongest, cleanest negative result in the project — no remaining implementation caveat.

**Un-centered mean across ALL runs** (true 1.205; data CI [1.179, 1.234]):
0.245 · 0.82 · 1.00 · 1.07 · **1.12** · 1.17 · 1.47 · 1.57 · 5.4 (final) · 6.0 (final)
Every value outside the CI; no two runs agree. Confirms the mean is an unconstrained direction of the
un-centered objective, exactly as diagnosed in Parts 4/10/13.

**New diagnostic works and is informative.** The corrected Fig-8 covariance-matrix spectra for `eval_best`
is an **excellent match to ground truth across ~4 decades** of eigenvalue magnitude. Note this diagnostic
centres by the empirical mean at each time, so it is INSENSITIVE to the mean offset — it isolates the
fluctuation/temporal-correlation structure. Reading: **the model gets the covariance STRUCTURE right while
getting the MEAN wrong**, which is precisely the two-channel split the project has been documenting
(variance/shape channel healthy; mean channel unconstrained). For `eval_final` the spectrum is inflated
~10³ at low k — the signature of the runaway.

**Conclusion.** The paper-faithful reproduction of OU is COMPLETE and the verdict is negative: as published,
the method does not converge for us at the paper's own budget with the paper's own algorithm. The working
reproduction requires centering (mean) and a stronger critic + increment scaling (variance) — Part II of the
report. Remaining OU work: none. Move to Config B for the working results and to the other examples.

---

## Part 33 — Fig-8 window fix VERIFIED; the discrepancy is in the PAPER's reference curve, not our model

Re-ran eval-only with the corrected Fig-8 construction (L=40 window = T=0.40, t_0 excluded, 40 components).

**The fix is verified against theory.**
- Ground-truth λ₁ (measured, from the EM test ensemble) ≈ **0.44**; analytic OU value = **0.4370**. ✔
- Spectrum decays smoothly 0.44 → ~2e-4 over 40 components, exactly as the analytic covariance
  `Cov(X_s,X_t) = (σ²/2θ)(e^{−θ|t−s|} − e^{−θ(t+s)})` predicts.
- The t_0 zero-eigenvalue cliff is gone (t_0 has Var=0 by construction; now excluded).

**Predicted-and-confirmed:** I predicted the k=1 gap would persist and equal the std deficit squared.
Measured std 0.19 vs true 0.212 ⇒ (0.19/0.212)² = **0.803**. Observed λ₁ ratio (pred/true) ≈ 0.35/0.44 =
**0.80**. ✔ The covariance spectrum is now a clean quantitative readout of the variance error.

**eval_best (ep ~11.6k):** spectrum tracks ground truth in SHAPE across 3+ decades but sits uniformly ~20%
low (the std deficit). Mean 1.12, std 0.19, diffusion 0.30→0.283, drift slightly steep.
**eval_final (ep 100k):** spectrum sits ABOVE ground truth (λ₁ ≈ 0.77 vs 0.44) and the gap widens with k —
the signature of the runaway. Mean 5.4, std 6.9, drift turns positive past x≈1.65.

### ★ The important new observation: OUR ground truth ≠ THE PAPER'S ground truth
Our reference curve is computed from the true EM ensemble and **matches the analytic OU covariance exactly**
(0.44 vs 0.4370). The paper's Fig-8 "Ground Truth" curve instead drops ~400× from λ₁≈4e-1 to λ₂≈1e-3 and
then goes flat. **No construction we can derive from the OU process produces that shape** — states from
deterministic x0 (λ₁/λ₂ = 7), from random x0 (7), stationary (13), or pure increments (flat, no dominant
mode). ⇒ **The discrepancy is in their REFERENCE curve, not in our model.** Whatever Fig. 8 plots, it is not
the per-time-centred covariance matrix of the OU state sequence. Unresolved; a precise question for the authors.

---

## Part 34 — ★ CORRECTION to Parts 26/32/33 (stale checkpoint file) + quantitative confirmation of the ÷Δ mechanism

**Data-handling error (mine).** The `best_checkpoint.txt` I read for Parts 26/32/33 was a STALE upload
(epoch 11,600). The actual best checkpoint of the paper-faithful 100k run is **epoch 26,200**. Corrected
numbers:

| metric | logged (stale, ep 11,600) | ACTUAL (ep 26,200) |
|---|---|---|
| selection_score | 6.69e-3 | **4.04e-3** |
| increment_std_rel_error | 4.21e-3 | **6.45e-4**  (~6.5× better) |
| increment_mean_abs_error | 1.77e-4 | 3.84e-4 |
| real / fake increment std | 0.030027 / 0.030153 | **0.0300274 / 0.0300467** |

⇒ Any statement in Parts 26/32/33 quoting "std_rel_err 4.2e-3" or "best ckpt ~11.6k" is WRONG; use the above.

### The correction SHARPENS the diagnosis: the variance channel is excellent, the failure is entirely mean/drift
With std_rel_err = **6.4e-4**, the generator's **per-step conditional variance is essentially exact** (fake
0.0300467 vs real 0.0300274). So the residual errors are NOT a variance-learning failure:
- **rollout std 0.19 vs 0.212 (−10%)** is caused by the DRIFT being too steep (over-reversion). The plots show
  b̂ below the analytic line, i.e. an effective θ larger than 1; the stationary std σ/√(2θ_eff) then falls.
  A variance deficit produced by a drift error, not by the noise model.
- **rollout mean 1.12** is the ÷Δ amplification, now confirmed QUANTITATIVELY:

```
per-step generator mean error  δ = 3.84e-4      (= 1.3% of the increment std 0.030)
effective drift bias           δ/Δ = 0.0384
fixed-point shift              δ/(θΔ) = 0.0384
predicted mean = D̃ fixed pt − shift = 1.183 − 0.038 = 1.145
OBSERVED mean  = 1.12                                        ✔ (within the un-centered run-to-run scatter)
```

**This is the Part-4 mechanism measured end-to-end.** A per-step mean error amounting to 1.3% of the noise
amplitude — utterly invisible in the pooled statistics, and far below anything an adversarial objective can
resolve — produces an 0.08 error in the long-run fixed point. The 1/Δ factor turns a negligible per-step bias
into a first-order error in the quantity of interest. **This is the single cleanest quantitative statement of
why the un-centered design cannot work, and why the mean must be routed through regression (D̃) rather than
through the GAN.**

### Revised summary of the paper-faithful OU run
- Variance / conditional-distribution channel: **excellent** (per-step std to 6e-4; covariance spectrum matches
  ground truth in shape across 3+ decades).
- Mean / drift channel: **fails** (mean 1.12, outside the data CI [1.179, 1.234]); `final` model diverges (5.4).
- The two-channel split is now quantitative, not qualitative.

**Note to self:** verify uploaded artefacts by timestamp/content before quoting them; stale files silently
corrupted three log entries.

---

## Part 35 — ★ CHECKPOINT SWEEP: the run is METASTABLE, not convergent. Training longer is actively harmful.

Swept 100 of the 1000 saved checkpoints from the Algorithm 4.1 100k OU run (every 10th), rolling each
out to T=4 from x0=1.5 and recording the long-run mean/std. **No retraining — this was free.**
(`examples/OU/sweep_checkpoints.py`, output `gan_model/checkpoint_sweep.{csv,png}`.)

### Four distinct phases
| phase | epochs | rollout mean range | max std |
|---|---|---|---|
| early chaos | 0 – 9,000 | 0.10 → **6494** | 462 |
| **metastable plateau** | **9,000 – 55,000** | **1.05 – 1.79** | 2.98 |
| escape / blow-up | 56,000 – 90,000 | 2.19 – **57.6** | 32.5 |
| partial recovery | 90,000 – 100,000 | 2.19 – 11.2 | 15.6 |

True values: mean 1.2049, std 0.2116.

### What this establishes
1. **There is no convergence — there is a metastable basin.** From ~9k to ~55k the model hovers near the
   right answer, then **escapes irreversibly** at ~56k. The paper's 100k budget lands *past the escape*
   (mean 5.59, std 7.01 — a 364% mean error).
2. **Even inside the plateau the mean wanders**: range 1.05–1.79, mean |error| **9.9%**, worst 48%.
   So the basin is not a converged solution — the mean random-walks within it, exactly as the
   unconstrained-mean diagnosis (Parts 4/10/13) predicts. It is *usable*, not *correct*.
3. **Longer training is actively harmful** — the failure is not under-training, it is over-training past
   the escape. This inverts the original hypothesis (Part 3) that we were ~100× under-budget.
4. **Strong support for the budget/epochs ambiguity.** If the paper's effective number of generator
   updates lands anywhere in 9k–55k epochs (plausible given batch size B and "epochs vs iterations" are
   both unspecified), their *final* model would sit in the plateau and look fine at figure resolution —
   which is exactly the standard of agreement the paper reports.
5. **Our model selection is a poor proxy for rollout quality.** The selection score (per-step increment
   statistics) chose **ep 26,200**; the rollout-best checkpoint is **ep 11,000**. The score is computed on
   *training states* (all at low x) and therefore cannot see the high-x extrapolation instability that
   actually destroys the rollout. Honest limitation to report.

### Mechanism of the escape
The per-step training metrics at 100k are unremarkable (std_rel_err 6.8e-3), yet the rollout explodes.
The instability is purely **extrapolative**: the effective drift turns positive beyond x≈1.65, outside the
training range (max 1.47), so trajectories that wander high run away. Training statistics are blind to it
because no training state is ever out there.

### Consequence for the next experiment
Do not run another 100k job. The decisive follow-up is the **activation function** (GELU is unbounded and
asymptotically linear ⇒ extrapolated increments grow without limit; tanh saturates ⇒ they stay bounded),
and it only needs to run to ~60k to see whether the escape still occurs. Also worth testing whether the
escape epoch is seed-dependent (n=1 so far).

### Part 35 addendum — the escape is INTERMITTENT, and the VARIANCE channel goes first
Fine-grained look at epochs 38k–62k (1k resolution):

| epoch | mean err | std err |    | epoch | mean err | std err |
|---|---|---|---|---|---|---|
| 38–45k | 2–13% | 0.5–8.6% |  | 50–51k | **0.8–2.3%** | **9–11%** |
| 46k | 18% | **59%** |            | 52–54k | 22–38% | **570–1309%** |
| 47k | 48% | **688%** |           | 55k | **4.0%** | **14%** |
| 48–49k | 9–22% | 224–349% |     | 56k+ | 82%→925% | 2149%→9290% |

Two things this shows:
1. **It bursts and recovers before escaping.** Instability appears at 46–49k, *recovers* at 50–51k,
   bursts again at 52–54k, *recovers* at 55k, then escapes permanently from 56k. This is intermittent
   bursting at a stability boundary, not a monotone degradation.
2. **The variance channel destabilises first and far more violently.** At 47k the std error is **688%**
   while the mean error is 48%; at 53k std is 1309% vs mean 38%. The blow-up is driven by the
   generator's *noise amplitude* exploding in the extrapolation region, with the mean dragged along.

**Consequence — this rules out centering as the fix for the escape.** Hard-centering removes the
generator's mean degree of freedom; it does nothing to $\sigma(x)$ at high $x$. So our Part-II fix
addresses accuracy in the plateau, not the escape. The escape is an **extrapolation instability in the
variance**, which points squarely at the activation (unbounded vs saturating) as the next test.

---

## Part 36 — Activation experiment (tanh): HYPOTHESIS REFUTED. tanh is far worse than GELU.

Ran `config_tanh.yaml` (identical to the Algorithm 4.1 baseline except `activation: gelu -> tanh`,
60k epochs), then swept all 60 checkpoints as in Part 35.

**Hypothesis (Part 35):** the escape is an extrapolation instability; GELU is unbounded and
asymptotically linear so $\sigma(x)$ can grow outside the data, while tanh saturates so increments stay
bounded and the runaway becomes impossible.

**Result: the opposite.** tanh never forms a metastable plateau at all.

| | GELU (baseline) | tanh |
|---|---|---|
| checkpoints within 5% mean / 20% std | 14/100 | **0/60** |
| median \|mean error\| | ~10% (in plateau) | **86.1%** |
| max \|mean error\| | 4678% | 4280% |
| behaviour | chaos → plateau (9k–55k) → escape | **chaotic throughout**; bounces 0.87–50 with no stable window |
| best rollout | ep 11,000, mean 1.214 | ep 47,000, mean 1.175 |

**Note the training metrics do NOT show this.** tanh's best checkpoint (ep 52,100) has
`increment_std_rel_error` 1.4e-3 and `increment_mean_abs_error` 6.4e-4, comparable to GELU's best
(6.4e-4 / 3.8e-4). Per-step statistics again fail to see the rollout behaviour (cf. Part 35).

### Why tanh is a poor choice *here* (two confounds, both worth stating)
1. **The inputs are already in the saturating region.** OU runs with `normalization: none`, so the
   network sees raw $x$. Measured: median $|x|=0.53$ (tanh$'=0.77$), 99th pct $1.07$ (tanh$'=0.38$),
   **test IC $1.5$ (tanh$=0.905$, tanh$'=0.18$)**. At the test initial condition the first-layer
   gradient is ~5x smaller than at the origin, so the network has very little resolution exactly where
   the rollout lives. Saturation buys boundedness at the cost of expressivity in the region that matters.
2. **Phase 1 changed too.** `activation` is global, so $\tilde{\mathbf D}$ was also retrained with tanh.
   Final Phase-1 losses are indistinguishable (1.4419e-2 vs 1.4417e-2), so the *fit* is equally good,
   but the extrapolation behaviour of $\tilde{\mathbf D}$ beyond the data is not controlled for.

### What this does and does not establish
- **Does:** bounded activations do not cure the instability; "unbounded extrapolation of the activation"
  is not a sufficient explanation for the escape. The activation ambiguity in the paper is unlikely to
  be the missing ingredient.
- **Does not:** cleanly test the boundedness idea, because tanh is confounded with (i) saturation on
  unnormalised inputs and (ii) a retrained $\tilde{\mathbf D}$. A fairer test would normalise the state
  first, or change the activation in the generator/critic only.

**Next candidates**, in order: (a) the gradient-penalty constant $\lambda$ (never given a value in the
paper); (b) batch size $B$ (also unspecified, and it sets the effective number of generator updates);
(c) seed dependence of the escape epoch (n=1 so far).
