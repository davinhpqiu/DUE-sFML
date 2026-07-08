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
**GBM Phase-1 mean solved by raw units + gated_resnet (Part 15).** Decisive open test: run the
full `GBM.py` in raw and check whether Phase-2 diffusion survives without YJ — if not, decouple
coordinates (D̃ raw / S̃ YJ). Then the nonlinear / double-well / non-Gaussian / 2D examples
(each stresses a different component: D̃ curvature, generator multimodality/skew, dimension).

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
