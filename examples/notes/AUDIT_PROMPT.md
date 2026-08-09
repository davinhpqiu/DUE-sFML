# Independent Audit Request — sFML (Chen & Xiu 2024) Reproduction

## Your task

You are auditing a Python/PyTorch reproduction of the method in the attached paper:

> Y. Chen and D. Xiu (2024), *"Learning stochastic dynamical system via flow map operator"*,
> Journal of Computational Physics 508, 112984.

**Goal:** determine, line by line, whether the code faithfully implements the paper's method —
especially **Algorithm 4.1** and **equations 4.6–4.23** — and identify every place where the
implementation deviates from, contradicts, or under-specifies the paper.

Be adversarial and skeptical. Do **not** assume the code is correct. Do **not** assume prior
audits were correct. Verify each claim against the paper text yourself. Quote the specific
paper line/equation and the specific code line for every finding.

## Background (what the method is)

The method learns an unknown stochastic differential equation `dx = a(x)dt + b(x)dW` from
trajectory data only. The one-step stochastic flow map is split into two sub-maps:

    x_{n+1} = D̃_Δ(x_n) + S̃_Δ(x_n, z),     z ~ N(0, I)

- **Phase 1**: `D̃_Δ` (deterministic sub-map) — a ResNet trained with a multi-step rollout MSE
  loss (paper eqs. 4.7–4.10).
- **Phase 2**: `S̃_Δ` (stochastic sub-map) — a generator trained adversarially with WGAN-GP
  against a critic/discriminator that scores the pair `(x_0, y_{1:L})`, where `y` are increments
  (paper Algorithm 4.1, eq. 4.19).

Evaluation recovers the "effective drift and diffusion" via eq. 4.23 and compares against the
known analytical `a(x)`, `b(x)` for each test SDE.

## Specific things to verify (do not take any of these on trust)

### A. Algorithm 4.1 — the training loop (`due/models/sde.py`)
For **each numbered line 1–23** of Algorithm 4.1, locate the corresponding code and state whether
it matches. In particular:
1. Line 6 — the increment formula. What exactly is `ŷ_{j+1}`? Check signs and which state is used.
2. Line 7 — the state update.
3. Lines 4–8 — is the fake sequence generated **recurrently**, feeding the model's **own**
   generated states forward (not the real data states)?
4. Line 5 — is fresh noise `z` drawn at **every** step `j`, or reused across steps?
5. Lines 10–11 — the interpolation for the gradient penalty. Is `ε` per-sample or per-element?
   Is `ε ~ U(0,1)`? **Which quantities are interpolated** — only `y`, or also `x_0`?
6. Line 12 — the gradient penalty. **With respect to which variable(s) is the gradient taken?**
   Read the subscript of `∇` in the paper very carefully and compare to the code.
7. Line 13 — the critic loss. Check the **sign convention** (which of real/fake is positive).
8. Lines 15/20 — Adam updates; are they applied to the correct network with the correct loss?
9. Line 16 — the generator update cadence relative to `n_ct`.
10. Line 18 — the generator loss.
11. Is the fake data generated once per batch and reused for both updates (as the pseudocode
    implies), or regenerated? Does this matter?

### B. Networks (`due/networks/gan.py`, `due/networks/fcn.py`)
12. Generator input: does it receive the state and the noise as the paper's Fig. 3 shows?
13. Critic input: does it receive `(x_0, y_{1:L})` as the paper's Fig. 3 shows?
14. `D̃_Δ = I + N_Δ` (eq. 4.10) — is the ResNet skip connection implemented correctly?
15. Layer counts/widths vs paper §5 ("3 layers, 20 nodes each", except 2D examples).
16. Are there any activation functions, output transforms, or initialisations that could bias the
    conditional mean or variance?

### C. Phase 1 (`due/models/ode.py`)
17. Is the loss the **multi-step rollout** MSE of eqs. 4.7–4.9 (roll `D̃` forward L steps from
    `x_0`, compare the whole trajectory), or a single-step loss?
18. Is `D̃` frozen during Phase 2?

### D. Data (`due/datasets/sde.py`, `due/datasets/normalizer.py`, `examples/*/generate_data.py`)
19. Does the data generation match paper §5 (Euler–Maruyama, Δ=0.01, 100 steps, a randomly chosen
    length-L=40 sub-sequence, N=10,000 trajectories)? Check the initial-condition distributions
    against the per-example sections (§5.1.1, §5.1.2, §5.2.1, §5.2.2, §5.2.3).
20. What exactly are `trainX` and `trainY`? Does the critic ultimately see **increments**
    (`y_n = x_n − x_{n−1}`) as the paper specifies, or states?
21. With `normalization: "none"`, are transform/inverse genuinely the identity?
22. Any train/test leakage or index errors in the windowing?

### E. Evaluation (`examples/OU/OU.py`, `examples/NonlinearDiffusion/NonlinearDiffusion.py`)
23. Does the rollout implement eq. 4.21 exactly?
24. Does the effective drift/diffusion computation implement **eq. 4.23** exactly?
    (`â(x) = E_z[G̃(x,z) − x]/Δ`, `b̂(x) = Std_z[G̃(x,z)]/√Δ`)
25. Are the analytical reference curves for each example correct?
    - OU §5.1.1: `dx = θ(μ−x)dt + σdW`, θ=1, μ=1.2, σ=0.3
    - Nonlinear diffusion §5.2.1: `dx = −μx dt + σe^{−x²}dW`, μ=5, σ=0.5
    Verify the analytical mean/std/covariance formulas used in the plots.
26. Is the model evaluated in a way that mirrors how it was trained (any train/eval mismatch)?

### F. Hyperparameters (`examples/*/config.yaml`)
27. Cross-check every value against paper §5: `n_ct=5`, `β₁=0.5`, `β₂=0.999`, `lr=5e-5`,
    gradient-penalty constant λ, batch size, number of epochs, network sizes.
28. Flag any parameter the paper does **not** specify (these are legitimate ambiguities, but list
    them explicitly).
29. Several config keys are toggles for features **not** in the paper. For each, state whether its
    current default value reproduces the paper's behaviour.

## Output format

Please produce:

1. **Table 1 — Algorithm 4.1 line-by-line**: paper line | code location | match? (✓ / ✗ / N/A) | note.
2. **Table 2 — Equations 4.7–4.23**: equation | code location | match? | note.
3. **Bugs / genuine mismatches**, ordered by severity. For each: the paper's requirement (quoted),
   what the code does, the likely numerical consequence, and a suggested fix.
4. **Deviations that are deliberate or ambiguous** — where the paper is silent or the code adds
   something extra.
5. **Anything suspicious you cannot resolve** from the provided material.

Do not summarise the paper back to me. Focus on discrepancies.

## After your independent audit

Only *after* completing the above, review the list below, which is the current
reproduction team's own record of known deviations. **Confirm, refute, or add to it.** Please do
not let it anchor your independent findings.

1. **Fake-data regeneration**: the paper generates the fake rollout once per batch (lines 4–8) and
   reuses it for both the critic (line 13) and generator (line 18) updates; the code regenerates it
   with fresh noise for the generator update.
2. **Activation function**: code uses GELU; the paper says only "fully connected feedforward DNN".
3. **Batch size**: code uses 1000; the paper lists `B` as a parameter but never states its value.
4. **Noise dimension Z**: code uses 1; the paper states this is a user choice (§4.3).
5. **Model selection**: the code checkpoints and can select a "best" model by a composite score;
   the paper appears to use the final model after 100,000 epochs.
6. **Phase-1 training length**: code uses 5,000 epochs for the 1D examples; the paper specifies
   5,000 only for a 2D example and is silent for 1D.
7. **Precision / seeding**: float32 with a fixed seed; the paper is silent.
8. **Paper line 10 wording**: says "sample `n_B` random numbers" while indexing `k = 1..B`;
   the code samples `B` (one per sample in the batch).
9. **Optional non-paper features** (all default-off in config): an MMD regularisation term, a
   "hard-centering" of the generator to force zero conditional mean, an increment-rescaling factor
   applied to the critic's inputs, a single-step critic variant, and a Muon optimizer option.

## Files provided

- `sFML_GAN.pdf` — the paper being reproduced (primary reference)
- `sde.py` — Phase-2 WGAN-GP trainer (**the core file — Algorithm 4.1**)
- `gan.py` — Generator and Critic network definitions
- `fcn.py` — ResNet / deterministic sub-map architectures
- `ode.py` — Phase-1 deterministic sub-map trainer
- `sde_dataset.py` (`due/datasets/sde.py`) — data loading and windowing
- `normalizer.py` — normalisation modes
- `OU_config.yaml` — hyperparameters, Ornstein–Uhlenbeck example (§5.1.1)
- `OU.py` — training driver + evaluation/figures for OU
- `OU_generate_data.py` — synthetic data generation for OU
- `NLD_config.yaml` — hyperparameters, nonlinear-diffusion example (§5.2.1)
- `NonlinearDiffusion.py` — training driver + evaluation for the nonlinear-diffusion example
- `NLD_generate_data.py` — synthetic data generation for the nonlinear-diffusion example
