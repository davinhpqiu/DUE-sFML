# sFML implementation comparison — `haoping-sfml` vs `sfml-sequence`

Comparison of my branch (`haoping-sfml`) and the professor's branch
(`sfml-sequence`, commit `3a1fcc68` "Implement sequence-level sFML training with
checkpoints"), both on the OU example (§5.1.1 of Chen & Xiu 2024).

## TL;DR

Both implement the same sFML method (ResNet deterministic sub-map D̃ + WGAN-GP
stochastic sub-map, multi-step/sequence training). We made **complementary**
changes:

- **My branch** `latent_dim = 1`
  (GAN-paper Z = dim(x)), cosine LR decay in Phase 2, multi-step D̃.
- **Professor's branch**
  checkpoint-selection of the best model, rich live diagnostics, and a large
  batch that makes runs ~5× faster.

Neither branch fully fixes the OU mean: the generator retains a mean bias that
the professor's selection metric does not target.

## Runtime

| | My branch | Professor's branch |
|---|---|---|
| Phase-1 epochs | 500 | 250 |
| Phase-2 epochs | 1000 | 1000 |
| Phase-2 batch size | 256 | 1000 |
| Phase-2 batches / epoch | 39 | 10 |
| Generator updates / epoch | 39 (1 per batch) | 2 (1 per 5 batches) |
| Total generator updates | ~39,000 | ~2,000 |
| Phase-2 time / epoch | ~2.6 s | ~0.49 s |
| **Total wall-clock (OU)** | **~50 min** | **~10 min** |

The ~5× speedup comes from Phase 2: a large batch (1000 → only 10 batches/epoch)
plus updating the generator **once every 5 critic batches** (the `updates C/G:
10/2` line), i.e. ~20× fewer generator updates per epoch. Fewer updates are
tolerable because the run selects the best checkpoint rather than the final one.

## What is the same in both

- Two-phase sFML: Phase 1 deterministic ResNet D̃ (multi-step MSE, eq. 4.11),
  Phase 2 WGAN-GP on full L-step increment sequences.
- Rollout is Algorithm 4.1: `increment = D̃(x) − x + G(x,z)`,
  `x_next = x + increment`.
- Critic sees `(x0, y_{1:L})`; gradient penalty over the joint input.
- `n_critic = 5`, `gp_lambda = 10`, Adam β = (0.5, 0.999), lr = 5e-5,
  GELU, `[-1,1]` normalization, `condition_on_state = true`.
- **Phase-1 (deterministic) is multi-step in both** — the professor's branch uses
  a new `load_sequence` loader returning `(x0, trajectory)` and feeds it to the
  ODE model, exactly like my manual slicing.

## What the professor's branch ADDS (not in my version)

1. **Checkpoint-selection module (the main change).** Saves a checkpoint every
   `checkpoint_interval` (100) epochs, scores each, keeps the best, writes
   `generator_best`, and evaluates *that* (not the final model). The selection
   score is:

   `score = w_std · std_rel_error + w_wgap · |Wasserstein gap| + w_gp · gradient_penalty`

   (weights all = 1 in config). It selects for correct increment **variance**, a
   converged critic, and a satisfied Lipschitz constraint — **but has no
   mean/drift term.**

2. **Rich per-epoch diagnostics** — logs Wasserstein gap, GP, C(real), C(fake),
   and real-vs-fake increment std (`dy std real/fake`). The increment-std readout
   directly monitors over/under-dispersion.

3. **Speed** — large batch (1000) + generator update every 5th batch → ~10 min.

4. **`load_sequence` dataset API** — cleaner than slicing the sequence by hand.

## What the professor's branch does NOT have (older base)

He branched from an earlier copy of my code, so it lacks:

- `latent_dim = 8` (old default) instead of `latent_dim = 1`
  (GAN-paper convention Z = dim(x); for scalar OU that is 1).
- **No Phase-2 cosine LR decay** — constant 5e-5 (my branch decays 5e-5 → 1e-5).
- Phase-1 epochs 250 vs 500.

## Provenance of the disputed settings

- **Phase-2 cosine decay (5e-5 → 1e-5):** comes from the **2022 GAN paper**
  (Chen & Xiu, SIAM SISC), which states it explicitly. The **sFML (2024) paper
  gives only the initial 5e-5** and says nothing about a schedule. So a *missing*
  Phase-2 cosine (professor's branch) is consistent with sFML as literally
  written; adding it (my branch) is an inference from the predecessor. Open detail.
- **Phase-1 cosine (lr 1e-3):** documented in the DUE paper — solid, and both
  branches have it.
- **`latent_dim = dim(x)`:** the 2022 GAN paper states dim(z) = dim(x); sFML
  leaves the noise dimension as a free user choice. So `1` (mine) is a defensible
  convention, `8` (his) is also allowed.

## Problems that still exist (both branches, and the reproduction overall)

What *does* reproduce on OU: the one-step conditional distribution, the effective
diffusion, and (roughly) the std are close. The hard parts remain:

1. **Generator mean bias → wrong long-time mean.** The generator carries a small
   spurious mean (e.g. +0.031 normalized at x=0 on the professor's selected
   model). Near the OU steady state the true per-step drift is ~0.1σ, so this
   bias is amplified ~100× by 1/Δ and relocates the map's fixed point — the mean
   lands anywhere from ~1.15 to ~2.3 instead of μ=1.2. This is the single most
   stubborn issue and appears in every run.

2. **Over-dispersion (std / diffusion too high).** The generator produces ~8–24%
   too much noise, and it's *x-dependent* (inflating at larger x). Neither more
   epochs nor a larger batch fixed it — the batch-1024 test made it worse.

3. **WGAN training is non-monotonic and unstable.** Model quality oscillates and
   *degrades* with more training (5000-epoch OU and the GBM run both diverge).
   So "how long to train" is effectively an un-principled hyperparameter. The
   professor's checkpoint-selection mitigates this — but only along the axes its
   score measures (variance, W-gap, GP), not the mean.

4. **Under-powered critic.** The 3×20 critic is a weak Wasserstein witness (its
   loss collapses to ~0 while the distributions clearly differ). It cannot sharply
   police the mean or the x-dependent variance, which is the *root enabler* of
   problems 1–3. A stronger critic and/or an explicit consistency (MMD) term is
   the likely cure, but both deviate from the paper's stated setup.

5. **Phase-1 D̃ fixed point is off (~1.15, not 1.2), and gets WORSE with more
   training.** The multi-step-from-window-starts loss under-constrains the
   fixed-point region, so extra epochs let it drift (1.15 → 1.10 at 3000 epochs).
   Per Remark 4.1 D̃ need not be accurate, but its bias still feeds the mean error.

6. **The method does not generalize to GBM as-is.** With the plain pipeline, GBM
   (multiplicative noise, exponentially wide range) *diverges* — the model
   produces wrong-sign / negative trajectories, because `[-1,1]` min-max
   normalization crushes the low-value dynamics. A log-transform would fix it but
   is **not allowed**: it uses knowledge of the true equation, which violates the
   whole premise (learn the flow map from data of an *unknown* system). The paper
   runs GBM raw, so this is a training/stability failure, not a missing transform.

7. **Selection metric omits the mean (professor's branch).** `score = std_rel_err
   + |W-gap| + GP` has no drift/mean term, so it can (and did) select a
   good-variance but wrong-mean checkpoint.

8. **Cannot reach the paper's training budget.** The paper trains to 100k epochs;
   on CPU that's infeasible, and our WGAN destabilizes well before then. It is
   also unclear whether the paper's "epochs" means full data passes or SGD
   iterations (the 2022 paper counts 40k *iterations*), which changes the required
   compute by orders of magnitude. So we can't currently confirm whether faithful
   reproduction is simply a compute problem or a deeper one.

**Net:** OU is reproduced qualitatively (all figures show the right behavior) but
not tightly on the mean; GBM is not reproduced; and the common root causes are
the WGAN's instability + a weak critic, neither of which is fully solved by
either branch. The professor's checkpoint-selection is the best partial fix so
far and should be merged with a mean-aware selection score and stronger critic.

## Suggested merge / next steps

- Combine the professor's **checkpoint-selection + diagnostics** (the right tools
  for the WGAN's non-monotonic training) with a **mean/drift term in the
  selection score**, so the chosen checkpoint is good on *both* moments.
- Decide the Phase-2 cosine question explicitly (it's an inference, not stated by
  sFML) — worth confirming against the authors' intent.
- `latent_dim`: 1 vs 8 is a free choice; 1 matches the scalar noise dimension.
