"""
Stochastic Flow Map Learning — Double-Well SDE (Chen & Xiu 2024, sec 5.2.3)

  dx = (x - x^3) dt + sigma dW,   sigma=0.5,  dt=0.01
  Test: x0 = 1.5 (stable well), following paper protocol.

Drift f(x) = x - x^3: stable equilibria at x=+-1, unstable at x=0.
Constant diffusion g(x) = sigma.
Stationary distribution: p(x) ∝ exp(4x^2 - 2x^4)  (bimodal, peaks at +-1).

Key result: training data (T_window=0.4) contains NO inter-well transitions.
sFML nonetheless learns to produce transitions at the correct timescale (~7 time units).
Paper uses T=300; we use T=20 (2000 steps) for CPU tractability — enough to show transitions.

Run:  python DoubleWell.py
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from yaml import safe_load
from pathlib import Path
from scipy import integrate
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import due

# ---- config ----
CONFIG_PATH = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
print(f">>> Using config: {CONFIG_PATH}")
conf_data, conf_net, conf_train = due.utils.read_config(CONFIG_PATH)
config_raw = safe_load(Path(CONFIG_PATH).read_text())
conf_gan = config_raw["gan"]
conf_gan["seed"] = config_raw["seed"]
conf_gan["dtype"] = config_raw["dtype"]
conf_gan["device"] = conf_train["device"]
conf_gan["latent_dim"] = conf_net["latent_dim"]

# ---- data ----
data_loader = due.datasets.sde.sde_dataset(conf_data)
trainX, trainY, test_data, vmin, vmax = data_loader.load_sequence("DW_train.mat", "DW_test.mat")
conf_net["sequence_length"] = trainY.shape[-1]
NZ = data_loader.normalizer
print(f"Normalization: {data_loader.normalization}")

# ---- eval_only mode ----
EVAL_ONLY = bool(config_raw.get("eval_only", False))
if EVAL_ONLY:
    print(">>> eval_only=true: skipping Phase 1 and Phase 2 training, loading saved models.")
else:
    # ---- Phase 1: deterministic sub-map D_theta ----
    _arch = conf_net.get("det_arch", "resnet")
    print(f"Phase-1 architecture: {_arch}")
    det_net = getattr(due.networks.fcn, _arch)(vmin, vmax, conf_net)
    phase1 = due.models.ODE(trainX, trainY, det_net, conf_train)
    phase1.train()
    phase1.save_hist()
    det_net = torch.load(conf_train["save_path"] + "/model", map_location=conf_train["device"], weights_only=False)

    # ---- Phase 2: WGAN-GP ----
    if conf_gan.get("single_step_critic", False):
        conf_net["sequence_length"] = 1   # single-step critic: (x_t, dx_t) pairs -> critic input dim 2d
    generator = due.networks.gan.Generator(conf_net)
    critic    = due.networks.gan.Critic(conf_net)
    sde_model = due.models.SDE(trainX, trainY, det_net, generator, critic, conf_gan)
    sde_model.train()
    sde_model.save_hist()

# ---- load saved models for eval ----
device = conf_train["device"]
gpath = Path(conf_gan["save_path"]) / "generator_best"
if not gpath.exists():
    gpath = Path(conf_gan["save_path"]) / "generator_final"
print("Loading generator from", gpath)
generator = torch.load(str(gpath), map_location=device, weights_only=False)
det_net   = torch.load(conf_train["save_path"] + "/model", map_location=device, weights_only=False)
generator.eval(); det_net.eval()

# ---- SDE / eval constants ----
SIGMA, DT = 0.5, 0.01
X0_TEST, N_SAMPLES = 1.5, 10_000
N_STEPS = 2000   # T = 20.0 — enough to show inter-well transitions (mean escape time ~7)
LATENT_DIM = conf_net["latent_dim"]
CENTER_GEN = bool(conf_gan.get("center_generator", False))
CENTER_K   = int(conf_gan.get("center_K", 16))
torch_dtype = torch.float64 if conf_data["dtype"] == "double" else torch.float32
save_path = conf_gan["save_path"]


def to_norm(x_phys):
    return NZ.transform(np.asarray(x_phys, dtype=np.float64).reshape(-1, 1))


def to_phys(u):
    return NZ.inverse(np.asarray(u, dtype=np.float64).reshape(-1, 1))[:, 0]


def gen_increment(u, n):
    """Hard-centered stochastic sub-map draw (matches training)."""
    if not CENTER_GEN:
        z = torch.randn(n, LATENT_DIM, device=device, dtype=torch_dtype)
        return generator(u, z)
    z = torch.randn(n, CENTER_K, LATENT_DIM, device=device, dtype=torch_dtype)
    u_rep = u.unsqueeze(1).expand(-1, CENTER_K, -1).reshape(n * CENTER_K, u.size(-1))
    out = generator(u_rep, z.reshape(n * CENTER_K, LATENT_DIM)).reshape(n, CENTER_K, -1)
    return out[:, 0, :] - out.mean(dim=1)


def stochastic_predict(x0_raw, n_samples, n_steps):
    u = torch.tensor(to_norm(x0_raw), dtype=torch_dtype, device=device).expand(n_samples, -1)
    traj = [to_phys(u.cpu().numpy())]
    with torch.no_grad():
        for _ in range(n_steps):
            u = det_net(u) + gen_increment(u, n_samples)
            traj.append(to_phys(u.cpu().numpy()))
    return np.stack(traj, axis=1)   # (n_samples, n_steps+1)


t_arr = np.arange(N_STEPS + 1) * DT

# ---- analytical stationary distribution ----
Z_const, _ = integrate.quad(lambda x: np.exp(4*x**2 - 2*x**4), -4.0, 4.0)


def p_stationary(x):
    return np.exp(4*x**2 - 2*x**4) / Z_const


# ---- inline EM ground truth from x0=1.5 ----
# test_data was generated from x0=0 so we produce EM ground truth on-the-fly (numpy only).
print("Generating EM ground truth from x0=1.5 ...")
rng_em = np.random.default_rng(0)
em_traj = np.empty((N_SAMPLES, N_STEPS + 1), dtype=np.float64)
em_traj[:, 0] = X0_TEST
x_em = np.full(N_SAMPLES, float(X0_TEST))
for _s in range(N_STEPS):
    x_em = x_em + (x_em - x_em**3) * DT + SIGMA * np.sqrt(DT) * rng_em.standard_normal(N_SAMPLES)
    em_traj[:, _s + 1] = x_em
mean_gt = em_traj.mean(axis=0)
std_gt  = em_traj.std(axis=0)
print("EM ground truth done.")


# ============================================================
# Plot 1: Phase-1 diagnostic — D_theta rollout from x0=1.5
# x0=1.5 is in the right well (f(1.5)=-1.875, strongly attracted to +1).
# D_theta should track the EM mean closely — good Phase 1 test.
# ============================================================
u0 = torch.tensor(to_norm(X0_TEST), dtype=torch_dtype, device=device)
det_traj = [float(X0_TEST)]
with torch.no_grad():
    u = u0.clone()
    for _ in range(N_STEPS):
        u = det_net(u)
        det_traj.append(float(to_phys(u.cpu().numpy())[0]))

plt.figure(figsize=(8, 5))
plt.plot(t_arr, mean_gt, 'k-', label='EM ground truth mean')
plt.plot(t_arr, det_traj, 'b--', label='D_theta rollout (det)')
plt.axhline(1.0, color='gray', lw=0.8, ls=':', label='stable fixed point x=1')
plt.xlabel('t'); plt.ylabel('x')
plt.title('Phase 1: D_theta rollout vs EM mean  (x0=1.5, double-well)')
plt.legend(); plt.tight_layout()
plt.savefig(save_path + "/det_rollout.png", dpi=150); plt.close()
print("Saved det_rollout.png")

# ============================================================
# Plot 2: Mean & Std from x0=1.5 vs EM ground truth.
# Mean starts at 1.5 → decays to ~1 (mean first-passage away from right well
# is ~7 time units, so for T=20 some transitions appear and the mean drops).
# Std rises as trajectories eventually begin switching to -1 well.
# ============================================================
print("Running stochastic prediction ...")
ens = stochastic_predict(X0_TEST, N_SAMPLES, N_STEPS)
mean_pred, std_pred = ens.mean(axis=0), ens.std(axis=0)

fig, ax = plt.subplots(1, 2, figsize=(12, 5))
ax[0].plot(t_arr, mean_gt, 'k-', label='EM ground truth'); ax[0].plot(t_arr, mean_pred, 'r--', label='sFML')
ax[0].axhline(1.0, color='gray', lw=0.8, ls=':', label='x=+1 (well)')
ax[0].set_xlabel('t'); ax[0].set_ylabel('Mean'); ax[0].set_title('Mean from x0=1.5 (double-well)'); ax[0].legend()
ax[1].plot(t_arr, std_gt, 'k-', label='EM ground truth'); ax[1].plot(t_arr, std_pred, 'r--', label='sFML')
ax[1].set_xlabel('t'); ax[1].set_ylabel('Std'); ax[1].set_title('Std from x0=1.5 (double-well)'); ax[1].legend()
plt.tight_layout(); plt.savefig(save_path + "/mean_std.png", dpi=150); plt.close()
print("Saved mean_std.png")

# ============================================================
# Plot 3: Effective drift f(x)=x-x^3 and diffusion g(x)=sigma
# Recovered from sFML ensemble at grid of initial conditions.
# ============================================================
x_grid = np.linspace(-1.8, 1.8, 37)
N_DD = 20_000
drift_pred = np.zeros_like(x_grid)
diff_pred  = np.zeros_like(x_grid)
with torch.no_grad():
    for i, xg in enumerate(x_grid):
        u = torch.tensor(to_norm(xg), dtype=torch_dtype, device=device).expand(N_DD, -1)
        nxt = to_phys((det_net(u) + gen_increment(u, N_DD)).cpu().numpy())
        dx = nxt - xg
        drift_pred[i] = dx.mean() / DT
        diff_pred[i]  = np.sqrt(dx.var() / DT)

fig, ax = plt.subplots(1, 2, figsize=(12, 5))
ax[0].plot(x_grid, x_grid - x_grid**3, 'k-', label='Analytical  x - x^3')
ax[0].plot(x_grid, drift_pred, 'r.', ms=6, label='sFML')
ax[0].axhline(0, color='gray', lw=0.5, ls=':')
ax[0].set_xlabel('x'); ax[0].set_ylabel('Drift f(x)'); ax[0].set_title('Effective Drift (double-well)'); ax[0].legend()

ax[1].axhline(SIGMA, color='k', label=f'Analytical  sigma={SIGMA}')
ax[1].plot(x_grid, diff_pred, 'r.', ms=6, label='sFML')
ax[1].set_xlabel('x'); ax[1].set_ylabel('Diffusion g(x)'); ax[1].set_title('Effective Diffusion (double-well)'); ax[1].legend()
plt.tight_layout(); plt.savefig(save_path + "/drift_diffusion.png", dpi=150); plt.close()
print("Saved drift_diffusion.png")

# ============================================================
# Plot 4: Stationary distribution — long rollout from random ICs
# Start from ICs spread across [-2,2], run N_BURN steps, collect
# final states and compare histogram to analytical p(x).
# ============================================================
N_STAT = 20_000
N_BURN = 1000   # T=10 — well past mixing time
rng = np.random.default_rng(42)
x_init = rng.uniform(-2.0, 2.0, N_STAT)
u = torch.tensor(to_norm(x_init), dtype=torch_dtype, device=device)
with torch.no_grad():
    for _ in range(N_BURN):
        u = det_net(u) + gen_increment(u, N_STAT)
stat_samples = to_phys(u.cpu().numpy())

x_plot = np.linspace(-2.5, 2.5, 400)
fig, axs = plt.subplots(figsize=(7, 5))
axs.hist(stat_samples, bins=100, density=True, alpha=0.6, label='sFML (stationary)')
axs.plot(x_plot, p_stationary(x_plot), 'k-', lw=2, label=r'Analytical $p \propto e^{4x^2-2x^4}$')
axs.set_xlabel('x'); axs.set_ylabel('Density')
axs.set_title('Stationary distribution (double-well)')
axs.legend(); plt.tight_layout()
plt.savefig(save_path + "/stationary_dist.png", dpi=150); plt.close()
print("Saved stationary_dist.png")

# ============================================================
# Plot 5: Conditional distributions from x0=1.5 at T=1 and T=20.
# T=1 (100 steps): narrow near +1, still in right well.
# T=20 (2000 steps, full rollout): bimodal — some trajectories have
# crossed to -1 well, showing sFML has learned transition dynamics.
# Key test: paper's headline result (§5.2.3). Both panels use `ens`
# already computed above (no extra stochastic_predict call).
# ============================================================
H1, H2 = 100, N_STEPS   # T=1, T=20

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, H in zip(axes, [H1, H2]):
    ax.hist(em_traj[:, H], bins=80, density=True, alpha=0.5, label='EM ground truth')
    ax.hist(ens[:, H],     bins=80, density=True, alpha=0.5, label='sFML')
    ax.set_xlabel('x'); ax.set_ylabel('Density')
    ax.set_title(f'Conditional from x0=1.5 at T={H*DT:.1f}')
    ax.legend()
plt.tight_layout()
plt.savefig(save_path + "/conditional_dist.png", dpi=150); plt.close()
print("Saved conditional_dist.png")

print(f"\nAll DoubleWell evaluation figures saved to {save_path}")
