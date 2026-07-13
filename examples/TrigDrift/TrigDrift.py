"""
Stochastic Flow Map Learning — Trigonometric Drift & Diffusion (Chen & Xiu 2024, sec 5.2.2)

  dx = sin(2*pi*x) dt + sigma*cos(2*pi*x) dW,   k=1, sigma=0.5,  dt=0.01
  Test: x0 = 0.6, T = 10.

Drift f(x) = sin(2*pi*x): stable equilibria at x = 1/2 + n (integer n).
Diffusion g(x) = 0.5*cos(2*pi*x): state-dependent, varies ~3x across IC range.

The paper trains from IC U(0.35, 0.7) — entirely within the basin of x=0.5 — and
tests from x0=0.6 (slightly off the stable point). The key test is whether sFML
recovers both the sinusoidal drift and the cosine diffusion across the IC range.
"Accuracy deteriorates near the endpoints" (paper, §5.2.2) — a coverage effect.

Run:  python TrigDrift.py
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from yaml import safe_load
from pathlib import Path
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
trainX, trainY, test_data, vmin, vmax = data_loader.load_sequence("TD_train.mat", "TD_test.mat")
conf_net["sequence_length"] = trainY.shape[-1]
NZ = data_loader.normalizer
print(f"Normalization: {data_loader.normalization}")

# ---- eval_only mode ----
EVAL_ONLY = bool(config_raw.get("eval_only", False))
if EVAL_ONLY:
    print(">>> eval_only=true: skipping Phase 1 and Phase 2 training, loading saved models.")
else:
    # ---- Phase 1 ----
    _arch = conf_net.get("det_arch", "resnet")
    print(f"Phase-1 architecture: {_arch}")
    det_net = getattr(due.networks.fcn, _arch)(vmin, vmax, conf_net)
    phase1 = due.models.ODE(trainX, trainY, det_net, conf_train)
    phase1.train()
    phase1.save_hist()
    det_net = torch.load(conf_train["save_path"] + "/model", map_location=conf_train["device"], weights_only=False)

    # ---- Phase 2 ----
    if conf_gan.get("single_step_critic", False):
        conf_net["sequence_length"] = 1   # single-step critic: (x_t, dx_t) pairs -> critic input dim 2d
    generator = due.networks.gan.Generator(conf_net)
    critic    = due.networks.gan.Critic(conf_net)
    sde_model = due.models.SDE(trainX, trainY, det_net, generator, critic, conf_gan)
    sde_model.train()
    sde_model.save_hist()

# ---- load models ----
device = conf_train["device"]
gpath = Path(conf_gan["save_path"]) / "generator_best"
if not gpath.exists():
    gpath = Path(conf_gan["save_path"]) / "generator_final"
print("Loading generator from", gpath)
generator = torch.load(str(gpath), map_location=device, weights_only=False)
det_net   = torch.load(conf_train["save_path"] + "/model", map_location=device, weights_only=False)
generator.eval(); det_net.eval()

# ---- constants ----
K, SIGMA, DT = 1, 0.5, 0.01
W = 2 * K * np.pi
X0_TEST, N_SAMPLES = 0.6, 10_000
N_STEPS = 1000   # T = 10.0
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
    return np.stack(traj, axis=1)


t_arr = np.arange(N_STEPS + 1) * DT

# ---- Plot 1: Phase-1 diagnostic — D_theta rollout vs EM mean ----
# Analytical mean from x0=0.6: no closed form for sin(2πx), compare to EM
u0 = torch.tensor(to_norm(X0_TEST), dtype=torch_dtype, device=device)
det_traj = [float(X0_TEST)]
with torch.no_grad():
    u = u0.clone()
    for _ in range(N_STEPS):
        u = det_net(u)
        det_traj.append(float(to_phys(u.cpu().numpy())[0]))

gt = test_data[:, 0, :N_STEPS + 1]
mean_gt = gt.mean(axis=0)

plt.figure(figsize=(8, 5))
plt.plot(t_arr, mean_gt, 'k-', label='EM ground truth mean')
plt.plot(t_arr, det_traj, 'b--', label='D_theta rollout')
plt.axhline(0.5, color='gray', lw=0.8, ls=':', label='stable point x=0.5')
plt.xlabel('t'); plt.ylabel('x')
plt.title('Phase 1: D_theta rollout vs EM mean  (x0=0.6, trig drift)')
plt.legend(); plt.tight_layout()
plt.savefig(save_path + "/det_rollout.png", dpi=150); plt.close()
print("Saved det_rollout.png")

# ---- Plot 2: Mean & Std vs EM ground truth ----
print("Running stochastic prediction ...")
ens = stochastic_predict(X0_TEST, N_SAMPLES, N_STEPS)
mean_pred, std_pred = ens.mean(axis=0), ens.std(axis=0)
std_gt = gt.std(axis=0)

fig, ax = plt.subplots(1, 2, figsize=(12, 5))
ax[0].plot(t_arr, mean_gt, 'k-', label='EM ground truth')
ax[0].plot(t_arr, mean_pred, 'r--', label='sFML')
ax[0].axhline(0.5, color='gray', lw=0.7, ls=':', label='x=0.5 (stable)')
ax[0].set_xlabel('t'); ax[0].set_ylabel('Mean')
ax[0].set_title('Mean (trig drift, x0=0.6)'); ax[0].legend()
ax[1].plot(t_arr, std_gt, 'k-', label='EM ground truth')
ax[1].plot(t_arr, std_pred, 'r--', label='sFML')
ax[1].set_xlabel('t'); ax[1].set_ylabel('Std')
ax[1].set_title('Std (trig drift, x0=0.6)'); ax[1].legend()
plt.tight_layout(); plt.savefig(save_path + "/mean_std.png", dpi=150); plt.close()
print("Saved mean_std.png")

# ---- Plot 3: Effective drift f(x)=sin(2πx) and diffusion |g(x)|=0.5|cos(2πx)| ----
# Grid over IC range plus some margin; paper notes accuracy degrades near endpoints
x_grid = np.linspace(0.25, 0.75, 41)
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
ax[0].plot(x_grid, np.sin(W * x_grid), 'k-', lw=2, label='Analytical  sin(2πx)')
ax[0].plot(x_grid, drift_pred, 'r.', ms=6, label='sFML')
ax[0].axvline(0.5, color='gray', lw=0.5, ls=':')
ax[0].axhline(0, color='gray', lw=0.5, ls=':')
# mark IC boundaries
ax[0].axvline(0.35, color='blue', lw=0.7, ls='--', alpha=0.5, label='IC boundary')
ax[0].axvline(0.70, color='blue', lw=0.7, ls='--', alpha=0.5)
ax[0].set_xlabel('x'); ax[0].set_ylabel('Drift f(x)')
ax[0].set_title('Effective Drift (trig drift)'); ax[0].legend(fontsize=8)

ax[1].plot(x_grid, SIGMA * np.abs(np.cos(W * x_grid)), 'k-', lw=2, label='Analytical  0.5|cos(2πx)|')
ax[1].plot(x_grid, diff_pred, 'r.', ms=6, label='sFML')
ax[1].axvline(0.35, color='blue', lw=0.7, ls='--', alpha=0.5, label='IC boundary')
ax[1].axvline(0.70, color='blue', lw=0.7, ls='--', alpha=0.5)
ax[1].set_xlabel('x'); ax[1].set_ylabel('Diffusion g(x)')
ax[1].set_title('Effective Diffusion (trig drift)'); ax[1].legend(fontsize=8)
plt.tight_layout(); plt.savefig(save_path + "/drift_diffusion.png", dpi=150); plt.close()
print("Saved drift_diffusion.png")

# ---- Plot 4: Conditional distribution at x=0.5 (paper's choice) ----
# One-step EM: x1 | x0=0.5 ~ N(x0 + f(0.5)*dt, g(0.5)^2*dt) = N(0.5, 0.25*dt)
# (f(0.5)=sin(pi)=0, g(0.5)=0.5*cos(pi)=-0.5, |g|=0.5)
X_COND = 0.5
N_COND_STEPS = 1   # one-step conditional
cond_sfml = stochastic_predict(X_COND, N_SAMPLES, N_COND_STEPS)[:, 1]

# EM one-step reference
rng_c = np.random.default_rng(1)
x0_c = np.full(N_SAMPLES, X_COND)
m_c = x0_c + np.sin(W * x0_c) * DT          # = 0.5 + 0
s_c = SIGMA * np.abs(np.cos(W * x0_c)) * np.sqrt(DT)   # = 0.5 * sqrt(0.01) = 0.05
em_cond = m_c + s_c * rng_c.standard_normal(N_SAMPLES)

xp = np.linspace(
    min(cond_sfml.min(), em_cond.min()) - 0.02,
    max(cond_sfml.max(), em_cond.max()) + 0.02, 200)
pdf = np.exp(-0.5 * ((xp - m_c[0]) / s_c[0])**2) / (s_c[0] * np.sqrt(2 * np.pi))

fig, axc = plt.subplots(figsize=(7, 5))
axc.hist(em_cond, bins=80, density=True, alpha=0.5, label='EM one-step')
axc.hist(cond_sfml, bins=80, density=True, alpha=0.5, label='sFML one-step')
axc.plot(xp, pdf, 'k-', lw=2, label=f'Analytical N({m_c[0]:.3f}, {s_c[0]:.4f}²)')
axc.set_xlabel('$x_{n+1}$'); axc.set_ylabel('Density')
axc.set_title(f'Conditional distribution at $x_n={X_COND}$ (trig drift)')
axc.legend(); plt.tight_layout()
plt.savefig(save_path + "/conditional_dist.png", dpi=150); plt.close()
print("Saved conditional_dist.png")

print(f"\nAll TrigDrift evaluation figures saved to {save_path}")
