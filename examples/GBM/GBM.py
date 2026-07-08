"""
Stochastic Flow Map Learning — Geometric Brownian Motion (GBM)

  dx = mu x dt + sigma x dW,   mu = 2.0, sigma = 1.0,  dt = 0.01
  Test: x0 = 0.5, marched to T = 1.0.

Same two-phase pipeline as OU (learned D_theta + hard-centered WGAN-GP), the ONLY
difference being data normalization: GBM's state spans [~0, ~170], which min-max
crushes into a sliver near -1. We use Yeo-Johnson (config data.normalization), which
discovers a log-like transform from the data (lambda ~ -0.4) and variance-stabilises
the multiplicative noise into additive noise in transformed space.

GBM has NO mean reversion / no fixed point: the mean grows exponentially, so we
compare against the analytical lognormal moments rather than an OU-style fixed point.

Run:  python GBM.py
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

# ---- data (Phase-2 coordinate) ----
data_loader = due.datasets.sde.sde_dataset(conf_data)
trainX, trainY, test_data, vmin, vmax = data_loader.load_sequence("GBM_train.mat", "GBM_test.mat")
conf_net["sequence_length"] = trainY.shape[-1]
NZ = data_loader.normalizer   # fitted Phase-2 transform (Yeo-Johnson / minmax)
print(f"Phase-2 normalization: {data_loader.normalization}  (lambda={float(NZ.lam[0]):.3f})")

# ---- eval_only mode: skip all training, load saved models directly ----
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
    generator = due.networks.gan.Generator(conf_net)
    critic    = due.networks.gan.Critic(conf_net)
    sde_model = due.models.SDE(trainX, trainY, det_net, generator, critic, conf_gan)
    sde_model.train()
    sde_model.save_hist()

# ---- evaluation ----
device = conf_train["device"]
gpath = Path(conf_gan["save_path"]) / "generator_best"
if not gpath.exists():
    gpath = Path(conf_gan["save_path"]) / "generator_final"
print("Loading generator from", gpath)
generator = torch.load(str(gpath), map_location=device, weights_only=False)
det_net   = torch.load(conf_train["save_path"] + "/model", map_location=device, weights_only=False)
generator.eval(); det_net.eval()

# GBM truth
MU, SIGMA, DT = 2.0, 1.0, 0.01
X0_TEST, N_SAMPLES, N_STEPS = 0.5, 10_000, 100
LATENT_DIM = conf_net["latent_dim"]
CENTER_GEN = bool(conf_gan.get("center_generator", False))
CENTER_K   = int(conf_gan.get("center_K", 16))
torch_dtype = torch.float64 if conf_data["dtype"] == "double" else torch.float32
save_path = conf_gan["save_path"]

def to_norm(x_phys):
    a = np.asarray(x_phys, dtype=np.float64).reshape(-1, 1)
    return NZ.transform(a)

def to_phys(u):
    a = np.asarray(u, dtype=np.float64).reshape(-1, 1)
    return NZ.inverse(a)[:, 0]

def gen_increment(u, n):
    """Stochastic sub-map draw, hard-centered per state when CENTER_GEN (matches training)."""
    if not CENTER_GEN:
        z = torch.randn(n, LATENT_DIM, device=device, dtype=torch_dtype)
        return generator(u, z)
    z = torch.randn(n, CENTER_K, LATENT_DIM, device=device, dtype=torch_dtype)
    u_rep = u.unsqueeze(1).expand(-1, CENTER_K, -1).reshape(n * CENTER_K, u.size(-1))
    out = generator(u_rep, z.reshape(n * CENTER_K, LATENT_DIM)).reshape(n, CENTER_K, -1)
    return out[:, 0, :] - out.mean(dim=1)

def stochastic_predict(x0_raw, n_samples, n_steps):
    """Run n_samples trajectories from x0_raw for n_steps."""
    u = torch.tensor(to_norm(x0_raw), dtype=torch_dtype, device=device).expand(n_samples, -1)
    traj = [to_phys(u.cpu().numpy())]
    with torch.no_grad():
        for _ in range(n_steps):
            u = det_net(u) + gen_increment(u, n_samples)
            traj.append(to_phys(u.cpu().numpy()))
    return np.stack(traj, axis=1)  # (n_samples, n_steps+1) physical

t_arr = np.arange(N_STEPS + 1) * DT

# ---- Phase-1 diagnostic: D_theta rollout vs analytical mean (exponential, no fixed pt) ----
u = torch.tensor(to_norm(X0_TEST), dtype=torch_dtype, device=device)
det_traj = [X0_TEST]
with torch.no_grad():
    for _ in range(N_STEPS):
        u = det_net(u)
        det_traj.append(float(to_phys(u.cpu().numpy())[0]))
mean_true = X0_TEST * np.exp(MU * t_arr)
plt.figure(figsize=(8, 5))
plt.plot(t_arr, mean_true, 'k-', label='Analytical mean  x0 e^{mu t}')
plt.plot(t_arr, det_traj, 'b--', label='D_theta rollout')
plt.xlabel('t'); plt.ylabel('x'); plt.title('Phase 1: D_theta vs analytical mean (GBM)')
plt.legend(); plt.tight_layout(); plt.savefig(save_path + "/det_rollout.png", dpi=150); plt.close()
print("Saved det_rollout.png")

# ---- ensemble ----
print("Running stochastic prediction ...")
ens = stochastic_predict(X0_TEST, N_SAMPLES, N_STEPS)

# ---- Mean & Std vs analytical lognormal ----
mean_pred, std_pred = ens.mean(axis=0), ens.std(axis=0)
mean_ana = X0_TEST * np.exp(MU * t_arr)
std_ana  = X0_TEST * np.exp(MU * t_arr) * np.sqrt(np.exp(SIGMA**2 * t_arr) - 1.0)
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
ax[0].plot(t_arr, mean_ana, 'k-', label='Analytical'); ax[0].plot(t_arr, mean_pred, 'r--', label='sFML')
ax[0].set_xlabel('t'); ax[0].set_ylabel('Mean'); ax[0].set_title('Mean (GBM)'); ax[0].legend()
ax[1].plot(t_arr, std_ana, 'k-', label='Analytical'); ax[1].plot(t_arr, std_pred, 'r--', label='sFML')
ax[1].set_xlabel('t'); ax[1].set_ylabel('Std'); ax[1].set_title('Standard Deviation (GBM)'); ax[1].legend()
plt.tight_layout(); plt.savefig(save_path + "/mean_std.png", dpi=150); plt.close()
print("Saved mean_std.png")

# ---- Effective drift f(x)=mu x and diffusion g(x)=sigma x (linear through origin) ----
xv = ens[:, :-1].flatten(); dxv = (ens[:, 1:] - ens[:, :-1]).flatten()
nb = 50
edges = np.linspace(np.percentile(xv, 0.5), np.percentile(xv, 99.5), nb + 1)
xc = 0.5 * (edges[:-1] + edges[1:]); drift = np.full(nb, np.nan); diff = np.full(nb, np.nan)
for i in range(nb):
    m = (xv >= edges[i]) & (xv < edges[i + 1])
    if m.sum() > 20:
        drift[i] = dxv[m].mean() / DT
        diff[i] = np.sqrt(dxv[m].var() / DT)
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
ax[0].plot(xc, MU * xc, 'k-', label='Analytical  mu x'); ax[0].plot(xc, drift, 'r.', label='sFML')
ax[0].set_xlabel('x'); ax[0].set_ylabel('Drift f(x)'); ax[0].set_title('Effective Drift'); ax[0].legend()
ax[1].plot(xc, SIGMA * xc, 'k-', label='Analytical  sigma x'); ax[1].plot(xc, diff, 'r.', label='sFML')
ax[1].set_xlabel('x'); ax[1].set_ylabel('Diffusion g(x)'); ax[1].set_title('Effective Diffusion'); ax[1].legend()
plt.tight_layout(); plt.savefig(save_path + "/drift_diffusion.png", dpi=150); plt.close()
print("Saved drift_diffusion.png")

# ---- Conditional distribution one step from x_n = 6 (paper's choice, Fig. 12) ----
X_COND = 6.0
cond = stochastic_predict(X_COND, N_SAMPLES, 1)[:, 1]
# EM one-step: x1 = x0 + mu x0 dt + sigma x0 sqrt(dt) N  -> Normal(x0(1+mu dt), (sigma x0 sqrt dt)^2)
m_c = X_COND * (1 + MU * DT); s_c = SIGMA * X_COND * np.sqrt(DT)
xp = np.linspace(cond.min() - 0.01, cond.max() + 0.01, 200)
pdf = np.exp(-0.5 * ((xp - m_c) / s_c) ** 2) / (s_c * np.sqrt(2 * np.pi))
fig, ax = plt.subplots(figsize=(7, 5))
ax.hist(cond, bins=80, density=True, alpha=0.6, label='sFML samples')
ax.plot(xp, pdf, 'k-', lw=2, label='Analytical (EM)')
ax.set_xlabel('$x_{n+1}$'); ax.set_ylabel('Density'); ax.set_title(f'Conditional distribution at $x_n={X_COND}$ (GBM)')
ax.legend(); plt.tight_layout(); plt.savefig(save_path + "/conditional_dist.png", dpi=150); plt.close()
print("Saved conditional_dist.png")

print("\nAll GBM evaluation figures saved to", save_path)
