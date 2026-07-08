"""
Stochastic Flow Map Learning — SDE with nonlinear diffusion (Chen & Xiu 2024, sec 5.2.1)

  dx = -mu x dt + sigma e^{-x^2} dW,   mu=5, sigma=0.5,  dt=0.01
  Test: x0 = -0.4.

Same full pipeline as OU/GBM (learned D_theta + hard-centered WGAN-GP + MMD + decay).
Analytical mean: E[x(t)] = x0 e^{-mu t} (drift is linear). Std has no closed form, so it
is compared against a ground-truth Euler-Maruyama ensemble (the test set). Effective
drift a(x) = -mu x and diffusion b(x) = sigma e^{-x^2} are recovered on a state grid.

Run:  python NonlinearDiffusion.py
"""
import numpy as np, torch, matplotlib.pyplot as plt
from yaml import safe_load
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
import due

CONFIG_PATH = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
print(f">>> Using config: {CONFIG_PATH}")
conf_data, conf_net, conf_train = due.utils.read_config(CONFIG_PATH)
config_raw = safe_load(Path(CONFIG_PATH).read_text())
conf_gan = config_raw["gan"]
conf_gan["seed"] = config_raw["seed"]; conf_gan["dtype"] = config_raw["dtype"]
conf_gan["device"] = conf_train["device"]; conf_gan["latent_dim"] = conf_net["latent_dim"]

data_loader = due.datasets.sde.sde_dataset(conf_data)
trainX, trainY, test_data, vmin, vmax = data_loader.load_sequence("NLD_train.mat", "NLD_test.mat")
conf_net["sequence_length"] = trainY.shape[-1]
NZ = data_loader.normalizer
print(f"Normalization: {data_loader.normalization} (lambda={float(NZ.lam[0]):.3f})")

# ---- Phase 1 ----
_arch = conf_net.get("det_arch", "resnet")
print(f"Phase-1 architecture: {_arch}")
det_net = getattr(due.networks.fcn, _arch)(vmin, vmax, conf_net)
phase1 = due.models.ODE(trainX, trainY, det_net, conf_train)
phase1.train(); phase1.save_hist()
det_net = torch.load(conf_train["save_path"] + "/model", map_location=conf_train["device"], weights_only=False)

# ---- Phase 2 ----
generator = due.networks.gan.Generator(conf_net); critic = due.networks.gan.Critic(conf_net)
sde_model = due.models.SDE(trainX, trainY, det_net, generator, critic, conf_gan)
sde_model.train(); sde_model.save_hist()

# ---- evaluation ----
device = conf_train["device"]
gpath = Path(conf_gan["save_path"]) / "generator_best"
if not gpath.exists(): gpath = Path(conf_gan["save_path"]) / "generator_final"
print("Loading generator from", gpath)
generator = torch.load(str(gpath), map_location=device, weights_only=False)
det_net = torch.load(conf_train["save_path"] + "/model", map_location=device, weights_only=False)
generator.eval(); det_net.eval()

MU, SIGMA, DT = 5.0, 0.5, 0.01
X0_TEST, N_SAMPLES = -0.4, 10_000
N_STEPS = 400   # T = 4.0 (10x the training window; reversion is fast)
LATENT_DIM = conf_net["latent_dim"]
CENTER_GEN = bool(conf_gan.get("center_generator", False)); CENTER_K = int(conf_gan.get("center_K", 16))
torch_dtype = torch.float64 if conf_data["dtype"] == "double" else torch.float32
save_path = conf_gan["save_path"]

def to_norm(x): return NZ.transform(np.asarray(x, dtype=np.float64).reshape(-1, 1))
def to_phys(u): return NZ.inverse(np.asarray(u, dtype=np.float64).reshape(-1, 1))[:, 0]

def gen_increment(u, n):
    if not CENTER_GEN:
        z = torch.randn(n, LATENT_DIM, device=device, dtype=torch_dtype); return generator(u, z)
    z = torch.randn(n, CENTER_K, LATENT_DIM, device=device, dtype=torch_dtype)
    u_rep = u.unsqueeze(1).expand(-1, CENTER_K, -1).reshape(n * CENTER_K, u.size(-1))
    out = generator(u_rep, z.reshape(n * CENTER_K, LATENT_DIM)).reshape(n, CENTER_K, -1)
    return out[:, 0, :] - out.mean(dim=1)

def stochastic_predict(x0, n_samples, n_steps):
    u = torch.tensor(to_norm(x0), dtype=torch_dtype, device=device).expand(n_samples, -1)
    traj = [to_phys(u.cpu().numpy())]
    with torch.no_grad():
        for _ in range(n_steps):
            u = det_net(u) + gen_increment(u, n_samples)
            traj.append(to_phys(u.cpu().numpy()))
    return np.stack(traj, axis=1)

t_arr = np.arange(N_STEPS + 1) * DT

# ---- Phase-1 diagnostic ----
u = torch.tensor(to_norm(X0_TEST), dtype=torch_dtype, device=device); det_traj = [X0_TEST]
with torch.no_grad():
    for _ in range(N_STEPS):
        u = det_net(u); det_traj.append(float(to_phys(u.cpu().numpy())[0]))
mean_ana = X0_TEST * np.exp(-MU * t_arr)
plt.figure(figsize=(8, 5))
plt.plot(t_arr, mean_ana, 'k-', label='Analytical mean  x0 e^{-mu t}')
plt.plot(t_arr, det_traj, 'b--', label='D_theta rollout')
plt.xlabel('t'); plt.ylabel('x'); plt.title('Phase 1: D_theta vs analytical mean (NLD)')
plt.legend(); plt.tight_layout(); plt.savefig(save_path + "/det_rollout.png", dpi=150); plt.close()
print("Saved det_rollout.png")

# ---- Mean & Std (std vs ground-truth EM ensemble = test set) ----
print("Running stochastic prediction ...")
ens = stochastic_predict(X0_TEST, N_SAMPLES, N_STEPS)
mean_pred, std_pred = ens.mean(axis=0), ens.std(axis=0)
gt = test_data[:, 0, :N_STEPS + 1]           # ground-truth EM ensemble from x0=-0.4
mean_gt, std_gt = gt.mean(axis=0), gt.std(axis=0)
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
ax[0].plot(t_arr, mean_ana, 'k-', label='Analytical'); ax[0].plot(t_arr, mean_pred, 'r--', label='sFML')
ax[0].set_xlabel('t'); ax[0].set_ylabel('Mean'); ax[0].set_title('Mean (NLD)'); ax[0].legend()
ax[1].plot(t_arr, std_gt, 'k-', label='Ground truth (EM)'); ax[1].plot(t_arr, std_pred, 'r--', label='sFML')
ax[1].set_xlabel('t'); ax[1].set_ylabel('Std'); ax[1].set_title('Standard Deviation (NLD)'); ax[1].legend()
plt.tight_layout(); plt.savefig(save_path + "/mean_std.png", dpi=150); plt.close()
print("Saved mean_std.png")

# ---- Effective drift a(x)=-mu x and diffusion b(x)=sigma e^{-x^2}, recovered on a grid ----
x_grid = np.linspace(-1.0, 1.0, 41); N_DD = 20_000
drift_pred = np.zeros_like(x_grid); diff_pred = np.zeros_like(x_grid)
with torch.no_grad():
    for i, xg in enumerate(x_grid):
        u = torch.tensor(to_norm(xg), dtype=torch_dtype, device=device).expand(N_DD, -1)
        nxt = to_phys((det_net(u) + gen_increment(u, N_DD)).cpu().numpy())
        dx = nxt - xg
        drift_pred[i] = dx.mean() / DT
        diff_pred[i] = dx.std() / np.sqrt(DT)
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
ax[0].plot(x_grid, -MU * x_grid, 'k-', label='Analytical  -mu x'); ax[0].plot(x_grid, drift_pred, 'r.', label='sFML')
ax[0].set_xlabel('x'); ax[0].set_ylabel('Drift a(x)'); ax[0].set_title('Effective Drift'); ax[0].legend()
ax[1].plot(x_grid, SIGMA * np.exp(-x_grid**2), 'k-', label='Analytical  sigma e^{-x^2}'); ax[1].plot(x_grid, diff_pred, 'r.', label='sFML')
ax[1].set_xlabel('x'); ax[1].set_ylabel('Diffusion b(x)'); ax[1].set_title('Effective Diffusion'); ax[1].legend()
plt.tight_layout(); plt.savefig(save_path + "/drift_diffusion.png", dpi=150); plt.close()
print("Saved drift_diffusion.png")

# ---- Conditional distribution one step from x=-0.3 (paper: G(-0.3)) ----
X_COND = -0.3
cond = stochastic_predict(X_COND, N_SAMPLES, 1)[:, 1]
m_c = X_COND * (1 - MU * DT); s_c = SIGMA * np.exp(-X_COND**2) * np.sqrt(DT)
xp = np.linspace(cond.min() - 0.01, cond.max() + 0.01, 200)
pdf = np.exp(-0.5 * ((xp - m_c) / s_c) ** 2) / (s_c * np.sqrt(2 * np.pi))
fig, axc = plt.subplots(figsize=(7, 5))
axc.hist(cond, bins=80, density=True, alpha=0.6, label='sFML samples')
axc.plot(xp, pdf, 'k-', lw=2, label='Analytical (EM)')
axc.set_xlabel('$x_{n+1}$'); axc.set_ylabel('Density'); axc.set_title(f'Conditional distribution at $x_n={X_COND}$ (NLD)')
axc.legend(); plt.tight_layout(); plt.savefig(save_path + "/conditional_dist.png", dpi=150); plt.close()
print("Saved conditional_dist.png")

print("\nAll NLD evaluation figures saved to", save_path)
