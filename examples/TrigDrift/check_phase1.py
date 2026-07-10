"""
Phase 1 diagnostic for Trigonometric Drift example.

Checks:
  1. D_theta rollout from 5 ICs vs EM mean
  2. Effective drift recovery: (D_theta(x)-x)/dt vs f(x)=sin(2πx) on a grid

D_theta should converge toward x=0.5 from any IC in the training range.
Near x=0.5, f(x) ≈ -2π(x-0.5), so the gated_resnet affine branch should
capture the dominant linear decay with MLP providing the nonlinear correction.

Run from examples/TrigDrift/:  python check_phase1.py
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from yaml import safe_load
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import due

K, SIGMA, DT = 1, 0.5, 0.01
W = 2 * K * np.pi

CONFIG_PATH = "config.yaml"
conf_data, conf_net, conf_train = due.utils.read_config(CONFIG_PATH)
config_raw = safe_load(Path(CONFIG_PATH).read_text())

device = conf_train["device"]
model_path = conf_train["save_path"] + "/model"
if not Path(model_path).exists():
    print(f"ERROR: no Phase 1 model at {model_path}. Run TrigDrift.py first (or Phase 1 alone).")
    sys.exit(1)

det_net = torch.load(model_path, map_location=device, weights_only=False)
det_net.eval()

data_loader = due.datasets.sde.sde_dataset(conf_data)
trainX, trainY, test_data, vmin, vmax = data_loader.load_sequence("TD_train.mat", "TD_test.mat")
NZ = data_loader.normalizer
torch_dtype = torch.float64 if conf_data["dtype"] == "double" else torch.float32

save_path = config_raw.get("gan", {}).get("save_path", "./gan_model")


def to_norm(x):
    return NZ.transform(np.asarray(x, dtype=np.float64).reshape(-1, 1))


def to_phys(u):
    return NZ.inverse(np.asarray(u, dtype=np.float64).reshape(-1, 1))[:, 0]


# ---- 1. Rollout comparison ----
N_STEPS = 1000
ICs = [0.35, 0.45, 0.50, 0.55, 0.60, 0.70]
t_arr = np.arange(N_STEPS + 1) * DT

# EM mean for each IC using test_data (x0=0.6 only), otherwise simulate
rng = np.random.default_rng(99)

fig, ax = plt.subplots(figsize=(9, 5))
for x0 in ICs:
    # D_theta rollout
    u = torch.tensor(to_norm(x0), dtype=torch_dtype, device=device)
    traj = [x0]
    with torch.no_grad():
        for _ in range(N_STEPS):
            u = det_net(u)
            traj.append(float(to_phys(u.cpu().numpy())[0]))
    ax.plot(t_arr, traj, ls='--', lw=1.2, label=f'D_θ x0={x0}')

    # EM mean for same IC
    x = np.full(2000, x0)
    em_traj = np.empty((2000, N_STEPS + 1)); em_traj[:, 0] = x
    sq = np.sqrt(DT)
    for n in range(N_STEPS):
        xn = em_traj[:, n]
        em_traj[:, n + 1] = xn + np.sin(W * xn) * DT + SIGMA * np.cos(W * xn) * sq * rng.standard_normal(2000)
    ax.plot(t_arr, em_traj.mean(axis=0), ls='-', lw=1.0, alpha=0.55)

ax.axhline(0.5, color='k', lw=0.7, ls=':', label='stable x=0.5')
ax.axhline(0.35, color='blue', lw=0.5, ls=':', alpha=0.5, label='IC boundaries')
ax.axhline(0.70, color='blue', lw=0.5, ls=':', alpha=0.5)
ax.set_xlabel('t'); ax.set_ylabel('x')
ax.set_title('Phase 1: D_theta rollout (dashed) vs EM mean (solid) — trig drift')
ax.legend(fontsize=7, ncol=2); plt.tight_layout()
plt.savefig(save_path + "/p1_rollout.png", dpi=150); plt.close()
print("Saved p1_rollout.png")

# ---- 2. Effective drift recovery on grid ----
# Estimate from D_theta:  f_pred(x) ≈ (E[D_theta(x)] - x) / dt
x_grid = np.linspace(0.25, 0.75, 61)
drift_pred = np.zeros_like(x_grid)
with torch.no_grad():
    for i, xg in enumerate(x_grid):
        u = torch.tensor(to_norm(xg), dtype=torch_dtype, device=device)
        u_next = det_net(u)
        xnext = float(to_phys(u_next.cpu().numpy())[0])
        drift_pred[i] = (xnext - xg) / DT

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(x_grid, np.sin(W * x_grid), 'k-', lw=2, label='Analytical  sin(2πx)')
ax.plot(x_grid, drift_pred, 'b.', ms=5, label='(D_theta(x)−x)/dt')
ax.axvline(0.5, color='gray', lw=0.5, ls=':'); ax.axhline(0, color='gray', lw=0.5, ls=':')
ax.axvline(0.35, color='blue', lw=0.7, ls='--', alpha=0.4, label='IC boundary')
ax.axvline(0.70, color='blue', lw=0.7, ls='--', alpha=0.4)
ax.set_xlabel('x'); ax.set_ylabel('Effective drift f(x)')
ax.set_title('Phase 1 drift recovery — trig drift'); ax.legend(fontsize=8)
plt.tight_layout(); plt.savefig(save_path + "/p1_drift_recovery.png", dpi=150); plt.close()
print("Saved p1_drift_recovery.png")

print("Phase 1 check complete.")
