"""
Phase-1 diagnostic for the double-well example.

Loads the saved D_theta and checks:
  1. Det rollout from multiple ICs vs EM mean trajectories.
  2. Effective drift recovery: D_theta(x)-x / dt vs f(x)=x-x^3 on a grid.

Run AFTER training Phase 1 (or after full DoubleWell.py run):
    python check_phase1.py
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

CONFIG_PATH = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
conf_data, conf_net, conf_train = due.utils.read_config(CONFIG_PATH)
config_raw = safe_load(Path(CONFIG_PATH).read_text())
conf_gan = config_raw["gan"]
conf_gan["device"] = conf_train["device"]
save_path = conf_train["save_path"]

data_loader = due.datasets.sde.sde_dataset(conf_data)
trainX, trainY, test_data, vmin, vmax = data_loader.load_sequence("DW_train.mat", "DW_test.mat")
NZ = data_loader.normalizer

device = conf_train["device"]
torch_dtype = torch.float64 if conf_data["dtype"] == "double" else torch.float32

det_net = torch.load(save_path + "/model", map_location=device, weights_only=False)
det_net.eval()

DT = 0.01
N_STEPS = 200   # T = 2.0

def to_norm(x):
    return NZ.transform(np.asarray(x, dtype=np.float64).reshape(-1, 1))

def to_phys(u):
    return NZ.inverse(np.asarray(u, dtype=np.float64).reshape(-1, 1))[:, 0]

# ---- Plot 1: D_theta rollout from several ICs vs EM mean ----
# For a symmetric double-well with x0 in each well:
# x0 = +1.0: EM mean stays near +1; det rollout should track it
# x0 = -1.0: EM mean stays near -1; det rollout should track it
# x0 =  0.0: EM mean stays at 0 (symmetry); det rollout should also stay ~0
# x0 =  0.5: EM mean moves toward +1 (noisy); det rollout should roughly track

ICS = [0.0, 0.5, -0.5, 1.0, -1.0, 1.5]
t_arr = np.arange(N_STEPS + 1) * DT

fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharey=False)
axes = axes.flatten()
for ax, x0 in zip(axes, ICS):
    # EM mean from 5000 trajectories
    rng = np.random.default_rng(0)
    tr = np.empty((5000, N_STEPS + 1)); tr[:, 0] = x0; sq = np.sqrt(DT)
    for n in range(N_STEPS):
        x = tr[:, n]
        tr[:, n + 1] = x + (x - x**3) * DT + 0.5 * sq * rng.standard_normal(5000)
    em_mean = tr.mean(axis=0)

    # D_theta rollout (single trajectory, deterministic)
    u = torch.tensor(to_norm(x0), dtype=torch_dtype, device=device)
    det_r = [x0]
    with torch.no_grad():
        for _ in range(N_STEPS):
            u = det_net(u)
            det_r.append(float(to_phys(u.cpu().numpy())[0]))

    ax.plot(t_arr, em_mean, 'k-', lw=1.5, label='EM mean')
    ax.plot(t_arr, det_r,   'b--', lw=1.5, label='D_theta')
    ax.axhline(1,  color='gray', lw=0.7, ls=':')
    ax.axhline(-1, color='gray', lw=0.7, ls=':')
    ax.axhline(0,  color='gray', lw=0.5, ls=':')
    ax.set_title(f'x0 = {x0}'); ax.set_xlabel('t'); ax.set_ylabel('x')
    if ax is axes[0]: ax.legend(fontsize=8)

plt.suptitle('Phase-1 diagnostic: D_theta vs EM mean (double-well)', fontsize=12)
plt.tight_layout()
plt.savefig(save_path + "/phase1_rollouts.png", dpi=150); plt.close()
print("Saved phase1_rollouts.png")

# ---- Plot 2: Effective drift recovery f(x) = x - x^3 ----
# Estimated as (D_theta(x) - x) / DT on a grid
x_grid = np.linspace(-2.0, 2.0, 81)
drift_det = np.zeros_like(x_grid)
with torch.no_grad():
    for i, xg in enumerate(x_grid):
        u_in  = torch.tensor(to_norm(xg), dtype=torch_dtype, device=device)
        u_out = det_net(u_in)
        x_out = float(to_phys(u_out.cpu().numpy())[0])
        drift_det[i] = (x_out - xg) / DT

plt.figure(figsize=(7, 5))
plt.plot(x_grid, x_grid - x_grid**3, 'k-', lw=2, label='Analytical  x - x^3')
plt.plot(x_grid, drift_det, 'b.', ms=5, label='D_theta estimate')
plt.axhline(0, color='gray', lw=0.5, ls=':')
plt.axvline(1,  color='gray', lw=0.5, ls=':')
plt.axvline(-1, color='gray', lw=0.5, ls=':')
plt.xlabel('x'); plt.ylabel('f(x)'); plt.title('Phase-1 drift recovery (double-well)')
plt.legend(); plt.tight_layout()
plt.savefig(save_path + "/phase1_drift.png", dpi=150); plt.close()
print("Saved phase1_drift.png")

print(f"\nPhase-1 diagnostics saved to {save_path}")
