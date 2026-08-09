"""
Phase 1 check — Deterministic sub-map D_theta for GBM.

Trains D_theta (gated_resnet) with multi-step rollout loss and produces:
  1. Single-step accuracy: D_theta(x_0) vs analytical E[x_1|x_0] = x_0*(1+mu*dt)
  2. Rollout from x_0=0.5 vs analytical GBM mean E[x(t)] = x_0*exp(mu*t)

GBM has no fixed point (mean grows exponentially), so there is no fixed-point
error to diagnose — the check is purely whether D_theta tracks the exponential mean.

Run from examples/GBM/ directory:
  python check_phase1.py
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import due

# GBM parameters
MU    = 2.0
DT    = 0.01
X0_TEST = 0.5
N_STEPS = 100   # T = 1.0

# Config
CONFIG_PATH = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
conf_data, conf_net, conf_train = due.utils.read_config(CONFIG_PATH)

# Data — load_sequence applies Yeo-Johnson and returns normalized trainX/trainY
data_loader = due.datasets.sde.sde_dataset(conf_data)
trainX, trainY, test_data, vmin, vmax = data_loader.load_sequence("GBM_train.mat", "GBM_test.mat")
NZ = data_loader.normalizer
conf_net["seq_len"] = trainY.shape[-1]   # L = 40
print(f"Normalization: {data_loader.normalization}  (lambda={float(NZ.lam[0]):.3f})")
print(f"trainX: {trainX.shape},  trainY: {trainY.shape}")

# Train Phase 1 — D_theta architecture from config ("resnet" | "gated_resnet")
_arch = conf_net.get("det_arch", "resnet")
print(f"Phase-1 architecture: {_arch}")
det_net = getattr(due.networks.fcn, _arch)(vmin, vmax, conf_net)
phase1_model = due.models.ODE(trainX, trainY, det_net, conf_train)
phase1_model.train()
phase1_model.save_hist()

device     = conf_train["device"]
det_net    = torch.load(conf_train["save_path"] + "/model", map_location=device, weights_only=False)
det_net.eval()
torch_dtype = torch.float32

def to_norm(x_phys):
    """Physical (N, 1) ndarray -> normalised (N, 1) float32."""
    return NZ.transform(np.asarray(x_phys, dtype=np.float64)).astype(np.float32)

def to_phys(x_norm):
    """Normalised (N, 1) ndarray -> physical (N, 1)."""
    return NZ.inverse(np.asarray(x_norm, dtype=np.float64))

# --- Diagnostic 1: single-step accuracy ---
x0_vals = np.linspace(0.2, 2.5, 10)
x1_true = x0_vals * (1.0 + MU * DT)   # E[x_1|x_0] for GBM

with torch.no_grad():
    x0_norm = to_norm(x0_vals[:, None])
    x1_norm = det_net(torch.tensor(x0_norm, dtype=torch_dtype, device=device)).cpu().numpy()
    x1_pred = to_phys(x1_norm)[:, 0]

fig, ax = plt.subplots(figsize=(7, 5))
ax.scatter(x0_vals, x1_true, c='k', s=60, zorder=3, label='Analytical E[x₁|x₀]')
ax.scatter(x0_vals, x1_pred, c='b', s=60, marker='^', zorder=3, label='D_θ(x₀)')
for xv, ya, yp in zip(x0_vals, x1_true, x1_pred):
    ax.plot([xv, xv], [ya, yp], 'r-', lw=1)
ax.set_xlabel('x₀'); ax.set_ylabel('x₁')
ax.set_title('GBM Phase 1: single-step accuracy')
ax.legend()
plt.tight_layout()
plt.savefig(conf_train["save_path"] + "/phase1_single_step.png", dpi=150)
plt.close()
print("Saved phase1_single_step.png")

# --- Diagnostic 2: rollout vs analytical mean ---
t_arr     = np.arange(N_STEPS + 1) * DT
mean_true = X0_TEST * np.exp(MU * t_arr)

det_traj = [X0_TEST]
x_det = torch.tensor(to_norm(np.array([[X0_TEST]])), dtype=torch_dtype, device=device)
with torch.no_grad():
    for _ in range(N_STEPS):
        x_det = det_net(x_det)
        det_traj.append(float(to_phys(x_det.cpu().numpy())[0, 0]))

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(t_arr, mean_true, 'k-',  label='Analytical mean: x₀·exp(μt)')
ax.plot(t_arr, det_traj,  'b--', label='D_θ rollout')
ax.set_xlabel('t'); ax.set_ylabel('x')
ax.set_title(f'GBM Phase 1: D_θ vs analytical mean (x₀={X0_TEST})')
ax.legend()
plt.tight_layout()
plt.savefig(conf_train["save_path"] + "/phase1_rollout.png", dpi=150)
plt.close()
print("Saved phase1_rollout.png")

final_pred = det_traj[-1]
final_true = mean_true[-1]
print(f"\nD_θ at T=1.0:        {final_pred:.4f}")
print(f"Analytical mean at T=1.0: {final_true:.4f}")
print(f"Relative error:           {abs(final_pred - final_true) / final_true * 100:.2f}%")
print(f"(GBM has no fixed point — mean grows as x₀·exp(μt))")
