"""
Phase 1 check — Deterministic sub-map D_theta only.

Trains D_theta with multi-step rollout loss (eq. 4.11) and produces two diagnostics:
  1. Single-step accuracy: D_theta(x_0) vs analytical E[x_1 | x_0]
  2. Long-rollout accuracy: rolling D_theta 400 steps from x_0=1.5 vs analytical mean

Run from examples/OU/ directory:
  python check_phase1.py
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from yaml import safe_load
from pathlib import Path
import due

# OU parameters
THETA = 1.0
MU    = 1.2
SIGMA = 0.3
DT    = 0.01
X0_TEST = 1.5
N_STEPS = 400  # T = 4.0

# Config
conf_data, conf_net, conf_train = due.utils.read_config("config.yaml")

# Data — same recurrent loader + normaliser OU.py uses (respects config's
# normalization mode). Fixes the old KeyError (config's data block no longer
# carries seq_len) and the previous hardcoded min-max normalisation.
data_loader = due.datasets.sde.sde_dataset(conf_data)
trainX_p1, trainY_p1, test_data, vmin, vmax = data_loader.load_sequence(
    "OU_train.mat", "OU_test.mat"
)
# trainX_p1: (N, d) normalised x0 ; trainY_p1: (N, d, L) normalised x1..xL
conf_net["sequence_length"] = trainY_p1.shape[-1]

# Train Phase 1
det_net = due.networks.fcn.gated_resnet(vmin, vmax, conf_net)
phase1_model = due.models.ODE(trainX_p1, trainY_p1, det_net, conf_train)
phase1_model.train()
phase1_model.save_hist()

# Load best saved model
device = conf_train["device"]
det_net = torch.load(conf_train["save_path"] + "/model", map_location=device, weights_only=False)
det_net.eval()

torch_dtype = torch.float64 if conf_data["dtype"] == "double" else torch.float32

# Normalise/denormalise via the fitted normaliser (identical to OU.py) so the
# diagnostics match the trained model's coordinate system in every mode.
NZ = data_loader.normalizer

def normalize(x):
    a = np.asarray(x, dtype=np.float64)
    return NZ.transform(a.reshape(-1, 1)).reshape(a.shape)

def denormalize(x):
    a = np.asarray(x, dtype=np.float64)
    return NZ.inverse(a.reshape(-1, 1)).reshape(a.shape)

# --- Diagnostic 1: single-step accuracy across a range of x_0 ---
x0_test_vals = np.linspace(0.4, 2.0, 10)
analytical_x1 = x0_test_vals + THETA * (MU - x0_test_vals) * DT  # E[x1|x0] for OU

with torch.no_grad():
    x0_norm = torch.tensor(normalize(x0_test_vals[:, None]), dtype=torch_dtype, device=device)
    x1_pred_norm = det_net(x0_norm).cpu().numpy()
    x1_pred = denormalize(x1_pred_norm[:, 0])

fig, ax = plt.subplots(figsize=(7, 5))
ax.scatter(x0_test_vals, analytical_x1, c='k', s=60, zorder=3, label='Analytical E[x₁|x₀]')
ax.scatter(x0_test_vals, x1_pred,       c='b', s=60, marker='^', zorder=3, label='D_θ(x₀)')
for xv, ya, yp in zip(x0_test_vals, analytical_x1, x1_pred):
    ax.plot([xv, xv], [ya, yp], 'r-', lw=1)
ax.set_xlabel('x₀'); ax.set_ylabel('x₁')
ax.set_title('Phase 1 check: single-step prediction for 10 random x₀')
ax.legend()
plt.tight_layout()
plt.savefig(conf_train["save_path"] + "/phase1_single_step.png", dpi=150)
plt.close()
print("Saved phase1_single_step.png")

# --- Diagnostic 2: long rollout from x_0=1.5 vs analytical mean ---
t_arr = np.arange(N_STEPS + 1) * DT
mean_true = (X0_TEST - MU) * np.exp(-THETA * t_arr) + MU

det_traj = [X0_TEST]
x_det = torch.tensor(normalize(np.array([[X0_TEST]])), dtype=torch_dtype, device=device)
with torch.no_grad():
    for _ in range(N_STEPS):
        x_det = det_net(x_det)
        det_traj.append(float(denormalize(x_det.cpu().numpy()[0, 0])))

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(t_arr, mean_true, 'k-',  label='Analytical mean')
ax.plot(t_arr, det_traj,  'b--', label='D_θ rollout')
ax.set_xlabel('t'); ax.set_ylabel('x')
ax.set_title('Phase 1: D_θ vs analytical mean (x₀=1.5)')
ax.legend()
plt.tight_layout()
plt.savefig(conf_train["save_path"] + "/phase1_rollout.png", dpi=150)
plt.close()
print("Saved phase1_rollout.png")

print(f"\nSteady-state error: D_θ → {det_traj[-1]:.4f}, analytical → {MU:.4f}, error = {abs(det_traj[-1]-MU):.4f}")
