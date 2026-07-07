"""
Phase-1-only diagnostic for the OU example.

Trains ONLY the deterministic sub-map D_theta (ResNet, multi-step MSE eq. 4.11)
and reports how good it is as a conditional-mean model — no GAN. Because Phase 2
is now hard-centered (S_delta zero-mean by construction), the long-run mean of the
full model equals D_theta's fixed point, so this script tells us the attributable
Phase-1 bias directly.

Outputs:
  - det_model/det_rollout_phase1.png : D_theta rollout from x0=1.5 vs analytical mean
  - printed: fixed point, rollout endpoint, effective drift (slope/intercept),
             and drift extrapolation at x = 1.2 .. 1.6 (the test region, off-data)

Run:  python phase1_only.py
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

# ---- config + data ----
conf_data, conf_net, conf_train = due.utils.read_config("config.yaml")
data_loader = due.datasets.sde.sde_dataset(conf_data)
trainX, trainY, test_data, vmin, vmax = data_loader.load_sequence("OU_train.mat", "OU_test.mat")
conf_net["sequence_length"] = trainY.shape[-1]

# ---- train Phase 1 (learned ResNet D_theta), ALWAYS (ignore use_oracle_det) ----
# phase1_single_step: fit the one-step conditional mean directly (eq. 4.12) on ALL
#   consecutive pairs, instead of the multi-step rollout loss (eq. 4.11). With hard-
#   centering, D_theta only needs the one-step mean, for which this is the direct
#   estimator (and it measured a better fixed point: OLS 1.18 vs multi-step 1.15).
single_step = conf_train.get("phase1_single_step", False)
if single_step:
    print(">>> Phase-1-only: SINGLE-STEP MSE (all one-step pairs, eq. 4.12)")
    d = trainX.shape[1]
    states = np.concatenate([trainX[:, :, None], trainY], axis=2)   # (N, d, L+1)
    p1X = np.ascontiguousarray(states[:, :, :-1].transpose(0, 2, 1).reshape(-1, d))       # (N*L, d)
    p1Y = np.ascontiguousarray(states[:, :, 1:].transpose(0, 2, 1).reshape(-1, d, 1))     # (N*L, d, 1)
    print("   one-step pairs:", p1X.shape[0])
else:
    print(">>> Phase-1-only: MULTI-STEP MSE (eq. 4.11, L=%d)" % trainY.shape[-1])
    p1X, p1Y = trainX, trainY

det_net = due.networks.fcn.resnet(vmin, vmax, conf_net)
phase1 = due.models.ODE(p1X, p1Y, det_net, conf_train)
phase1.train()
phase1.save_hist()

det_net = torch.load(conf_train["save_path"] + "/model",
                     map_location=conf_train["device"], weights_only=False)
det_net.eval()

# ---- OU truth ----
THETA, MU, SIGMA, DT = 1.0, 1.2, 0.3, 0.01
X0_TEST, N_STEPS = 1.5, 400
vmin_val = float(vmin.flatten()[0]); vmax_val = float(vmax.flatten()[0])
tdt = torch.float64 if conf_data["dtype"] == "double" else torch.float32

def normalize(x):   return 2 * (x - 0.5 * (vmax_val + vmin_val)) / (vmax_val - vmin_val)
def denormalize(u): return u * 0.5 * (vmax_val - vmin_val) + 0.5 * (vmax_val + vmin_val)

def D_phys(x_phys):
    """Apply the learned D_theta in physical space (scalar or array)."""
    x = np.atleast_1d(np.asarray(x_phys, dtype=np.float64))
    u = torch.tensor(normalize(x).reshape(-1, 1), dtype=tdt, device=conf_train["device"])
    with torch.no_grad():
        y = det_net(u).cpu().numpy().reshape(-1)
    return denormalize(y)

# ---- rollout from x0 = 1.5 ----
t_arr = np.arange(N_STEPS + 1) * DT
det_traj = [X0_TEST]
x = X0_TEST
for _ in range(N_STEPS):
    x = float(D_phys(x)[0]); det_traj.append(x)
det_traj = np.array(det_traj)
mean_true = (X0_TEST - MU) * np.exp(-THETA * t_arr) + MU

# ---- fixed point: solve D(x) = x ----
xs = np.linspace(0.8, 1.6, 8001)
g = D_phys(xs) - xs            # increment; zero at the fixed point
sign = np.sign(g)
idx = np.where(np.diff(sign) != 0)[0]
if len(idx):
    i = idx[0]
    fp = xs[i] - g[i] * (xs[i+1] - xs[i]) / (g[i+1] - g[i])   # linear root
else:
    fp = float('nan')

# ---- effective drift f(x) = (D(x)-x)/dt, fit slope/intercept over the DATA range ----
xd = np.linspace(0.3, 1.0, 200)              # on-data region
fd = (D_phys(xd) - xd) / DT
A = np.vstack([xd, np.ones_like(xd)]).T
slope, intercept = np.linalg.lstsq(A, fd, rcond=None)[0]
theta_eff = -slope
mu_eff = intercept / theta_eff if theta_eff != 0 else float('nan')

print("\n================ PHASE-1 D_theta DIAGNOSTIC ================")
print(f"Fixed point  D(x)=x           : {fp:.4f}   (true mu = 1.2000)")
print(f"Rollout endpoint from x0=1.5  : {det_traj[-1]:.4f}   (true -> 1.2000)")
print(f"Effective theta (on-data fit) : {theta_eff:.4f}   (true 1.0)")
print(f"Effective mu    (on-data fit) : {mu_eff:.4f}   (true 1.2)")
print("\nDrift extrapolation  f(x)=theta*(mu-x), true vs learned  (x=1.5 is the test IC, OFF data):")
for xq in [1.0, 1.2, 1.3, 1.44, 1.5, 1.6]:
    f_true = THETA * (MU - xq)
    f_learn = float((D_phys(xq)[0] - xq) / DT)
    flag = "  <- test IC (off-data)" if abs(xq - 1.5) < 1e-9 else ("  (off-data)" if xq > 1.44 else "")
    print(f"  x={xq:.2f}   true f={f_true:+.4f}   learned f={f_learn:+.4f}{flag}")
print("===========================================================")

# ---- plot ----
plt.figure(figsize=(8, 5))
plt.plot(t_arr, mean_true, 'k-',  label='Analytical mean')
plt.plot(t_arr, det_traj,  'b--', label='D_theta rollout (learned)')
plt.axhline(MU, color='gray', ls=':', lw=1)
plt.xlabel('t'); plt.ylabel('x')
plt.title(f'Phase 1 (learned D_theta): fixed point {fp:.3f} vs mu=1.2')
plt.legend(); plt.tight_layout()
out = conf_train["save_path"] + "/det_rollout_phase1.png"
plt.savefig(out, dpi=150); plt.close()
print("Saved", out)
