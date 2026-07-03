"""
Phase-1-ONLY diagnostic: D_theta fixed point + rollout.

Loads only det_model/model (the frozen deterministic sub-map). Does NOT touch
the generator, so it is safe to run WHILE Phase 2 is still training — Phase 2
freezes D_theta and never re-saves it, so this file is stable on disk.

Run from examples/OU/:
    python phase1_check.py
"""
import numpy as np
import torch
import matplotlib.pyplot as plt
import due

conf_data, conf_net, conf_train = due.utils.read_config("config.yaml")
device = conf_train["device"]
torch_dtype = torch.float64 if conf_data["dtype"] == "double" else torch.float32

# vmin/vmax exactly as OU.py computes them
loader = due.datasets.sde.sde_dataset(conf_data)
_, _, _, vmin, vmax = loader.load("OU_train.mat", "OU_test.mat")
vmin_val = float(vmin.flatten()[0]); vmax_val = float(vmax.flatten()[0])

def normalize(x):    return 2 * (x - 0.5 * (vmax_val + vmin_val)) / (vmax_val - vmin_val)
def denormalize(x):  return x * 0.5 * (vmax_val - vmin_val) + 0.5 * (vmax_val + vmin_val)

det_net = torch.load(conf_train["save_path"] + "/model", map_location=device, weights_only=False)
det_net.eval()

THETA, MU, DT = 1.0, 1.2, 0.01
X0, N = 1.5, 400
t = np.arange(N + 1) * DT

# Deterministic rollout from x0 = 1.5
x = torch.tensor(normalize(np.array([[X0]])), dtype=torch_dtype, device=device)
traj = [X0]
with torch.no_grad():
    for _ in range(N):
        x = det_net(x)
        traj.append(float(denormalize(x.cpu().numpy()[0, 0])))
mean_true = (X0 - MU) * np.exp(-THETA * t) + MU

# Fixed point: where D_theta(x) = x
xs = np.linspace(0.5, 2.0, 600)
d_incr = np.zeros_like(xs)
with torch.no_grad():
    for i, xv in enumerate(xs):
        xn = torch.tensor(normalize(np.array([[xv]])), dtype=torch_dtype, device=device)
        d_incr[i] = float(denormalize(det_net(xn).cpu().numpy()[0, 0])) - xv
fp = xs[np.argmin(np.abs(d_incr))]

print(f"D_theta fixed point  D(x)=x : {fp:.4f}   (target mu = {MU}; was ~1.15 at 500 epochs)")
print(f"D_theta rollout settles at  : {traj[-1]:.4f}")

plt.figure(figsize=(8, 5))
plt.plot(t, mean_true, 'k-',  label='Analytical mean')
plt.plot(t, traj,      'b--', label='D_theta rollout')
plt.axhline(MU, color='gray', ls=':', label='mu = 1.2')
plt.xlabel('t'); plt.ylabel('x'); plt.legend()
plt.title('Phase 1 check: D_theta rollout vs analytical mean')
plt.tight_layout()
plt.savefig(conf_train["save_path"] + "/phase1_check.png", dpi=150)
plt.close()
print("Saved -> " + conf_train["save_path"] + "/phase1_check.png")
