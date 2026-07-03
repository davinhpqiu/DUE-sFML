"""
Standalone diagnostics for the trained OU sFML model.

Does NOT retrain anything. Loads the two networks already saved by OU.py:
    det_model/model            (deterministic sub-map D_theta, ResNet)
    gan_model/generator_final  (generator G_phi)
and produces two diagnostic curves that explain the flat-mean pathology:

  (A) Generator conditional mean and std vs state x.
      Expectation for OU: mean ~ 0 for ALL x, std ~ sigma*sqrt(dt).
      Pathology: a positive lift in the mean past the edge of data coverage.

  (B) Effective one-step drift of the FULL map  x -> D_theta(x) + G(x,z)
      compared with the true OU drift theta*(mu - x).
      The state where this crosses zero with negative slope is the learned
      map's STABLE fixed point -- i.e. the value the ensemble mean gets
      trapped at. It should sit near your observed ~1.51.

  Also prints D_theta's own fixed point (solve D_theta(x) = x).

Run from examples/OU/:
    python diagnose.py
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from yaml import safe_load
import due

# ----- config / normalization (mirrors OU.py) -----
conf_data, conf_net, conf_train = due.utils.read_config("config.yaml")
conf_net["seq_len"] = conf_data["seq_len"]
config_raw = safe_load(Path("config.yaml").read_text())

device     = conf_train["device"]
LATENT_DIM = conf_net["latent_dim"]
torch_dtype = torch.float64 if conf_data["dtype"] == "double" else torch.float32

# vmin/vmax computed exactly as in OU.py (joint min-max over all training states)
data_loader = due.datasets.sde.sde_dataset(conf_data)
_, _, _, vmin, vmax = data_loader.load("OU_train.mat", "OU_test.mat")
vmin_val = float(vmin.flatten()[0])
vmax_val = float(vmax.flatten()[0])
half_range = 0.5 * (vmax_val - vmin_val)   # increments scale by this factor

def normalize(x):    # physical -> [-1,1]
    return 2 * (x - 0.5 * (vmax_val + vmin_val)) / (vmax_val - vmin_val)
def denormalize(x):  # [-1,1] -> physical
    return x * half_range + 0.5 * (vmax_val + vmin_val)

# ----- OU truth -----
THETA, MU, SIGMA, DT = 1.0, 1.2, 0.3, 0.01
noise_std_phys = SIGMA * np.sqrt(DT)

# actual coverage of the training data (for annotating the plots)
raw = data_loader.load("OU_train.mat", "OU_test.mat")  # noqa (already loaded above)
import scipy.io as sio
train_states = sio.loadmat("OU_train.mat")["trajectories"].ravel()
cover_max = np.percentile(train_states, 99.9)
data_max  = train_states.max()

# ----- load trained networks (no retraining) -----
generator = torch.load(config_raw["gan"]["save_path"] + "/generator_final",
                       map_location=device, weights_only=False)
det_net   = torch.load(conf_train["save_path"] + "/model",
                       map_location=device, weights_only=False)
generator.eval(); det_net.eval()

# ----- sweep x over physical range, including the OOD test point 1.5 -----
x_phys = np.linspace(0.0, 1.65, 200)
x_norm = torch.tensor(normalize(x_phys)[:, None], dtype=torch_dtype, device=device)

M = 20000  # z-samples per x for stable statistics
gen_mean_phys = np.zeros_like(x_phys)
gen_std_phys  = np.zeros_like(x_phys)
eff_drift     = np.zeros_like(x_phys)   # E[ x_next - x ] / dt  for full map
d_incr_phys   = np.zeros_like(x_phys)   # D_theta one-step increment (physical)

with torch.no_grad():
    for i in range(len(x_phys)):
        xi = x_norm[i:i+1].expand(M, -1)                 # (M,1) normalized state
        z  = torch.randn(M, LATENT_DIM, dtype=torch_dtype, device=device)
        g  = generator(xi, z)[:, 0].cpu().numpy()        # normalized residual
        d  = det_net(x_norm[i:i+1])[0, 0].item()         # normalized D_theta(x)

        gen_mean_phys[i] = g.mean() * half_range         # residual mean (physical)
        gen_std_phys[i]  = g.std()  * half_range
        d_incr_phys[i]   = (d - x_norm[i, 0].item()) * half_range
        x_next_phys      = denormalize(d + g)            # full stochastic map
        eff_drift[i]     = (x_next_phys - x_phys[i]).mean() / DT

true_drift = THETA * (MU - x_phys)

# ----- D_theta fixed point (solve D_theta(x) = x) -----
d_fixed = x_phys[np.argmin(np.abs(d_incr_phys))]
# stable fixed point of the FULL mean map: eff_drift crosses 0 with negative slope
sign = np.sign(eff_drift)
crossings = np.where(np.diff(sign) < 0)[0]  # + -> - crossings (stable)
full_fp = [x_phys[k] for k in crossings]

print("=" * 62)
print("DIAGNOSTIC SUMMARY")
print("=" * 62)
print(f"data coverage: 99.9 pct = {cover_max:.3f}, max = {data_max:.3f}")
print(f"noise std (sigma*sqrt(dt)) = {noise_std_phys:.4f} (physical)\n")
print(f"D_theta fixed point  D(x)=x  at x = {d_fixed:.3f}   (true mu = {MU})")
print(f"full-map stable fixed point(s) (trapped mean) = "
      f"{[f'{v:.3f}' for v in full_fp]}\n")
print(" x     gen_mean   (xnoise)   true_drift  eff_drift")
for xq in (0.8, 1.2, 1.44, 1.5):
    j = int(np.argmin(np.abs(x_phys - xq)))
    print(f"{x_phys[j]:.2f}   {gen_mean_phys[j]:+.4f}   "
          f"{gen_mean_phys[j]/noise_std_phys:+.2f}x   "
          f"{true_drift[j]:+.4f}    {eff_drift[j]:+.4f}")

# ----- plots -----
fig, ax = plt.subplots(1, 3, figsize=(17, 5))

ax[0].axhline(0, color="k", lw=0.8)
ax[0].plot(x_phys, gen_mean_phys, "b-", label=r"$E_z[G(x,z)]$ (should be ~0)")
ax[0].axvline(data_max, color="gray", ls=":", label="data max")
ax[0].axvline(1.5, color="r", ls="--", label="test IC 1.5")
ax[0].set_xlabel("x"); ax[0].set_ylabel("generator mean (physical)")
ax[0].set_title("(A) Generator conditional mean vs x"); ax[0].legend()

ax[1].plot(x_phys, gen_std_phys, "b-", label="generator std")
ax[1].axhline(noise_std_phys, color="k", ls="--", label=r"$\sigma\sqrt{\Delta}$")
ax[1].axvline(data_max, color="gray", ls=":")
ax[1].set_xlabel("x"); ax[1].set_ylabel("generator std (physical)")
ax[1].set_title("(B) Generator std vs x"); ax[1].legend()

ax[2].axhline(0, color="k", lw=0.8)
ax[2].plot(x_phys, true_drift, "k-", label=r"true $\theta(\mu-x)$")
ax[2].plot(x_phys, eff_drift, "r-", label="effective drift (full map)")
ax[2].axvline(1.5, color="r", ls="--")
for v in full_fp:
    ax[2].plot(v, 0, "mo", ms=8)
ax[2].set_xlabel("x"); ax[2].set_ylabel("drift  f(x)")
ax[2].set_title("(C) Effective drift; magenta = trapped mean"); ax[2].legend()

plt.tight_layout()
out = config_raw["gan"]["save_path"] + "/diagnostics.png"
plt.savefig(out, dpi=150)
print(f"\nSaved -> {out}")
