"""
Checkpoint sweep — no retraining required.

Rolls out every k-th saved generator checkpoint and records the long-run mean/std.
Answers two questions at once:
  (1) Is there a training budget at which the FINAL model would have been fine?
      (i.e. does the paper's smaller effective budget explain the divergence?)
  (2) When does the extrapolation instability set in, and is it monotone or oscillatory?

The per-step training metrics CANNOT see this: they are evaluated on training states
(all at low x), while the instability lives at high x, outside the data.

Run from examples/OU/:   python sweep_checkpoints.py [config.yaml]
"""
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from yaml import safe_load

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
import due

CONFIG_PATH = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
conf_data, conf_net, conf_train = due.utils.read_config(CONFIG_PATH)
raw = safe_load(Path(CONFIG_PATH).read_text())
conf_gan = raw["gan"]

# ---- knobs ----
EVERY = 10          # evaluate every EVERY-th checkpoint (1000 total -> 100 points)
N_SAMPLES = 2_000   # rollout ensemble per checkpoint (enough for the mean/std trend)
N_STEPS = 400       # T = 4.0
THETA, MU, SIGMA, DT, X0 = 1.0, 1.2, 0.3, 0.01, 1.5

device = conf_train["device"]
torch_dtype = torch.float64 if raw["dtype"] == "double" else torch.float32
CENTER = bool(conf_gan.get("center_generator", False))
CENTER_K = int(conf_gan.get("center_K", 16))
LATENT = conf_net["latent_dim"]

data_loader = due.datasets.sde.sde_dataset(conf_data)
_, _, _, vmin, vmax = data_loader.load_sequence("OU_train.mat", "OU_test.mat")
NZ = data_loader.normalizer
det_net = torch.load(conf_train["save_path"] + "/model", map_location=device, weights_only=False)
det_net.eval()


def norm(x):
    a = np.asarray(x, dtype=np.float64)
    return NZ.transform(a.reshape(-1, 1)).reshape(a.shape)


def denorm(x):
    a = np.asarray(x, dtype=np.float64)
    return NZ.inverse(a.reshape(-1, 1)).reshape(a.shape)


def rollout(gen, n, steps):
    """Mean/std of the ensemble at the final time, mirroring the training-time centering."""
    x = torch.tensor(norm(np.array([[X0]])), dtype=torch_dtype).expand(n, -1).to(device)
    with torch.no_grad():
        for _ in range(steps):
            if not CENTER:
                z = torch.randn(n, LATENT, device=device, dtype=torch_dtype)
                r = gen(x, z)
            else:
                z = torch.randn(n, CENTER_K, LATENT, device=device, dtype=torch_dtype)
                xr = x.unsqueeze(1).expand(-1, CENTER_K, -1).reshape(n * CENTER_K, x.size(-1))
                out = gen(xr, z.reshape(n * CENTER_K, LATENT)).reshape(n, CENTER_K, -1)
                r = out[:, 0, :] - out.mean(dim=1)
            x = det_net(x) + r
            if not torch.isfinite(x).all():          # blown up
                return np.nan, np.nan
    final = denorm(x.cpu().numpy()[:, 0])
    return float(final.mean()), float(final.std())


ckpt_dir = Path(conf_gan["save_path"]) / "checkpoints"
files = sorted(ckpt_dir.glob("generator_epoch_*"),
               key=lambda p: int(re.findall(r"(\d+)$", p.stem)[0]))
files = files[EVERY - 1::EVERY]
print(f"sweeping {len(files)} checkpoints (every {EVERY}th of {len(list(ckpt_dir.glob('generator_epoch_*')))})")

eps, means, stds = [], [], []
for i, f in enumerate(files):
    ep = int(re.findall(r"(\d+)$", f.stem)[0])
    gen = torch.load(str(f), map_location=device, weights_only=False)
    gen.eval()
    m, s = rollout(gen, N_SAMPLES, N_STEPS)
    eps.append(ep); means.append(m); stds.append(s)
    if i % 10 == 0 or not np.isfinite(m):
        print(f"  ep {ep:6d}:  mean={m:8.3f}  std={s:8.3f}")

eps = np.array(eps); means = np.array(means); stds = np.array(stds)
np.savetxt(Path(conf_gan["save_path"]) / "checkpoint_sweep.csv",
           np.column_stack([eps, means, stds]), header="epoch rollout_mean rollout_std")

mean_true = MU + (X0 - MU) * np.exp(-THETA * N_STEPS * DT)
std_true = (SIGMA / np.sqrt(2 * THETA)) * np.sqrt(1 - np.exp(-2 * THETA * N_STEPS * DT))

fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
ax[0].axhline(mean_true, color="k", ls="-", label=f"true ({mean_true:.3f})")
ax[0].plot(eps, means, "r.-", ms=4, lw=1, label="sFML rollout mean")
ax[0].set_xlabel("training epoch"); ax[0].set_ylabel(f"mean at T={N_STEPS*DT}")
ax[0].set_title("Long-run mean vs training budget"); ax[0].legend()
ax[0].set_ylim(min(0.9, np.nanmin(means) * 0.95), max(1.5, np.nanpercentile(means[np.isfinite(means)], 95) * 1.1))

ax[1].axhline(std_true, color="k", ls="-", label=f"true ({std_true:.3f})")
ax[1].plot(eps, stds, "r.-", ms=4, lw=1, label="sFML rollout std")
ax[1].set_xlabel("training epoch"); ax[1].set_ylabel(f"std at T={N_STEPS*DT}")
ax[1].set_title("Long-run std vs training budget"); ax[1].legend()
ax[1].set_yscale("log")

plt.tight_layout()
out = Path(conf_gan["save_path"]) / "checkpoint_sweep.png"
plt.savefig(out, dpi=150)
print("\nSaved", out)

ok = np.isfinite(means) & (np.abs(means - mean_true) < 0.05)
if ok.any():
    print(f"epochs whose FINAL model would have been within 0.05 of the truth: "
          f"{eps[ok].min()}–{eps[ok].max()} ({ok.sum()}/{len(eps)} checkpoints)")
else:
    print("no checkpoint lands within 0.05 of the true mean")
