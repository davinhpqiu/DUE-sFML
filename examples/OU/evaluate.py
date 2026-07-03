
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

# Config
conf_data, conf_net, conf_train = due.utils.read_config("config.yaml")
config_raw = safe_load(Path("config.yaml").read_text())
conf_gan   = config_raw["gan"]

device      = conf_train["device"]
torch_dtype = torch.float64 if conf_data["dtype"] == "double" else torch.float32

# Reload training data to recover vmin/vmax (for normalize/denormalize)
data_loader = due.datasets.sde.sde_dataset(conf_data)
_, _, _, vmin, vmax = data_loader.load("OU_train.mat", "OU_test.mat")

vmin_val = float(vmin.flatten()[0])
vmax_val = float(vmax.flatten()[0])

def normalize(x):
    """Map physical state to normalized [-1, 1] space."""
    return 2 * (x - 0.5 * (vmax_val + vmin_val)) / (vmax_val - vmin_val)

def denormalize(x):
    """Map normalized [-1, 1] state back to physical space."""
    return x * 0.5 * (vmax_val - vmin_val) + 0.5 * (vmax_val + vmin_val)

# Load D_theta trained in Phase 1
det_net = torch.load(
    conf_train["save_path"] + "/model",
    map_location=device,
    weights_only=False  # required for PyTorch >= 2.6
)
det_net.eval()

# OU parameters
THETA   = 1.0
MU      = 1.2
DT      = 0.01
X0_TEST = 1.5
N_STEPS = 400 

t_arr     = np.arange(N_STEPS + 1) * DT
save_path = conf_gan["save_path"]

# Single-step: for 10 random x_0 values, roll out one step each and compare
# D_theta(x_0) to the analytical one-step mean E[x_1 | x_0] = x_0 + theta*(mu - x_0)*dt

np.random.seed(0)
# Sample 10 random starting points spread across the state space
x0_samples = np.random.uniform(-0.5, 2.0, size=10)

# Analytical one-step mean for each x_0
x1_true = x0_samples + THETA * (MU - x0_samples) * DT

# D_theta one-step prediction for each x_0
x0_norm = normalize(x0_samples[:, None]) # shape (10, 1), normalized
x0_tensor = torch.tensor(x0_norm, dtype=torch_dtype, device=device)

with torch.no_grad():
    x1_pred = det_net(x0_tensor).cpu().numpy()[:, 0]  # shape (10,), normalized

x1_pred = denormalize(x1_pred) # back to physical space

# Plot: one point per x_0, comparing predicted vs analytical next step
fig, ax = plt.subplots(figsize=(8, 5))
for i, (x0, true, pred) in enumerate(zip(x0_samples, x1_true, x1_pred)):
    ax.plot([x0], [true], 'ko', markersize=6) # analytical
    ax.plot([x0], [pred], 'b^', markersize=6) # D_theta
    ax.plot([x0, x0], [true, pred], 'r-', alpha=0.4, linewidth=1) # error line

# Dummy handles for legend
ax.plot([], [], 'ko', label='Analytical $E[x_1 | x_0]$')
ax.plot([], [], 'b^', label='$D_\\theta(x_0)$')
ax.plot([], [], 'r-', alpha=0.6, label='Error')

ax.set_xlabel('$x_0$')
ax.set_ylabel('$x_1$')
ax.set_title('Phase 1 check: single-step prediction for 10 random $x_0$')
ax.legend()
plt.tight_layout()
plt.savefig(save_path + "/det_single_step.png", dpi=150)
plt.close()
print("Saved", save_path + "/det_single_step.png")
