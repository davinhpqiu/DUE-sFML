"""
Stochastic Flow Map Learning — Ornstein-Uhlenbeck Process

Follows Section 5.1.1 of:
  Chen & Xiu (2024), "Learning stochastic flow map from data",
  J. Comput. Phys., 514, 113218.

Two-phase training:
  Phase 1: Train deterministic sub-map D_theta (ResNet) via MSE.
  Phase 2: Train Generator G_phi and Critic C_psi (WGAN-GP) on residuals.

Run from examples/OU/ directory:
  python OU.py
"""

import numpy as np
import scipy.io as sio
import torch
import matplotlib.pyplot as plt
from yaml import safe_load
from pathlib import Path
import due

# Config

conf_data, conf_net, conf_train = due.utils.read_config("config.yaml")

# Build Phase 2 config by extending 'gan' section global fields

config_raw = safe_load(Path("config.yaml").read_text())
conf_gan = config_raw["gan"]
conf_gan["seed"]       = config_raw["seed"]
conf_gan["dtype"]      = config_raw["dtype"]
conf_gan["device"]     = conf_train["device"]
conf_gan["latent_dim"] = conf_net["latent_dim"]

# Propagate seq_len into conf_net so the Critic knows its input dimension
conf_net["seq_len"] = conf_data["seq_len"]

# Data
# Call sde_dataset only to obtain vmin/vmax (joint min-max over all states)
# and the raw test trajectories. We do NOT use the pair-extracted trainX/trainY.
data_loader = due.datasets.sde.sde_dataset(conf_data)
_, _, test_data, vmin, vmax = data_loader.load(
    "OU_train.mat", "OU_test.mat"
)
# test_data: shape (N_test, d, T_test+1), raw (unnormalized)

# Load raw training trajectories and normalise to [-1, 1] using the same
# vmin/vmax computed above. Shape: (N, d, L+1).
raw_seqs   = sio.loadmat("OU_train.mat")["trajectories"]          # (N, d, L+1), raw
# vmin/vmax are (1, d); broadcast over the time axis with [:, :, None]
train_seqs = (2 * (raw_seqs - 0.5 * (vmax[:, :, None] + vmin[:, :, None]))
              / (vmax[:, :, None] - vmin[:, :, None])).astype(np.float32)
# train_seqs: (N, d, L+1) normalised — used for both Phase 1 and Phase 2

# Phase 1 — Deterministic Sub-map D_theta (ResNet), multi-step rollout loss
#
# The paper's loss (eq. 4.11) rolls D_theta from x_0^(i) for L steps and
# minimises sum_{n=1}^{L} ||x_n^(i) - D_theta^[n](x_0^(i))||^2.
# This trains the network to stay accurate over multi-step predictions
# (the "recurrent structure" shown in Figure 2 of the paper).
#
# trainX_p1: initial conditions x_0, shape (N, d)
# trainY_p1: full trajectory targets (x_1,...,x_L), shape (N, d, L)
trainX_p1 = train_seqs[:, :, 0]    # (N, d)
trainY_p1  = train_seqs[:, :, 1:]  # (N, d, L) — ODE model sees multi_steps = L = 40

det_net = due.networks.fcn.resnet(vmin, vmax, conf_net)
phase1_model = due.models.ODE(trainX_p1, trainY_p1, det_net, conf_train)
phase1_model.train()
phase1_model.save_hist()
# det_net is frozen inside SDE.__init__

generator = due.networks.gan.Generator(conf_net)
critic    = due.networks.gan.Critic(conf_net)   # input_dim = d*(1+L) = 41 for OU

sde_model = due.models.SDE(train_seqs, det_net, generator, critic, conf_gan)
sde_model.train()
sde_model.save_hist()

# Evaluation

device = conf_train["device"]
generator = torch.load(conf_gan["save_path"] + "/generator_final", map_location=device, weights_only=False)
det_net   = torch.load(conf_train["save_path"] + "/model",         map_location=device, weights_only=False)
generator.eval()
det_net.eval()

# OU analytical parameters (from config / paper)
THETA = 1.0
MU    = 1.2
SIGMA = 0.3
DT    = 0.01
X0_TEST   = 1.5
N_SAMPLES = 100_000 # ensemble size (paper value; 10,000 for fast check)
N_STEPS   = 400 # T=4.0 / DT=0.01
LATENT_DIM = conf_net["latent_dim"]

vmin_val = float(vmin.flatten()[0])
vmax_val = float(vmax.flatten()[0])
torch_dtype = torch.float64 if conf_data["dtype"] == "double" else torch.float32


def normalize(x):
    return 2 * (x - 0.5 * (vmax_val + vmin_val)) / (vmax_val - vmin_val)

def denormalize(x):
    return x * 0.5 * (vmax_val - vmin_val) + 0.5 * (vmax_val + vmin_val)


def stochastic_predict(x0_raw, n_samples, n_steps):
    """
    Run n_samples independent trajectories from x0_raw for n_steps.

    Returns:
        numpy array of shape (n_samples, n_steps+1) — unnormalized state values.
    """
    x0_norm = normalize(np.array([[x0_raw]], dtype=np.float64))
    x = torch.tensor(x0_norm, dtype=torch_dtype).expand(n_samples, -1).to(device)

    traj = [denormalize(x.cpu().numpy()[:, 0])]

    with torch.no_grad():
        for _ in range(n_steps):
            z      = torch.randn(n_samples, LATENT_DIM, device=device, dtype=torch_dtype)
            r_fake = generator(x, z)
            x      = det_net(x) + r_fake
            traj.append(denormalize(x.cpu().numpy()[:, 0]))

    return np.stack(traj, axis=1)  # (n_samples, n_steps+1)


# generator residual statistics 
# check generator output has mean ~0 and std ~ SIGMA * sqrt(DT) / (0.5*(vmax-vmin)) in normalized space
with torch.no_grad():
    x_test = torch.zeros(10000, 1, dtype=torch_dtype, device=device)
    z_test = torch.randn(10000, LATENT_DIM, dtype=torch_dtype, device=device)
    r_test = generator(x_test, z_test).cpu().numpy()
print(f"\nGenerator residual check (normalized space, x=0):")
print(f"  mean = {r_test.mean():.6f}  (expected ~0)")
print(f"  std  = {r_test.std():.6f}   (expected ~{0.3 * (0.01**0.5) / (0.5*(vmax_val-vmin_val)):.4f})")

print("\nRunning stochastic prediction ...")
ensemble = stochastic_predict(X0_TEST, N_SAMPLES, N_STEPS)
# ensemble: (N_SAMPLES, N_STEPS+1)

t_arr = np.arange(N_STEPS + 1) * DT
save_path = conf_gan["save_path"]


# Phase 1 diagnostic: deterministic rollout vs analytical mean

det_traj = [X0_TEST]

# Normalized [-1,1] space; shape (1,1) for the network
x_det = torch.tensor(normalize(np.array([[X0_TEST]])), dtype=torch_dtype, device=device)

with torch.no_grad(): # no gradients needed
    for _ in range(N_STEPS):
        x_det = det_net(x_det) # one-step: x_{n+1} = D_theta(x_n)
        det_traj.append(float(denormalize(x_det.cpu().numpy()[0, 0]))) # convert back to physical space, store

# Analytical OU mean starting from X0_TEST: E[x(t)] = (x0 - mu)*exp(-theta*t) + mu
mean_true_det = (X0_TEST - MU) * np.exp(-THETA * t_arr) + MU

plt.figure(figsize=(8, 5))
plt.plot(t_arr, mean_true_det, 'k-',  label='Analytical mean')
plt.plot(t_arr, det_traj,      'b--', label='D_θ rollout')
plt.xlabel('t')
plt.ylabel('x')
plt.title('Phase 1: D_θ vs analytical mean')
plt.legend()
plt.tight_layout()
plt.savefig(save_path + "/det_rollout.png", dpi=150)   # save to gan_model
plt.close()
print("Saved det_rollout.png")


#  Fig 5: Mean and Standard Deviation
mean_pred = ensemble.mean(axis=0)
std_pred  = ensemble.std(axis=0)
mean_true = (X0_TEST - MU) * np.exp(-THETA * t_arr) + MU
std_true  = (SIGMA / np.sqrt(2 * THETA)) * np.sqrt(1 - np.exp(-2 * THETA * t_arr))

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].plot(t_arr, mean_true, 'k-',  label='Analytical')
axes[0].plot(t_arr, mean_pred, 'r--', label='sFML')
axes[0].set_xlabel('t'); axes[0].set_ylabel('Mean'); axes[0].legend()
axes[0].set_title('Mean')

axes[1].plot(t_arr, std_true, 'k-',  label='Analytical')
axes[1].plot(t_arr, std_pred, 'r--', label='sFML')
axes[1].set_xlabel('t'); axes[1].set_ylabel('Std'); axes[1].legend()
axes[1].set_title('Standard Deviation')

plt.tight_layout()
plt.savefig(save_path + "/mean_std.png", dpi=150)
plt.close()
print("Saved mean_std.png")

# Fig 4: Sample trajectories — training data (left) vs sFML model (right)
t_train = np.arange(raw_seqs.shape[2]) * DT
n_show  = 50
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for i in range(min(n_show, raw_seqs.shape[0])):
    axes[0].plot(t_train, raw_seqs[i, 0, :], lw=0.7, alpha=0.6)
axes[0].set_xlabel('t'); axes[0].set_ylabel('x')
axes[0].set_title('Training data samples')
for i in range(min(n_show, ensemble.shape[0])):
    axes[1].plot(t_arr, ensemble[i, :], lw=0.7, alpha=0.6)
axes[1].set_xlabel('t'); axes[1].set_ylabel('x')
axes[1].set_title(f'sFML model samples ($x_0={X0_TEST}$, T={N_STEPS * DT:.0f})')
plt.tight_layout()
plt.savefig(save_path + "/trajectories.png", dpi=150)
plt.close()
print("Saved trajectories.png")

# Fig 6: Effective Drift and Diffusion Recovery
# Estimate from the ensemble at each state value using binning
x_vals  = ensemble[:, :-1].flatten()
dx_vals = (ensemble[:, 1:] - ensemble[:, :-1]).flatten()

# Bin by x value
n_bins = 50
x_edges = np.linspace(x_vals.min(), x_vals.max(), n_bins + 1)
x_centers  = 0.5 * (x_edges[:-1] + x_edges[1:])
drift_pred = np.zeros(n_bins)
diff_pred  = np.zeros(n_bins)

for i in range(n_bins):
    mask = (x_vals >= x_edges[i]) & (x_vals < x_edges[i+1])
    if mask.sum() > 10:
        drift_pred[i] = dx_vals[mask].mean() / DT
        diff_pred[i]  = np.sqrt(dx_vals[mask].var() / DT)

drift_true = THETA * (MU - x_centers)
diff_true  = SIGMA * np.ones_like(x_centers)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].plot(x_centers, drift_true, 'k-',  label='Analytical')
axes[0].plot(x_centers, drift_pred, 'r.', label='sFML (binned)')
axes[0].set_xlabel('x'); axes[0].set_ylabel('Drift f(x)'); axes[0].legend()
axes[0].set_title('Effective Drift')

axes[1].plot(x_centers, diff_true, 'k-',  label='Analytical')
axes[1].plot(x_centers, diff_pred, 'r.', label='sFML (binned)')
axes[1].set_xlabel('x'); axes[1].set_ylabel('Diffusion g(x)'); axes[1].legend()
axes[1].set_title('Effective Diffusion')

plt.tight_layout()
plt.savefig(save_path + "/drift_diffusion.png", dpi=150)
plt.close()
print("Saved drift_diffusion.png")

# Fig 7: Conditional Distribution at x = 0.8
X_COND = 0.8
cond_samples = stochastic_predict(X_COND, N_SAMPLES, 1)[:, 1]  # one step from x=0.8

mean_cond_true = (1 - THETA * DT) * X_COND + THETA * MU * DT
std_cond_true  = SIGMA * np.sqrt(DT)
x_plot = np.linspace(cond_samples.min() - 0.01, cond_samples.max() + 0.01, 200)
pdf_true = (1 / (std_cond_true * np.sqrt(2 * np.pi))) * np.exp(
    -0.5 * ((x_plot - mean_cond_true) / std_cond_true) ** 2
)

fig, ax = plt.subplots(figsize=(7, 5))
ax.hist(cond_samples, bins=80, density=True, alpha=0.6, label='sFML samples')
ax.plot(x_plot, pdf_true, 'k-', linewidth=2, label='Analytical')
ax.set_xlabel('$x_{n+1}$')
ax.set_ylabel('Density')
ax.set_title(f'Conditional distribution at $x_n = {X_COND}$')
ax.legend()
plt.tight_layout()
plt.savefig(save_path + "/conditional_dist.png", dpi=150)
plt.close()
print("Saved conditional_dist.png")

# Fig 8: Covariance-matrix spectra (paper's Fig 8)
# Eigenvalue spectrum of the covariance matrix of the length-(N_STEPS+1) solution
# sequence: Prediction (sFML from x0=1.5) vs Ground Truth (test data from x0=1.5).
pred_seq  = ensemble                        # (M, N_STEPS+1)
truth_seq = test_data[:, 0, :]              # (N_test, N_STEPS+1) ground-truth OU paths
cov_pred  = np.cov(pred_seq,  rowvar=False)
cov_truth = np.cov(truth_seq, rowvar=False)
eig_pred  = np.clip(np.sort(np.linalg.eigvalsh(cov_pred))[::-1],  1e-16, None)
eig_truth = np.clip(np.sort(np.linalg.eigvalsh(cov_truth))[::-1], 1e-16, None)
k = np.arange(1, len(eig_pred) + 1)

fig, ax = plt.subplots(figsize=(7, 5))
ax.semilogy(k, eig_truth, 'k-',  label='Ground Truth')
ax.semilogy(k, eig_pred,  'r--', label='Prediction')
ax.set_xlabel('Index')
ax.set_ylabel('Eigenvalue')
ax.set_title('Covariance matrix spectra')
ax.legend()
plt.tight_layout()
plt.savefig(save_path + "/covariance_spectra.png", dpi=150)
plt.close()
print("Saved covariance_spectra.png (eigenvalue spectrum)")

# Bonus (not a paper figure): covariance function C(tau) vs analytical, from x0=1.5
mu_stat = MU
burnin  = N_STEPS // 2
X_stat  = ensemble[:, burnin:]              # near-stationary tail
T_cov   = X_stat.shape[1]
C_pred  = np.array([
    np.mean((X_stat[:, :T_cov - lag] - mu_stat) * (X_stat[:, lag:] - mu_stat))
    for lag in range(min(T_cov, 100))
])
tau    = np.arange(len(C_pred)) * DT
C_true = (SIGMA ** 2 / (2 * THETA)) * np.exp(-THETA * tau)
fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(tau, C_true, 'k-',  label='Analytical')
ax.plot(tau, C_pred, 'r--', label='sFML')
ax.set_xlabel('Lag $\\tau$')
ax.set_ylabel('Covariance $C(\\tau)$')
ax.set_title('Covariance function')
ax.legend()
plt.tight_layout()
plt.savefig(save_path + "/covariance_function.png", dpi=150)
plt.close()
print("Saved covariance_function.png")

print("\nAll evaluation figures saved to", save_path)
