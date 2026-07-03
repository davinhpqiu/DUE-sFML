"""
Stochastic Flow Map Learning — Geometric Brownian Motion (sFML paper §5.1.2)

    dx = mu*x dt + sigma*x dW,   mu = 2.0, sigma = 1.0   (multiplicative noise)

Same two-phase pipeline as the OU example; only the SDE and the analytical
reference formulas change. Run from examples/GBM/:  python GBM.py
"""
import numpy as np
import scipy.io as sio
import torch
import matplotlib.pyplot as plt
from yaml import safe_load
from pathlib import Path
import due

# ---- config ----
conf_data, conf_net, conf_train = due.utils.read_config("config.yaml")
config_raw = safe_load(Path("config.yaml").read_text())
conf_gan = config_raw["gan"]
conf_gan["seed"]       = config_raw["seed"]
conf_gan["dtype"]      = config_raw["dtype"]
conf_gan["device"]     = conf_train["device"]
conf_gan["latent_dim"] = conf_net["latent_dim"]
conf_net["seq_len"]    = conf_data["seq_len"]

# ---- data (vmin/vmax + raw test) ----
data_loader = due.datasets.sde.sde_dataset(conf_data)
_, _, test_data, vmin, vmax = data_loader.load("GBM_train.mat", "GBM_test.mat")

raw_seqs   = sio.loadmat("GBM_train.mat")["trajectories"]
train_seqs = (2 * (raw_seqs - 0.5 * (vmax[:, :, None] + vmin[:, :, None]))
              / (vmax[:, :, None] - vmin[:, :, None])).astype(np.float32)

# ---- Phase 1: deterministic sub-map (multi-step, eq 4.11) ----
trainX_p1 = train_seqs[:, :, 0]
trainY_p1 = train_seqs[:, :, 1:]
det_net = due.networks.fcn.resnet(vmin, vmax, conf_net)
phase1_model = due.models.ODE(trainX_p1, trainY_p1, det_net, conf_train)
phase1_model.train()
phase1_model.save_hist()

# ---- Phase 2: WGAN-GP ----
generator = due.networks.gan.Generator(conf_net)
critic    = due.networks.gan.Critic(conf_net)
sde_model = due.models.SDE(train_seqs, det_net, generator, critic, conf_gan)
sde_model.train()
sde_model.save_hist()

# ---- Evaluation ----
device = conf_train["device"]
generator = torch.load(conf_gan["save_path"] + "/generator_final", map_location=device, weights_only=False)
det_net   = torch.load(conf_train["save_path"] + "/model",         map_location=device, weights_only=False)
generator.eval(); det_net.eval()

MU, SIGMA, DT = 2.0, 1.0, 0.01
X0_TEST   = 0.5
N_SAMPLES = 100_000
N_STEPS   = 100          # T = 1.0
X_COND    = 6.0          # state for the conditional-distribution figure
LATENT_DIM = conf_net["latent_dim"]
torch_dtype = torch.float64 if conf_data["dtype"] == "double" else torch.float32

vmin_val = float(vmin.flatten()[0]); vmax_val = float(vmax.flatten()[0])
def normalize(x):    return 2 * (x - 0.5 * (vmax_val + vmin_val)) / (vmax_val - vmin_val)
def denormalize(x):  return x * 0.5 * (vmax_val - vmin_val) + 0.5 * (vmax_val + vmin_val)

def stochastic_predict(x0_raw, n_samples, n_steps):
    x0_norm = normalize(np.array([[x0_raw]], dtype=np.float64))
    x = torch.tensor(x0_norm, dtype=torch_dtype).expand(n_samples, -1).to(device)
    traj = [denormalize(x.cpu().numpy()[:, 0])]
    with torch.no_grad():
        for _ in range(n_steps):
            z = torch.randn(n_samples, LATENT_DIM, device=device, dtype=torch_dtype)
            x = det_net(x) + generator(x, z)
            traj.append(denormalize(x.cpu().numpy()[:, 0]))
    return np.stack(traj, axis=1)

print("\nRunning stochastic prediction ...")
ensemble = stochastic_predict(X0_TEST, N_SAMPLES, N_STEPS)
t_arr = np.arange(N_STEPS + 1) * DT
save_path = conf_gan["save_path"]

# Phase-1 diagnostic: D_theta rollout vs analytical mean
det_traj = [X0_TEST]
x_det = torch.tensor(normalize(np.array([[X0_TEST]])), dtype=torch_dtype, device=device)
with torch.no_grad():
    for _ in range(N_STEPS):
        x_det = det_net(x_det)
        det_traj.append(float(denormalize(x_det.cpu().numpy()[0, 0])))
mean_true_det = X0_TEST * np.exp(MU * t_arr)
plt.figure(figsize=(8, 5))
plt.plot(t_arr, mean_true_det, 'k-',  label='Analytical mean')
plt.plot(t_arr, det_traj,      'b--', label='D_theta rollout')
plt.xlabel('t'); plt.ylabel('x'); plt.title('Phase 1: D_theta vs analytical mean'); plt.legend()
plt.tight_layout(); plt.savefig(save_path + "/det_rollout.png", dpi=150); plt.close()
print("Saved det_rollout.png")

# Fig 4: trajectories
t_train = np.arange(raw_seqs.shape[2]) * DT
n_show = 50
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for i in range(min(n_show, raw_seqs.shape[0])):
    axes[0].plot(t_train, raw_seqs[i, 0, :], lw=0.7, alpha=0.6)
axes[0].set_xlabel('t'); axes[0].set_ylabel('x'); axes[0].set_title('Training data samples')
for i in range(min(n_show, ensemble.shape[0])):
    axes[1].plot(t_arr, ensemble[i, :], lw=0.7, alpha=0.6)
axes[1].set_xlabel('t'); axes[1].set_ylabel('x')
axes[1].set_title(f'sFML model samples ($x_0={X0_TEST}$, T={N_STEPS*DT:.0f})')
plt.tight_layout(); plt.savefig(save_path + "/trajectories.png", dpi=150); plt.close()
print("Saved trajectories.png")

# Fig 5: mean and std (analytical GBM moments)
mean_pred = ensemble.mean(axis=0); std_pred = ensemble.std(axis=0)
mean_true = X0_TEST * np.exp(MU * t_arr)
std_true  = X0_TEST * np.exp(MU * t_arr) * np.sqrt(np.exp(SIGMA**2 * t_arr) - 1)
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].plot(t_arr, mean_true, 'k-', label='Analytical'); axes[0].plot(t_arr, mean_pred, 'r--', label='sFML')
axes[0].set_xlabel('t'); axes[0].set_ylabel('Mean'); axes[0].set_title('Mean'); axes[0].legend()
axes[1].plot(t_arr, std_true, 'k-', label='Analytical'); axes[1].plot(t_arr, std_pred, 'r--', label='sFML')
axes[1].set_xlabel('t'); axes[1].set_ylabel('Std'); axes[1].set_title('Standard Deviation'); axes[1].legend()
plt.tight_layout(); plt.savefig(save_path + "/mean_std.png", dpi=150); plt.close()
print("Saved mean_std.png")

# Fig 6: effective drift a(x)=mu*x and diffusion b(x)=sigma*x
x_vals  = ensemble[:, :-1].flatten()
dx_vals = (ensemble[:, 1:] - ensemble[:, :-1]).flatten()
n_bins = 50
x_edges = np.linspace(np.percentile(x_vals, 0.5), np.percentile(x_vals, 99.5), n_bins + 1)
x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
drift_pred = np.zeros(n_bins); diff_pred = np.zeros(n_bins)
for i in range(n_bins):
    mask = (x_vals >= x_edges[i]) & (x_vals < x_edges[i+1])
    if mask.sum() > 10:
        drift_pred[i] = dx_vals[mask].mean() / DT
        diff_pred[i]  = np.sqrt(dx_vals[mask].var() / DT)
drift_true = MU * x_centers
diff_true  = SIGMA * x_centers
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].plot(x_centers, drift_true, 'k-', label='Analytical'); axes[0].plot(x_centers, drift_pred, 'r.', label='sFML (binned)')
axes[0].set_xlabel('x'); axes[0].set_ylabel('Drift f(x)'); axes[0].set_title('Effective Drift  a(x)=mu*x'); axes[0].legend()
axes[1].plot(x_centers, diff_true, 'k-', label='Analytical'); axes[1].plot(x_centers, diff_pred, 'r.', label='sFML (binned)')
axes[1].set_xlabel('x'); axes[1].set_ylabel('Diffusion g(x)'); axes[1].set_title('Effective Diffusion  b(x)=sigma*x'); axes[1].legend()
plt.tight_layout(); plt.savefig(save_path + "/drift_diffusion.png", dpi=150); plt.close()
print("Saved drift_diffusion.png")

# Fig 7: conditional distribution at x = X_COND (Euler-Maruyama one-step law is Gaussian)
cond_samples = stochastic_predict(X_COND, N_SAMPLES, 1)[:, 1]
mean_cond_true = X_COND * (1 + MU * DT)
std_cond_true  = SIGMA * X_COND * np.sqrt(DT)
x_plot = np.linspace(cond_samples.min() - 0.05, cond_samples.max() + 0.05, 300)
pdf_true = (1 / (std_cond_true * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x_plot - mean_cond_true) / std_cond_true) ** 2)
fig, ax = plt.subplots(figsize=(7, 5))
ax.hist(cond_samples, bins=80, density=True, alpha=0.6, label='sFML samples')
ax.plot(x_plot, pdf_true, 'k-', lw=2, label='Analytical (EM one-step)')
ax.set_xlabel('$x_{n+1}$'); ax.set_ylabel('Density')
ax.set_title(f'Conditional distribution at $x_n = {X_COND}$'); ax.legend()
plt.tight_layout(); plt.savefig(save_path + "/conditional_dist.png", dpi=150); plt.close()
print("Saved conditional_dist.png")

# Fig 8: covariance-matrix spectra (Prediction vs Ground Truth from test data)
pred_seq  = ensemble
truth_seq = test_data[:, 0, :]
eig_pred  = np.clip(np.sort(np.linalg.eigvalsh(np.cov(pred_seq,  rowvar=False)))[::-1], 1e-16, None)
eig_truth = np.clip(np.sort(np.linalg.eigvalsh(np.cov(truth_seq, rowvar=False)))[::-1], 1e-16, None)
k = np.arange(1, len(eig_pred) + 1)
fig, ax = plt.subplots(figsize=(7, 5))
ax.semilogy(k, eig_truth, 'k-', label='Ground Truth'); ax.semilogy(k, eig_pred, 'r--', label='Prediction')
ax.set_xlabel('Index'); ax.set_ylabel('Eigenvalue'); ax.set_title('Covariance matrix spectra'); ax.legend()
plt.tight_layout(); plt.savefig(save_path + "/covariance_spectra.png", dpi=150); plt.close()
print("Saved covariance_spectra.png")

print("\nAll evaluation figures saved to", save_path)
