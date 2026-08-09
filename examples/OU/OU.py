"""
Stochastic Flow Map Learning — Ornstein-Uhlenbeck Process

Follows Section 5.1.1 of:
  Chen & Xiu (2024), "Learning stochastic dynamical system via flow map operator",
  J. Comput. Phys., 508, 112984.

Two-phase training:
  Phase 1: Train deterministic sub-map D_theta (ResNet) via MSE.
  Phase 2: Train S_delta and Critic C_psi (WGAN-GP) on full increment
           sequences (x0, y1:L), following Algorithm 4.1.

Run from examples/OU/ directory:
  python OU.py
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

# Config  (optional path arg:  python OU.py [config.yaml])

CONFIG_PATH = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
print(f">>> Using config: {CONFIG_PATH}")
conf_data, conf_net, conf_train = due.utils.read_config(CONFIG_PATH)

# Build Phase 2 config by extending 'gan' section global fields

config_raw = safe_load(Path(CONFIG_PATH).read_text())
conf_gan = config_raw["gan"]
conf_gan["seed"] = config_raw["seed"]
conf_gan["dtype"] = config_raw["dtype"]
conf_gan["device"] = conf_train["device"]
conf_gan["latent_dim"] = conf_net["latent_dim"]

# Data

data_loader = due.datasets.sde.sde_dataset(conf_data)
trainX, trainY, test_data, vmin, vmax = data_loader.load_sequence(
    "OU_train.mat", "OU_test.mat"
)
# trainX: shape (N, d), normalized x0
# trainY: shape (N, d, L), normalized x1,...,xL
# test_data: shape (N_test, d, T_test+1), raw (unnormalized)

conf_net["sequence_length"] = trainY.shape[-1]

# Phase 1 — Deterministic Sub-map D_theta (ResNet)

import os as _os

class OracleDet(torch.nn.Module):
    """Phase-1 ABLATION: exact OU Euler conditional-mean map
    D(x) = x + theta*(mu - x)*dt  (fixed point = mu, linear -> extrapolates exactly).
    Operates in NORMALISED space: forward(u) = normalize(D(denormalize(u)))."""
    def __init__(self, vmin, vmax, theta=1.0, mu=1.2, dt=0.01):
        super().__init__()
        c = 0.5 * (float(vmax) + float(vmin))
        h = 0.5 * (float(vmax) - float(vmin))
        self.register_buffer("c", torch.tensor(c))
        self.register_buffer("h", torch.tensor(h))
        self.theta, self.mu, self.dt = theta, mu, dt
    def forward(self, u):
        x = u * self.h + self.c
        y = x + self.theta * (self.mu - x) * self.dt
        return (y - self.c) / self.h
    def count_params(self):
        return 0

# ---- eval_only mode ----
EVAL_ONLY = bool(config_raw.get("eval_only", False))
if EVAL_ONLY:
    print(">>> eval_only=true: skipping Phase 1 and Phase 2 training, loading saved models.")
else:
    if conf_train.get("use_oracle_det", False):
        print(">>> Phase-1 ABLATION: D_theta replaced by the EXACT OU oracle (no Phase-1 training)")
        # OracleDet applies (de)norm internally via h/c. For normalization="none" the
        # network receives physical x directly, so we need h=1, c=0 (pass vmin=-1, vmax=1).
        _norm_type = conf_data.get("normalization", "none")
        if _norm_type == "none":
            _ov, _ov2 = -1.0, 1.0   # → h=1, c=0: forward(u) = u + θ(μ-u)dt
        else:
            _ov, _ov2 = float(vmin.flatten()[0]), float(vmax.flatten()[0])
        det_net = OracleDet(_ov, _ov2)
        _os.makedirs(conf_train["save_path"], exist_ok=True)
        torch.save(det_net, conf_train["save_path"] + "/model")
    else:
        _arch = conf_net.get("det_arch", "resnet")
        print(f"Phase-1 architecture: {_arch}")
        det_net = getattr(due.networks.fcn, _arch)(vmin, vmax, conf_net)
        phase1_model = due.models.ODE(trainX, trainY, det_net, conf_train)
        phase1_model.train()
        phase1_model.save_hist()

    # Use the best deterministic map saved during Phase 1, then freeze it in SDE.__init__.
    det_net = torch.load(
        conf_train["save_path"] + "/model",
        map_location=conf_train["device"],
        weights_only=False,
    )

    # Phase 2 — WGAN-GP (Generator + Critic)
    # When single_step_critic is on, the critic sees (x_t, Δx_t) pairs — input dim = 2d.
    # Override sequence_length to 1 so the Critic is sized correctly.
    if conf_gan.get("single_step_critic", False):
        conf_net["sequence_length"] = 1
    generator = due.networks.gan.Generator(conf_net)
    critic    = due.networks.gan.Critic(conf_net)

    sde_model = due.models.SDE(trainX, trainY, det_net, generator, critic, conf_gan)
    sde_model.train()
    sde_model.save_hist()

# Evaluation

device = conf_train["device"]
det_net = torch.load(conf_train["save_path"] + "/model", map_location=device, weights_only=False)
det_net.eval()

# Which trained model(s) to evaluate. The trainer ALWAYS saves both:
#   "best"  = checkpoint chosen by the selection score (our pick)
#   "final" = model at the last epoch, no selection (what the paper uses)
# eval_model: "best" | "final" | "both" (default). Each writes to gan_model/eval_<tag>/.
_em = str(conf_gan.get("eval_model", "both")).lower()
EVAL_TAGS = {"best": ["best"], "final": ["final"], "both": ["best", "final"]}.get(_em, ["best", "final"])
generator = None  # assigned per-tag in the eval loop below (global, read by gen_increment)

# OU analytical parameters (from config / paper)
THETA = 1.0
MU    = 1.2
SIGMA = 0.3
DT    = 0.01
X0_TEST   = 1.5
N_SAMPLES = 100_000 # paper §5.1.1: statistics averaged over 100,000 simulation samples
N_STEPS   = 400 # T=4.0 / DT=0.01
LATENT_DIM = conf_net["latent_dim"]

vmin_val = float(vmin.flatten()[0])
vmax_val = float(vmax.flatten()[0])
torch_dtype = torch.float64 if conf_data["dtype"] == "double" else torch.float32

# Hard-centering toggle (must MIRROR the training setting in config.yaml so the
# model is evaluated the same way it was trained).
CENTER_GEN = bool(conf_gan.get("center_generator", False))
CENTER_K   = int(conf_gan.get("center_K", 16))
print(f"Evaluation hard-centering: {'ON (K=%d)' % CENTER_K if CENTER_GEN else 'OFF'}")


# Route through the fitted normaliser (identical to the old min-max formula for
# normalization="minmax"; enables "none"/raw and "yeojohnson" via config).
NZ = data_loader.normalizer

def normalize(x):
    a = np.asarray(x, dtype=np.float64)
    return NZ.transform(a.reshape(-1, 1)).reshape(a.shape)

def denormalize(x):
    a = np.asarray(x, dtype=np.float64)
    return NZ.inverse(a.reshape(-1, 1)).reshape(a.shape)


def gen_increment(x, n_samples):
    """Stochastic sub-map draw S_delta(x, z), hard-centered per state when CENTER_GEN."""
    if not CENTER_GEN:
        z = torch.randn(n_samples, LATENT_DIM, device=device, dtype=torch_dtype)
        return generator(x, z)
    z = torch.randn(n_samples, CENTER_K, LATENT_DIM, device=device, dtype=torch_dtype)
    x_rep = x.unsqueeze(1).expand(-1, CENTER_K, -1).reshape(n_samples * CENTER_K, x.size(-1))
    out = generator(x_rep, z.reshape(n_samples * CENTER_K, LATENT_DIM))
    out = out.reshape(n_samples, CENTER_K, -1)
    return out[:, 0, :] - out.mean(dim=1)


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
            r_fake = gen_increment(x, n_samples)
            x      = det_net(x) + r_fake
            traj.append(denormalize(x.cpu().numpy()[:, 0]))

    return np.stack(traj, axis=1)  # (n_samples, n_steps+1)


def run_eval(save_path):
    # generator residual statistics 
    # check generator output has mean ~0 and std ~ SIGMA * sqrt(DT) / (0.5*(vmax-vmin)) in normalized space
    with torch.no_grad():
        x_test = torch.zeros(10000, 1, dtype=torch_dtype, device=device)
        r_test = gen_increment(x_test, 10000).cpu().numpy()
    print(f"\nGenerator residual check (normalized space, x=0, centering={'ON' if CENTER_GEN else 'OFF'}):")
    print(f"  mean = {r_test.mean():.6f}  (expected ~0)")
    print(f"  std  = {r_test.std():.6f}   (expected ~{0.3 * (0.01**0.5) / (0.5*(vmax_val-vmin_val)):.4f})")

    print("\nRunning stochastic prediction ...")
    ensemble = stochastic_predict(X0_TEST, N_SAMPLES, N_STEPS)
    # ensemble: (N_SAMPLES, N_STEPS+1)

    t_arr = np.arange(N_STEPS + 1) * DT


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

    # Fig 6: Effective Drift and Diffusion Recovery
    # Paper eq. (4.23): for each x on a fixed grid, draw N_z fresh z samples and compute
    #   â(x) = E_z[G̃(x,z) - x] / Δ
    #   b̂(x) = Std_z[G̃(x,z)] / √Δ
    # This avoids trajectory binning artifacts (sparse/correlated samples at tail states).
    x_grid = np.linspace(0.5, 2.0, 200)   # densified grid (professor: finer resolution)
    N_Z = 1_000_000                        # MC z draws per grid point (professor: 1e6 -> ~7x tighter std estimate than 20k)
    N_CHUNK = 100_000                      # process N_Z in chunks (each pass is N_CHUNK * center_K rows) to bound memory
    drift_pred = np.zeros_like(x_grid)
    diff_pred  = np.zeros_like(x_grid)

    with torch.no_grad():
        for i, xg in enumerate(x_grid):
            u_norm = normalize(np.array([[xg]]))
            s1 = 0.0; s2 = 0.0; drawn = 0        # running sum / sum-of-squares (no 1e6-vector stored)
            while drawn < N_Z:
                m = min(N_CHUNK, N_Z - drawn)
                u = torch.tensor(u_norm, dtype=torch_dtype, device=device).expand(m, -1)
                dx = denormalize((det_net(u) + gen_increment(u, m)).cpu().numpy())[:, 0] - xg
                s1 += dx.sum(); s2 += np.square(dx).sum(); drawn += m
            mean = s1 / N_Z
            var  = max(s2 / N_Z - mean * mean, 0.0)
            drift_pred[i] = mean / DT
            diff_pred[i]  = np.sqrt(var / DT)

    drift_true = THETA * (MU - x_grid)
    diff_true  = SIGMA * np.ones_like(x_grid)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(x_grid, drift_true, 'k-',  label='Analytical')
    axes[0].plot(x_grid, drift_pred, 'r.', ms=5, label='sFML')
    axes[0].set_xlabel('x'); axes[0].set_ylabel('Drift f(x)'); axes[0].legend()
    axes[0].set_title('Effective Drift')

    axes[1].plot(x_grid, diff_true, 'k-',  label='Analytical')
    axes[1].plot(x_grid, diff_pred, 'r.', ms=5, label='sFML')
    axes[1].set_xlabel('x'); axes[1].set_ylabel('Diffusion g(x)'); axes[1].legend()
    axes[1].set_ylim(0.25, 0.35)   # fixed window (sigma=0.3 +-0.05) so auto-scale can't magnify a small tilt
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

    # Fig 8 (paper): COVARIANCE MATRIX SPECTRA.
    # The paper's caption/text: "the spectra of the covariance function of the simulated
    # sequences by the sFML model, along with that of the true solution sequence" — i.e.
    # the EIGENVALUES of the (T x T) covariance matrix of the solution SEQUENCE,
    # prediction vs ground truth. (Previously this plotted the lag-covariance function
    # C(tau), which is a DIFFERENT object — flagged by independent audit.)
    #
    # Both ensembles start from the same x0 and are compared over the same time grid, so
    # no stationarity assumption is needed (the old burn-in + centre-by-mu approximation
    # is gone: we centre by the empirical mean at each time, which is exact).
    # WINDOW (fix 1): the paper's Fig. 8 has exactly 40 components = the recurrent length
    # L, i.e. the sequence spans T = L*dt = 0.4, NOT the full T=4.0 rollout. This matters:
    # the OU correlation time is 1/theta = 1. Over a window SHORT relative to it (0.4),
    # all time points stay strongly correlated, so C is nearly rank-1 and one mode carries
    # most of the variance (analytically lambda_1 = 0.437, matching the paper's ~4e-1).
    # Over T=4 (4 correlation times) the process decorrelates, the variance spreads over
    # many modes, and lambda_1 = 1.77 instead. Using the L-window reproduces the paper.
    #
    # t_0 EXCLUDED (fix 2): every trajectory starts at exactly x0, so Var(X_{t_0}) = 0 and
    # the t_0 row/column of C vanishes identically -> an exact zero eigenvalue (log scale:
    # -inf, previously clipped to 1e-18 and visible as a cliff). The paper indexes 1..40.
    L_WIN = int(trainY.shape[-1])                    # = L = 40 (paper's recurrent length)
    idx_t = np.arange(1, L_WIN + 1)                  # t_1 .. t_L, excluding the deterministic t_0
    gt_cov = test_data[:, 0, :]                      # ground-truth EM ensemble from the same x0

    def cov_spectrum(traj):
        """Eigenvalues (descending) of the time-time covariance matrix of an ensemble.

        C_{jk} = Cov(X_{t_j}, X_{t_k}) is the discrete Karhunen-Loeve operator: its
        eigenvalues are the variances carried by each temporal mode, and they sum to the
        total variance, trace(C) = sum_j Var(X_{t_j}).
        """
        A = traj[:, idx_t]
        A = A - A.mean(axis=0, keepdims=True)        # centre by the empirical mean at each time
        C = (A.T @ A) / max(A.shape[0] - 1, 1)       # (L, L) covariance matrix
        w = np.linalg.eigvalsh(C)[::-1]              # eigvalsh -> ascending; reverse
        return np.maximum(w, 1e-16)                  # guard only; no exact zeros now that t_0 is gone

    ev_pred = cov_spectrum(ensemble)
    ev_true = cov_spectrum(gt_cov)
    k = np.arange(1, L_WIN + 1)
    print(f"  cov spectra over L={L_WIN} window (T={L_WIN*DT:.2f}), t_0 excluded")
    print(f"    lambda_1  true={ev_true[0]:.4e}  sFML={ev_pred[0]:.4e}   (analytic 4.370e-01)")
    print(f"    trace     true={ev_true.sum():.4e}  sFML={ev_pred.sum():.4e}")

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.semilogy(k, ev_true, 'b-o', ms=4, mfc='none', label='Ground Truth')
    ax.semilogy(k, ev_pred, 'r--s', ms=4, mfc='none', label='Prediction')
    ax.set_xlabel('Component')
    ax.set_ylabel('Spectra of Cov Matrix')
    ax.set_title('Covariance Matrix Spectra')
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path + "/covariance_spectra.png", dpi=150)
    plt.close()
    print("Saved covariance_spectra.png  (eigenvalue spectrum, paper Fig. 8)")

    print("\nAll evaluation figures saved to", save_path)


import os as _os
_base = conf_gan["save_path"]
for _tag in EVAL_TAGS:
    _gp = Path(_base) / f"generator_{_tag}"
    if not _gp.exists():
        print(f"[skip] {_gp} not found"); continue
    generator = torch.load(str(_gp), map_location=device, weights_only=False)
    generator.eval()
    _out = _os.path.join(_base, f"eval_{_tag}")
    _os.makedirs(_out, exist_ok=True)
    print(f"\n===== Evaluating '{_tag}' model  ->  {_out} =====")
    run_eval(_out)
