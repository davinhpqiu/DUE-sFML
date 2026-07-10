"""
Data generation — Double-Well SDE (Chen & Xiu 2024, sec 5.2.3):
    dx = (x - x^3) dt + sigma dW,   sigma=0.5,  dt=0.01

Drift f(x) = x - x^3 = x(1-x^2).  Stable equilibria at x=+-1, unstable at x=0.
Constant diffusion — minmax normalization is fine.
Stationary: p(x) ∝ exp(4x^2 - 2x^4)  (bimodal, peaks at +-1).

Train: IC ~ U(-2,2), 100 EM steps (T=1.0), random L=40 window, N=10000.
Test : x0=0.0 (top of barrier), 500 EM steps (T=5.0), N=10000.
"""
import numpy as np, scipy.io as sio, os

SIGMA, DT = 0.5, 0.01
N_TRAIN, STEPS, L = 10_000, 100, 40
N_TEST, X0_TEST, STEPS_TEST = 10_000, 0.0, 500
SEED = 0


def euler_maruyama(x0, n_steps, rng):
    N = len(x0)
    tr = np.empty((N, n_steps + 1))
    tr[:, 0] = x0
    sq = np.sqrt(DT)
    for n in range(n_steps):
        x = tr[:, n]
        tr[:, n + 1] = x + (x - x**3) * DT + SIGMA * sq * rng.standard_normal(N)
    return tr


if __name__ == "__main__":
    rng = np.random.default_rng(SEED)
    here = os.path.dirname(os.path.abspath(__file__))

    # train: random windows of length L from trajectories started at U(-2, 2)
    full = euler_maruyama(rng.uniform(-2, 2, N_TRAIN), STEPS, rng)
    starts = rng.integers(0, STEPS - L + 1, N_TRAIN)
    idx = starts[:, None] + np.arange(L + 1)[None, :]
    sub = full[np.arange(N_TRAIN)[:, None], idx]
    sio.savemat(os.path.join(here, "DW_train.mat"), {"trajectories": sub[:, None, :]})

    # test: x0=0 (top of barrier), long trajectories to see mixing into both wells
    test = euler_maruyama(np.full(N_TEST, X0_TEST), STEPS_TEST, rng)
    sio.savemat(os.path.join(here, "DW_test.mat"), {"trajectories": test[:, None, :]})

    print(f"Saved DW_train.mat {sub.shape} and DW_test.mat {test.shape}")

    # quick sanity check: stationary histogram of all training states
    try:
        import matplotlib.pyplot as plt
        all_x = full.flatten()
        x_plot = np.linspace(-2.5, 2.5, 400)
        log_p = 4 * x_plot**2 - 2 * x_plot**4
        p = np.exp(log_p - log_p.max())
        from scipy import integrate
        Z = integrate.quad(lambda x: np.exp(4*x**2 - 2*x**4), -4, 4)[0]
        p_norm = np.exp(4 * x_plot**2 - 2 * x_plot**4) / Z
        plt.figure(figsize=(7, 4))
        plt.hist(all_x, bins=100, density=True, alpha=0.5, label="EM training states")
        plt.plot(x_plot, p_norm, 'k-', lw=2, label=r"$p \propto e^{4x^2-2x^4}$")
        plt.xlabel("x"); plt.ylabel("density")
        plt.title("Stationary distribution check (training data)")
        plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(here, "stationary_check.png"), dpi=150)
        plt.close()
        print("Saved stationary_check.png")
    except Exception as e:
        print(f"(plot skipped: {e})")
