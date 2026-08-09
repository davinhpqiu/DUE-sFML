"""
Data generation — SDE with trigonometric drift and diffusion (Chen & Xiu 2024, sec 5.2.2):
    dx = sin(2*k*pi*x) dt + sigma*cos(2*k*pi*x) dW,   k=1, sigma=0.5,  dt=0.01

Drift f(x) = sin(2*pi*x): zeros at x = n/2, stable equilibria at x = 1/2 + n (integer n).
Diffusion g(x) = 0.5*cos(2*pi*x): state-dependent, zero at x = 1/4 + n/2.
IC range U(0.35, 0.7) keeps trajectories near the stable point x=0.5.

Train: IC ~ U(0.35, 0.7), 100 EM steps (T=1.0), random L=40 window, N=10000.
Test : x0=0.6, 1000 EM steps (T=10.0), N=10000.
"""
import numpy as np, scipy.io as sio, os

K, SIGMA, DT = 1, 0.5, 0.01
N_TRAIN, STEPS, L = 10_000, 100, 40
N_TEST, X0_TEST, STEPS_TEST = 10_000, 0.6, 1000
SEED = 0


def euler_maruyama(x0, n_steps, rng):
    N = len(x0)
    tr = np.empty((N, n_steps + 1))
    tr[:, 0] = x0
    sq = np.sqrt(DT)
    w = 2 * K * np.pi
    for n in range(n_steps):
        x = tr[:, n]
        tr[:, n + 1] = x + np.sin(w * x) * DT + SIGMA * np.cos(w * x) * sq * rng.standard_normal(N)
    return tr


if __name__ == "__main__":
    rng = np.random.default_rng(SEED)
    here = os.path.dirname(os.path.abspath(__file__))

    # train: random L=40 windows from U(0.35, 0.7) trajectories
    full = euler_maruyama(rng.uniform(0.35, 0.7, N_TRAIN), STEPS, rng)
    starts = rng.integers(0, STEPS - L + 1, N_TRAIN)
    idx = starts[:, None] + np.arange(L + 1)[None, :]
    sub = full[np.arange(N_TRAIN)[:, None], idx]
    sio.savemat(os.path.join(here, "TD_train.mat"), {"trajectories": sub[:, None, :]})

    # test: x0=0.6, long rollout to T=10
    test = euler_maruyama(np.full(N_TEST, X0_TEST), STEPS_TEST, rng)
    sio.savemat(os.path.join(here, "TD_test.mat"), {"trajectories": test[:, None, :]})

    print(f"Saved TD_train.mat {sub.shape} and TD_test.mat {test.shape}")

    # quick sanity: show training state coverage and analytical drift/diffusion
    try:
        import matplotlib.pyplot as plt
        x_plot = np.linspace(0.2, 0.8, 200)
        w = 2 * np.pi
        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
        ax[0].hist(full.flatten(), bins=80, density=True, alpha=0.6)
        ax[0].set_xlabel('x'); ax[0].set_ylabel('density')
        ax[0].set_title('Training state distribution')
        ax[1].plot(x_plot, np.sin(w * x_plot), label='drift sin(2πx)')
        ax[1].plot(x_plot, SIGMA * np.abs(np.cos(w * x_plot)), label='|diffusion| 0.5|cos(2πx)|')
        ax[1].axhline(0, color='gray', lw=0.5); ax[1].axvline(0.5, color='gray', lw=0.5, ls=':')
        ax[1].set_xlabel('x'); ax[1].legend(); ax[1].set_title('Analytical drift & diffusion')
        plt.tight_layout()
        plt.savefig(os.path.join(here, "data_check.png"), dpi=150); plt.close()
        print("Saved data_check.png")
    except Exception as e:
        print(f"(plot skipped: {e})")
