"""
Data generation — SDE with nonlinear diffusion (Chen & Xiu 2024, sec. 5.2.1):
    dx = -mu x dt + sigma e^{-x^2} dW,   mu=5, sigma=0.5,  dt=0.01
Train: IC ~ U(-1,1), 100 EM steps (T=1.0), random L=40 window, N=10000.
Test : x0=-0.4, 1000 EM steps (T=10).
"""
import numpy as np, scipy.io as sio, os

MU, SIGMA, DT = 5.0, 0.5, 0.01
N_TRAIN, STEPS, L = 10_000, 100, 40
N_TEST, X0_TEST, STEPS_TEST = 10_000, -0.4, 1000
SEED = 0

def euler_maruyama(x0, n_steps, rng):
    N = len(x0); tr = np.empty((N, n_steps + 1)); tr[:, 0] = x0; sq = np.sqrt(DT)
    for n in range(n_steps):
        x = tr[:, n]
        tr[:, n + 1] = x + (-MU * x) * DT + SIGMA * np.exp(-x**2) * sq * rng.standard_normal(N)
    return tr

if __name__ == "__main__":
    rng = np.random.default_rng(SEED)
    here = os.path.dirname(os.path.abspath(__file__))
    # train
    full = euler_maruyama(rng.uniform(-1, 1, N_TRAIN), STEPS, rng)
    starts = rng.integers(0, STEPS - L + 1, N_TRAIN)
    idx = starts[:, None] + np.arange(L + 1)[None, :]
    sub = full[np.arange(N_TRAIN)[:, None], idx]
    sio.savemat(os.path.join(here, "NLD_train.mat"), {"trajectories": sub[:, None, :]})
    # test
    test = euler_maruyama(np.full(N_TEST, X0_TEST), STEPS_TEST, rng)
    sio.savemat(os.path.join(here, "NLD_test.mat"), {"trajectories": test[:, None, :]})
    print("Saved NLD_train.mat", sub.shape, "and NLD_test.mat", test.shape)
