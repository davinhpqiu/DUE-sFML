"""
Data generation for Geometric Brownian Motion (GBM).  sFML paper, §5.1.2.

SDE:  dx = mu*x dt + sigma*x dW      (multiplicative noise)
Parameters: mu = 2.0, sigma = 1.0

Same protocol as the OU example:
  - N = 10,000 trajectories
  - Initial conditions x0 ~ Uniform(0, 2)
  - Euler-Maruyama, dt = 0.01, 100 steps, then a random length-40 window
  - Saved shape (N, 1, 41)

Test data:
  - N_test trajectories from x0 = 0.5, 100 steps (T = 1.0; GBM grows exponentially)
  - Saved shape (N_test, 1, 101)
"""
import numpy as np
import scipy.io as sio
import os

MU, SIGMA, DT = 2.0, 1.0, 0.01

N_TRAIN       = 10_000
STEPS_FULL    = 100
L             = 40
X0_MIN, X0_MAX = 0.0, 2.0

N_TEST        = 10_000
X0_TEST       = 0.5
STEPS_TEST    = 100          # T = 1.0

SEED = 0


def euler_maruyama(x0_arr, n_steps, rng):
    """GBM Euler-Maruyama: x_{n+1} = x_n + mu*x_n*dt + sigma*x_n*sqrt(dt)*eps."""
    N = len(x0_arr)
    traj = np.empty((N, n_steps + 1), dtype=np.float64)
    traj[:, 0] = x0_arr
    sqrt_dt = np.sqrt(DT)
    for n in range(n_steps):
        x = traj[:, n]
        noise = rng.standard_normal(N)
        traj[:, n + 1] = x + MU * x * DT + SIGMA * x * sqrt_dt * noise
    return traj


def generate_train(rng):
    print(f"Generating GBM training data: N={N_TRAIN}, steps={STEPS_FULL}, L={L} ...")
    x0 = rng.uniform(X0_MIN, X0_MAX, size=N_TRAIN)
    full = euler_maruyama(x0, STEPS_FULL, rng)
    max_start = STEPS_FULL - L
    starts = rng.integers(0, max_start + 1, size=N_TRAIN)
    idx = starts[:, None] + np.arange(L + 1)[None, :]
    sub = full[np.arange(N_TRAIN)[:, None], idx]
    data = sub[:, np.newaxis, :]
    print(f"  Training array shape: {data.shape}")
    print(f"  State range: [{data.min():.4f}, {data.max():.4f}]")
    return data


def generate_test(rng):
    print(f"Generating GBM test data: N={N_TEST}, x0={X0_TEST}, steps={STEPS_TEST} ...")
    x0 = np.full(N_TEST, X0_TEST)
    traj = euler_maruyama(x0, STEPS_TEST, rng)
    data = traj[:, np.newaxis, :]
    print(f"  Test array shape: {data.shape}")
    print(f"  State range: [{data.min():.4f}, {data.max():.4f}]")

    # Sanity check vs analytical GBM moments
    t = np.arange(STEPS_TEST + 1) * DT
    mean_true = X0_TEST * np.exp(MU * t)
    std_true  = X0_TEST * np.exp(MU * t) * np.sqrt(np.exp(SIGMA**2 * t) - 1)
    me, se = data[:, 0, :].mean(0), data[:, 0, :].std(0)
    print("  Sanity (analytical vs empirical mean / std):")
    for s in [0, 50, 100]:
        print(f"    t={s*DT:.1f}: mean {mean_true[s]:.3f} vs {me[s]:.3f} | "
              f"std {std_true[s]:.3f} vs {se[s]:.3f}")
    return data


if __name__ == "__main__":
    rng = np.random.default_rng(SEED)
    save_dir = os.path.dirname(os.path.abspath(__file__))

    train_data = generate_train(rng)
    sio.savemat(os.path.join(save_dir, "GBM_train.mat"), {"trajectories": train_data})
    print("  Saved -> GBM_train.mat\n")

    test_data = generate_test(rng)
    sio.savemat(os.path.join(save_dir, "GBM_test.mat"), {"trajectories": test_data})
    print("  Saved -> GBM_test.mat\n")
    print("Done.")
