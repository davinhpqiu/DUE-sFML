"""
Data generation for the Ornstein-Uhlenbeck (OU) process.

Follows Section 5.1.1 of:
  Chen & Xiu (2024), "Learning stochastic dynamical system via flow map operator",
  J. Comput. Phys., 508, 112984.

SDE:  dx = theta*(mu - x)*dt + sigma*dW
Parameters: theta=1.0, mu=1.2, sigma=0.3

Training data
-------------
- N = 10,000 trajectories
- Initial conditions: x0 ~ Uniform(0, 0.25)
- Euler-Maruyama: dt=0.01, 100 steps  (total time T_sim = 1.0)
- Randomly select a contiguous subsequence of length L=40 from each trajectory
- Saved shape: (N, 1, L+1) = (10000, 1, 41)

Test data
---------
- N_test trajectories starting from x0 = 1.5
- Euler-Maruyama: dt=0.01, 400 steps  (total time T_pred = 4.0)
- Saved shape: (N_test, 1, 401)
- Used to compute ground-truth statistics for Figures 5-8 of the paper.

Output
------
  OU_train.mat   key: 'trajectories'  shape (10000, 1, 41)
  OU_test.mat    key: 'trajectories'  shape (N_test, 1, 401)
"""

import numpy as np
import scipy.io as sio
import os

# Parameters

THETA  = 1.0     # mean reversion rate
MU     = 1.2     # mean reversion level
SIGMA  = 0.3     # noise coefficient
DT     = 0.01    # Euler-Maruyama time step

# Training
N_TRAIN       = 10_000   # number of training trajectories (paper §5.1.1)
STEPS_FULL    = 100      # E-M steps per trajectory before subsampling
L             = 40       # length of randomly chosen subsequence (in steps)
X0_MIN, X0_MAX = 0.0, 0.25  # initial condition range

# Test
N_TEST        = 10_000   # number of test trajectories from x0=1.5
X0_TEST       = 1.5      # test initial condition (paper §5.1.1)
STEPS_TEST    = 400      # E-M steps  (T=4.0 / dt=0.01)

SEED = 1


# Euler-Maruyama integrator

def euler_maruyama(x0_arr, n_steps, theta, mu, sigma, dt, rng):
    """
    Integrate the OU SDE for a batch of initial conditions.
    
    Parameters
    ----------
    x0_arr : ndarray, shape (N,)
    n_steps : int
    Returns
    -------
    traj : ndarray, shape (N, n_steps+1)
    """
    N = len(x0_arr)
    traj = np.empty((N, n_steps + 1), dtype=np.float64)
    traj[:, 0] = x0_arr
    sqrt_dt = np.sqrt(dt)
    for n in range(n_steps):
        x = traj[:, n]
        noise = rng.standard_normal(N)
        traj[:, n + 1] = x + theta * (mu - x) * dt + sigma * sqrt_dt * noise
    return traj  # shape (N, n_steps+1)

# Generate training data

def generate_train(rng):
    print(f"Generating training data: N={N_TRAIN}, steps={STEPS_FULL}, L={L} ...")

    # Initial conditions x0 ~ U(0, 0.25)
    x0 = rng.uniform(X0_MIN, X0_MAX, size=N_TRAIN)

    # Run full trajectories: shape (N_TRAIN, STEPS_FULL+1)
    full_traj = euler_maruyama(x0, STEPS_FULL, THETA, MU, SIGMA, DT, rng)

    # Randomly select subsequence of length L from each trajectory.
    # Valid start indices: 0 .. STEPS_FULL - L inclusive
    max_start = STEPS_FULL - L
    starts = rng.integers(0, max_start + 1, size=N_TRAIN)

    # Extract subsequences: shape (N_TRAIN, L+1)
    idx = starts[:, None] + np.arange(L + 1)[None, :]  # (N_TRAIN, L+1)
    sub_traj = full_traj[np.arange(N_TRAIN)[:, None], idx]  # (N_TRAIN, L+1)

    # DUE format: (N, d, T+1)  with d=1
    data = sub_traj[:, np.newaxis, :]  # (N_TRAIN, 1, L+1)

    print(f"  Training array shape: {data.shape}")
    print(f"  State range: [{data.min():.4f}, {data.max():.4f}]")
    return data

# Generate test data

def generate_test(rng):
    print(f"Generating test data: N={N_TEST}, x0={X0_TEST}, steps={STEPS_TEST} ...")

    x0 = np.full(N_TEST, X0_TEST)
    traj = euler_maruyama(x0, STEPS_TEST, THETA, MU, SIGMA, DT, rng)

    # DUE format: (N, d, T+1)  with d=1
    data = traj[:, np.newaxis, :]  # (N_TEST, 1, STEPS_TEST+1)

    print(f"  Test array shape: {data.shape}")
    print(f"  State range: [{data.min():.4f}, {data.max():.4f}]")

    # Quick sanity check against analytical moments
    t_arr = np.arange(STEPS_TEST + 1) * DT
    mean_analytical = (X0_TEST - MU) * np.exp(-THETA * t_arr) + MU
    std_analytical  = (SIGMA / np.sqrt(2 * THETA)) * np.sqrt(1 - np.exp(-2 * THETA * t_arr))
    mean_empirical  = data[:, 0, :].mean(axis=0)
    std_empirical   = data[:, 0, :].std(axis=0)

    # Report error at a few time points
    check_steps = [0, 100, 200, 400]
    print("  Sanity check (analytical vs. empirical mean / std):")
    for s in check_steps:
        print(f"    t={s*DT:.1f}: "
              f"mean {mean_analytical[s]:.4f} vs {mean_empirical[s]:.4f} | "
              f"std  {std_analytical[s]:.4f} vs {std_empirical[s]:.4f}")

    return data

# Main

if __name__ == "__main__":
    rng = np.random.default_rng(SEED)

    save_dir = os.path.dirname(os.path.abspath(__file__))

    train_data = generate_train(rng)
    train_path = os.path.join(save_dir, "OU_train.mat")
    sio.savemat(train_path, {"trajectories": train_data})
    print(f"  Saved -> {train_path}\n")

    test_data = generate_test(rng)
    test_path = os.path.join(save_dir, "OU_test.mat")
    sio.savemat(test_path, {"trajectories": test_data})
    print(f"  Saved -> {test_path}\n")

    print("Done.")
