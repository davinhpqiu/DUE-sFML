"""
Data-derived, invertible, per-coordinate state normaliser for sFML.

Two modes:
  - "minmax"      : affine to [-1,1] (the original DUE behaviour). Identical math.
  - "yeojohnson"  : fit a Yeo-Johnson power transform per coordinate (lambda by MLE
                    on the training states), THEN affine to [-1,1] in transformed
                    space. Equation-agnostic: for well-behaved data lambda~1 (near
                    identity); for multiplicative/skewed data (e.g. GBM) lambda~0
                    (log-like), which variance-stabilises multiplicative noise into
                    additive noise — turning GBM into an OU-like problem for the GAN.

The transform is applied to STATES, identically at every time step, so the flow-map
autonomy is preserved. Prediction is done in normalised space and inverted for output.

Coordinate axis is axis=1 for both trajectory arrays (N,d,L) and state arrays (n,d).
"""

import numpy as np
from scipy.stats import yeojohnson


def _yj_fwd(x, lam):
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x)
    pos = x >= 0
    if abs(lam) < 1e-6:
        out[pos] = np.log1p(x[pos])
    else:
        out[pos] = (np.power(x[pos] + 1.0, lam) - 1.0) / lam
    if abs(lam - 2.0) < 1e-6:
        out[~pos] = -np.log1p(-x[~pos])
    else:
        out[~pos] = -(np.power(-x[~pos] + 1.0, 2.0 - lam) - 1.0) / (2.0 - lam)
    return out


def _yj_inv(y, lam):
    y = np.asarray(y, dtype=np.float64)
    out = np.empty_like(y)
    pos = y >= 0
    if abs(lam) < 1e-6:
        out[pos] = np.expm1(y[pos])
    else:
        out[pos] = np.power(y[pos] * lam + 1.0, 1.0 / lam) - 1.0
    if abs(lam - 2.0) < 1e-6:
        out[~pos] = -np.expm1(-y[~pos])
    else:
        out[~pos] = 1.0 - np.power(-(2.0 - lam) * y[~pos] + 1.0, 1.0 / (2.0 - lam))
    return out


class Normalizer:
    def __init__(self, mode="minmax"):
        assert mode in ("minmax", "yeojohnson", "none")
        self.mode = mode

    def fit(self, data):
        """data: (N, d, L) trajectory array (physical units)."""
        d = data.shape[1]
        self.d = d
        if self.mode == "none":
            self.lam = np.ones(d)
            return self  # raw: transform/inverse are identity
        flat = np.moveaxis(data, 1, 0).reshape(d, -1)  # (d, N*L)
        if self.mode == "yeojohnson":
            self.lam = np.array([yeojohnson(flat[i])[1] for i in range(d)])
            tf = np.stack([_yj_fwd(flat[i], self.lam[i]) for i in range(d)])
        else:
            self.lam = np.ones(d)
            tf = flat
        self.smin = tf.min(axis=1)
        self.smax = tf.max(axis=1)
        self.scale = np.where(self.smax > self.smin, self.smax - self.smin, 1.0)
        return self

    def _coord(self, x, i, fn):
        # apply scalar fn to coordinate i, coord axis = 1
        if x.ndim == 3:
            return fn(x[:, i, :])
        return fn(x[:, i])

    def transform(self, x):
        """physical (..., coord axis 1) -> normalised [-1,1]."""
        x = np.asarray(x, dtype=np.float64)
        if self.mode == "none":
            return x
        out = np.empty_like(x)
        for i in range(self.d):
            def fwd(xi):
                yi = _yj_fwd(xi, self.lam[i]) if self.mode == "yeojohnson" else xi
                # NO clip: training data is within [smin,smax] by construction (clip
                # would be a no-op there), but eval states outside the training range
                # (e.g. the OOD test IC x0=1.5) must map past +-1 so the flow map can
                # extrapolate, matching the original eval behaviour.
                return 2.0 * (yi - 0.5 * (self.smax[i] + self.smin[i])) / self.scale[i]
            if x.ndim == 3:
                out[:, i, :] = fwd(x[:, i, :])
            else:
                out[:, i] = fwd(x[:, i])
        return out

    def inverse(self, u):
        """normalised [-1,1] (..., coord axis 1) -> physical."""
        u = np.asarray(u, dtype=np.float64)
        if self.mode == "none":
            return u
        out = np.empty_like(u)
        for i in range(self.d):
            def inv(ui):
                yi = 0.5 * ui * self.scale[i] + 0.5 * (self.smax[i] + self.smin[i])
                return _yj_inv(yi, self.lam[i]) if self.mode == "yeojohnson" else yi
            if u.ndim == 3:
                out[:, i, :] = inv(u[:, i, :])
            else:
                out[:, i] = inv(u[:, i])
        return out
