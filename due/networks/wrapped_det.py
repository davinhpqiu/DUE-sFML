"""
Cross-coordinate wrapper for the deterministic sub-map D_theta.

Lets a D_theta trained in ONE coordinate (e.g. raw physical units, where a linear
SDE mean stays linear so the net fits and extrapolates it cleanly) be used inside a
Phase-2 recursion that runs in ANOTHER coordinate (e.g. Yeo-Johnson, which the GAN
needs to variance-stabilise multiplicative diffusion).

Given a Phase-2-coordinate state u, the wrapped map returns the Phase-2-coordinate
conditional mean by sandwiching the raw map between the inverse and forward transforms:

    D_wrapped(u) = T( D_raw( T^{-1}(u) ) )

where T is the Phase-2 normaliser (transform) and T^{-1} its inverse. D_raw is frozen
in Phase 2, so only forward evaluation of the transforms is needed; they are plain
differentiable torch ops. The wrapper exposes `output_dim` and inherits `count_params`
so it drops into the SDE model exactly where a plain det-net would — no changes to the
SDE / ODE trainers.

Only used when Phase-1 and Phase-2 use different `normalization` modes (the decouple
path). The Phase-2 normaliser must be "minmax" or "yeojohnson" (it carries smin/smax/
scale); "none" needs no wrapping (raw both phases = coupled).
"""

import torch
from .nn import nn

_EPS = 1e-6


def _yj_fwd(x, lam):
    """Yeo-Johnson forward (physical -> transformed), per-coordinate lam. Torch, guarded."""
    pos = x >= 0
    inv_lam = 1.0 / torch.where(lam.abs() < _EPS, torch.ones_like(lam), lam)
    inv_2ml = 1.0 / torch.where((lam - 2.0).abs() < _EPS, torch.ones_like(lam), 2.0 - lam)
    xp = torch.clamp(x + 1.0, min=_EPS)
    out_p = torch.where(lam.abs() < _EPS, torch.log1p(torch.clamp(x, min=_EPS - 1.0)),
                        (torch.pow(xp, lam) - 1.0) * inv_lam)
    xn = torch.clamp(-x + 1.0, min=_EPS)
    out_n = torch.where((lam - 2.0).abs() < _EPS, -torch.log1p(torch.clamp(-x, min=_EPS - 1.0)),
                        -(torch.pow(xn, 2.0 - lam) - 1.0) * inv_2ml)
    return torch.where(pos, out_p, out_n)


def _yj_inv(y, lam):
    """Yeo-Johnson inverse (transformed -> physical), per-coordinate lam. Torch, guarded."""
    pos = y >= 0
    inv_lam = 1.0 / torch.where(lam.abs() < _EPS, torch.ones_like(lam), lam)
    inv_2ml = 1.0 / torch.where((lam - 2.0).abs() < _EPS, torch.ones_like(lam), 2.0 - lam)
    base_p = torch.clamp(y * lam + 1.0, min=_EPS)
    out_p = torch.where(lam.abs() < _EPS, torch.expm1(y), torch.pow(base_p, inv_lam) - 1.0)
    base_n = torch.clamp(-(2.0 - lam) * y + 1.0, min=_EPS)
    out_n = torch.where((lam - 2.0).abs() < _EPS, -torch.expm1(-y),
                        1.0 - torch.pow(base_n, inv_2ml))
    return torch.where(pos, out_p, out_n)


class WrappedDet(nn):
    """Wrap a raw-coordinate D_theta so it can be called in the Phase-2 coordinate.

    Args:
        det_raw : trained deterministic sub-map operating in the Phase-1 coordinate.
        normalizer : the fitted Phase-2 `Normalizer` (mode "minmax" or "yeojohnson").
    """

    def __init__(self, det_raw, normalizer):
        super().__init__()
        assert getattr(normalizer, "mode", "none") in ("minmax", "yeojohnson"), \
            "WrappedDet requires a minmax/yeojohnson Phase-2 normaliser (none needs no wrap)."
        self.det = det_raw
        self.mode = normalizer.mode
        self.d = int(normalizer.d)
        self.output_dim = getattr(det_raw, "output_dim", self.d)
        dt = torch.get_default_dtype()
        self.register_buffer("lam",   torch.as_tensor(normalizer.lam,   dtype=dt))
        self.register_buffer("smin",  torch.as_tensor(normalizer.smin,  dtype=dt))
        self.register_buffer("smax",  torch.as_tensor(normalizer.smax,  dtype=dt))
        self.register_buffer("scale", torch.as_tensor(normalizer.scale, dtype=dt))

    def _p(self, buf, ref):
        # reshape a (d,) buffer to broadcast on coord axis 1, matching ref's dtype/device
        shape = [1] * ref.ndim
        shape[1] = self.d
        return buf.reshape(shape).to(dtype=ref.dtype, device=ref.device)

    def _to_phys(self, u):
        """Phase-2 coordinate -> physical."""
        smin, smax, scale = self._p(self.smin, u), self._p(self.smax, u), self._p(self.scale, u)
        y = 0.5 * u * scale + 0.5 * (smax + smin)
        return _yj_inv(y, self._p(self.lam, u)) if self.mode == "yeojohnson" else y

    def _to_norm(self, x):
        """Physical -> Phase-2 coordinate."""
        smin, smax, scale = self._p(self.smin, x), self._p(self.smax, x), self._p(self.scale, x)
        y = _yj_fwd(x, self._p(self.lam, x)) if self.mode == "yeojohnson" else x
        return 2.0 * (y - 0.5 * (smax + smin)) / scale

    def forward(self, u):
        return self._to_norm(self.det(self._to_phys(u)))
