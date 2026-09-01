from azula.denoise import Denoiser, DiracPosterior
from azula.noise import Schedule

import math
import torch

from torch import Tensor
from typing import Optional


EDM_SIGMA_MIN = 0.002
EDM_SIGMA_MAX = 80.0


def _usable(value, fallback: float) -> float:
    r"""Accepts a declared noise level only if it is finite and strictly positive."""

    if value is None:
        return fallback

    value = float(value)

    return value if 0 < value < math.inf else fallback


class EDMSchedule(Schedule):
    r"""Karras spacing over a variance-exploding schedule.

    Maps azula time onto the noise level with the EDM rho-spacing, so that
    ``t = 0`` is clean data and ``t = 1`` is the noisiest state:

        sigma(t) = (sigma_max^(1/rho) + (1 - t) (sigma_min^(1/rho) - sigma_max^(1/rho)))^rho

    Karras-style preconditioners are variance exploding, so ``alpha_t = 1``.
    """

    def __init__(self, sigma_min: float, sigma_max: float, rho: float = 7.0) -> None:
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.rho = rho

    def __call__(self, t: Tensor) -> tuple[Tensor, Tensor]:
        lo, hi = self.sigma_min ** (1 / self.rho), self.sigma_max ** (1 / self.rho)

        sigma_t = (hi + (1 - t) * (lo - hi)) ** self.rho
        alpha_t = torch.ones_like(sigma_t)

        return alpha_t, sigma_t


class EDMNetDenoiser(Denoiser):
    r"""Wraps an InverseBench preconditioned net into an azula Denoiser.

    InverseBench nets (``VPPrecond``, ``VEPrecond``, ``iDDPMPrecond``, ``EDMPrecond``)
    are all Karras-style: they take ``(x, sigma)`` with ``x = x_0 + sigma eps`` and
    return the denoised ``x_0``.

    The noise range is read from the net where it is meaningful -- ``VPPrecond`` works
    out to roughly [0.001, 152], ``VEPrecond`` declares [0.02, 100], ``iDDPMPrecond``
    computes it -- but ``EDMPrecond`` uses ``sigma_min = 0`` and ``sigma_max = inf`` as
    sentinels for "unbounded", not as a usable range. Those would make the Karras
    spacing ``inf - inf``, so anything non-finite or non-positive falls back to the EDM
    defaults [0.002, 80].

    Arguments:
        net: A preconditioned InverseBench net.
        sigma_min: Lowest noise level. Defaults to the net's own.
        sigma_max: Highest noise level. Defaults to the net's own.
    """

    def __init__(
        self,
        net,
        sigma_min: Optional[float] = None,
        sigma_max: Optional[float] = None,
        rho: float = 7.0,
    ) -> None:
        super().__init__()

        self.net = net

        if sigma_min is None:
            sigma_min = _usable(getattr(net, "sigma_min", None), EDM_SIGMA_MIN)
        if sigma_max is None:
            sigma_max = _usable(getattr(net, "sigma_max", None), EDM_SIGMA_MAX)

        self._schedule = EDMSchedule(sigma_min=sigma_min, sigma_max=sigma_max, rho=rho)

    @property
    def schedule(self) -> Schedule:
        return self._schedule

    def forward(self, x_t: Tensor, t: Tensor, **kwargs) -> DiracPosterior:
        alpha_t, sigma_t = self.schedule(t)

        shape = (-1, *(1,) * (x_t.ndim - 1))
        alpha = alpha_t.reshape(shape).to(x_t)
        sigma = sigma_t.reshape(shape).to(x_t)

        # azula parameterises x_t = alpha_t x_0 + sigma_t eps, while the net expects
        # x = x_0 + s eps. Dividing by alpha_t gives that form with an effective noise
        # level s = sigma_t / alpha_t. Both collapse to identity when alpha_t = 1.
        x0 = self.net(x_t / alpha, sigma / alpha)

        return DiracPosterior(mean=x0)
