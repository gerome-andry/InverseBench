from azula.denoise import Denoiser, DiracPosterior
from azula.noise import Schedule

import torch

from torch import Tensor
from typing import Optional


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
    return the denoised ``x_0``. Each exposes its own ``sigma_min`` / ``sigma_max``,
    which differ by preconditioner -- EDM defaults to [0.002, 80] while VP works out
    to roughly [0.001, 152] -- so the range is read from the net rather than assumed.

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
            sigma_min = float(net.sigma_min)
        if sigma_max is None:
            sigma_max = float(net.sigma_max)

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
