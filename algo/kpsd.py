# algo/kpsd.py

import torch

from torch import Tensor

from algo.base import Algo
from algo.azula_bridge import EDMNetDenoiser
from algo.kps import _by_chunk
from azula.sample import DDIMSampler
from kps.denoiser import KPSDenoiser


class KPSDenoiserAlgo(Algo):
    r"""KPS as a posterior DENOISER, driven by an azula sampler.

        E[x | x_t, y]_i  =  x_hat_i + V_t,i A^T (A V_t,i A^T + Sigma_y)^-1 (y - A x_hat_i - b)

    with x_hat_i = E[x | x_t,i] the per-particle Tweedie mean, V_t,i its local Tweedie
    covariance, and ONLY (A, b, Sigma_y) estimated from the cloud. See `kps.denoiser`.

    This is a different object from `KPSAlgo`, not a re-parameterisation of it. That one runs
    a split-Gibbs ladder whose analysis returns a sample-like point; this returns a conditional
    mean and lets the sampler supply every bit of the dispersion.

    ETA IS THE KERNEL KNOB, AND 1 IS NOT THE LADDER. With alpha = 1, azula's step keeps
    (sigma_s/sigma_t)^2 of x_t; the ladder discards x_t and re-noises from the analysis point,
    which is the tau = 1 corner and needs a LARGE eta. Measured on one Navier-Stokes instance
    at 32 steps: eta = 1 scored err 0.952 against the full re-noise's 0.705, a difference of
    kernel rather than of method. The default here is therefore a saturating eta.

    BUDGET. steps * num_particles simulator calls, plus one denoiser evaluation and `rank`
    vector-Jacobian products per step. There is no inner ODE, so a step is cheaper than a
    ladder sweep at the same N.
    """

    def __init__(
        self,
        net,
        forward_op,
        num_particles: int = 128,
        steps: int = 64,
        rank: int = 32,
        mode: str = "h",
        solve_iter: int = 16,
        eta: float = 1e6,
        sigma_max: float = 80.0,
        sigma_min: float = 0.05,
        sim_batch: int = 0,
        **kwargs,
    ):
        super().__init__(net, forward_op, **kwargs)

        self.num_particles = num_particles
        self.steps = steps
        self.rank = rank
        self.mode = mode
        self.solve_iter = solve_iter
        self.eta = eta
        self.sigma_max = sigma_max
        self.sigma_min = sigma_min
        self.sim_batch = sim_batch

    def _simulator(self, obs: Tensor):
        r"""Same contract as KPSAlgo: queried WITH observation noise, complex packed to real.

        The noise matters less here than in the sampler -- the innovation's base point is the
        cloud average, so the per-particle draw is divided by sqrt(N) -- but the operator is
        the benchmark's and it is not ours to make noiseless.
        """

        if torch.is_complex(obs):
            y = torch.cat([obs.real, obs.imag], dim=1).to(torch.float32)

            def once(x: Tensor) -> Tensor:
                out = self.forward_op({"target": x})

                return torch.cat([out.real, out.imag], dim=1).to(torch.float32)
        else:
            y = obs.to(torch.float32)

            def once(x: Tensor) -> Tensor:
                return self.forward_op({"target": x}).to(torch.float32)

        return y, lambda x: _by_chunk(once, x, self.sim_batch)

    @torch.no_grad()
    def inference(self, obs: Tensor, num_samples: int = 1) -> Tensor:
        r"""Returns the whole cloud, (num_particles, *shape). `num_samples` is ignored."""

        y, simulate = self._simulator(obs)
        device = self.forward_op.device

        prior = EDMNetDenoiser(self.net, sigma_min=self.sigma_min, sigma_max=self.sigma_max)
        post = KPSDenoiser(prior, y, simulate, rank=self.rank, mode=self.mode,
                           solve_iter=self.solve_iter)

        sampler = DDIMSampler(post, eta=self.eta, steps=self.steps, silent=True, device=device)
        x_1 = torch.randn(self.num_particles, *tuple(self.net.shape),
                          device=device) * self.sigma_max

        return sampler(x_1)
