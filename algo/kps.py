# algo/kps.py

import torch

from torch import Tensor

from algo.base import Algo
from kps import KPS, EDMPrior


class KPSAlgo(Algo):
    r"""KPS for InverseBench.

        z_i = x_i + V_t A^T (A V_t A^T + Sigma_y)^-1 (y - Y_i)

    inside an annealed split-Gibbs ladder. See the `kps` package for the derivation and for
    the measurements behind each default.

    ONE CLOUD, NOT ONE CLOUD PER SAMPLE. A maintained ensemble is carried through the whole
    ladder and every particle of it is a posterior sample, so `num_particles` simulator calls
    per sweep yield `num_particles` samples. The earlier arrangement rebuilt a fresh cloud at
    every level per requested sample and returned one member of it; the maintained cloud
    measured 2.7-3.8x better at equal cost.

    BUDGET. levels * sweeps * num_particles simulator calls. `draw_steps` buys denoiser
    evaluations, which the benchmark does not count.

    mode    "h"  slope fitted from the cloud, rank-truncated. Derivative-free.
            "g"  slope is the simulator's exact Jacobian. Needs a differentiable simulator.
                 Measured 355x better on inverse scattering and 6.8x worse on Navier-Stokes,
                 where no linear map represents the dynamics and the fit's implicit
                 regularisation helps instead.
    """

    def __init__(
        self,
        net,
        forward_op,
        mode: str = "h",
        num_particles: int = 128,
        levels: int = 4,
        sweeps: int = 4,
        rank: int = 32,
        draw_steps: int = 8,
        sigma_max: float = 80.0,
        sigma_min: float = 0.05,
        solve_iter: int = 2,
        churn: float = 0.0,
        **kwargs,
    ):
        super().__init__(net, forward_op, **kwargs)

        self.mode = mode
        self.num_particles = num_particles
        self.levels = levels
        self.sweeps = sweeps
        self.rank = rank
        self.draw_steps = draw_steps
        self.sigma_max = sigma_max
        self.sigma_min = sigma_min
        self.solve_iter = solve_iter
        self.churn = churn

    def _simulator(self, obs: Tensor):
        r"""Queried WITH observation noise -- the innovation then carries its own correctly
        covaried perturbation and no inflation is needed. The noiseless operator is not used.
        """

        if torch.is_complex(obs):                                   # inv-scatter
            y = torch.cat([obs.real, obs.imag], dim=1).to(torch.float32)

            def simulate(x: Tensor) -> Tensor:
                out = self.forward_op({"target": x})

                return torch.cat([out.real, out.imag], dim=1).to(torch.float32)
        else:                                                       # navier-stokes, blackhole
            y = obs.to(torch.float32)

            def simulate(x: Tensor) -> Tensor:
                return self.forward_op({"target": x}).to(torch.float32)

        return y, simulate

    @torch.no_grad()
    def inference(self, obs: Tensor, num_samples: int = 1) -> Tensor:
        y, simulate = self._simulator(obs)

        n = max(self.num_particles, num_samples)

        sampler = KPS(
            EDMPrior(self.net),
            simulate,
            y,
            mode=self.mode,
            num_particles=n,
            levels=self.levels,
            sweeps=self.sweeps,
            rank=self.rank,
            draw_steps=self.draw_steps,
            sigma_max=self.sigma_max,
            sigma_min=self.sigma_min,
            solve_iter=self.solve_iter,
            churn=self.churn,
        )

        x = sampler.sample(tuple(self.net.shape), device=self.forward_op.device)

        return x[:num_samples]
