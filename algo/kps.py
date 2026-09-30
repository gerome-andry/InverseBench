# algo/kps.py

import torch

from torch import Tensor

from algo.base import Algo
from algo.undecorate import differentiable
from kps import KPS, EDMPrior


def _by_chunk(fn, x: Tensor, batch: int) -> Tensor:
    r"""Apply `fn` to particle slices of `x` and concatenate. Particles are independent."""

    if not batch or batch >= x.shape[0]:
        return fn(x)

    return torch.cat([fn(x[i:i + batch]) for i in range(0, x.shape[0], batch)], dim=0)


class _BatchedPrior(EDMPrior):
    r"""EDMPrior whose prior-factor draw runs in particle chunks.

    The draw integrates the probability-flow ODE with the denoiser applied to the whole cloud
    at once, so its activation memory is linear in N and it is the second of the two places
    that scales (the first is the simulator). Chunking it is EXACT at churn=0: the ODE is
    deterministic and every particle is integrated independently, so the concatenation of the
    chunk results is bit-identical to the unchunked draw. At churn > 0 the draws are still
    valid but consume the RNG stream in a different order, so they will not match bitwise.

    The Jacobian path does NOT need this. `kps.vt.vt_columns` already chunks over particles
    (pchunk=64) and over directions (chunk=8), so the autograd graph is bounded there no
    matter how large N is.
    """

    def __init__(self, net, batch: int = 0) -> None:
        super().__init__(net)

        self.batch = batch

    @torch.no_grad()
    def draw(self, x_t: Tensor, sigma: float, steps: int = 8, churn: float = 0.0,
             sigma_eps: float = 2e-3) -> Tensor:
        draw = super().draw          # bound outside the comprehension: super() needs __class__

        return _by_chunk(lambda xb: draw(xb, sigma, steps, churn, sigma_eps), x_t, self.batch)


class KPSAlgo(Algo):
    r"""KPS for InverseBench.

        z_i = x_i + V_t A^T (A V_t A^T + Sigma_y)^-1 (y - Y_i)

    inside an annealed split-Gibbs ladder. See the `kps` package for the derivation and for
    the measurements behind each default.

    ONE CLOUD, AND ALL OF IT IS THE ANSWER. A maintained ensemble is carried through the whole
    ladder and every particle of it is a posterior sample, so `inference` returns all
    `num_particles` of them. The earlier arrangement rebuilt a fresh cloud at every level per
    requested sample and returned one member of it; the maintained cloud measured 2.7-3.8x
    better at equal cost.

    `num_samples` IS IGNORED. It is part of the Algo interface and InverseBench passes it, but
    there is no particles-versus-samples distinction in this method: the ensemble size IS the
    number of posterior samples, and returning a subset of an interacting cloud would only
    discard work already paid for. Set `num_particles` to control both.

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
        solve_iter: int = 16,
        churn: float = 0.0,
        sim_batch: int = 0,
        draw_batch: int = 0,
        progress: bool = True,
        holdout: bool = True,
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
        self.sim_batch = sim_batch
        self.draw_batch = draw_batch
        self.progress = progress
        self.holdout = holdout

    def _simulator(self, obs: Tensor):
        r"""Queried WITH observation noise -- the innovation then carries its own correctly
        covaried perturbation and no inflation is needed. The noiseless operator is not used.

        Goes through `differentiable`, which strips the `@torch.no_grad()` that Navier-Stokes
        puts on its operator. Without it `torch.func.vjp` returns ZEROS there rather than
        raising, so mode "g" builds A^T = 0 and its update is identically zero -- it does not
        run at all. Values are unchanged and the other operators are returned untouched.
        """

        op = differentiable(self.forward_op)

        if torch.is_complex(obs):                                   # inv-scatter
            y = torch.cat([obs.real, obs.imag], dim=1).to(torch.float32)

            def once(x: Tensor) -> Tensor:
                out = op({"target": x})

                return torch.cat([out.real, out.imag], dim=1).to(torch.float32)
        else:                                                       # navier-stokes, blackhole
            y = obs.to(torch.float32)

            def once(x: Tensor) -> Tensor:
                return op({"target": x}).to(torch.float32)

        def simulate(x: Tensor) -> Tensor:
            return _by_chunk(once, x, self.sim_batch)

        return y, simulate

    @torch.no_grad()
    def inference(self, obs: Tensor, num_samples: int = 1) -> Tensor:
        r"""Returns the whole cloud, (num_particles, *shape). `num_samples` is ignored."""

        y, simulate = self._simulator(obs)

        sampler = KPS(
            _BatchedPrior(self.net, self.draw_batch),
            simulate,
            y,
            mode=self.mode,
            num_particles=self.num_particles,
            levels=self.levels,
            sweeps=self.sweeps,
            rank=self.rank,
            draw_steps=self.draw_steps,
            sigma_max=self.sigma_max,
            sigma_min=self.sigma_min,
            solve_iter=self.solve_iter,
            churn=self.churn,
            progress=self.progress,
        )

        # the sampler reads this off itself; see kps/sampler.py. holdout=False scores
        # Sigma_y in sample and is an ablation hook, not a production setting.
        sampler.holdout = self.holdout

        return sampler.sample(tuple(self.net.shape), device=self.forward_op.device)
