# algo/bslr.py

import torch

from torch import Tensor

from algo.base import Algo
from kps.bayes import BayesSLR
from kps.prior import EDMPrior


class _BatchSigma(torch.nn.Module):
    """The preconditioned net with sigma expanded to the batch, as every backbone expects."""

    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x: Tensor, sigma: Tensor) -> Tensor:
        return self.net(x, sigma.reshape(-1).expand(x.shape[0]))


class BayesSLRAlgo(Algo):
    """
    Bayesian SLR ladder for InverseBench (KPS branch `bayes-slr`: kps/bayes.py, BAYES.md).

    The likelihood's Gaussian-linear surrogate is estimated as a posterior carried down an
    annealed split-Gibbs ladder: matrix-normal prior from the previous level, forgetting
    factor and ridge by exact leave-one-out, LOOC noise, cross-fitted per-particle Tweedie
    gains. The forward operator is queried WITH its noise (fresh at every call); its noise
    level is never used. Returns the whole cloud, (num_particles, *shape): every particle
    is a posterior sample.

    Config parameters (configs/algorithm/bslr.yaml)
    -----------------------------------------------
    num_particles   N, simulator calls per level. The error falls like 1/N at high dimension.
    levels, sweeps  the ladder (EDM rho-schedule from sigma_max to sigma_min).
    folds           cross-fitting folds.
    rank            cap on the slope's rank: k denoiser vjps per particle and level.
    draw_steps      denoiser evaluations per draw of p(x | x_t) (not simulator calls).
    """

    def __init__(self, net, forward_op, num_particles: int = 128, levels: int = 20,
                 sweeps: int = 1, folds: int = 8, rank: int = 128, draw_steps: int = 8,
                 sigma_max: float = 80.0, sigma_min: float = 0.05, churn: float = 0.0,
                 progress: bool = True, **kwargs):
        super().__init__(net, forward_op)
        self.kw = dict(num_particles=num_particles, levels=levels, sweeps=sweeps, folds=folds,
                       rank=rank, draw_steps=draw_steps, sigma_max=sigma_max,
                       sigma_min=sigma_min, churn=churn, progress=progress)
        self.prior = EDMPrior(_BatchSigma(self.net))

    @torch.no_grad()
    def inference(self, obs: Tensor, num_samples: int = 1) -> Tensor:
        if torch.is_complex(obs):  # inv_scatter: real and imaginary parts as channels
            obs_in = torch.cat([obs.real, obs.imag], dim=1).to(torch.float32)

            def simulator(x: Tensor) -> Tensor:
                y = self.forward_op({"target": x})
                return torch.cat([y.real, y.imag], dim=1).to(torch.float32)
        else:
            obs_in = obs.to(torch.float32)

            def simulator(x: Tensor) -> Tensor:
                return self.forward_op({"target": x}).to(torch.float32)

        sampler = BayesSLR(self.prior, simulator, obs_in, **self.kw)
        x, self.trace = sampler.sample(self.net.shape, device=self.forward_op.device, trace=True)

        return x
