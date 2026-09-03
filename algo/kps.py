# algo/kps.py

import os
import torch

from torch import Tensor
from typing import Optional

from algo.azula_bridge import EDMNetDenoiser
from algo.base import Algo
from kps.sampler import PosteriorGibbsSampler
from kps.update import GIPLFUpdate, HIPLFUpdate, PIPLFUpdate


class KPSAlgo(Algo):
    """
    Plug-and-play KPS sampler for InverseBench.

    Config parameters (configs/algorithm/kps{p,h,g}.yaml)
    -----------------------------------------------------
    num_steps       Outer diffusion steps.
    posterior_iter  Linearisation refinements per posterior update.
    gibbs_iter      Gibbs sweeps per diffusion step.
    num_particles   Ensemble size. Drives the rank of the fitted slope.
    prior_mode      "particles" (ensemble covariance) or "gradient" (analytic, via vjp).
    slope_mode      "particles" (statistical linear regression) or "gradient" (Jacobian).
    solve_iter      Krylov iterations in the Kalman solve.
    ridge_x         Ridge on the state covariance. None (default) uses the smallest
                    eigenvalue above the numerical rank tolerance -- scale-free and
                    precision-free. Unused by GIPLF, whose slope is an autodiff Jacobian
                    and never forms Cxx.
    ridge_y         Floor on the observation-noise covariance. None (default) uses
                    mean(dR^2), the ML estimate of the noise variance from the regression
                    residual, which tracks the data scale on its own.
    importance      Importance-sample the returned particle instead of taking x_k[0].

    The (prior_mode, slope_mode) pair selects the update:
        (particles, particles) -> PIPLF
        (gradient,  particles) -> HIPLF
        (gradient,  gradient)  -> GIPLF
    """

    def __init__(
        self,
        net,
        forward_op,
        num_steps: int = 100,
        posterior_iter: int = 2,
        gibbs_iter: int = 2,
        num_particles: Optional[int] = None,
        prior_mode: str = "particles",
        slope_mode: str = "particles",
        solve_iter: int = 2,
        ridge_x: Optional[float] = None,
        ridge_y: Optional[float] = None,
        importance: bool = True,
        **kwargs,
    ):
        super().__init__(net, forward_op, **kwargs)

        if os.environ.get("KPS_PROBE"):
            import kps_probe

            kps_probe.arm()

        self.num_steps = num_steps
        self.num_particles = num_particles if num_particles else 2
        self.posterior_iter = posterior_iter
        self.solve_iter = solve_iter
        self.ridge_x = ridge_x
        self.ridge_y = ridge_y
        self.importance = importance
        self.gibbs_iter = gibbs_iter

        updates = {
            ("particles", "particles"): PIPLFUpdate,
            ("gradient", "particles"): HIPLFUpdate,
            ("gradient", "gradient"): GIPLFUpdate,
        }

        if (prior_mode, slope_mode) not in updates:
            raise NotImplementedError(
                f"No update for prior_mode={prior_mode!r}, slope_mode={slope_mode!r}."
            )

        self.update = updates[prior_mode, slope_mode]

        # Convert the InverseBench net into an azula Denoiser once, at construction.
        # The noise range comes from the net, since it differs per preconditioner.
        self.denoiser = EDMNetDenoiser(net=self.net)

    @torch.no_grad()
    def inference(self, obs: Tensor, num_samples: int = 1) -> Tensor:
        device = self.forward_op.device

        # obs stays a single observation: KPS broadcasts one y over the whole particle
        # cloud, and every sample is drawn from the posterior for that same observation.
        if torch.is_complex(obs):
            # inv_scatter case
            obs_in = torch.cat([obs.real, obs.imag], dim=1).to(torch.float32)

            def likelihood(x: Tensor) -> Tensor:
                y = self.forward_op({"target": x})

                return torch.cat([y.real, y.imag], dim=1).to(torch.float32)
        else:
            # blackhole (and NS) — already real
            obs_in = obs.to(torch.float32)

            def likelihood(x: Tensor) -> Tensor:
                return self.forward_op({"target": x}).to(torch.float32)

        post_update = self.update(
            y=obs_in,
            likelihood=likelihood,
            solve_iter=self.solve_iter,
            posterior_iter=self.posterior_iter,
            ridge_x=self.ridge_x,
            ridge_y=self.ridge_y,
            importance=self.importance,
        )

        sampler = PosteriorGibbsSampler(
            denoiser=self.denoiser,
            posterior_update=post_update,
            gibbs_iter=self.gibbs_iter,
            inner_steps_factor=1,
            steps=self.num_steps,
            num_particles=self.num_particles,
        )

        self._last_update = post_update

        x1 = sampler.init((num_samples, *self.net.shape), device=device)
        x0 = sampler(x1)

        if os.environ.get("KPS_PROBE"):
            import kps_probe

            kps_probe.report(self)

        return x0
