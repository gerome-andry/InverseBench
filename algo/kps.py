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
    importance      Importance-sample the cloud. With `maintain` this is SIR over the whole
                    ensemble (`select_all`), which is what carries calibration; without it,
                    one particle is drawn and the rest discarded.
    maintain        Keep one persistent ensemble across the trajectory instead of rebuilding
                    it from a single x_t at every step. Every particle persists and is
                    returned, so the output is N samples rather than a point estimate. A
                    conditional cloud degenerates as t -> 0: its width is set by the
                    schedule, so dY eventually falls below the observation noise and the
                    regression fits noise (measured dR/dY 7% -> 113% on navier-stokes).
    resample_noise  Whether each sweep redraws the particle's noise. False keeps it, shifting
                    x_t by alpha_t * (x_new - x). Only meaningful with `maintain`.
    cov_mode        How the analytic prior covariance is taken with a maintained cloud.
                    "point" linearises at the cloud mean; "ensemble" averages the Tweedie
                    covariance over every particle, as GIPLF already does for its Jacobian.
                    Unused by PIPLF, which never calls cov_x.
    inner_steps_factor
                    Divides the inner denoising depth. A maintained cloud does not need an
                    accurate redraw of p(x0|x_t), so 4 measured 2.1x cheaper at equal quality.

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
        maintain: bool = False,
        resample_noise: bool = True,
        cov_mode: str = "point",
        inner_steps_factor: int = 1,
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
        self.maintain = maintain
        self.resample_noise = resample_noise
        self.cov_mode = cov_mode
        self.inner_steps_factor = inner_steps_factor
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
            return_all=self.maintain,
        )

        sampler = PosteriorGibbsSampler(
            denoiser=self.denoiser,
            posterior_update=post_update,
            gibbs_iter=self.gibbs_iter,
            inner_steps_factor=self.inner_steps_factor,
            maintain=self.maintain,
            resample_noise=self.resample_noise,
            cov_mode=self.cov_mode,
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
