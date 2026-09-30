# algo/mmps_sim.py

import torch

from torch import Tensor
from tqdm.auto import tqdm

from algo.base import Algo
from algo.azula_bridge import EDMNetDenoiser
from algo.undecorate import differentiable
from azula.guidance.mmps import MMPSDenoiser
from azula.linalg.covariance import IsotropicCovariance
from azula.sample import DDIMSampler


class InnovationScale:
    r"""Keeps azula's own IsotropicCovariance and rewrites its scalar each step.

    Sigma_y = var(y - h(x_hat)) * inflate, read off the innovation MMPS already needs. The
    covariance stays an `IsotropicCovariance` over a 0-d tensor that is filled in place, so
    azula's solver sees exactly the type it expects -- passing a bare callable instead gets
    coerced into `IsotropicCovariance(lmbda=<the callable>)` and fails inside __matmul__.
    """

    def __init__(self, prior, y, op, inflate: float = 1.0, device=None) -> None:
        self.prior, self.y, self.op, self.inflate = prior, y, op, inflate
        self.lmbda = torch.tensor(1.0, device=device)
        self.cov = IsotropicCovariance(self.lmbda)

    @torch.no_grad()
    def set(self, x_t: Tensor, t: Tensor) -> float:
        resid = self.y - self.op(self.prior(x_t, t).mean)
        var = float(resid.pow(2).mean()) * self.inflate
        self.lmbda.fill_(max(var, 1e-12))

        return var


class MMPSSimAlgo(Algo):
    r"""MMPS driven by the forward operator itself, with the stated observation noise.

    This is the simplest thing that can work on these problems, and it is here because it may
    never have been tried. MMPS already supports a NONLINEAR operator: it linearises per
    particle at the Tweedie mean by jvp/vjp and its innovation is y - h(x_hat_i), which
    carries no linearisation error at all. So handing it the simulator needs

        no statistical linear regression        no Sigma_y estimation
        no held-out residual, ridge or rank     no coupling between particles

    and the particles are independent, so one batched jvp/vjp pair per Krylov iteration covers
    the whole cloud -- against N^2 simulator passes for an averaged-Jacobian variant. Each
    particle is an independent posterior sample; `num_particles` only sets how many you draw.

    WHAT IT GIVES UP against the fitted-likelihood variants: A is the LOCAL Jacobian at
    x_hat_i, not an average over the cloud, so nothing here represents the curvature of h
    across the ensemble, and Sigma_y is the stated noise rather than noise plus linearisation
    error. `inflate` is the one knob for the latter -- Sigma_y = inflate * noise_var -- and it
    exists because the linearisation residual has measured several times larger than the
    observation noise on these problems.

    SIGMA_Y IS READ OFF THE INNOVATION, which is what `noise_var=None` means and is the
    default. The stated observation noise is the wrong scale by a mile -- Navier-Stokes ships
    sigma_noise^2 = 1e-8 while the measured residual variance per component runs 0.1 to 27, so
    the gain is told the data are perfect and the chain breaks (relative l2 1.225, worse than
    the prior). Multiplying that tiny number does nothing either: 1x and 1000x measured 1.2250
    and 1.2253, because both stay far below tr(A V A^T)/D_y.

    And no CONSTANT would work, because the residual moves 250x along the chain:

        sigma        76.4    9.67    2.49    0.46    0.09
        resid/D_y    26.7    20.5    6.76    1.19    0.106
        tr AVA/D_y    107    90.1    5.48    0.42    0.059

    so Sigma_y is comparable to A V A^T (ratio 0.22 to 2.8) when set correctly -- it is never
    negligible and never dominant.

    The innovation MMPS already forms, y - h(x_hat), IS that mismatch, so its variance is
    Sigma_y, per step, for free. It needs no ensemble and no extra simulator call, and unlike
    the regression variants it cannot interpolate its way to zero: h(x_hat) has no free
    parameters. Setting `noise_var` to a number overrides it with a constant, and `inflate`
    scales whichever is used.
    """

    def __init__(
        self,
        net,
        forward_op,
        num_particles: int = 128,
        steps: int = 64,
        solve_iter: int = 8,
        noise_var: float | None = None,
        inflate: float = 1.0,
        eta: float = 1.0,
        progress: bool = True,
        sigma_max: float = 80.0,
        sigma_min: float = 0.05,
        **kwargs,
    ):
        super().__init__(net, forward_op, **kwargs)

        self.num_particles = num_particles
        self.steps = steps
        self.solve_iter = solve_iter
        self.noise_var = noise_var
        self.inflate = inflate
        self.eta = eta
        self.progress = progress
        self.sigma_max = sigma_max
        self.sigma_min = sigma_min

    def _operator(self, obs: Tensor):
        r"""(y, op) with y packed to real and the operator's no_grad bypassed."""

        # `raw` and `packed` must not share a name: `raw` is captured by the closures below,
        # and rebinding that name to the wrapper makes the wrapper call itself.
        raw = differentiable(self.forward_op)

        if torch.is_complex(obs):
            y = torch.cat([obs.real, obs.imag], dim=1).to(torch.float32)

            def packed(x: Tensor) -> Tensor:
                out = raw({"target": x})

                return torch.cat([out.real, out.imag], dim=1).to(torch.float32)
        else:
            y = obs.to(torch.float32)

            def packed(x: Tensor) -> Tensor:
                return raw({"target": x}).to(torch.float32)

        return y.reshape(1, -1), lambda x: packed(x).reshape(x.shape[0], -1)

    @torch.no_grad()
    def inference(self, obs: Tensor, num_samples: int = 1) -> Tensor:
        y, op = self._operator(obs)
        device = self.forward_op.device

        prior = EDMNetDenoiser(self.net, sigma_min=self.sigma_min, sigma_max=self.sigma_max)

        scale = None
        if self.noise_var is None:
            scale = InnovationScale(prior, y, op, self.inflate, device)
            cov = scale.cov
        else:
            cov = IsotropicCovariance(
                torch.tensor(max(self.noise_var, 1e-12) * self.inflate, device=device))

        post = MMPSDenoiser(prior, y, op, cov, iterations=self.solve_iter)

        sampler = DDIMSampler(post, eta=self.eta, steps=self.steps, device=device, silent=True)
        x = torch.randn(self.num_particles, *tuple(self.net.shape),
                        device=device) * self.sigma_max

        ts = sampler.timesteps
        bar = tqdm(total=self.steps, desc="MMPS-sim", unit="step",
                   leave=False, disable=not self.progress)

        for t_cur, t_nxt in zip(ts[:-1], ts[1:]):
            t = torch.tensor(float(t_cur), device=device)

            if scale is not None:
                scale.set(x, t)

            x = sampler.step(x, t, torch.full_like(t, float(t_nxt)))

            bar.set_postfix(sigma=f"{float(post.schedule(t)[1].reshape(-1)[0]):.3g}",
                            sy=f"{float(cov.lmbda) if scale else 0:.3g}",
                            spread=f"{float(x.std(0).mean()):.3g}", refresh=False)
            bar.update(1)

        bar.close()

        return x
