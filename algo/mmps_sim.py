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


class MMPSSimAlgo(Algo):
    r"""MMPS driven by the forward operator itself, with the stated observation noise.

    This is the simplest thing that can work on these problems, and it is here because it may
    never have been tried. MMPS already supports a NONLINEAR operator: it linearises per
    particle at the Tweedie mean by jvp/vjp and its innovation is y - h(x_hat_i), which
    carries no linearisation error at all. So handing it the simulator needs

        no statistical linear regression        no Sigma_y estimation
        no held-out residual, ridge or rank     no coupling between particles

    and the particles are independent, so one batched jvp/vjp pair per Krylov iteration covers
    the whole cloud -- against N^2 simulator passes for the averaged-Jacobian variant in
    `kpsd`. Each particle is an independent posterior sample; `num_particles` only sets how
    many you draw.

    WHAT IT GIVES UP against the fitted-likelihood variants: A is the LOCAL Jacobian at
    x_hat_i, not an average over the cloud, so nothing here represents the curvature of h
    across the ensemble, and Sigma_y is the stated noise rather than noise plus linearisation
    error. `inflate` is the one knob for the latter -- Sigma_y = inflate * noise_var -- and it
    exists because the linearisation residual has measured several times larger than the
    observation noise on these problems.

    `noise_var` defaults to the problem's own sigma_noise^2, floored, since a zero Sigma_y
    makes the gain singular and blackhole ships sigma_noise = 0.
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

        var = self.noise_var
        if var is None:
            var = float(getattr(self.forward_op, "sigma_noise", 0.0)) ** 2
        var = max(var, 1e-8) * self.inflate

        prior = EDMNetDenoiser(self.net, sigma_min=self.sigma_min, sigma_max=self.sigma_max)
        post = MMPSDenoiser(prior, y, op,
                            IsotropicCovariance(torch.tensor(var, device=device)),
                            iterations=self.solve_iter)

        sampler = DDIMSampler(post, eta=self.eta, steps=self.steps, device=device, silent=True)
        x = torch.randn(self.num_particles, *tuple(self.net.shape),
                        device=device) * self.sigma_max

        ts = sampler.timesteps
        bar = tqdm(total=self.steps, desc=f"MMPS-sim (var={var:.3g})", unit="step",
                   leave=False, disable=not self.progress)

        for t_cur, t_nxt in zip(ts[:-1], ts[1:]):
            t = torch.tensor(float(t_cur), device=device)
            x = sampler.step(x, t, torch.full_like(t, float(t_nxt)))

            bar.set_postfix(sigma=f"{float(post.schedule(t)[1].reshape(-1)[0]):.3g}",
                            spread=f"{float(x.std(0).mean()):.3g}", refresh=False)
            bar.update(1)

        bar.close()

        return x
