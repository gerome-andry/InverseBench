# algo/kpsd.py

import torch

from torch import Tensor

from tqdm.auto import tqdm

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

    `eta` HERE IS THE SAMPLER'S, not the inner draw's. The KPSDenoiser has its own eta for the
    probe integration, which stays at 0 (the probability-flow ODE); this one sets the outer
    kernel.

    ETA IS THE KERNEL KNOB, AND 1 IS NOT THE LADDER. With alpha = 1, azula's step keeps
    (sigma_s/sigma_t)^2 of x_t; the ladder discards x_t and re-noises from the analysis point,
    which is the tau = 1 corner and needs a LARGE eta. Measured on one Navier-Stokes instance
    at 32 steps: eta = 1 scored err 0.952 against the full re-noise's 0.705, a difference of
    kernel rather than of method. The default here is therefore a saturating eta.

    BUDGET. steps * sweeps * num_particles simulator calls, plus one denoiser evaluation and `rank`
    vector-Jacobian products per step. There is no inner ODE, so a step is cheaper than a
    ladder sweep at the same N.
    """

    def __init__(
        self,
        net,
        forward_op,
        num_particles: int = 128,
        steps: int = 64,
        sweeps: int = 1,
        rank: int = 32,
        probe_steps: int = 8,
        sigma_mode: str = "iso",
        mode: str = "h",
        solve_iter: int = 16,
        eta: float = 1e6,
        progress: bool = True,
        sigma_max: float = 80.0,
        sigma_min: float = 0.05,
        sim_batch: int = 0,
        **kwargs,
    ):
        super().__init__(net, forward_op, **kwargs)

        self.num_particles = num_particles
        self.steps = steps
        self.sweeps = sweeps
        self.rank = rank
        self.probe_steps = probe_steps
        self.sigma_mode = sigma_mode
        self.mode = mode
        self.solve_iter = solve_iter
        self.eta = eta
        self.progress = progress
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
                           probe_steps=self.probe_steps, sigma_mode=self.sigma_mode,
                           solve_iter=self.solve_iter)

        sampler = DDIMSampler(post, eta=self.eta, steps=self.steps, device=device,
                              silent=True)
        x = torch.randn(self.num_particles, *tuple(self.net.shape),
                        device=device) * self.sigma_max

        ts = sampler.timesteps
        bar = tqdm(total=self.steps * self.sweeps, desc="KPSD", unit="sweep",
                   leave=False, disable=not self.progress)

        for t_cur, t_nxt in zip(ts[:-1], ts[1:]):
            # SCALAR, following azula: its samplers multiply alpha_t straight into x without
            # reshaping, so a per-particle time vector broadcasts against the image axes and
            # fails. The denoiser expands it to the batch itself.
            t = torch.tensor(float(t_cur), device=device)

            # GIBBS SWEEPS AT FIXED SIGMA. A DDIM step cannot express these: at s = t its
            # tau = 1 - (sigma_s/sigma_t)^2 is zero, so the step is the identity. The ladder's
            # sweep is a full refresh at the SAME noise level -- analyse, then re-noise from
            # the analysis point -- which is this.
            for _ in range(self.sweeps - 1):
                alpha_t, sigma_t = post.schedule(t)
                shape = (-1, *(1,) * (x.ndim - 1))
                mean = post(x, t).mean
                x = alpha_t.reshape(shape).to(x) * mean \
                    + sigma_t.reshape(shape).to(x) * torch.randn_like(x)
                bar.set_postfix(sigma=f"{float(sigma_t.reshape(-1)[0]):.3g}",
                                spread=f"{float(x.std(0).mean()):.3g}", refresh=False)
                bar.update(1)

            x = sampler.step(x, t, torch.full_like(t, float(t_nxt)))
            bar.set_postfix(sigma=f"{float(post.schedule(t)[1].reshape(-1)[0]):.3g}",
                            spread=f"{float(x.std(0).mean()):.3g}", refresh=False)
            bar.update(1)

        bar.close()

        return x
