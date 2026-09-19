# algo/kpsd.py

import torch

from torch import Tensor
from tqdm.auto import tqdm

from algo.base import Algo
from algo.undecorate import differentiable
from algo.azula_bridge import EDMNetDenoiser
from azula.sample import DDIMSampler
from kpsd import KPSDenoiserG, KPSDenoiserH


class KPSDenoiserAlgo(Algo):
    r"""The KPSD posterior denoiser, driven by an azula sampler.

    Each step returns E[x | x_t, y] per particle, with the likelihood -- A, b, Sigma_y --
    fitted from the batch by plain SLR and shared across it. See the `kpsd` package.

    This exists to be compared against KPSH/KPSG, the handcrafted ladder in `kps`, on the same
    problems and at the same simulator budget. It is a different object, not a re-tuning of it:
    the ladder's analysis returns a sample-like point and carries its own re-noising schedule,
    while this returns a conditional mean and lets the sampler supply every bit of the
    dispersion.

    THE SIMULATOR BUDGET IS  steps * sweeps * probe_steps_cost * num_particles, and it has to
    be matched by hand against the ladder before any table means anything. The ladder's
    `gen_N128` spends levels 32 * sweeps 2 * N 128 = 8192 calls with draw_steps 4, so the
    matching setting here is steps=32, sweeps=2, probe_steps=4, num_particles=128 -- one
    simulator call per (step, sweep, particle), plus `probe_steps` PRIOR (not simulator) passes
    to draw the point it is called on.

        steps         outer DDIM steps: how finely the noise level is annealed.
        sweeps        analyses per noise level. sweeps=1 advances every step (the DDIM
                      transition); sweeps>1 re-noises BACK to the current level in between,
                      which is the ladder's split-Gibbs inner loop. It buys a better analysis
                      at one level rather than more levels, and it is the knob that makes the
                      call counts equal.
        probe_steps   inner backward-diffusion steps drawing the x each (x, y) pair sits at.
                      1 collapses onto the Tweedie mean, which is the wrong point to call a
                      nonlinear simulator on -- see `kpsd.denoiser.draw`.
        holdout       folds for the HELD-OUT Sigma_y. BOTH modes need it, for different
                      reasons. mode="h": in sample the residual is structurally ZERO once
                      D_y >= N, because the fit interpolates the cloud (measured 1e-11 on
                      blackhole) -- the update is then told the data are noiseless.
                      mode="g": the slope is not fitted to the pairs, so the in-sample
                      residual looks defensible, but on a NONLINEAR operator each particle's
                      own Jacobian biases the residual scored at that particle -- blackhole
                      cp_chi2 60.6 in sample against 7.2 held out. 1 reverts to in-sample as
                      a control, and is a trap on both paths.
        solve_iter    Krylov iterations for (A V A^T + Sigma_y)^-1. Costs no simulator calls,
                      but on the ladder's G variant it was worth psnr 19.3 -> 34.0 from 2 to
                      16, so it is not a knob to leave small.

    BUDGET IS NOT COMPARABLE BETWEEN THE TWO VARIANTS EITHER:

        mode="h"    one simulator call per step. steps * num_particles calls in total, which
                    is the ladder's accounting exactly.
        mode="g"    the Stein slope applies the average of the per-particle Jacobians, which
                    costs N batched passes per application and two per solver iteration. So a
                    step costs 2 * N * solve_iter jvp/vjp passes over the cloud on top of the
                    one forward call -- hundreds of times the ladder's cost per step. Use a
                    small N and few iterations, and read its wall time as part of the result.

    `ridge` STABILISES THE PARTICLE SLOPE'S INVERSE and is the difference between that variant
    working and collapsing. With the plain pseudo-inverse the fit interpolates the cloud
    whenever D >= N, Sigma_y goes to rank one, and the gain is told the data are noiseless. A
    ridge breaks that: the fit no longer interpolates, so residual reappears inside the cloud's
    span, graded so that poorly explored directions contribute their full variance. Measured on
    a solvable problem with D = 128 > N: energy to the exact posterior 0.803 -> 0.177 at N=16
    and 0.547 -> 0.122 at N=32, with an interior optimum near 0.3 and the spread ratio restored
    from 0.005 to 0.68. It does nothing for mode="g", which inverts nothing.

    `eta` defaults to a saturating value, which clips azula's tau to 1 and makes each step a
    FULL re-noise from the posterior mean -- the split-Gibbs kernel the ladder uses. eta = 1 is
    ordinary DDPM and keeps (sigma_s/sigma_t)^2 of x_t instead; on Navier-Stokes that measured
    0.952 against 0.705 for the full re-noise, a difference of kernel rather than of method.
    """

    def __init__(
        self,
        net,
        forward_op,
        num_particles: int = 128,
        steps: int = 64,
        mode: str = "h",
        solve_iter: int = 8,
        probe_steps: int = 4,
        sweeps: int = 1,
        holdout: int = 2,
        ridge: float = 0.0,
        eta: float = 1e6,
        progress: bool = True,
        sigma_max: float = 80.0,
        sigma_min: float = 0.05,
        **kwargs,
    ):
        super().__init__(net, forward_op, **kwargs)

        if mode not in ("h", "g"):
            raise ValueError(f"mode must be 'h' or 'g', got {mode!r}")

        self.num_particles = num_particles
        self.steps = steps
        self.mode = mode
        self.solve_iter = solve_iter
        self.probe_steps = probe_steps
        self.sweeps = sweeps
        self.holdout = holdout
        self.ridge = ridge
        self.eta = eta
        self.progress = progress
        self.sigma_max = sigma_max
        self.sigma_min = sigma_min

    def _simulator(self, obs: Tensor):
        r"""x -> y, queried WITH the operator's observation noise.

        Sigma_y is estimated from these samples, so a noiseless query would report no noise and
        hand the gain a covariance that claims certainty it does not have. Measured on the
        solvable case: with a noiseless simulator the error stops falling with N.

        Complex observations are packed to real, as in KPSAlgo -- the regression and the solve
        are real-valued.
        """

        # `differentiable` strips Navier-Stokes' @torch.no_grad(), without which
        # torch.func.vjp returns zeros there instead of raising and mode="g" gets A^T = 0.
        op = differentiable(self.forward_op)

        if torch.is_complex(obs):
            y = torch.cat([obs.real, obs.imag], dim=1).to(torch.float32)

            def simulate(x: Tensor) -> Tensor:
                out = op({"target": x})

                return torch.cat([out.real, out.imag], dim=1).to(torch.float32)
        else:
            y = obs.to(torch.float32)

            def simulate(x: Tensor) -> Tensor:
                return op({"target": x}).to(torch.float32)

        return y.reshape(1, -1), simulate

    @torch.no_grad()
    def inference(self, obs: Tensor, num_samples: int = 1) -> Tensor:
        r"""Returns the whole cloud, (num_particles, *shape). `num_samples` is ignored."""

        y, simulate = self._simulator(obs)
        device = self.forward_op.device

        prior = EDMNetDenoiser(self.net, sigma_min=self.sigma_min, sigma_max=self.sigma_max)
        cls = KPSDenoiserH if self.mode == "h" else KPSDenoiserG
        post = cls(prior, y, simulate, iterations=self.solve_iter, ridge=self.ridge,
                   probe_steps=self.probe_steps, holdout=self.holdout)

        sampler = DDIMSampler(post, eta=self.eta, steps=self.steps, device=device, silent=True)
        x = torch.randn(self.num_particles, *tuple(self.net.shape),
                        device=device) * self.sigma_max

        ts = sampler.timesteps
        calls = self.steps * self.sweeps * self.num_particles
        bar = tqdm(total=self.steps, desc=f"KPSD-{self.mode} ({calls} calls)", unit="step",
                   leave=False, disable=not self.progress)

        for t_cur, t_nxt in zip(ts[:-1], ts[1:]):
            # SCALAR time, following azula: its samplers multiply alpha_t straight into x
            # without reshaping, so a per-particle time vector broadcasts against the image
            # axes and fails.
            t = torch.tensor(float(t_cur), device=device)

            # The extra sweeps land BACK at t: analyse, re-noise to the SAME level, repeat.
            # Only the last one transitions to t_nxt, so `steps` still sets the ladder's
            # resolution and `sweeps` only sets how hard each rung is worked.
            #
            # This CANNOT go through sampler.step with s = t: azula's tau is
            # 1 - (alpha_t/alpha_s * sigma_s/sigma_t)^2, which is exactly 0 there, so the step
            # returns x_t unchanged and the sweep would silently cost calls and do nothing.
            # Written out, it is the split-Gibbs kernel -- the same full re-noise the eta
            # default produces, at t instead of t_nxt.
            alpha_t, sigma_t = post.schedule(t)

            for _ in range(self.sweeps - 1):
                x = alpha_t * post(x, t).mean + sigma_t * torch.randn_like(x)

            x = sampler.step(x, t, torch.full_like(t, float(t_nxt)))

            bar.set_postfix(sigma=f"{float(post.schedule(t)[1].reshape(-1)[0]):.3g}",
                            spread=f"{float(x.std(0).mean()):.3g}", refresh=False)
            bar.update(1)

        bar.close()

        return x
