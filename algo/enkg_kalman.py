import torch

from .enkg import EnKG

import wandb


# ----------------------------------------------------------------------------------------
# EnKG with the KPS Kalman solve injected, to isolate what the inverse actually buys.
#
# EnKG builds its increment from the TRANSPOSE, `coef = err @ dY^T / N`, which is the
# alpha -> infinity limit of the damped Kalman step `P A^T (A P A^T + alpha Omega)^-1 d`.
# Re-introducing a finite alpha turns it back into an ensemble Kalman analysis, done here in
# the N x N ensemble subspace so it costs one small dense solve and no Krylov iterations:
#
#     coef_kalman = (err @ dY^T / N) @ (dY dY^T / N + alpha I)^-1
#
# alpha large  -> recovers EnKG (the inverse tends to I / alpha, a pure rescale)
# alpha small  -> full Kalman analysis, the sharp KPS-style step
#
# `normalize` keeps or removes EnKG's own `lr = scale / |coef|` trust region, so the solve
# and the step bound can be attributed separately.
# ----------------------------------------------------------------------------------------


class EnKGKalman(EnKG):
    def __init__(self, *args, alpha: float = 1.0, normalize: bool = True,
                 renoise: bool = False, **kwargs):
        super().__init__(*args, **kwargs)

        self.alpha = alpha
        self.normalize = normalize
        self.renoise = renoise

    @torch.no_grad()
    def update_particles(self, particles, observation, num_steps, sigma_start, guidance_scale=1.0):
        x0s = torch.zeros_like(particles)
        num_batchs = particles.shape[0] // self.batch_size
        N, *spatial = particles.shape
        t_hat = sigma_start

        from .enkg import ode_sampler

        for j in range(self.num_updates):
            for i in range(num_batchs):
                start, end = i * self.batch_size, (i + 1) * self.batch_size
                x0s[start:end] = ode_sampler(
                    self.net, particles[start:end], num_steps=num_steps, sigma_start=sigma_start
                )

            ys = self.forward_op.forward(x0s)

            # EnKG's `gradient_m` autodiffs `sum((m - o)^2)`, which is analytically
            # `2 (m - o)` and which it then halves -- so the residual is just `m - o`.
            # Computing it directly also sidesteps autograd's refusal to differentiate a
            # complex loss, which is what stops stock EnKG running on inv-scatter. Complex
            # measurements are embedded as real/imag, exactly as `algo/kps.py` does, so the
            # ensemble algebra stays real.
            obs = observation

            if torch.is_complex(ys):
                ys_r = torch.cat([ys.real, ys.imag], dim=1).float()
                obs_r = torch.cat([obs.real, obs.imag], dim=1).float()
            else:
                ys_r, obs_r = ys.float(), obs.float()

            xs_diff = particles - particles.mean(dim=0, keepdim=True)
            ys_diff = ys_r - ys_r.mean(dim=0, keepdim=True)
            ys_err = ys_r - obs_r

            Y = ys_diff.reshape(N, -1)
            E = ys_err.reshape(N, -1)

            coef = torch.matmul(E, Y.T) / N  # (N, N) -- EnKG stops here

            # the Kalman correction: divide by the observation-space Gram instead of
            # leaving the transpose bare
            G = torch.matmul(Y, Y.T) / N
            scale = G.diagonal(dim1=-2, dim2=-1).mean().clamp(min=1e-30)
            eye = torch.eye(N, device=G.device, dtype=G.dtype)

            coef = torch.linalg.solve(G + self.alpha * scale * eye, coef.T).T

            if self.normalize:
                lr = guidance_scale / torch.linalg.matrix_norm(coef)
            else:
                lr = guidance_scale

            if self.renoise:
                # KPS's Gibbs structure: correct the CLEAN state, then re-noise. EnKG instead
                # perturbs the noisy particle directly and lets the next ODE pass absorb it.
                # Re-noising is an explicit projection back onto the prior manifold, which on
                # KPS was the single most effective navier-stokes knob (y-misfit 0.917 ->
                # 0.181 from gibbs 2 -> 16). EDM parametrises x_t = x0 + sigma * eps.
                x0_diff = x0s - x0s.mean(dim=0, keepdim=True)
                dxs = coef @ x0_diff.reshape(N, -1)

                x0_new = x0s - lr * dxs.reshape(N, *spatial)
                particles = x0_new + sigma_start * torch.randn_like(x0_new)
            else:
                dxs = coef @ xs_diff.reshape(N, -1)
                particles = particles - lr * dxs.reshape(N, *spatial)

        return particles, t_hat
