#!/usr/bin/env python
"""Component-level diagnostic for the KPS update.

End-to-end PSNR cannot resolve anything small here: re-running one identical config spans
~5 dB. These tests are exact instead -- adjointness, symmetry, slope error against the true
Jacobian, solver residual -- so each component either passes or fails on its own terms.

The tests run inside the real pipeline: the update's `__call__` is patched, so the cloud and
`cov_x` are exactly what the sampler produced at the first posterior step. Then it aborts,
so a diagnostic costs one Gibbs step, not a full sample.

    uv run python diagnose.py problem=inv-scatter algorithm=kpsp
    uv run python diagnose.py problem=inv-scatter algorithm=kpsp ++algorithm.method.num_particles=256
"""

import os
import pickle

import hydra
import torch

from hydra.utils import instantiate
from torch.utils.data import DataLoader

import kps.update as U

from kps.localize import estimate_radius, grid_modulators
from kps.covariance import DeltaCovariance, RidgeDeltaCovariance
from kps.update import anomaly, flat

from utils.helper import open_url

ROWS = []


class Done(Exception):
    r"""Raised once the diagnostics have run, to abort the sample."""


def row(name, value, ok, note=""):
    ROWS.append((name, value, ok, note))


def rel(a, b):
    r"""Relative error, complex-safe."""

    a, b = a.reshape(-1).to(torch.complex128), b.reshape(-1).to(torch.complex128)

    return (torch.linalg.norm(a - b) / torch.linalg.norm(b).clamp(min=1e-30)).real.item()


def rand_like_cloud(t):
    return torch.randn_like(t)


# --------------------------------------------------------------------------------------
# the tests
# --------------------------------------------------------------------------------------


def t_operator_linearity(upd, x):
    r"""Is the forward map linear? Decides whether the slope can be scored against truth.

    A linear (Born) operator makes the true Jacobian the operator itself, which turns the
    slope test from a sanity check into an exact error measurement.
    """

    x1, x2 = x[:1], x[1:2]
    a, b = 0.7, -1.3

    lhs = upd.stack(a * x1 + b * x2)
    rhs = a * upd.stack(x1) + b * upd.stack(x2)

    # an affine operator leaves a constant that cancels in the combination above only when
    # a + b == 1, so compare against the affine-corrected form as well
    off = upd.stack(torch.zeros_like(x1))
    rhs_affine = rhs + (1 - a - b) * off

    e_lin, e_aff = rel(lhs, rhs), rel(lhs, rhs_affine)
    e = min(e_lin, e_aff)

    # float32 through a physics solve will not superpose to machine precision; anything at
    # the percent level still lets a finite-difference Jacobian stand in for the truth,
    # since the rank-deficiency signal we are after is orders of magnitude larger
    ok = e < 2e-2

    row("operator linearity", f"{e:.2e}", ok,
        "~linear -> slope scored vs finite-diff J" if ok else "nonlinear -> slope test skipped")

    return ok


def t_adjoint(A, At, x, y_k):
    r"""<u, A v> == <At u, v>. Catches a silently zero vjp (the @torch.no_grad() trap)."""

    v = torch.randn_like(flat(x))
    u = torch.randn_like(flat(y_k))

    lhs = (u * A(v)).sum()
    rhs = (At(u) * v).sum()

    err = ((lhs - rhs).abs() / lhs.abs().clamp(min=1e-30)).item()

    # a zero adjoint passes the identity trivially when both sides vanish, so check the norm
    at_norm = At(u).abs().max().item()

    # the ridged inverse is itself only ~1% accurate in float32, so adjointness cannot
    # be better than that; a structural mismatch would show up far above this scale
    row("adjoint <u,Av>=<Atu,v>", f"{err:.2e}", err < 2e-2, "float32 floor ~1e-2")
    row("  At non-degenerate", f"{at_norm:.2e}", at_norm > 0,
        "zero => vjp is dead (no_grad on the operator)" if at_norm == 0 else "")


def t_cov_x(cov_x, x):
    r"""The analytic prior covariance must be symmetric PSD, or cg and cov_kalman are invalid."""

    if cov_x is None:
        row("cov_x symmetry", "n/a", True, "particle prior, cov_x unused")
        return

    N, B, *shape = x.shape
    v, w = torch.randn(B, *shape, device=x.device), torch.randn(B, *shape, device=x.device)

    sym = ((v * cov_x(w)).sum() - (w * cov_x(v)).sum()).abs()
    scale = (v * cov_x(w)).sum().abs().clamp(min=1e-30)
    quad = (v * cov_x(v)).sum().item()

    row("cov_x symmetry", f"{(sym / scale).item():.2e}", (sym / scale).item() < 1e-2)
    row("cov_x PSD <v,Cv>", f"{quad:.3e}", quad >= 0)

    # how negative, relative to the top of the spectrum: a mild dip can be shifted away,
    # a large one means the operator is not a perturbed covariance at all
    def power(op, iters=25):
        u = torch.randn(B, *shape, device=x.device)
        u = u / u.norm()

        for _ in range(iters):
            u = op(u)
            n = u.norm()

            if n < 1e-30:
                return 0.0

            u = u / n

        return (u * op(u)).sum().item()

    lam_max = power(cov_x)
    # the most negative eigenvalue, via the spectrum shifted to make it dominant
    shifted = lambda v: abs(lam_max) * v - cov_x(v)
    lam_min = abs(lam_max) - power(shifted)

    ratio = lam_min / abs(lam_max) if lam_max != 0 else float("nan")

    row("  cov_x lambda_max", f"{lam_max:.3e}", True)
    row("  cov_x lambda_min", f"{lam_min:.3e}", lam_min >= 0,
        f"lambda_min/|lambda_max| = {ratio:.2f}")


def t_slope_quality(upd, A, x, y_k):
    r"""How good the fitted slope is, and the structural bound on how good it could be.

    `dY ~= A dX` is the regression's defining property and needs no finite differences, so
    it measures the fit itself rather than my ability to probe it. The rank line is the
    bound no amount of conditioning can move: a slope fitted from N members spans at most
    N-1 directions of a Dy-dimensional observation space.
    """

    dX, dY = anomaly(x), anomaly(y_k)

    e_fit = rel(A(dX), dY)

    row("slope: A(dX) vs dY", f"{e_fit:.3f}", e_fit < 0.3,
        "in-span fit quality; = dR/dY in the probe")
    row("  rank vs obs dim", f"{x.shape[0] - 1}/{upd.y.numel()}", True,
        "the fit can only ever span the former")


def t_residual_reachable(upd, S, y_k):
    r"""How much of the data misfit lies in span(dY), the only place the update can act.

    The correction enters observation space through A, whose range is span(dY). Whatever
    part of `y - g(x)` lies outside it cannot be removed by any solver or ridge -- it is a
    ceiling set purely by the ensemble size.
    """

    dY = anomaly(y_k)
    r = upd.y.reshape(1, 1, -1) - S.y_k.mean(1, keepdim=True)

    Gram = torch.einsum("bne,bme->bnm", dY, dY)
    c = torch.einsum("bne,bme->bnm", dY, r)
    alpha = torch.linalg.lstsq(Gram, c).solution
    r_par = torch.einsum("bnm,bne->bme", alpha, dY)

    frac = (r_par.norm() / r.norm().clamp(min=1e-30)).item()

    row("residual in span(dY)", f"{frac:.3f}", frac > 0.5,
        "fraction of the misfit the update can even address")



def t_increment_anatomy(upd, S, prior, x, y_k):
    r"""Where the high-frequency energy in the update comes from.

    On NS the reconstruction carries 14x the target's fine-scale energy. Three candidates:
    the ensemble span is already contaminated, the solve's coefficients excite its wiggly
    members, or the increment leaves the span entirely (float32 through Cxx^-1).
    """

    from azula.linalg.solve import gmres

    def hf(flat_img, shape):
        img = flat_img.reshape(-1, *shape)[0]
        P = torch.fft.fftshift(torch.fft.fft2(img.squeeze())).abs() ** 2
        n = P.shape[-1]
        yy, xx = torch.meshgrid(torch.arange(n) - n // 2, torch.arange(n) - n // 2, indexing="ij")
        rad = (yy.float() ** 2 + xx.float() ** 2).sqrt().to(P.device) / (n / 2)

        return (P[rad > 0.35].sum() / P.sum().clamp(min=1e-30)).item()

    N, B, *shape = x.shape
    grid = shape[-2:]

    def cov_kalman(u):
        return S.A(prior(S.At(u))) + S.omega(u)

    b = upd.y.reshape(1, 1, -1) - S.y_k
    dx = gmres(A=cov_kalman, b=b, iterations=2)
    incr = prior(S.At(dx))  # (B, N, Dx) -- what actually gets added to the state

    dX = anomaly(x)

    # 1. is the ensemble span itself fine-scale?
    row("HF of cloud member", f"{hf(flat(x)[0, :1], shape):.4f}", True, "the state being corrected")
    row("HF of dX anomaly", f"{hf(dX[0, :1], shape):.4f}", True, "the span the update lives in")

    # 2. is the increment fine-scale relative to the span it is built from?
    row("HF of increment", f"{hf(incr[0, :1], shape):.4f}", True, "prior(At(dx)), one solve")

    # 3. does the increment actually stay in span(dX)?
    G = torch.einsum("bne,bme->bnm", dX, dX)
    c = torch.einsum("bne,bme->bnm", dX, incr[:, :1])
    alpha = torch.linalg.lstsq(G, c).solution
    proj = torch.einsum("bnm,bne->bme", alpha, dX)
    out_frac = ((incr[:, :1] - proj).norm() / incr[:, :1].norm().clamp(min=1e-30)).item()

    row("increment out of span", f"{out_frac:.4f}", out_frac < 0.1,
        "0 => confined to ensemble span, as ETKF would guarantee")

    # 4. relative step size, for a gamma_t-based trust region
    rel_step = (incr.norm() / flat(x).norm().clamp(min=1e-30)).item()
    row("step / |x|", f"{rel_step:.4f}", True, "size of one increment")

    # the trust region measures against the ensemble spread, so report that ratio too
    spread = dX.norm()
    row("step / spread", f"{(incr.norm() / spread.clamp(min=1e-30)).item():.3f}", True,
        "the quantity max_step caps")


def t_kalman_solve(upd, S, prior, y):
    r"""Residual left by the Kalman solve. A barely-solved system is a barely-applied update."""

    from azula.linalg.solve import gmres

    def cov_kalman(u):
        return S.A(prior(S.At(u))) + S.omega(u)

    b = y.reshape(1, 1, -1) - S.y_k

    for it in (1, 2, 4, 8, 16, 32):
        dx = gmres(A=cov_kalman, b=b, iterations=it)
        res = rel(cov_kalman(dx), b)
        row(f"solve residual @{it} iter", f"{res:.3f}", res < 0.2,
            "fraction of the Kalman system left unsolved")

    # Omega = dR dR^T + ridge_y I has an exact inverse already, so left-preconditioning by it
    # is free. K = A P A^T + Omega, so Omega^-1 K = Omega^-1 A P A^T + I -- a spectrum
    # clustered around 1, which is what GMRES converges fast on. Residuals below are the
    # TRUE ones, ||K x - b||, not the preconditioned ones GMRES minimises.
    Minv = S.omega.inv

    pre_op = lambda u: Minv(cov_kalman(u))
    pre_b = Minv(b)

    for it in (1, 2, 4, 8):
        dx = gmres(A=pre_op, b=pre_b, iterations=it)
        res = rel(cov_kalman(dx), b)
        row(f"  precond @{it} iter", f"{res:.3f}", res < 0.2,
            "same system, left-preconditioned by Omega^-1")

    # A PIPLF prior IS the particle covariance, so A P A^T = A dX dX^T A^T ~= dY dY^T and the
    # whole Kalman operator is low-rank + ridge: dY dY^T + dR dR^T + ridge_y I. That has an
    # exact closed-form inverse here -- no Krylov iterations at all.
    dY_ = anomaly(y_k_global[0])
    dR_ = dY_ - S.A(anomaly(x_global[0]))

    lowrank = torch.cat([dY_, dR_], dim=1)  # (B, 2N, Dy)
    K_hat = RidgeDeltaCovariance(lowrank.mT, ridge=float(S.omega.r.lmbda))

    dx_exact = K_hat.inv(b)
    row("  closed-form solve", f"{rel(cov_kalman(dx_exact), b):.3f}", True,
        "dY dY^T + dR dR^T + ridge, inverted exactly (no Krylov)")

    # Is the operator even a fixed linear map? Krylov methods assume it is. A drifting
    # operator breaks Arnoldi outright, and would also explain the fixed-seed
    # irreproducibility seen end to end.
    v_probe = torch.randn_like(b)
    k1, k2 = cov_kalman(v_probe), cov_kalman(v_probe)

    drift = rel(k1, k2)
    row("operator repeatability", f"{drift:.2e}", drift < 1e-6,
        "K(v) called twice; nonzero => Krylov assumptions broken")

    # GMRES minimises the residual over a growing Krylov space, so the true residual can
    # never increase. If it does, the culprit is loss of orthogonality: azula's Arnoldi
    # uses classical Gram-Schmidt with no reorthogonalisation. This reference solve uses
    # modified Gram-Schmidt with a second pass, so the comparison isolates that.
    def gmres_reortho(op, rhs, iters):
        r0 = rhs.to(torch.float64)
        beta = r0.norm()
        V = [r0 / beta.clamp(min=1e-300)]
        H = torch.zeros(iters + 1, iters, dtype=torch.float64, device=rhs.device)

        for j in range(iters):
            w = op(V[j].to(rhs)).to(torch.float64)

            for _ in range(2):  # modified Gram-Schmidt, twice
                for i in range(j + 1):
                    h = (w * V[i]).sum()
                    H[i, j] = H[i, j] + h
                    w = w - h * V[i]

            H[j + 1, j] = w.norm()
            V.append(w / H[j + 1, j].clamp(min=1e-300))

        e1 = torch.zeros(iters + 1, dtype=torch.float64, device=rhs.device)
        e1[0] = beta
        y = torch.linalg.lstsq(H[: iters + 1, :iters], e1.unsqueeze(-1)).solution

        out = sum(y[i, 0] * V[i] for i in range(iters))

        return out.to(rhs)

    for it in (4, 8, 16, 32):
        dx_ref = gmres_reortho(cov_kalman, b, it)
        res_ref = rel(cov_kalman(dx_ref), b)
        row(f"  reortho gmres @{it}", f"{res_ref:.3f}", res_ref < 0.2,
            "same system, reorthogonalised Arnoldi")

    # symmetry of the operator cg would assume
    u1, u2 = torch.randn_like(b), torch.randn_like(b)
    sym = ((u1 * cov_kalman(u2)).sum() - (u2 * cov_kalman(u1)).sum()).abs()
    scale = (u1 * cov_kalman(u2)).sum().abs().clamp(min=1e-30)

    row("cov_kalman symmetry", f"{(sym / scale).item():.2e}", (sym / scale).item() < 1e-2,
        "cg is only valid if this is symmetric")


def t_floor_placement(upd, S, x, y_k):
    r"""Where the observation-noise floor lands: on span(dY), or only off it.

    Current adds ridge_y isotropically; legacy floored only the complement. On the resolved
    directions that difference is the whole disagreement between the two implementations.
    """

    dY = anomaly(y_k)
    dR = dY - S.A(anomaly(x))
    tau = dR.pow(2).mean()

    # a direction the ensemble spans, and one orthogonal to the span
    u_in = dY[:, :1].clone()  # (B, 1, Dy), literally a member of span(dY)

    u_rand = torch.randn_like(u_in)
    Gram = torch.einsum("bne,bme->bnm", dY, dY)  # (B, N, N)
    c = torch.einsum("bne,bme->bnm", dY, u_rand)  # (B, N, 1)
    alpha = torch.linalg.lstsq(Gram, c).solution  # (B, N, 1)
    u_out = u_rand - torch.einsum("bnm,bne->bme", alpha, dY)  # residual after projection

    def q(op, u):
        return ((u * op(u)).sum() / (u * u).sum().clamp(min=1e-30)).item()

    Omega = S.omega
    lo_in, lo_out = q(Omega, u_in), q(Omega, u_out)

    # legacy's shape: no ridge on span(dY), tau only on the complement
    Lam = RidgeDeltaCovariance(dR.mT, ridge=0.0)
    leg_in, leg_out = q(Lam, u_in), q(Lam, u_out) + tau.item()

    row("floor on span(dY)", f"{lo_in:.3e}", True, f"legacy would give {leg_in:.3e}")
    row("floor off span(dY)", f"{lo_out:.3e}", True, f"legacy would give {leg_out:.3e}")
    row("ridge_y / tau", f"{float(S.omega.r.lmbda):.3e}", True, f"tau = mean(dR^2) = {tau.item():.3e}")


def t_ridge_inverse(upd, x):
    r"""Round-trip of the ridged inverse on the span, and the conditioning it implies."""

    dX = anomaly(x)
    C = RidgeDeltaCovariance(dX.mT, ridge=upd.ridge_x)

    v = dX[:, :1].clone()  # a direction the covariance actually resolves
    err = rel(C(C.inv(v)), v)

    w, _ = C.eigh
    lmbda = float(C.r.lmbda)
    cond = (w.max().item() + lmbda) / lmbda

    row("ridge inverse round-trip", f"{err:.2e}", err < 1e-2, "C C^-1 v == v on span(dX)")
    row("ridge_x", f"{lmbda:.3e}", True, "adaptive: smallest trusted eigenvalue")
    row("implied condition number", f"{cond:.2e}", cond < 1e12)


def t_localization_radius(upd, x, y_k):
    r"""Is the estimated taper radius meaningful, or does it degenerate to the grid size?

    A radius at max_lag means the autocorrelation never decayed, which makes the taper
    near-uniform and the modulation rank ~1 -- localization then does nothing by construction.
    """

    dX = anomaly(x)
    N, B, *shape = x.shape

    if len(shape) < 2:
        row("localization radius", "n/a", True, "state is not a grid")
        return

    grid = tuple(shape[-2:])
    r = estimate_radius(dX, dX, grid)
    _, keep = grid_modulators(grid, r, energy=upd.localize_energy, cap=upd.localize_cap)

    degenerate = r >= max(grid) // 2

    row("taper radius (auto)", f"{r:.1f}", not degenerate,
        f"grid {grid}, max_lag {max(grid) // 2}" + ("  DEGENERATE" if degenerate else ""))
    row("  modulation rank", f"{keep}", keep > 1,
        f"cap {upd.localize_cap}; rank 1 => taper is uniform => no-op")


# --------------------------------------------------------------------------------------


y_k_global = []
x_global = []


def run(upd, x, cov_x):
    print(f"\ncloud: N={x.shape[0]} B={x.shape[1]} state={tuple(x.shape[2:])}")

    y_k, A, At = upd.slope(x)
    y_k_global.clear(); y_k_global.append(y_k)
    x_global.clear(); x_global.append(x)
    print(f"observation: Dy={flat(y_k).shape[-1]}  ridge_x={upd.ridge_x}  ridge_y={upd.ridge_y}")

    linear = t_operator_linearity(upd, x)
    t_adjoint(A, At, x, y_k)
    t_cov_x(cov_x, x)

    if linear:
        t_slope_quality(upd, A, x, y_k)

    S = upd.linearize(x)

    # the prior the update actually uses: particle covariance for PIPLF, analytic otherwise
    if isinstance(upd, U.PIPLFUpdate):
        prior = DeltaCovariance(anomaly(x).mT)
    else:
        prior = upd.lift(x, cov_x)

    t_residual_reachable(upd, S, y_k)
    t_increment_anatomy(upd, S, prior, x, y_k)
    t_kalman_solve(upd, S, prior, upd.y)
    t_floor_placement(upd, S, x, y_k)
    t_ridge_inverse(upd, x)
    t_localization_radius(upd, x, y_k)


def report():
    print("\n================ KPS COMPONENT DIAGNOSTIC ================")
    print(f"{'test':<28} {'value':>12}  {'':<4} {'note'}")

    for name, value, ok, note in ROWS:
        print(f"{name:<28} {value:>12}  {'ok ' if ok else 'FAIL':<4} {note}")

    bad = [r[0] for r in ROWS if not r[2]]
    print("\n" + (f"FAILED: {', '.join(bad)}" if bad else "all checks passed"))
    print("=========================================================\n")


@hydra.main(version_base="1.3", config_path="configs", config_name="config")
def main(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if config.tf32:
        torch.set_float32_matmul_precision("high")

    # KPS_DET=1 pins every nondeterministic kernel it can. If operator repeatability is
    # what breaks the Krylov solve, this is the switch that shows it.
    if os.environ.get("KPS_DET"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.set_float32_matmul_precision("highest")
        print("determinism: on")

    torch.manual_seed(config.seed)

    forward_op = instantiate(config.problem.model, device=device)
    testset = instantiate(config.problem.data)

    try:
        with open_url(config.problem.prior, "rb") as f:
            net = pickle.load(f)["ema"].to(device)
    except Exception:
        net = instantiate(config.pretrain.model)
        ckpt = torch.load(config.problem.prior, map_location=device)
        net.load_state_dict(ckpt.get("ema", ckpt.get("net")))
        net = net.to(device)

    net.eval()

    # the DataLoader's collation is what turns the dataset's numpy into batched tensors
    loader = DataLoader(testset, batch_size=1, shuffle=False)
    data = next(iter(loader))

    if isinstance(data, torch.Tensor):
        data = data.to(device)
    else:
        data = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in data.items()}

    observation = forward_op(data)

    algo = instantiate(config.algorithm.method, forward_op=forward_op, net=net)

    # patch every update flavour: the diagnostics run on the first real posterior step
    for cls in (U.PIPLFUpdate, U.HIPLFUpdate, U.GIPLFUpdate):

        def patched(self, x, cov_x, _cls=cls):
            run(self, x, cov_x)

            raise Done()

        cls.__call__ = patched

    try:
        algo.inference(observation, num_samples=1)
    except Done:
        pass

    report()


if __name__ == "__main__":
    main()
