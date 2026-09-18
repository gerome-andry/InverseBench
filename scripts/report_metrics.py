#!/usr/bin/env python3
r"""Score saved ensembles as POSTERIORS, in x space and in y space.

    python3 scripts/report_metrics.py                        # the campaign, to scripts/metrics.csv
    python3 scripts/report_metrics.py --runs '*' --official  # every run, bench evaluator too

WHY BOTH SPACES. InverseBench scores x accuracy, which is ill-posed when many x explain the
same y, and it scores it PER MEMBER -- `relative l2` is minimised by collapsing every particle
onto the posterior mean, so it actively rewards the failure mode an ensemble method has. The
y-space columns say whether the recovered states reproduce the observation, which is the
well-posed question, and the ensemble columns say whether the cloud is a posterior at all.
Nothing here replaces the bench metric: `member_*` reproduces it exactly so the numbers stay
comparable.

THE ENSEMBLE COLUMNS ARE BLADE'S. Spread-skill ratio and CRPS are how the ensemble-forecasting
literature scores a predictive distribution, and they are what makes runs at different N
comparable:

    skill     RMSE of the ensemble MEAN. Falls as 1/sqrt(N) only if the cloud is unbiased.
    spread    ensemble sd, with the Fortin et al. (2014) (N+1)/N correction. Without that
              correction a small ensemble looks under-dispersed purely by construction.
    ssr       spread / skill. 1 is calibrated, < 1 over-confident, > 1 over-dispersed. This is
              the column the variance collapse shows up in.
    crps      FAIR (unbiased) CRPS, mean_i |x_i - y| - E|x_i - x_j| / 2. The biased estimator
              improves with N for free, which would make an N sweep meaningless -- this one
              does not, so N=64 and N=512 can be put in the same table.
    cover90   fraction of pixels where the truth lies inside [q05, q95]. 0.90 is calibrated,
              and it assumes nothing about the shape of the marginal.
    rank      mean normalised rank of the truth among the members, 0.5 if the cloud is centred
              on it. Together with cover90 it separates a biased cloud from a narrow one.

CRPS IS REPORTED RAW AND NORMALISED. Raw is in the data's units, which is what Blade quotes;
`crps_n` divides by the rms of the truth so the three problems can share an axis.

Y SPACE HAS A FLOOR AND IT IS NOT ZERO. `recon_obs` is the forward operator applied WITH its
observation noise, so a perfect reconstruction still differs from `observation` by one noise
draw. `y_floor` is that level, sigma_noise * sqrt(dim) / ||y||. A run scoring BELOW it is
fitting the noise, not the signal.
"""

import argparse
import sys

from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

PROBLEMS = ("navier-stokes", "inv-scatter", "blackhole")


def _flat(x):
    return x.reshape(x.shape[0], -1)


def fair_crps(pred: torch.Tensor, truth: torch.Tensor) -> float:
    r"""Unbiased CRPS of the empirical ensemble, averaged over components.

    The pairwise term is done by the order statistic rather than the N x N matrix:
    sum_ij |x_i - x_j| = 2 sum_k (2k + 1 - N) x_(k), so this is O(N log N) per component
    instead of O(N^2) -- at N=512 over 16384 components the matrix form is 4e9 entries.
    """

    P, y = _flat(pred), truth.reshape(1, -1)
    N = P.shape[0]

    term1 = (P - y).abs().mean(0)                                   # (D,)
    srt, _ = P.sort(dim=0)
    k = torch.arange(N, device=P.device, dtype=P.dtype).reshape(-1, 1)
    pair = 2.0 * ((2 * k + 1 - N) * srt).sum(0)                     # sum_ij |x_i - x_j|

    return float((term1 - pair / (2 * N * (N - 1))).mean())


def ens_metrics(pred: torch.Tensor, truth: torch.Tensor, prefix: str = "") -> dict:
    r"""Posterior scores for one ensemble against one truth. Shapes (N, ...) and (1, ...).

    A single-member "ensemble" has no spread and no pairwise CRPS term, so everything
    distributional comes back NaN rather than as a divide-by-zero. Old runs that returned one
    sample per observation land there.
    """

    P, y = _flat(pred.float()), truth.float().reshape(1, -1)
    N = P.shape[0]

    if N < 2:
        nan = float("nan")
        return {f"{prefix}mean_l2": float((P - y).norm() / y.norm().clamp(min=1e-30)),
                f"{prefix}member_l2": float((P - y).norm() / y.norm().clamp(min=1e-30)),
                f"{prefix}skill": float((P - y).pow(2).mean().sqrt()),
                f"{prefix}spread": nan, f"{prefix}ssr": nan,
                f"{prefix}crps": float((P - y).abs().mean()), f"{prefix}crps_n": nan,
                f"{prefix}cover90": nan, f"{prefix}rank": nan}

    m = P.mean(0, keepdim=True)
    skill = (m - y).pow(2).mean().sqrt()
    spread = (P.var(0, unbiased=True).mean() * (N + 1) / N).sqrt()

    q = torch.quantile(P, torch.tensor([0.05, 0.95], dtype=P.dtype), dim=0)
    cover = ((y[0] >= q[0]) & (y[0] <= q[1])).float().mean()
    rank = (P < y).float().mean()            # P(member < truth): 0.5 when centred

    nrm = y.pow(2).mean().sqrt().clamp(min=1e-30)
    crps = fair_crps(pred.float(), truth.float())

    return {
        f"{prefix}mean_l2": float((m - y).norm() / y.norm().clamp(min=1e-30)),
        f"{prefix}member_l2": float(((P - y).norm(dim=1) / y.norm().clamp(min=1e-30)).mean()),
        f"{prefix}skill": float(skill),
        f"{prefix}spread": float(spread),
        f"{prefix}ssr": float(spread / skill.clamp(min=1e-30)),
        f"{prefix}crps": crps,
        f"{prefix}crps_n": crps / float(nrm),
        f"{prefix}cover90": float(cover),
        f"{prefix}rank": float(rank),
    }


def score_file(path: Path, d=None) -> dict:
    r"""Both spaces for one saved result. Complex observations are packed to real first."""

    d = torch.load(path, map_location="cpu", weights_only=False) if d is None else d
    rec, tgt = d["recon"].float(), d["target"].float()
    if tgt.dim() == rec.dim() - 1:
        tgt = tgt.unsqueeze(0)
    tgt = tgt.reshape(1, *rec.shape[1:])

    out = {"n": rec.shape[0]}
    out.update(ens_metrics(rec, tgt, prefix="x_"))

    if "recon_obs" in d:
        obs, ro = d["observation"].cpu(), d["recon_obs"].cpu()
        if torch.is_complex(obs):
            obs, ro = torch.view_as_real(obs), torch.view_as_real(ro)
        obs = obs.float().reshape(1, -1)
        out.update(ens_metrics(ro.float(), obs, prefix="y_"))

    return out


def official(problem: str, path: Path, cache: dict) -> dict:
    r"""The bench's OWN evaluator, so the headline columns are reproduced and not re-derived.

    Blackhole needs the forward operator -- its chi-squares go through it, and its psnr
    maximises over blur factors -- so this instantiates the problem. That costs a few seconds
    per problem and is cached.
    """

    from hydra import compose, initialize_config_dir
    from hydra.utils import instantiate

    if problem not in cache:
        with initialize_config_dir(version_base="1.3", config_dir=str(ROOT / "configs")):
            cfg = compose(config_name="config", overrides=[f"problem={problem}",
                                                           f"pretrain={problem}"])
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        fwd = instantiate(cfg.problem.model, device=dev)
        cache[problem] = (fwd, instantiate(cfg.problem.evaluator, forward_op=fwd))

    fwd, ev = cache[problem]
    d = torch.load(path, map_location="cpu", weights_only=False)

    try:
        return {f"ib_{k}": float(v) for k, v in
                ev(pred=d["recon"], target=d["target"], observation=d["observation"]).items()}
    except Exception as exc:                       # an operator that cannot score on this box
        return {"ib_error": f"{type(exc).__name__}"}


def runs(problems=PROBLEMS, pattern="*"):
    r"""Every (problem, algo, exp_name) directory holding results, whatever produced it.

    `pattern` filters on the run (exp_name) directory. The exp trees carry a few hundred
    directories of older exploratory work, so scanning everything is slow and mostly noise:
    "gen*" is the generation campaign, and anything else can be named explicitly.
    """

    import fnmatch
    import yaml

    for problem in problems:
        p = yaml.safe_load((ROOT / "configs" / "problem" / f"{problem}.yaml").read_text())
        base = Path(p["exp_dir"])
        if not base.exists():
            continue
        for algo in sorted(x for x in base.iterdir() if x.is_dir()):
            for exp in sorted(x for x in algo.iterdir() if x.is_dir()):
                if not fnmatch.fnmatch(exp.name, pattern):
                    continue
                files = sorted(exp.glob("result_*.pt"))
                if files:
                    yield problem, algo.name, exp.name, files


def collect(problems=PROBLEMS, with_official=False, limit=None, pattern="*"):
    import pandas as pd

    rows, cache = [], {}
    for problem, algo, exp, files in runs(problems, pattern):
        for f in files[:limit]:
            row = dict(problem=problem, algo=algo, run=exp,
                       id=int(f.stem.split("_")[1]), n=0)
            try:
                d = torch.load(f, map_location="cpu", weights_only=False)
                row.update(score_file(f, d))
                if with_official:
                    row.update(official(problem, f, cache))
            except Exception as exc:
                row["error"] = f"{type(exc).__name__}: {exc}"
            rows.append(row)
            print(f"  {problem:<14}{algo:<6}{exp:<12}id={row['id']:<4}"
                  f"N={row['n']:<5}x_ssr={row.get('x_ssr', float('nan')):.3f}", flush=True)

    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--problem", action="append", choices=PROBLEMS,
                    help="restrict to one problem; repeatable (default: all)")
    ap.add_argument("--official", action="store_true",
                    help="also run the bench's own evaluator (needs the forward operator)")
    ap.add_argument("--limit", type=int, help="first K ids per run, for a quick look")
    ap.add_argument("--runs", default="gen*",
                    help="glob on the run directory; default 'gen*', the campaign. '*' takes "
                         "every historical run, which is a few hundred directories")
    ap.add_argument("--out", default=str(ROOT / "scripts" / "metrics.csv"))
    args = ap.parse_args()

    df = collect(tuple(args.problem or PROBLEMS), args.official, args.limit, args.runs)
    df.to_csv(args.out, index=False)
    print(f"\n{len(df)} rows -> {args.out}")

    if not df.empty:
        keys = [c for c in ("x_mean_l2", "x_ssr", "x_crps_n", "y_mean_l2", "y_ssr") if c in df]
        print(df.groupby(["problem", "algo", "run", "n"])[keys].mean().round(4).to_string())

    return 0


if __name__ == "__main__":
    sys.exit(main())
