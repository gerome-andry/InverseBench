#!/usr/bin/env python
"""Posterior-predictive misfit for saved KPS runs.

InverseBench scores x-accuracy, which is ill-posed when many x explain the same y. The
well-posed objective is whether the recovered x reproduces the observation. main.py
already saves `recon_obs = forward_op({"target": recon})`, so this needs no re-running.

    uv run python score_predictive.py <exp_dir> [<exp_dir> ...]
"""

import sys

from pathlib import Path

import torch


def misfit(path):
    d = torch.load(path, map_location="cpu", weights_only=False)

    if "recon_obs" not in d:
        return None

    y, yr = d["observation"].cpu(), d["recon_obs"].cpu()

    if torch.is_complex(y) != torch.is_complex(yr):
        y, yr = torch.view_as_real(y) if torch.is_complex(y) else y, \
                torch.view_as_real(yr) if torch.is_complex(yr) else yr

    return ((y - yr).norm() / y.norm().clamp(min=1e-30)).item()


def main(dirs):
    print(f"{'run':<52} {'n':>4} {'median':>9} {'min':>9} {'max':>9}")
    for d in dirs:
        vals = [m for p in sorted(Path(d).glob("result_*.pt")) if (m := misfit(p)) is not None]
        if not vals:
            print(f"{str(d):<52} {'-':>4} {'no recon_obs found':>29}")
            continue
        v = sorted(vals)
        print(f"{str(d)[-52:]:<52} {len(v):>4} {v[len(v) // 2]:>9.4f} {v[0]:>9.4f} {v[-1]:>9.4f}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1:])
