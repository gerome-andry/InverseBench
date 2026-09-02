"""Instrumentation for KPS runs. Enable with KPS_PROBE=1.

Prints one row per linearisation (subsampled), then a summary. Paste the whole block
back as feedback. Adds no cost beyond a few norms and one extra likelihood call per row.
"""

import os
import torch

import kps.update as U

from kps.update import anomaly, flat

_ROWS = []
_STATE = {"step": 0, "first_bad": None}
_EVERY = int(os.environ.get("KPS_PROBE_EVERY", "8"))
_MAXROWS = int(os.environ.get("KPS_PROBE_ROWS", "14"))


def _f(t):
    return torch.isfinite(t).all().item()


def _n(t):
    t = t[torch.isfinite(t)]
    return t.abs().max().item() if t.numel() else float("nan")


def arm():
    _orig_lin = U.PosteriorUpdate.linearize
    _orig_iter = U.PosteriorUpdate.iterate

    def linearize(self, x_k):
        i = _STATE["step"]
        _STATE["step"] += 1

        y_k = self.stack(x_k)
        dX, dY = anomaly(x_k), anomaly(y_k)

        if _STATE["first_bad"] is None:
            for tag, t in (("x_k", x_k), ("y_k", y_k)):
                if not _f(t):
                    _STATE["first_bad"] = f"{tag} at linearisation {i}"

        S = _orig_lin(self, x_k)

        if i % _EVERY == 0 and len(_ROWS) < _MAXROWS:
            w = torch.linalg.eigvalsh(torch.matmul(dX, dX.mT)).clamp(min=0)
            N = w.shape[-1]
            tol = torch.finfo(w.dtype).eps * N * w.max().clamp(min=1e-30)
            rank = int((w > tol).sum(-1).float().median())
            rx = torch.where(w > tol, w, torch.full_like(w, torch.inf)).min()
            dR = dY - S.A(dX)
            _ROWS.append(dict(
                i=i, rank=f"{rank}/{N}",
                x=_n(x_k), y=_n(y_k),
                dX=dX.pow(2).mean().sqrt().item(),
                dY=dY.pow(2).mean().sqrt().item(),
                dR=dR.pow(2).mean().sqrt().item(),
                rx=rx.item(), ry=dR.pow(2).mean().item(),
                cond=(w.max() / rx).item(),
                fin=int(_f(x_k)) * 10 + int(_f(y_k)),
            ))
        return S

    def iterate(self, x, prior):
        out = _orig_iter(self, x, prior)
        if _ROWS and "rel" not in _ROWS[-1]:
            with torch.no_grad():
                d = (out.unsqueeze(0) - x)
                _ROWS[-1]["step_rel"] = (d.norm() / x.norm().clamp(min=1e-30)).item()
        return out

    U.PosteriorUpdate.linearize = linearize
    U.PosteriorUpdate.iterate = iterate


def report(algo=None):
    print("\n================ KPS PROBE ================")
    if not _ROWS:
        print("no rows recorded")
        return
    hdr = f"{'lin':>4} {'rank':>8} {'|x|':>10} {'|y|':>10} {'rms dX':>10} {'rms dY':>10} " \
          f"{'rms dR':>10} {'ridge_x':>10} {'ridge_y':>10} {'cond':>9} {'step/|x|':>9} {'fin':>4}"
    print(hdr)
    for r in _ROWS:
        print(f"{r['i']:>4} {r['rank']:>8} {r['x']:>10.2e} {r['y']:>10.2e} {r['dX']:>10.2e} "
              f"{r['dY']:>10.2e} {r['dR']:>10.2e} {r['rx']:>10.2e} {r['ry']:>10.2e} "
              f"{r['cond']:>9.1e} {r.get('step_rel', float('nan')):>9.2e} {r['fin']:>4}")
    print(f"\nfirst non-finite: {_STATE['first_bad'] or 'none'}")
    if algo is not None and getattr(algo, "_last_update", None) is not None:
        tr = getattr(algo._last_update, "ess_trace", [])
        if tr:
            e = torch.stack(tr)
            print(f"ESS: mean {e.mean():.2f}  min {e.min():.2f}  max {e.max():.2f}  "
                  f"(N = {algo.num_particles})")
    print("fin column: 11 = x and y finite, 10 = y bad, 01 = x bad, 00 = both bad")
    print("===========================================\n")
