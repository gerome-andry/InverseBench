r"""Paper assets for the RIDGE-SLR campaign (KPSCV / KPSGF), kept apart from the ladder's.

The ladder campaign's assets are what the current paper draft uses. NOTHING here touches them:
this module imports `paper_assets` for its logic, points it at a different campaign and a
different metrics cache, and the driver notebook writes to a different output directory.

    paper_assets                scripts/metrics.csv        -> notebooks/paper/
    paper_assets_cv (here)      scripts/metrics_cv.csv     -> notebooks/paper_cv/

Importing this module REBINDS `paper_assets.CAMPAIGN` in the importing process, which is what
makes `config_table` and `sampler_table` describe the new arms. That is process-local and
writes nothing, so a notebook that imports `paper_assets` directly is unaffected -- but do not
import both into one kernel and expect the ladder's tables.

WHAT DIFFERS IN THE ARMS, and it is only the likelihood fit:

    ladder   KPSH  rank-32 truncated slope, two-fold held-out isotropic Sigma_y, direct solve
             KPSG  exact Jacobian slope, kps.fit.sigma_y (a REGRESSION residual) as a scalar
    new      KPSCV ridged SLR, lambda by K-fold CV, FULL S_y from the same joint, df-corrected,
                   matrix-free solve
             KPSGF exact Jacobian slope, FULL S_y from G's OWN linearisation residual

Ladder, draw, re-noising, handover and V_t are identical -- `gen_samples_cv.py` asserts that
against each baseline's stored config before it will write an array, so the comparison is
paired arm for arm. See `ridge_slr/RESULTS_BLACKHOLE.md`.
"""

from __future__ import annotations

import pathlib

import pandas as pd

import paper_assets as pa

ROOT = pa.ROOT
OUT = ROOT / "notebooks" / "paper_cv"
METRICS = ROOT / "scripts" / "metrics_cv.csv"

# One directory per (algorithm, N), mirroring the ladder campaign arm for arm. KPSGF has no
# Navier-Stokes row for the same reason KPSG had none: NS is H-only in both campaigns.
CAMPAIGN = {
    ("blackhole", "KPSGF"): ["gen_gf"],
    ("inv-scatter", "KPSGF"): ["gen_gf"],
    ("blackhole", "KPSCV"): ["gen_cv_N64", "gen_cv_N128", "gen_cv_N256", "gen_cv_N512"],
    ("inv-scatter", "KPSCV"): ["gen_cv_N64", "gen_cv_N128", "gen_cv_N256", "gen_cv_N512"],
    ("navier-stokes", "KPSCV"): ["gen_cv_N64", "gen_cv_N128", "gen_cv_N256", "gen_cv_N512"],
}

# The ladder's arms, for the head-to-head. Kept here rather than read from pa.CAMPAIGN because
# that global is about to be rebound.
LADDER = {
    ("blackhole", "KPSG"): ["gen"], ("inv-scatter", "KPSG"): ["gen"],
    ("blackhole", "KPSH"): ["gen_N64", "gen_N128", "gen_N256", "gen_N512"],
    ("inv-scatter", "KPSH"): ["gen_N64", "gen_N128", "gen_N256", "gen_N512"],
    ("navier-stokes", "KPSH"): ["gen_N64", "gen_N128", "gen_N256", "gen_N512"],
}

PAPER_NAME = {"KPSCV": "KPS-CV", "KPSGF": "KPS-GF", "KPSH": "KPS-H", "KPSG": "KPS-G"}

pa.CAMPAIGN = CAMPAIGN                      # process-local; see the module docstring
# Labels and colours for the new arms. The COLOURS DELIBERATELY MATCH their ladder
# counterparts -- KPS-CV takes KPS-H's blue, KPS-GF takes KPS-G's orange -- because the two
# notebooks are read side by side and the convention should be "blue = fitted slope, orange =
# Jacobian slope", not "blue = whichever campaign". No figure mixes the campaigns, so there is
# nothing to disambiguate. Both are slots 1 and 2 of the validated categorical palette.
pa.LBL = {**pa.LBL, "KPSCV": "KPS-CV", "KPSGF": "KPS-GF"}
pa.C = {**pa.C, "KPSCV": pa.C["KPSH"], "KPSGF": pa.C["KPSG"]}


def load_metrics(official: bool = True) -> pd.DataFrame:
    r"""The ridge-SLR cache. One file, because `--official` wrote the ib_ columns into it."""

    df = pd.read_csv(METRICS)
    df["method"] = df.algo + " (N=" + df.n.astype(str) + ")"

    return df


def load_both() -> pd.DataFrame:
    r"""New campaign and ladder in one frame, with a `campaign` column, for the head-to-head.

    The two caches are scored by the SAME `report_metrics.score_file`, so the columns mean the
    same thing. What differs between the rows is the likelihood fit and nothing else.
    """

    new = load_metrics()
    new["campaign"] = "ridge-SLR"

    old = pd.read_csv(ROOT / "scripts" / "metrics.csv")
    f = ROOT / "scripts" / "metrics_official_full.csv"
    if f.exists():
        off = pd.read_csv(f)
        ib = [c for c in off.columns if c.startswith("ib_")]
        old = old.merge(off[["problem", "algo", "run", "id"] + ib],
                        on=["problem", "algo", "run", "id"], how="left")
    old["method"] = old.algo + " (N=" + old.n.astype(str) + ")"
    old["campaign"] = "ladder"

    return pd.concat([new, old], ignore_index=True)


def paired(df: pd.DataFrame, problem: str, new_algo: str, old_algo: str, n: int,
           cols=("x_member_l2", "x_skill", "x_ssr", "x_crps_n", "x_cover90")) -> pd.DataFrame:
    r"""Per-id paired difference between one new arm and its ladder counterpart.

    `KPS/report/KPSTEST_PROMPT.md`: pair on instance id and report the per-id differences with
    their sign agreement, because arm means mislead -- on this project a 5-6% effect with an
    11-sigma headline turned out to be one instance of three. The sem below is of the PAIRED
    difference, so instance variation is already removed from it.
    """

    a = df[(df.problem == problem) & (df.algo == old_algo) & (df.n == n)].set_index("id")
    b = df[(df.problem == problem) & (df.algo == new_algo) & (df.n == n)].set_index("id")
    ids = a.index.intersection(b.index)

    rows = []
    for c in cols:
        if c not in a or c not in b:
            continue
        d = (b.loc[ids, c] - a.loc[ids, c]).astype(float)
        if c == "x_ssr":
            win = int(((b.loc[ids, c] - 1).abs() < (a.loc[ids, c] - 1).abs()).sum())
        elif c == "x_cover90":
            win = int(((b.loc[ids, c] - 0.9).abs() < (a.loc[ids, c] - 0.9).abs()).sum())
        else:
            win = int((d < 0).sum())
        rows.append(dict(metric=pa.NAME.get(c, c), ladder=float(a.loc[ids, c].median()),
                         ridge=float(b.loc[ids, c].median()), diff=float(d.mean()),
                         sem=float(d.std() / max(len(ids), 1) ** 0.5),
                         better=f"{win}/{len(ids)}"))

    return pd.DataFrame(rows)


def emit(out: pathlib.Path, name: str, body: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / name).write_text(body)
    print(f"wrote {out / name}  ({len(body.splitlines())} lines)")
