r"""Paper-ready assets from the KPS generation campaign: LaTeX tables, figures, calibration.

Everything reads the SCORED CACHES plus the stored result files; nothing re-runs a sampler.

    metrics.csv                distributional metrics, 1040 rows, all three problems
    metrics_official_full.csv  InverseBench's own evaluator over the same 1040

Two provenance facts the tables must carry, both established from git history and the stored
`config.yaml` of the runs themselves:

  * BLACKHOLE IS NOT THE BENCHMARK'S PROBLEM. Commit b93dfbb (2026-09-17) switched
    `noise_type` from 'eht' to 'vis_thermal' so that the Jacobian variant could run at all,
    and the campaign ran after it. The repo's own README says this changes the measurement
    model, not the solver, and that its numbers must not be compared against 'eht' results.
  * NAVIER-STOKES USES sigma_noise = 1e-4, not the benchmark's 0.0 (our commit ab79ef2).
  * Inverse scattering matches upstream exactly (its 1e-4 is upstream's own, 2025-03-30).
"""

from __future__ import annotations

import pathlib
import torch
import numpy as np
import pandas as pd
import yaml

ROOT = pathlib.Path("/mnt/home/gandry/kps/InverseBench")
EXP = pathlib.Path("/mnt/home/gandry/ceph/ibench/ibench_exp")

# exp-tree directory name -> the name used in the paper
EXPDIR = {"blackhole": "blackhole", "inv-scatter": "inv-scatter-linear",
          "navier-stokes": "navier-stokes-ds2"}
NICE = {"blackhole": "Black-hole imaging", "inv-scatter": "Inverse scattering",
        "navier-stokes": "Navier--Stokes"}
# matplotlib is not LaTeX: "--" renders as two hyphens, so plots use this one.
PLAIN = {k: v.replace("--", "\u2013") for k, v in NICE.items()}
CAMPAIGN = {("blackhole", "KPSG"): ["gen"], ("inv-scatter", "KPSG"): ["gen"],
            ("blackhole", "KPSH"): ["gen_N64", "gen_N128", "gen_N256", "gen_N512"],
            ("inv-scatter", "KPSH"): ["gen_N64", "gen_N128", "gen_N256", "gen_N512"],
            ("navier-stokes", "KPSH"): ["gen_N64", "gen_N128", "gen_N256", "gen_N512"]}


def tex_escape(s: str) -> str:
    return str(s).replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")


def load_metrics(official: bool = True) -> pd.DataFrame:
    r"""The two caches joined on (problem, algo, run, id)."""

    df = pd.read_csv(ROOT / "scripts" / "metrics.csv")
    f = ROOT / "scripts" / "metrics_official_full.csv"
    if official and f.exists():
        off = pd.read_csv(f)
        ib = [c for c in off.columns if c.startswith("ib_")]
        df = df.merge(off[["problem", "algo", "run", "id"] + ib],
                      on=["problem", "algo", "run", "id"], how="left")
    df["method"] = df.algo + " (N=" + df.n.astype(str) + ")"
    return df


def run_config(problem: str, algo: str, run: str) -> dict:
    f = EXP / EXPDIR[problem] / algo / run / "config.yaml"
    return yaml.safe_load(f.read_text()) if f.exists() else {}


# ----------------------------------------------------------------- 1. the config table

def config_table(df: pd.DataFrame) -> str:
    r"""Appendix table: the forward model, the prior and the sampler settings, per problem.

    Read from the runs' OWN stored config.yaml, not from the config files as they read today,
    so the table describes what actually produced the numbers. `id_list` is deliberately taken
    from the scored results rather than the config: the campaign was sharded across an array
    job, so each stored config records only its shard's slice.
    """

    rows = []
    for prob in ("blackhole", "inv-scatter", "navier-stokes"):
        # Whichever arm of THIS campaign has a stored config for the problem -- the table only
        # needs one, since the forward model and prior are per problem, not per arm. The order
        # reproduces the previous hardcoded "KPSG if present else KPSH" exactly for the ladder
        # campaign, and lets a different campaign (see paper_assets_cv.py) reuse this function
        # instead of copying it.
        algo = next(a for a in ("KPSG", "KPSGF", "KPSH", "KPSCV") if (prob, a) in CAMPAIGN)
        run = CAMPAIGN[(prob, algo)][0]
        c = run_config(prob, algo, run)
        m, pm = c.get("problem", {}).get("model", {}), c.get("problem", {})
        d = df[df.problem == prob]
        ids = sorted(d.id.unique())
        forward = {
            "blackhole": rf"EHT visibilities, {m.get('imsize','?')}$\times${m.get('imsize','?')} image, "
                         rf"\texttt{{{tex_escape(m.get('noise_type','?'))}}} noise, "
                         rf"\texttt{{{tex_escape(m.get('ttype','?'))}}} transform",
            "inv-scatter": rf"{m.get('Nx','?')}$\times${m.get('Ny','?')} permittivity, "
                           rf"{m.get('numTrans','?')} transmitters $\times$ {m.get('numRec','?')} receivers, "
                           rf"wavenumber {m.get('wave','?')}",
            "navier-stokes": rf"vorticity at $t={m.get('forward_time','?')}$, Re${{}}={m.get('Re','?')}$, "
                             rf"{m.get('resolution','?')}$^2$ downsampled $\times${m.get('downsample_factor','?')}",
        }[prob]
        rows.append(dict(
            problem=NICE[prob], forward=forward,
            sigma=f"{m.get('sigma_noise', float('nan')):g}",
            prior=pathlib.Path(str(pm.get("prior", "?"))).name.replace(".pt", ""),
            ids=f"{len(ids)} ({min(ids)}--{max(ids)})",
        ))

    # p{} columns so the forward-model description WRAPS; with plain l columns this table was
    # wider than the ICLR text block. \raggedright is applied INSIDE each cell rather than
    # through a >{...} column prefix, because that syntax needs the array package and the
    # draft's preamble does not load it -- the same trap that made \multirow print literally.
    # Justified p{} cells cannot hyphenate \texttt{inv-scatter-5m} and overflowed by 19pt.
    # \sloppy must go INSIDE the cell: a p{} cell starts its own paragraph, so a
    # \sloppy before \begin{tabular} never reaches it.
    rg = lambda txt: r"\raggedright\sloppy " + txt
    # WIDTHS MUST LEAVE ROOM FOR \tabcolsep. Five columns have four inter-column gaps of
    # 2*\tabcolsep each; at the 6pt default that is 48pt on a 397pt line, i.e. 0.12 of the
    # width, which is what pushed a 0.89 sum over the edge by 19pt. Narrow the separation and
    # size the columns to ~0.84 so the total lands inside the block.
    # \sloppy for the last few points: what remains are single unbreakable tokens inside
    # narrow cells (\texttt names, "transmitters"), which no column width fixes. \sloppy
    # stretches inter-word space instead of letting them protrude. Base LaTeX, no package.
    out = [r"\small", r"\setlength{\tabcolsep}{4pt}", r"\sloppy",
           r"\begin{tabular}{@{}p{0.17\linewidth}p{0.28\linewidth}"
           r"p{0.06\linewidth}p{0.22\linewidth}p{0.11\linewidth}@{}}",
           r"\toprule",
           r"Problem & Forward model & $\sigma_{\text{noise}}$ & Prior & Instances \\",
           r"\midrule"]
    eol = r" \\"
    for r in rows:
        prior = rg(r"\texttt{\scriptsize " + tex_escape(r["prior"]) + "}")
        cells = [rg(r["problem"]), rg(r["forward"]), r["sigma"], prior, r["ids"]]
        out.append(" & ".join(cells) + eol)
    out += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(out)


def sampler_table() -> str:
    r"""Appendix table: the sampler settings, read from the stored configs of every arm."""

    seen = {}
    for (prob, algo), runs in CAMPAIGN.items():
        for run in runs:
            c = run_config(prob, algo, run)
            m = c.get("algorithm", {}).get("method", {})
            if not m:
                continue
            key = (algo, m.get("num_particles"))
            seen.setdefault(key, dict(
                algo=algo, mode=m.get("mode"), slope=m.get("slope"),
                N=m.get("num_particles"),
                T=m.get("levels"), M=m.get("sweeps"), rank=m.get("rank", "--"),
                solve=m.get("solve_iter", "--"), draw=m.get("draw_steps"),
                smax=m.get("sigma_max"), smin=m.get("sigma_min"),
                seed=c.get("seed"), probs=set()))
            seen[key]["probs"].add(NICE[prob])

    out = [r"\begin{tabular}{llrrrrrrrr}", r"\toprule",
           r"Variant & Slope & $N$ & $T$ & $M$ & $r$ & Krylov & ODE & "
           r"$\sigma_{\max}$ & $\sigma_{\min}$ \\", r"\midrule"]
    for k in sorted(seen, key=lambda k: (k[0], k[1])):
        s = seen[k]
        # Read from the STORED config, because the two Jacobian variants are different
        # estimators: kps.fit.slope_gradient batches torch.func.vjp over the cloud and the
        # forward operators act per sample, so its Jacobian is block diagonal and every particle
        # uses its OWN exact slope. Only ridge_slr.kps_cv (slope='mean') averages the cloud.
        # Configs without the key predate it and are per-particle.
        slope = ("ensemble regression (SLR)" if s["mode"] != "g" else
                 "averaged Jacobian" if s.get("slope") == "mean" else "per-particle Jacobian")
        out.append(f"KPS-{'G' if s['mode']=='g' else 'H'} & {slope} & {s['N']} & {s['T']} & "
                   f"{s['M']} & {s['rank']} & {s['solve']} & {s['draw']} & "
                   f"{s['smax']:g} & {s['smin']:g} \\\\")
    out += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(out)


# ------------------------------------------------- 2/3. the result tables

DIVERGED = 2.0          # x_mean_l2 above this is a blown-up run, not a bad one

# Published baselines, InverseBench (arXiv:2503.11043) Table 3: linear inverse scattering at
# 360 receivers, sigma_y = 1e-4 -- exactly our configuration, so these rows are directly
# comparable and our inverse-scattering column can be read against them.
#
# VERIFY AGAINST THE TYPESET PDF BEFORE SUBMISSION. These were read out of the arXiv HTML by
# an automated fetch, not transcribed by hand from the paper. The blackhole and Navier-Stokes
# tables live in Appendix A.1, which that fetch could not reach, so those two problems still
# emit placeholder rows.
BASELINES = {
    # Table 3: linear inverse scattering, 360 receivers, sigma_y = 1e-4 -- our exact setting.
    "inv-scatter": [
        ("FISTA-TV",   {"ib_psnr": 32.126, "ib_ssim": 0.979}),
        ("DDRM",       {"ib_psnr": 32.598, "ib_ssim": 0.929}),
        ("DDNM",       {"ib_psnr": 36.381, "ib_ssim": 0.935}),
        (r"$\Pi$GDM",  {"ib_psnr": 27.925, "ib_ssim": 0.889}),
        ("DPS",        {"ib_psnr": 32.061, "ib_ssim": 0.846}),
        ("LGD",        {"ib_psnr": 27.901, "ib_ssim": 0.812}),
        ("DiffPIR",    {"ib_psnr": 34.241, "ib_ssim": 0.988}),
        ("PnP-DM",     {"ib_psnr": 33.914, "ib_ssim": 0.988}),
        ("DAPS",       {"ib_psnr": 34.641, "ib_ssim": 0.957}),
        ("RED-diff",   {"ib_psnr": 36.556, "ib_ssim": 0.981}),
        ("FPS",        {"ib_psnr": 33.242, "ib_ssim": 0.870}),
        ("MCG-diff",   {"ib_psnr": 30.937, "ib_ssim": 0.751}),
    ],
    # Table 6, OBSERVATION TIME RATIO 100%, which matches our observation_time_ratio = 1.0.
    # sd in parentheses in the paper, carried here as the second entry.
    # CAVEAT ON "Blur PSNR": the paper reports ONE blurred PSNR, at the telescope's target
    # resolution. Our evaluator emits f = 10, 15, 20 and f=10 is numerically identical to the
    # unblurred PSNR, so which of f=15 / f=20 corresponds to their column is NOT established.
    # Both are shown; do not claim a win on this row without pinning the blur factor.
    "blackhole": [
        ("SMILI",        {"ib_psnr": (22.67, 3.13), "blur": (27.79, 4.02),
                          "ib_cp_chi2": (1.878, 0.952), "ib_camp_chi2": (17.612, 10.299)}),
        ("EHT-Imaging",  {"ib_psnr": (24.28, 3.63), "blur": (28.57, 4.52),
                          "ib_cp_chi2": (1.251, 0.250), "ib_camp_chi2": (1.259, 0.316)}),
        ("DPS",          {"ib_psnr": (25.86, 3.90), "blur": (32.94, 6.19),
                          "ib_cp_chi2": (8.759, 37.784), "ib_camp_chi2": (5.456, 24.185)}),
        ("LGD",          {"ib_psnr": (21.22, 3.64), "blur": (26.06, 4.98),
                          "ib_cp_chi2": (13.239, 17.231), "ib_camp_chi2": (13.233, 39.107)}),
        ("RED-diff",     {"ib_psnr": (23.77, 4.13), "blur": (29.13, 6.22),
                          "ib_cp_chi2": (1.853, 0.938), "ib_camp_chi2": (2.050, 2.361)}),
        ("PnP-DM",       {"ib_psnr": (26.07, 3.70), "blur": (32.88, 6.02),
                          "ib_cp_chi2": (1.311, 0.195), "ib_camp_chi2": (1.199, 0.221)}),
        ("DAPS",         {"ib_psnr": (25.60, 3.64), "blur": (32.78, 5.68),
                          "ib_cp_chi2": (1.300, 0.324), "ib_camp_chi2": (1.229, 0.532)}),
        ("DiffPIR",      {"ib_psnr": (25.01, 4.64), "blur": (31.86, 6.56),
                          "ib_cp_chi2": (3.271, 1.623), "ib_camp_chi2": (2.970, 1.202)}),
    ],
    # Table 8, subsampling x2 (our downsample_factor = 2) at sigma = 0.0. Ours is 1e-4, which
    # is that column to four decimals.
    "navier-stokes": [
        ("EKI",       {"ib_relative l2": (0.577, 0.138)}),
        ("DPS-fGSG",  {"ib_relative l2": (1.687, 0.156)}),
        ("DPS-cGSG",  {"ib_relative l2": (2.203, 0.314)}),
        ("DPG",       {"ib_relative l2": (0.325, 0.188)}),
        ("SCG",       {"ib_relative l2": (0.908, 0.600)}),
        # N from the benchmark's own enkg.yaml (num_samples); the other baselines are not
        # ensemble methods, so they have no comparable count.
        ("EnKG",      {"ib_relative l2": (0.120, 0.085), "N": 2048}),
    ],
}


def _agg(df: pd.DataFrame, cols, key=("problem", "algo", "n")):
    r"""Median, and mean +/- sd over the runs that did not blow up, with the failure count.

    BOTH are reported on purpose. Blackhole KPS-H diverges on up to 4 of 100 instances and a
    single one of them reaches x_mean_l2 = 27614, which moves the mean by four orders of
    magnitude -- so a mean alone is not a description of the typical instance, and a median
    alone hides that the method sometimes fails outright.
    """

    key = list(key)
    g_all = df.groupby(key)
    ok = df[df.x_mean_l2 <= DIVERGED]
    g = ok.groupby(key)
    out = pd.DataFrame({"ids": g_all.size(),
                        "fail": g_all.x_mean_l2.apply(lambda s: int((s > DIVERGED).sum()))})
    for c in cols:
        if c not in df:
            continue
        out[c + "_med"] = g[c].median()
        out[c + "_mean"] = g[c].mean()
        out[c + "_sd"] = g[c].std()
    return out


def _group_table(spec, header, blocks, size=r"\small"):
    r"""A table whose groups are introduced by a spanning row rather than \multirow.

    TWO RENDERING BUGS THIS AVOIDS, both seen in the compiled draft:
      * `\multirow` needs the multirow package. Without it LaTeX prints the argument
        literally, so the draft showed "5*Black-hole imaging" in the first cell.
      * a dedicated Problem column makes the table wider than the text block; a spanning row
        carries the same information in the space that is already there.
    `size` shrinks the type: these tables have many numeric columns and the default size
    overflows the ICLR text width.
    """

    out = [size, rf"\begin{{tabular}}{{{spec}}}", r"\toprule", header + r" \\"]
    for title, rows in blocks:
        out.append(r"\midrule")
        if title:
            out.append(rf"\multicolumn{{{spec.count('c') + spec.count('l') + spec.count('r')}}}"
                       rf"{{l}}{{\itshape {title}}} \\")
        out.extend(rows)
    out += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(out)


def _pm(m, s, nd=3):
    if pd.isna(m):
        return "--"
    return rf"${m:.{nd}f} \pm {s:.{nd}f}$" if pd.notna(s) else f"${m:.{nd}f}$"


def benchmark_table(df: pd.DataFrame, problem: str, stat: str = "mean") -> str:
    r"""InverseBench's OWN metrics for our arms, laid out so published rows can be pasted in.

    The baseline rows are left as explicit placeholders rather than invented: no baseline was
    run at this scale locally (the exp tree has 4 EnKG results on Navier-Stokes and none
    elsewhere), so the published numbers have to come from the benchmark paper.

    `stat="median"` reports the median and inter-quartile range instead. Use it for blackhole:
    its closure-phase chi-square is heavy-tailed, with a standard deviation up to 39.9 on a
    quantity whose target is 1, so a mean there describes the tail rather than the method.
    """

    # BLUR PSNR, AND THE EVALUATOR'S KEYS ARE OFF BY ONE. `evaluate_psnr` returns five
    # columns -- the unblurred PSNR, then blur at factors 0, 10, 15, 20 -- while eval.py reads
    # indices 0..3 and names them psnr, f=10, f=15, f=20. So the key "f=10" is really blur
    # factor 0 and agrees with the unblurred PSNR to 5e-6 on all 500 rows; "f=15" is factor 10
    # and "f=20" is factor 15; factor 20 is computed and never recorded. The column below is
    # therefore labelled with the TRUE factor.
    #
    # ONE blur column, because the benchmark reports one. Which factor theirs is is not stated
    # in the paper, so putting their number beside ours asserts a correspondence that has not
    # been checked -- say so wherever this table is used.
    COLS = {"blackhole": [("ib_psnr", "PSNR $\\uparrow$", 2),
                          ("ib_blur_psnr (f=15)",
                           r"\shortstack{blur PSNR\\$f{=}10$ $\uparrow$}", 2),
                          ("ib_cp_chi2", r"$\chi^2_{\text{cp}}\to 1$", 2),
                          ("ib_camp_chi2", r"$\chi^2_{\text{camp}}\to 1$", 2)],
            "inv-scatter": [("ib_psnr", "PSNR $\\uparrow$", 2),
                            ("ib_ssim", "SSIM $\\uparrow$", 4)],
            "navier-stokes": [("ib_relative l2", r"rel.\ $L_2$ $\downarrow$", 4)]}[problem]

    d = df[df.problem == problem]
    t = _agg(d, [c for c, _, _ in COLS] + ["x_member_l2"])
    t = t.reset_index().sort_values(["algo", "n"])

    # A failures column of nothing but dashes is noise: drop it unless something failed.
    anyfail = bool((t["fail"] > 0).any())
    spec = "ll" + "c" * (len(COLS) + (1 if anyfail else 0))
    # blackhole carries the most metric columns; at \small it runs past the text block, so it
    # gets \footnotesize and a tighter column separation.
    out = [r"\footnotesize", r"\setlength{\tabcolsep}{3.5pt}"] if len(COLS) > 3 else [r"\small"]
    if problem == "blackhole":
        # The caveat travels INSIDE the fragment: a .tex file gets pasted into a draft months
        # later by someone who never read the notebook.
        out += [r"% CAUTION: these blackhole numbers use noise_type='vis_thermal', NOT the",
                r"% benchmark's 'eht' (our commit b93dfbb, 2026-09-17). That is a different",
                r"% MEASUREMENT MODEL, not just a different solver, so they are not directly",
                r"% comparable with published 'eht' baselines. State this where the table is used."]
    if problem == "navier-stokes":
        out += [r"% NOTE: sigma_noise = 1e-4 here, not the benchmark's 0.0 (our commit ab79ef2).",
                r"%",
                r"% THE BUDGETS ARE NOT MATCHED, and EnKG's advantage here is largely budget.",
                r"% Counted from algo/enkg.py with the benchmark's own config (num_samples=2048,",
                r"% num_steps=80, num_updates=2, threshold=0.1): 78 guided steps x 2 updates x",
                r"% 2048 particles = 319,488 simulator calls. Ours is T*M*N:",
                r"%     KPS-H N=512  32*2*512 =  32,768   (9.8x fewer than EnKG)",
                r"%     KPS-H N=64   32*2*64  =   4,096   (78x fewer)",
                r"% The other baselines' configs imply 1e5-1e6 calls but their batch_size means",
                r"% different things per algorithm and was NOT verified by reading each loop.",
                r"% State the call counts wherever this table is used."]
    out += [rf"\begin{{tabular}}{{{spec}}}", r"\toprule",
           "Method & $N$ & " + " & ".join(h for _, h, _ in COLS)
           + (r" & failures" if anyfail else "") + r" \\",
           r"\midrule"]
    base = BASELINES.get(problem)
    if base:
        # each problem's numbers come from a different table of the benchmark paper, and the
        # blackhole and NS rows are a specific block of theirs -- name it so it can be checked
        src = {"inv-scatter": "Table 3, 360 receivers",
               "blackhole": "Table 6, 100\\% observation time",
               "navier-stokes": r"Table 8, $\times 2$, $\sigma=0$"}[problem]
        # no %-formatting here: `src` contains a literal \% and would be read as a format spec
        ncol = len(COLS) + 2 + (1 if anyfail else 0)
        out.append(r"\multicolumn{" + str(ncol) + r"}{l}{\emph{published baselines "
                   r"(InverseBench, " + src + r")}} \\")
        for name, vals in base:
            cells = []
            for c, _, nd in COLS:
                v = vals.get("blur") if "blur_psnr" in c else vals.get(c)
                if v is None:
                    cells.append("--")
                elif isinstance(v, tuple):
                    cells.append(rf"${v[0]:.{nd}f} \pm {v[1]:.{nd}f}$")
                else:
                    cells.append(f"${v:.{nd}f}$")
            bn = vals.get("N", "--")
            out.append(f"{name} & {bn} & " + " & ".join(cells)
                       + (r" & --" if anyfail else "") + r" \\")

    # separate the published rows from ours; without this the two blocks run together
    out += [r"\midrule",
            r"\multicolumn{" + str(len(COLS) + 2 + (1 if anyfail else 0))
            + r"}{l}{\emph{this work}} \\"]

    med = None
    if stat == "median":
        ok = d[d.x_mean_l2 <= DIVERGED]
        med = ok.groupby(["algo", "n"])
    for _, r in t.iterrows():
        if stat == "median":
            gg = med.get_group((r.algo, r.n))
            cells = []
            for c, _, nd in COLS:
                q1, m, q3 = gg[c].quantile([.25, .5, .75])
                # stacked, not side by side: side by side this table ran 97pt past the block
                cells.append(rf"\shortstack{{{m:.{nd}f}\\"
                             rf"{{\tiny[{q1:.{nd}f},\,{q3:.{nd}f}]}}}}"
                             if pd.notna(m) else "--")
        else:
            cells = [_pm(r.get(c + "_mean"), r.get(c + "_sd"), nd) for c, _, nd in COLS]
        name = "KPS-G" if r.algo == "KPSG" else "KPS-H"
        fail = "--" if r["fail"] == 0 else rf"{int(r['fail'])}/{int(r['ids'])}"
        if stat == "median":          # match the stacked cells' baseline (see above)
            name = rf"\shortstack[l]{{{name}\\\strut}}"
            nn = rf"\shortstack{{{int(r.n)}\\\strut}}"
            fail = rf"\shortstack{{{fail}\\\strut}}"
        else:
            nn = str(int(r.n))
        out.append(f"{name} & {nn} & " + " & ".join(cells)
                   + (f" & {fail}" if anyfail else "") + r" \\")
    out += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(out)


def posterior_table(df: pd.DataFrame) -> str:
    r"""The distributional metrics, which are the claim the bench metrics cannot express."""

    COLS = [("x_member_l2", TEXHEAD["x_member_l2"], 3),
            ("x_mean_l2", TEXHEAD["x_mean_l2"], 3),
            ("x_crps_n", TEXHEAD["x_crps_n"], 4),
            ("x_ssr", TEXHEAD["x_ssr"], 3),
            ("x_cover90", TEXHEAD["x_cover90"], 3)]
    blocks = []
    for prob in ("blackhole", "inv-scatter", "navier-stokes"):
        d = df[df.problem == prob]
        if d.empty:
            continue
        t = _agg(d, [c for c, _, _ in COLS]).reset_index().sort_values(["algo", "n"])
        rows = []
        for _, r in t.iterrows():
            name = "KPS-G" if r.algo == "KPSG" else "KPS-H"
            cells = [_pm(r.get(c + "_mean"), r.get(c + "_sd"), nd) for c, _, nd in COLS]
            rows.append(f"{name} & {int(r.n)} & " + " & ".join(cells) + r" \\")
        blocks.append((NICE[prob], rows))
    return _group_table("ll" + "c" * len(COLS),
                        r"Method & $N$ & " + " & ".join(h for _, h, _ in COLS), blocks)


def posterior_table_median(df: pd.DataFrame) -> str:
    r"""Median with the inter-quartile range stacked beneath it, so the table stays inside the
    text width. Needed because blackhole KPS-H's spread/error standard deviation reaches 5.2:
    a few near-divergent instances carry an enormous ratio while their MEAN error is fine, so
    the `x_mean_l2` filter does not catch them. Quote this wherever an sd exceeds its mean."""

    COLS = [("x_member_l2", TEXHEAD["x_member_l2"], 3),
            ("x_mean_l2", TEXHEAD["x_mean_l2"], 3),
            ("x_crps_n", TEXHEAD["x_crps_n"], 4),
            ("x_ssr", TEXHEAD["x_ssr"], 3),
            ("x_cover90", TEXHEAD["x_cover90"], 3)]
    blocks = []
    for prob in ("blackhole", "inv-scatter", "navier-stokes"):
        d = df[(df.problem == prob) & (df.x_mean_l2 <= DIVERGED)]
        if d.empty:
            continue
        g = d.groupby(["algo", "n"])
        rows = []
        for k in sorted(g.groups):
            gg = g.get_group(k)
            name = "KPS-G" if k[0] == "KPSG" else "KPS-H"
            cells = []
            for c, _, nd in COLS:
                q1, med, q3 = gg[c].quantile([.25, .5, .75])
                # the IQR goes UNDER the median: side by side it overflows the text block
                # \shortstack is base LaTeX; \makecell would need a package the draft's
                # preamble does not load, which is exactly how \multirow printed literally.
                cells.append(rf"\shortstack{{{med:.{nd}f}\\"
                             rf"{{\tiny[{q1:.{nd}f},\,{q3:.{nd}f}]}}}}")
            lab = rf"\shortstack[l]{{{name}\\\strut}}"
            num = rf"\shortstack{{{int(k[1])}\\\strut}}"
            rows.append(f"{lab} & {num} & " + " & ".join(cells) + r" \\")
        blocks.append((NICE[prob], rows))
    return _group_table("ll" + "c" * len(COLS),
                        r"Method & $N$ & " + " & ".join(h for _, h, _ in COLS), blocks,
                        size=r"\footnotesize")


# ----------------------------------------------------------------- figures

# ICLR's text block. Figures are built AT this width so \includegraphics[width=\linewidth]
# is a 1:1 placement and the type renders at the size it was set in. Building at 6.9in and
# letting LaTeX shrink to 5.5in silently turned 8pt labels into 6.4pt, and the 9in-wide
# calibration figure into 4.9pt.
FIGW = 5.5

C = {"KPSH": "#2a78d6", "KPSG": "#eb6834"}
LBL = {"KPSH": "KPS-H", "KPSG": "KPS-G"}

# One name per metric, written so it needs no caption, with the direction as an arrow rather
# than a parenthetical. `member_l2` is the error of a SINGLE posterior sample (averaged over
# members); `mean_l2` is the error of the ensemble MEAN -- both relative L2 against the truth.
# `ssr` is literally spread / error, so that is what it is called.
NAME = {
    "x_member_l2": "single-sample error \u2193",
    "x_mean_l2":   "posterior-mean error \u2193",
    "x_crps_n":    "CRPS \u2193",
    "x_ssr":       "spread \u00f7 error \u2192 1",
    "x_cover90":   "90% coverage \u2192 0.9",
    "y_member_l2": "single-sample error \u2193",
    "y_crps_n":    "CRPS \u2193",
    "y_ssr":       "spread \u00f7 error \u2192 1",
    "y_cover90":   "90% coverage \u2192 0.9",
}
# STANDARD ENSEMBLE-VERIFICATION NAMES, so a label needs no caption to decode:
#   skill                = RMSE of the ENSEMBLE MEAN
#   spread               = ensemble sd, with the usual (N+1)/N finite-ensemble inflation
#   spread-skill ratio   = spread / skill, which is 1 for a calibrated ensemble
#   particle RMSE        = RMSE of a single member, averaged over members
# `particle RMSE` and `skill` are both RELATIVE (divided by ||x*||); say so in the caption.
# An earlier version used invented phrasings ("single-sample error", "spread / error") that
# had to be explained; these do not.
NAME = {
    "x_member_l2": "particle RMSE \u2193",
    "x_mean_l2":   "skill \u2193",
    "x_crps_n":    "CRPS \u2193",
    "x_ssr":       "spread-skill ratio \u2192 1",
    "x_cover90":   "90% coverage \u2192 0.9",
}
NAME.update({k.replace("x_", "y_"): v for k, v in list(NAME.items())})

TEXNAME = {
    "x_member_l2": r"particle RMSE $\downarrow$",
    "x_mean_l2":   r"skill $\downarrow$",
    "x_crps_n":    r"CRPS $\downarrow$",
    "x_ssr":       r"spread-skill ratio $\to 1$",
    "x_cover90":   r"90\% coverage $\to 0.9$",
}
TEXNAME.update({k.replace("x_", "y_"): v for k, v in list(TEXNAME.items())})

# stacked for table headers, which are narrow
TEXHEAD = {
    "x_member_l2": r"\shortstack{particle\\RMSE $\downarrow$}",
    "x_mean_l2":   r"\shortstack{skill\\$\downarrow$}",
    "x_crps_n":    r"\shortstack{CRPS\\$\downarrow$}",
    "x_ssr":       r"\shortstack{spread-skill\\ratio $\to 1$}",
    "x_cover90":   r"\shortstack{90\% coverage\\$\to 0.9$}",
}
TEXHEAD.update({k.replace("x_", "y_"): v for k, v in list(TEXHEAD.items())})



def style():
    import matplotlib.pyplot as plt
    import seaborn  # noqa: F401  -- registers 'icefire'
    plt.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 300, "font.size": 8,
        "axes.titlesize": 8.5, "axes.labelsize": 8, "axes.spines.top": False,
        "axes.spines.right": False, "axes.edgecolor": "#8a8a84",
        "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.frameon": False,
        "legend.fontsize": 7.5, "lines.linewidth": 1.5, "figure.constrained_layout.use": True,
        "pdf.fonttype": 42, "ps.fonttype": 42,        # embed real fonts, not type-3
    })


def _series(ax, d, col, rng, single_marker="D"):
    r"""One arm per algorithm: median over instances with a bootstrap 95% interval.

    SHARED by fig_particles and fig_ppc so the two cannot drift apart -- an earlier version
    drew state space with error bars and observation space with a shaded IQR band, which
    invited the reader to compare two different summaries. An algorithm that ran at a single
    ensemble size is drawn as a point, not a line, so it is not read as a trend.
    """

    for algo, ga in d.groupby("algo"):
        g = ga.groupby("n")[col]
        xs = np.array(sorted(g.groups))
        med = np.array([g.get_group(x).median() for x in xs])
        lo, hi = [], []
        for x in xs:
            v = g.get_group(x).values
            b = np.median(rng.choice(v, (2000, len(v))), axis=1)
            lo.append(np.percentile(b, 2.5)); hi.append(np.percentile(b, 97.5))
        err = np.vstack([med - lo, np.array(hi) - med])
        if len(xs) > 1:
            ax.errorbar(xs, med, err, marker="o", ms=3.5, capsize=2, lw=1.4,
                        color=C[algo], label=LBL[algo])
        else:
            ax.errorbar(xs, med, err, marker=single_marker, ms=5, ls="none", capsize=2,
                        color=C[algo], label=LBL[algo])


def fig_particles(df: pd.DataFrame, out: pathlib.Path | None = None,
                  metrics=("x_member_l2", "x_ssr", "x_cover90"), transpose=False):
    r"""The calibration claim in one figure: accuracy and honesty against ensemble size.

    Median over instances with a bootstrap 95% interval, because a few blackhole instances
    diverge and a mean would describe them rather than the method. KPS-G sits at a single N
    and is drawn as a point, not a line, so it is not read as a trend.
    """

    import matplotlib.pyplot as plt
    style()
    # `metrics` selects the columns. The main text wants the two that carry the claim --
    # accuracy holds, calibration improves with N -- and 90% coverage says the same thing as
    # spread/error, so it is appendix material when space is short.
    TGT = {"x_ssr": 1.0, "x_cover90": 0.90}
    PANELS = [(c, NAME[c], TGT.get(c), c == "x_member_l2") for c in metrics]
    probs = [p for p in ("blackhole", "inv-scatter", "navier-stokes") if (df.problem == p).any()]
    # TRANSPOSE for the main text: problems across, metrics down. Panel WIDTH is fixed by the
    # text block, so fewer columns means wider -- and therefore taller -- panels. The 2-metric
    # version came out 8in tall, worse than the 3-metric one it was meant to shrink. With the
    # three problems as columns the same content is about 3.5in.
    nrow, ncol = (len(PANELS), len(probs)) if transpose else (len(probs), len(PANELS))
    pw = FIGW / max(ncol, 1)
    fig, axes = plt.subplots(nrow, ncol, figsize=(FIGW, 0.95 * pw * nrow), squeeze=False)
    rng = np.random.default_rng(0)

    for i, prob in enumerate(probs):
        d = df[(df.problem == prob) & (df.x_mean_l2 <= DIVERGED)]
        for j, (col, lab, tgt, logy) in enumerate(PANELS):
            ax = axes[j][i] if transpose else axes[i][j]
            if tgt is not None:
                ax.axhline(tgt, color="#b0b0aa", ls=(0, (4, 3)), lw=1, zorder=0)
            _series(ax, d, col, rng)
            ax.set_xscale("log", base=2)
            if logy and col == "x_member_l2":
                ax.set_yscale("log")
            last_row = (j == len(PANELS) - 1) if transpose else (i == len(probs) - 1)
            if last_row:
                ax.set_xlabel("ensemble size $N$")
            if transpose:
                if i == 0:
                    ax.set_ylabel(lab)
                if j == 0:
                    ax.set_title(PLAIN[prob], loc="left", fontweight="bold")
            else:
                if j == 0:
                    ax.set_ylabel(PLAIN[prob], fontweight="bold")
                if i == 0:
                    ax.set_title(lab, loc="left")
    axes[0][0].legend(loc="best", fontsize=6.5)
    if out:
        fig.savefig(out, bbox_inches="tight")
    return fig


# ----------------------------------------------------------------- galleries

# Per-problem colour, chosen by DOMAIN CONVENTION where one exists:
#   navier-stokes  seaborn's `icefire`, diverging and dark at zero -- vorticity is signed and
#                  SIGNED[] keeps the scale centred, so dark really does mean zero. It shows
#                  the filaments that RdBu_r flattened.
#   blackhole      `afmhot`, which is ehtim's own default (Image.display cfun) and what the
#                  published EHT images use. `inferno` looks similar but is not the convention.
#   inv-scatter    `Greys_r`: the scatterer is bright on a dark background, the usual way this
#                  permittivity is shown, and it matches afmhot's dark field.
CMAP = {"navier-stokes": "icefire", "inv-scatter": "Greys_r", "blackhole": "afmhot"}
SIGNED = {"navier-stokes": True, "inv-scatter": False, "blackhole": False}


def load(problem: str, algo: str, run: str, i: int):
    r"""One stored ensemble as (samples, truth), unnormalised, channel dropped."""

    import torch
    f = EXP / EXPDIR[problem] / algo / run / f"result_{i}.pt"
    d = torch.load(f, map_location="cpu", weights_only=False)
    x = d["recon"].float()
    t = d["target"].float().reshape(1, *x.shape[1:])
    sq = lambda z: z[:, 0] if z.dim() == 4 else z
    return sq(x), sq(t), d


def _scale(arr, problem, kind="field"):
    r"""Shared colour limits. Whether a field is two-sided is PHYSICS, not a property of the
    sample: vorticity is signed and centred on zero, permittivity and intensity are not."""

    a = np.asarray(arr)
    if kind == "error":
        v = np.percentile(np.abs(a), 99.5)
        return dict(cmap="RdBu_r", vmin=-v, vmax=v)
    if kind == "mag":
        # UNCERTAINTY MUST NOT LOOK LIKE ANOTHER INTENSITY IMAGE. magma is dark-to-warm like
        # afmhot, so beside a blackhole field panel it read as a second brightness map;
        # `mako` collides with icefire's blue on Navier-Stokes and `cividis` washes out there.
        # viridis shares no hue with either field map, and its purple floor separates "zero
        # uncertainty" from afmhot's black "zero intensity".
        return dict(cmap="viridis", vmin=0, vmax=np.percentile(a, 99.5))
    if SIGNED[problem]:
        v = np.percentile(np.abs(a), 99.5)
        return dict(cmap=CMAP[problem], vmin=-v, vmax=v)
    return dict(cmap=CMAP[problem], vmin=np.percentile(a, 0.5),
                vmax=np.percentile(a, 99.5))


def _panel(ax, img, title="", **kw):
    ax.imshow(np.asarray(img), origin="lower", interpolation="nearest", **kw)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    if title:
        ax.set_title(title, fontsize=7, pad=2)


def fig_gallery_main(problem="blackhole", algo="KPSH", idx=0,
                     runs=("gen_N64", "gen_N128", "gen_N512"),
                     out: pathlib.Path | None = None):
    r"""MAIN-TEXT figure, deliberately small: what more particles buy, visually.

    Row 1 is the posterior mean against the truth as N grows; row 2 is the per-pixel standard
    deviation on a shared scale. The claim the figure has to carry is that the ensemble gains
    structure rather than merely shrinking, so the uncertainty panels share one colour scale
    across N -- scaling each panel to itself would hide exactly the effect being shown.
    """

    import matplotlib.pyplot as plt
    style()
    ens = [load(problem, algo, r, idx) for r in runs]
    truth = ens[0][1][0]
    means = [e[0].mean(0) for e in ens]
    stds = [e[0].std(0) for e in ens]

    fs = _scale(np.stack([truth.numpy()] + [m.numpy() for m in means]), problem)
    ss = _scale(np.stack([s.numpy() for s in stds]), problem, "mag")

    n = len(runs) + 1
    fig, ax = plt.subplots(2, n, figsize=(1.28 * n, 2.75))
    _panel(ax[0, 0], truth, "truth", **fs)
    ax[1, 0].axis("off")
    for j, (r, m, s) in enumerate(zip(runs, means, stds), start=1):
        N = int(r.split("_N")[-1])
        _panel(ax[0, j], m, f"$N={N}$", **fs)
        _panel(ax[1, j], s, "", **ss)
    ax[0, 0].set_ylabel("mean", fontsize=7.5)
    ax[1, 1].set_ylabel("sd", fontsize=7.5)
    for a in (ax[0, 0], ax[1, 1]):
        a.set_yticks([]); a.yaxis.set_visible(True)
    if out:
        fig.savefig(out, bbox_inches="tight")
    return fig


def fig_gallery_samples(problem, algo, run, idx=0, k=6, out=None):
    r"""APPENDIX figure: individual members, which is where multimodality is visible at all.

    A mean and a standard deviation cannot show two modes -- they average them into a blur
    with a wide error bar. Only the members can, so this draws them on the truth's colour
    scale and adds the mean for reference.
    """

    import matplotlib.pyplot as plt
    style()
    x, t, _ = load(problem, algo, run, idx)
    k = min(k, x.shape[0])
    sel = np.linspace(0, x.shape[0] - 1, k).astype(int)
    fs = _scale(np.concatenate([t.numpy(), x[sel].numpy()]), problem)

    fig, ax = plt.subplots(1, k + 2, figsize=(1.15 * (k + 2), 1.5))
    _panel(ax[0], t[0], "truth", **fs)
    _panel(ax[1], x.mean(0), "mean", **fs)
    for j, s in enumerate(sel):
        _panel(ax[j + 2], x[s], "posterior samples" if j == 0 else "", **fs)
    fig.suptitle(f"{PLAIN[problem]} — {LBL[algo]} $N={x.shape[0]}$, instance {idx}",
                 fontsize=8, x=0.005, ha="left")
    if out:
        fig.savefig(out, bbox_inches="tight")
    return fig


def bimodality(x, seed=0):
    r"""2-means separation over within-cluster spread, in the ensemble's top-2 PC plane.

    A scalar that says whether the members form TWO groups rather than one cloud. Used to
    PICK the instance shown in the multimodality figure, so the choice is reproducible and
    stated, rather than an eye-catching instance found by browsing.
    """

    v = x.reshape(x.shape[0], -1).numpy().astype(np.float64)
    v = v - v.mean(0)
    U, S, _ = np.linalg.svd(v, full_matrices=False)
    z = U[:, :2] * S[:2]
    rng = np.random.default_rng(seed)
    c = z[rng.choice(len(z), 2, replace=False)]
    for _ in range(50):
        lab = np.argmin(((z[:, None] - c[None]) ** 2).sum(-1), 1)
        if len(set(lab)) < 2:
            return 0.0
        c = np.stack([z[lab == k].mean(0) for k in (0, 1)])
    within = np.mean([np.linalg.norm(z[lab == k] - c[k], axis=1).mean() for k in (0, 1)])
    return float(np.linalg.norm(c[0] - c[1]) / max(within, 1e-12))


def scan_bimodal(problem, algo, run, ids=range(0, 100, 2)):
    r"""Rank instances by bimodality, skipping ensembles that diverged."""

    out = []
    for i in ids:
        try:
            x, _, _ = load(problem, algo, run, i)
        except Exception:
            continue
        if x.reshape(x.shape[0], -1).std() > 50:
            continue
        out.append((bimodality(x), int(i)))
    return sorted(out, reverse=True)


def fig_multimodal(problem="blackhole", algo="KPSH", run="gen_N512", idx=None,
                   k=4, out=None):
    r"""THE figure for the uncertainty claim: a posterior with two modes, and both captured.

    Members are split by the sign of their first principal component, which for a genuinely
    bimodal ensemble is the axis separating the modes. The mean is shown deliberately: it is
    the average of two incompatible images and resembles neither, which is the argument that a
    point estimate cannot represent this posterior.
    """

    import matplotlib.pyplot as plt
    style()
    if idx is None:
        idx = scan_bimodal(problem, algo, run)[0][1]
    x, t, _ = load(problem, algo, run, idx)
    v = x.reshape(x.shape[0], -1).numpy().astype(np.float64)
    v = v - v.mean(0)
    U, S, _ = np.linalg.svd(v, full_matrices=False)
    z = U[:, 0] * S[0]
    order = np.argsort(z)
    lo, hi = order[:k], order[-k:]

    # THE MEAN MUST BE IN THE COLOUR SCALE. Averaging two modes lowers the peak, so a scale
    # taken from the truth and the members alone renders the mean dim and hides the very
    # thing the figure exists to show -- that it is a blur of two incompatible images.
    fs = _scale(np.concatenate([t.numpy(), x.mean(0).numpy()[None],
                                x.numpy()[list(lo) + list(hi)]]), problem)
    fig, ax = plt.subplots(2, k + 2, figsize=(FIGW, 2.1 * FIGW / (k + 2)),
                           gridspec_kw=dict(width_ratios=[1, 0.18] + [1] * k))
    _panel(ax[0, 0], t[0], "truth", **fs)
    _panel(ax[1, 0], x.mean(0), "ensemble mean", **fs)
    ax[0, 1].axis("off"); ax[1, 1].axis("off")      # a gap, so the mean does not read as a mode
    for j, i in enumerate(lo):
        _panel(ax[0, j + 2], x[i], "mode A" if j == 0 else "", **fs)
    for j, i in enumerate(hi):
        _panel(ax[1, j + 2], x[i], "mode B" if j == 0 else "", **fs)
    if out:
        fig.savefig(out, bbox_inches="tight")
    return fig, idx


# ----------------------------------------------------------------- TARP

def tarp_curve(samples, truths, n_alpha=51, seed=0, ref="prior", n_draws=20):
    r"""The TARP statistic itself, on arrays. Separated from the file loading so it can be
    VALIDATED against cases whose answer is known -- see `tarp_selftest`.

        f_i = (1/N) sum_j 1[ ||x_ij - r|| < ||x_i* - r|| ]
        ECP(alpha) = (1/M) sum_i 1[ f_i <= alpha ]

    The credible region at level alpha is the ball about r holding alpha of the posterior
    mass, so the truth is inside exactly when f_i <= alpha. If the ensembles are posterior
    draws then f is uniform and ECP(alpha) = alpha.

    `n_draws` reference points per instance are averaged: one draw per instance is unbiased
    but noisy, and the curve is what is being read.
    """

    rng = np.random.default_rng(seed)
    S = [np.asarray(s, dtype=np.float64).reshape(len(s), -1) for s in samples]
    T = np.asarray(truths, dtype=np.float64).reshape(len(truths), -1)
    M = len(T)
    alpha = np.linspace(0, 1, n_alpha)
    acc = np.zeros(n_alpha)

    for _ in range(n_draws):
        fs = np.empty(M)
        for i in range(M):
            if ref == "gauss":
                r = T.mean(0) + T.std(0) * rng.standard_normal(T.shape[1])
            else:
                j = rng.choice([m for m in range(M) if m != i])
                r = T[j]
            d_ref = np.linalg.norm(T[i] - r)
            fs[i] = (np.linalg.norm(S[i] - r, axis=1) < d_ref).mean()
        acc += np.array([(fs <= a).mean() for a in alpha])
    return alpha, acc / n_draws, fs


def tarp_selftest(d=64, M=200, N=200, seed=0):
    r"""Does the estimator say what it should on cases whose answer is known?

    A correctly-sampled Gaussian posterior must sit on the diagonal; inflating the ensemble's
    spread must push the curve ABOVE it (too broad), shrinking it BELOW (over-confident).
    Returns the maximum deviation from the diagonal for each case.
    """

    rng = np.random.default_rng(seed)
    mu = rng.standard_normal((M, d))                 # one posterior centre per instance
    # THE TRUTH MUST BE A DRAW FROM THE SAME POSTERIOR AS THE SAMPLES. Centring the ensemble
    # ON the truth instead makes the truth artificially central, and the estimator correctly
    # reports that as over-dispersion -- which is a broken TEST, not a broken statistic.
    truths = mu + rng.standard_normal((M, d))
    out = {}
    for name, fac in (("calibrated", 1.0), ("over-dispersed x2", 2.0),
                      ("over-confident x0.5", 0.5)):
        samples = [m + fac * rng.standard_normal((N, d)) for m in mu]
        a, c, _ = tarp_curve(samples, truths, seed=seed, n_draws=10)
        out[name] = (a, c, float(np.max(c - a)), float(np.min(c - a)))
    return out


def tarp(problem, algo, run, ids, n_alpha=51, seed=0, ref="prior", n_draws=20,
         max_members=128):
    r"""Expected-coverage curve from STORED ensembles (Lemos et al., arXiv:2302.03026).

    The implementation in `KPS/kps/metrics.py` wants a callable posterior sampler, which an
    amortised estimator has and a stored campaign does not. The statistic is the same: for
    each instance draw a reference point r, and record the fraction of members closer to r
    than the truth is,

        f_i = (1/N) sum_j  1[ ||x_ij - r_i|| < ||x_i* - r_i|| ].

    If the ensembles are drawn from the true posterior then f is uniform on [0, 1], so the
    ECDF of f against the diagonal IS the calibration curve: above the diagonal the ensembles
    are too broad, below it too narrow.

    `ref="prior"` draws each reference from the OTHER instances' ground truths, which is a
    sample of the prior and needs no density. `ref="gauss"` uses a Gaussian matched to the
    truths. Streaming: only the scalar f_i is kept, so a 100 x 512 x 64 x 64 campaign never
    lands in memory at once.
    """

    import torch
    truths, samples = [], []
    for i in ids:
        try:
            x, t, _ = load(problem, algo, run, i)
        except Exception:
            continue
        v = x.reshape(x.shape[0], -1).numpy()
        if not np.isfinite(v).all() or v.std() > 50:
            continue                               # a diverged ensemble is not a posterior
        if max_members and v.shape[0] > max_members:      # the statistic needs a sample of
            v = v[::max(1, v.shape[0] // max_members)][:max_members]   # members, not all of
        samples.append(v.astype(np.float32))          # them; float32 halves 3.4 GB on 128^2
        truths.append(t[0].reshape(-1).numpy().astype(np.float64))
    return tarp_curve(samples, np.stack(truths), n_alpha=n_alpha, seed=seed,
                      ref=ref, n_draws=n_draws)


def fig_tarp(curves, out=None):
    r"""One panel per problem; a curve above the diagonal is over-dispersed, below is
    over-confident. The 1-sigma band is the binomial spread at the number of instances, so a
    deviation inside it is not evidence of miscalibration."""

    import matplotlib.pyplot as plt
    style()
    probs = sorted({p for p, _, _ in curves})
    fig, ax = plt.subplots(1, len(probs), figsize=(2.3 * len(probs), 2.4), squeeze=False)
    for a, prob in zip(ax[0], probs):
        a.plot([0, 1], [0, 1], color="#b0b0aa", ls=(0, (4, 3)), lw=1)
        m = None
        for p, lab, (al, cov, fs) in curves:
            if p != prob:
                continue
            m = len(fs)
            col = C["KPSG"] if "KPS-G" in lab else C["KPSH"]
            a.plot(al, cov, label=lab, lw=1.4, color=col)
        if m:
            al = np.linspace(0, 1, 51)
            s = np.sqrt(al * (1 - al) / m)
            a.fill_between(al, al - s, al + s, color="#c9c9c4", alpha=.35, lw=0, zorder=0)
        a.set_title(PLAIN[prob], loc="left", fontsize=8)
        a.set_xlabel("credibility level"); a.set_xlim(0, 1); a.set_ylim(0, 1)
        a.set_aspect("equal")
    ax[0][0].set_ylabel("expected coverage")
    for a in ax[0]:                       # every panel names its own curves
        if a.get_legend_handles_labels()[0]:
            a.legend(loc="upper left", fontsize=6.5)
    if out:
        fig.savefig(out, bbox_inches="tight")
    return fig


def rank_hist(problem, algo, run, ids, n_pix=400, seed=0, max_members=128):
    r"""A TRUE rank histogram: the rank of the truth among the members, PER PIXEL, pooled.

    Not to be confused with `x_rank` in the metrics cache, which is
    `(members < truth).mean()` over the WHOLE field. That scalar averages thousands of pixels,
    so it concentrates at 0.5 for any dispersion and is a diagnostic of MEAN BIAS, not of
    spread -- plotting it as a rank histogram reads as over-dispersion no matter what the
    ensemble does, which contradicts SSR and TARP on this campaign.

    The classical statistic ranks the truth among the N members at ONE location, and pools
    those ranks over locations and instances:

        flat      calibrated
        U-shaped  the truth falls outside the ensemble too often -> too narrow
        dome      the truth sits in the middle too often          -> too wide
    """

    rng = np.random.default_rng(seed)
    out = []
    for i in ids:
        try:
            x, t, _ = load(problem, algo, run, i)
        except Exception:
            continue
        v = x.reshape(x.shape[0], -1).numpy()
        if not np.isfinite(v).all() or v.std() > 50:
            continue
        if max_members and v.shape[0] > max_members:
            v = v[:: max(1, v.shape[0] // max_members)][:max_members]
        y = t[0].reshape(-1).numpy()
        idx = rng.choice(v.shape[1], min(n_pix, v.shape[1]), replace=False)
        r = (v[:, idx] < y[idx]).sum(0) / v.shape[0]      # in [0, 1]
        out.append(r)
    return np.concatenate(out) if out else np.array([])


def fig_rank(jobs, out=None):
    r"""Pooled rank histograms. A U shape is the signature of an over-confident ensemble."""

    import matplotlib.pyplot as plt
    style()
    probs = sorted({p for p, _, _, _ in jobs}, key=lambda p:
                   ["blackhole", "inv-scatter", "navier-stokes"].index(p))
    fig, ax = plt.subplots(1, len(probs), figsize=(2.3 * len(probs), 2.1), squeeze=False)
    for a, prob in zip(ax[0], probs):
        for p, algo, run, ids in jobs:
            if p != prob:
                continue
            r = rank_hist(p, algo, run, ids)
            if not len(r):
                continue
            N = run.split("_N")[-1] if "_N" in run else "16"
            a.hist(r, bins=15, range=(0, 1), histtype="step", density=True,
                   color=C[algo], lw=1.4, label=f"{LBL[algo]} N={N}")
        a.axhline(1.0, color="#b0b0aa", ls=(0, (4, 3)), lw=1)
        a.set_title(PLAIN[prob], loc="left", fontsize=8)
        a.set_xlabel("rank of the truth")
        a.legend(fontsize=6.5)
    ax[0][0].set_ylabel("density")
    if out:
        fig.savefig(out, bbox_inches="tight")
    return fig


# ----------------------------------------------------------------- MIRA

def mira_core(truth, posterior, num_runs=100, norm=True, seed=0, center="uniform"):
    r"""MIRA calibration score (Saidi et al., arXiv:2605.02014; github SammyS15/mira-score).

    Paired-sample calibration with no density, no evidence integral and no reference
    posterior. For each run and each ground truth: draw a random centre, let the radius be the
    distance from that centre to ONE HELD-OUT posterior sample, count how many of the
    remaining N = S-1 samples fall inside, and score the probability the ensemble assigned to
    what actually happened:

        p          = (counts + 1) / (N + 2)                 Laplace-smoothed mass inside
        score_i    = [ p*k + (1-p')*(1-k) ] / max_val,      k = 1[truth inside]
        max_val    = (N + 1) / (N + 2)

    WHY 2/3 IS THE CALIBRATED VALUE, and why it does not depend on the dimension: the held-out
    sample's distance rank among the others is uniform when the members are exchangeable, so
    p ~ U(0,1); and the truth is exchangeable with the members exactly when the posterior is
    calibrated, so P(inside | p) = p. Then E[score] = E[p^2 + (1-p)^2] = 2/3. If the truth is
    unrelated to the ensemble, k is independent of p with probability 1/2, giving 1/2. The
    centre distribution therefore affects the POWER of the test but not its reference value.

    `truth` is (T, q); `posterior` is (M, T, S, q) or a list of M lists of (S, q) arrays.
    Returns (score, std) per model, the std being the spread over `num_runs`.
    """

    rng = np.random.default_rng(seed)
    T = len(truth)
    Y = np.asarray(truth, dtype=np.float32).reshape(T, -1)
    models = [[np.asarray(s, dtype=np.float32).reshape(len(s), -1) for s in m]
              for m in posterior]

    if norm:                     # pooled z-score, which preserves relative positions
        allv = np.concatenate([Y] + [s for m in models for s in m], axis=0)
        mu, sd = allv.mean(), allv.std()
        sd = sd if sd > 0 else 1.0
        Y = (Y - mu) / sd
        models = [[(s - mu) / sd for s in m] for m in models]

    q = Y.shape[1]
    out = []
    for m in models:
        runs = np.empty(num_runs)
        for r in range(num_runs):
            vals = np.empty(T)
            for i in range(T):
                S = m[i]
                N = S.shape[0] - 1
                c = (rng.random(q).astype(np.float32) if center == "uniform"
                     else Y[rng.integers(T)])
                d = np.linalg.norm(S - c, axis=1)
                h = rng.integers(S.shape[0])            # the held-out sample sets the radius
                rad = d[h]
                keep = np.ones(S.shape[0], bool); keep[h] = False
                counts = int((d[keep] < rad).sum())
                k = float(np.linalg.norm(Y[i] - c) < rad)
                p_in = (counts + 1) / (N + 2)
                p_out = (N - counts + 1) / (N + 2)
                vals[i] = (p_in * k + p_out * (1 - k)) / ((N + 1) / (N + 2))
            runs[r] = vals.mean()
        out.append((float(runs.mean()), float(runs.std())))
    return out


def mira_selftest(d=32, T=120, S=64, num_runs=30, seed=0):
    r"""Does the implementation reproduce the paper's two reference values?

    calibrated   -> ~2/3   (truth drawn from the same posterior as the members)
    unpaired     -> ~1/2   (truth shuffled against the ensembles)
    over-confident and over-dispersed both fall BELOW 2/3, which is the point: MIRA scores
    calibration, and either failure direction is a miscalibration.
    """

    rng = np.random.default_rng(seed)
    # THE INSTANCES MUST BE WELL SEPARATED for the unpaired case to be detectable at all.
    # With mu ~ N(0, I) and a within-ensemble spread of 1, a shuffled ensemble sits almost on
    # top of the right one and MIRA correctly reports it as nearly calibrated -- that is a
    # weak TEST, not a wrong statistic. Separating the centres by 5 sigma makes "unpaired"
    # mean what the paper means by it.
    mu = 5.0 * rng.standard_normal((T, d))
    truth = mu + rng.standard_normal((T, d))
    cal = [mu[i] + rng.standard_normal((S, d)) for i in range(T)]
    nar = [mu[i] + 0.3 * rng.standard_normal((S, d)) for i in range(T)]
    wid = [mu[i] + 3.0 * rng.standard_normal((S, d)) for i in range(T)]
    shuf = [cal[j] for j in rng.permutation(T)]
    res = mira_core(truth, [cal, nar, wid, shuf], num_runs=num_runs, seed=seed)
    return dict(zip(["calibrated (->0.667)", "over-confident x0.3",
                     "over-dispersed x3", "unpaired (->0.5)"], res))


def mira_campaign(jobs, num_runs=30, max_members=64, seed=0):
    r"""MIRA over stored runs. `jobs` is a list of (problem, algo, run, ids)."""

    per_problem = {}
    for prob, algo, run, ids in jobs:
        per_problem.setdefault(prob, []).append((algo, run, ids))

    out = []
    for prob, arms in per_problem.items():
        truths, sets, labels = None, [], []
        for algo, run, ids in arms:
            T, S = [], []
            for i in ids:
                try:
                    x, t, _ = load(prob, algo, run, i)
                except Exception:
                    continue
                v = x.reshape(x.shape[0], -1).numpy()
                if not np.isfinite(v).all() or v.std() > 50:
                    continue
                if max_members and v.shape[0] > max_members:
                    v = v[:: max(1, v.shape[0] // max_members)][:max_members]
                S.append(v)
                T.append(t[0].reshape(-1).numpy())
            sets.append(S)
            labels.append((algo, run, len(S)))
            truths = np.stack(T)
        # the arms must share the same truths; take the shortest common prefix
        n = min(len(s) for s in sets)
        res = mira_core(truths[:n], [s[:n] for s in sets], num_runs=num_runs, seed=seed)
        for (algo, run, _), (sc, sd) in zip(labels, res):
            N = run.split("_N")[-1] if "_N" in run else "16"
            out.append((prob, f"{LBL[algo]} N={N}", sc, sd, n))
    return out


def _tarp_color(lab, ns):
    r"""KPS-H shaded by ensemble size, KPS-G keeping its own hue.

    The particle trend is the thing being shown, so N has to be readable off the line itself
    and not only off the legend. A sequential ramp does that; two arbitrary hues would not.
    The ramp is built from the FULL set of N present in the figure, not per panel, so a given
    shade means the same ensemble size in every panel.
    """

    import matplotlib.pyplot as plt
    if "KPS-G" in lab:
        return C["KPSG"]
    n = int(lab.split("N=")[-1])
    i = sorted(ns).index(n)
    return plt.cm.Blues(0.38 + 0.57 * i / max(1, len(ns) - 1))


def fig_calibration(curves, mira_res, out=None, width=None, scale=1.0, row_h=2.15):
    r"""TARP and MIRA in one figure: the whole calibration claim on one row.

    The MIRA panel shows each arm's score with its own spread, against the calibrated
    reference 2/3 and the band expected from a FINITE number of ground truths,
    sigma = sqrt(1 / (18 L)) (the paper's expression). A point inside that band is not
    distinguishable from perfectly calibrated at that L; one at 1/2 is indistinguishable from
    an ensemble unrelated to the truth.
    """

    import matplotlib.lines as mlines
    import matplotlib.pyplot as plt
    style()
    # BUILD AT THE WIDTH IT WILL BE PLACED AT. In a two-column layout this figure spans about
    # half the text block; drawing it at FIGW and letting LaTeX shrink to 0.48\linewidth would
    # render 8pt labels at under 4pt. Passing width=FIGW/2 makes the placement 1:1, so type set
    # at 8pt arrives on the page at 8pt.
    #
    # The LAYOUT IS THE SAME AT EITHER WIDTH -- same panels, same tick density, same label
    # text. An earlier half-width version also thinned the ticks to three and shortened the
    # labels; that bought room the figure did not need and read as cramped. The only thing
    # that changes with width is `scale`, which trims the type a little so that six tick
    # labels still clear each other in a 1.4in panel.
    W = width or FIGW
    mag = scale * FIGW / W          # how much bigger the type is RELATIVE to the panels
    plt.rcParams.update({k: v * scale for k, v in {
        "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
        "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
    }.items()})
    probs = [p for p in ("blackhole", "inv-scatter", "navier-stokes")
             if any(q == p for q, _, _ in curves)]
    # ONE ROW: the three TARP panels, then MIRA. A 2 x 2 gives each panel 2.75in and more
    # room than it needs, at the cost of a figure half a page tall; in one line each is about
    # 1.25in and the whole thing is under 2in, which is what a column wants.
    n = len(probs) + 1
    ncol, nrow = n, 1
    # NOT SQUARE, AND THAT IS THE POINT. Four panels across 5.5in leaves each 1.25in wide, and
    # with set_aspect("equal") that also caps their HEIGHT at 1.25in, which is where the curves
    # became unreadable -- the aspect, not the layout, was the binding constraint. Letting the
    # panels be taller than they are wide costs the 45-degree diagonal (it is still the
    # reference line, just steeper) and buys back every bit of vertical resolution.
    # MIRA GETS MORE WIDTH than a TARP panel: it carries five rotated category labels along
    # its x axis where a TARP panel carries three numbers, so at equal width its labels set
    # the crowding for the whole row.
    fig, axes = plt.subplots(nrow, ncol, figsize=(W, row_h),
                             gridspec_kw={"width_ratios": [1] * len(probs) + [1.35]})
    ax = np.atleast_1d(axes).ravel()
    # WHAT DECIDES THE THINNING IS PANEL WIDTH IN INCHES, not `mag` alone: the same type in a
    # 1.25in panel and in a 2.75in one are different problems, and one row crowds even at
    # scale 1. `mag` still counts, because magnified type crowds a wide panel too.
    pw = (W * (1 / (len(probs) + 1.35))) / max(mag, 1e-6)   # a TARP panel, the narrow kind
    dense = pw > 1.8
    tf = 1.0 if dense else 0.92                  # a common trim for every label in the figure

    # every KPS-H size anywhere in the figure, so a shade means one N across all panels
    ns = sorted({int(l.split("N=")[-1]) for _, l, _ in curves if "KPS-G" not in l})
    for a, prob in zip(ax[:len(probs)], probs):
        a.plot([0, 1], [0, 1], color="#b0b0aa", ls=(0, (4, 3)), lw=1)
        m = None
        for p, lab, (al, cov, fs) in sorted(
                (c for c in curves if c[0] == prob),
                key=lambda c: (-1, 0) if "KPS-G" in c[1] else (1, int(c[1].split("N=")[-1]))):
            m = len(fs)
            a.plot(al, cov, lw=1.4, color=_tarp_color(lab, ns), label=lab)
        if m:
            al = np.linspace(0, 1, 51)
            s = np.sqrt(al * (1 - al) / m)
            a.fill_between(al, al - s, al + s, color="#c9c9c4", alpha=.35, lw=0, zorder=0)
        a.set_title(PLAIN[prob], loc="left", fontsize=8 * scale * tf)
        a.set_xlabel("credibility level" if dense else "credibility",
                     fontsize=8 * scale * tf)
        a.set_xlim(0, 1); a.set_ylim(0, 1)
        # Pinned, not left to the locator, which drops to {0, 1} in a narrow panel. Six
        # labels overprint into "0.00.20.40.60.81.0" whenever the panel is small relative to
        # the type. In ONE ROW the panels are square AND small, so both axes drop together --
        # in the old 2 x 2 the y axis had room to spare and kept its full 0.2 grid.
        # X and Y are now limited by DIFFERENT things: width is tight (1.25in, so six labels
        # overprint) while height is not (the panels are ~1.8in tall), so the y axis keeps its
        # full 0.2 grid and only x thins. Under the old square aspect both were width-limited.
        a.set_xticks(np.arange(0, 1.01, 0.2) if dense else np.array([0.0, 0.5, 1.0]))
        a.set_yticks(np.arange(0, 1.01, 0.2))
        if not dense:
            a.tick_params(labelsize=7.5 * scale * tf, pad=1.5, length=2)
    ax[0].set_ylabel("TARP", fontsize=8 * scale * tf)

    a = ax[len(probs)]
    a.axhline(2 / 3, color="#4a4a45", ls=(0, (4, 3)), lw=1.1)
    a.axhline(0.5, color="#b0b0aa", ls=(0, (2, 3)), lw=1)
    xs, labs = [], []
    for i, (prob, lab, sc, sd, n) in enumerate(mira_res):
        band = np.sqrt(1.0 / (18.0 * n))          # expected spread at L ground truths
        a.fill_between([i - .42, i + .42], 2 / 3 - band, 2 / 3 + band,
                       color="#c9c9c4", alpha=.45, lw=0, zorder=0)
        a.errorbar(i, sc, sd, marker="o", ms=4, capsize=2.5, lw=1.3,
                   color=C["KPSG"] if "KPS-G" in lab else C["KPSH"])
        xs.append(i)
        abbr = {"blackhole": "BH", "inv-scatter": "IS", "navier-stokes": "NS"}[prob]
        labs.append(f"{abbr} {lab.replace('KPS-', '').replace(' N=', '-')}")
    a.set_xticks(xs)
    a.set_xticklabels(labs, fontsize=6.5 * scale * tf, rotation=45, ha="right")
    a.set_xlim(-0.6, len(mira_res) - 0.4)
    a.set_yticks([0.5, 0.55, 0.6, 2 / 3])        # height is ample now, so keep all four
    a.set_yticklabels(["0.50", "0.55", "0.60", "2/3"])
    if not dense:
        a.tick_params(axis="y", labelsize=7.5 * scale * tf, pad=1.5, length=2)
    # the metric is named ONCE, on the y axis, exactly as TARP is on the first panel. It was
    # on the title as well, which said the same word twice in a 1.7in panel.
    a.set_ylim(0.45, 0.72); a.set_ylabel("MIRA", fontsize=8 * scale * tf)
    for extra in ax[len(probs) + 1:]:
        extra.axis("off")

    # ONE KEY, CENTRED, OUTSIDE THE AXES. A per-panel legend covered the curves it described,
    # and spelling out "KPS-H N=" four times said the same thing four times. Only the first
    # blue entry carries the arm name; the rest are bare sizes, so the row reads left to right
    # as one arm then a ladder, and the ramp explains itself without being narrated.
    line = mlines.Line2D
    handles = ([line([], [], color=C["KPSG"], lw=1.6)]
               + [line([], [], color=_tarp_color(f"N={n}", ns), lw=1.6) for n in ns])
    # Arm names come from the CURVES, not from literals, so a different campaign (see
    # paper_assets_cv.py) labels itself. Identical output for the ladder, where the two arms
    # are exactly KPS-G and KPS-H.
    _g = next((l for _, l, _ in curves if "KPS-G" in l), "KPS-G N=16")
    _h = next((l for _, l, _ in curves if "KPS-G" not in l), "KPS-H N=64")
    labels = [f"{_g.split(' N=')[0]}  $N$={_g.split('N=')[-1]}",
              f"{_h.split(' N=')[0]}  $N$={ns[0]}"] + [str(n) for n in ns[1:]]
    # UNDER THE TARP PANELS, NOT THE WHOLE FIGURE. The key describes the curves only; centred
    # on the figure it drifts under MIRA, which has no curves in it and its own colour meaning.
    # The anchor is read off the laid-out axes rather than guessed, because panel 1 carries the
    # y label and the TARP block is therefore not centred on the first three quarters.
    eng = fig.get_layout_engine()
    if eng is not None:
        eng.set(rect=(0, 0.10, 1, 0.90))          # reserve the strip the legend will sit in
    fig.canvas.draw()
    b0, b1 = ax[0].get_position(), ax[len(probs) - 1].get_position()
    fig.legend(handles, labels, loc="lower center", ncol=len(handles),
               bbox_to_anchor=(0.5 * (b0.x0 + b1.x1), 0.0), bbox_transform=fig.transFigure,
               frameon=False, fontsize=6.5 * scale * tf, handlelength=1.6,
               columnspacing=1.2 * tf, handletextpad=0.5, borderpad=0.1)
    if out:
        fig.savefig(out, bbox_inches="tight")
    style()          # `scale` must not leak into whatever figure is drawn next
    return fig


def fig_gallery_instances(problem, algo, run, ids=(0, 1, 2, 3), k=4, out=None):
    r"""APPENDIX: several instances, several members each. One row per instance.

    Individual members, not the mean: a mean cannot show that two instances fail in different
    ways, and it cannot show multimodality at all.
    """

    import matplotlib.pyplot as plt
    style()
    rows = []
    for i in ids:
        try:
            x, t, _ = load(problem, algo, run, i)
        except Exception:
            continue
        sel = np.linspace(0, x.shape[0] - 1, k).astype(int)
        rows.append((i, t[0], x.mean(0), x[sel]))
    if not rows:
        return None
    allv = np.concatenate([np.concatenate([r[1].numpy()[None], r[3].numpy()]) for r in rows])
    fs = _scale(allv, problem)

    fig, ax = plt.subplots(len(rows), k + 2, figsize=(1.15 * (k + 2), 1.2 * len(rows)),
                           squeeze=False)
    for r, (i, t, m, mem) in enumerate(rows):
        _panel(ax[r][0], t, "truth" if r == 0 else "", **fs)
        _panel(ax[r][1], m, "mean" if r == 0 else "", **fs)
        for j in range(k):
            _panel(ax[r][j + 2], mem[j], "posterior samples" if (r == 0 and j == 0) else "",
                   **fs)
        ax[r][0].set_ylabel(f"id {i}", fontsize=6.5)
        ax[r][0].yaxis.set_visible(True); ax[r][0].set_yticks([])
    fig.suptitle(f"{PLAIN[problem]} — {LBL[algo]} $N$={run.split('_N')[-1] if '_N' in run else 16}",
                 fontsize=8, x=0.005, ha="left")
    if out:
        fig.savefig(out, bbox_inches="tight")
    return fig


# ------------------------------------------------- posterior predictive (y space)

def ppc_table(df: pd.DataFrame) -> str:
    r"""The posterior predictive check: do the ensembles reproduce the OBSERVATION, with an
    honest spread? Read against the state-space table, an arm calibrated here and collapsed
    there has lost width only along directions the data does not constrain.

    Mean and sd, not medians: the y-space metrics are far better behaved than the x-space ones
    (no divergences survive the filter here) so a mean describes them, and the sd is what the
    reader needs to judge the gap.
    """

    COLS = [("y_member_l2", TEXHEAD["y_member_l2"], 4),
            ("y_crps_n", TEXHEAD["y_crps_n"], 4),
            ("y_ssr", TEXHEAD["y_ssr"], 3),
            ("y_cover90", TEXHEAD["y_cover90"], 3)]
    blocks = []
    for prob in ("blackhole", "inv-scatter", "navier-stokes"):
        d = df[(df.problem == prob) & (df.x_mean_l2 <= DIVERGED)]
        if d.empty:
            continue
        g = d.groupby(["algo", "n"])
        rows = []
        for k in sorted(g.groups):
            gg = g.get_group(k)
            name = "KPS-G" if k[0] == "KPSG" else "KPS-H"
            cells = [_pm(gg[c].mean(), gg[c].std(), nd) for c, _, nd in COLS]
            rows.append(f"{name} & {int(k[1])} & " + " & ".join(cells) + r" \\")
        blocks.append((NICE[prob], rows))
    return _group_table("ll" + "c" * len(COLS),
                        r"Method & $N$ & " + " & ".join(h for _, h, _ in COLS), blocks)


def fig_ppc(df: pd.DataFrame, out=None):
    r"""Posterior predictive check: spread / error in OBSERVATION space against ensemble size.

    Observation space only. The state-space counterpart is already in `fig_particles`, and
    overlaying the two doubled the number of curves for a comparison the reader can make
    between figures.
    """

    import matplotlib.pyplot as plt
    style()
    probs = [p for p in ("blackhole", "inv-scatter", "navier-stokes") if (df.problem == p).any()]
    fig, ax = plt.subplots(1, len(probs),
                           figsize=(FIGW, 0.95 * FIGW / len(probs)), squeeze=False)
    rng = np.random.default_rng(0)
    for a, prob in zip(ax[0], probs):
        d = df[(df.problem == prob) & (df.x_mean_l2 <= DIVERGED)]
        a.axhline(1.0, color="#b0b0aa", ls=(0, (4, 3)), lw=1)
        _series(a, d, "y_ssr", rng)
        a.set_xscale("log", base=2); a.set_xlabel("ensemble size $N$")
        a.set_title(PLAIN[prob], loc="left", fontsize=8)
        a.legend(fontsize=6.5, loc="best")
    ax[0][0].set_ylabel(NAME["y_ssr"])
    if out:
        fig.savefig(out, bbox_inches="tight")
    return fig


def ppc_table_median(df: pd.DataFrame) -> str:
    r"""Median and IQR of the observation-space metrics.

    Needed for blackhole KPS-H at N = 256 and 512, where the mean-and-sd table shows an sd
    LARGER than its mean (0.116 +/- 0.516): a few instances that survive the state-space
    divergence filter still have wild observation-space errors, so the mean describes them
    rather than the method.
    """

    COLS = [("y_member_l2", TEXHEAD["y_member_l2"], 4),
            ("y_crps_n", TEXHEAD["y_crps_n"], 4),
            ("y_ssr", TEXHEAD["y_ssr"], 3),
            ("y_cover90", TEXHEAD["y_cover90"], 3)]
    blocks = []
    for prob in ("blackhole", "inv-scatter", "navier-stokes"):
        d = df[(df.problem == prob) & (df.x_mean_l2 <= DIVERGED)]
        if d.empty:
            continue
        g = d.groupby(["algo", "n"])
        rows = []
        for k in sorted(g.groups):
            gg = g.get_group(k)
            name = "KPS-G" if k[0] == "KPSG" else "KPS-H"
            cells = []
            for c, _, nd in COLS:
                q1, med, q3 = gg[c].quantile([.25, .5, .75])
                cells.append(rf"\shortstack{{{med:.{nd}f}\\"
                             rf"{{\tiny[{q1:.{nd}f},\,{q3:.{nd}f}]}}}}")
            lab = rf"\shortstack[l]{{{name}\\\strut}}"
            num = rf"\shortstack{{{int(k[1])}\\\strut}}"
            rows.append(f"{lab} & {num} & " + " & ".join(cells) + r" \\")
        blocks.append((NICE[prob], rows))
    return _group_table("ll" + "c" * len(COLS),
                        r"Method & $N$ & " + " & ".join(h for _, h, _ in COLS), blocks,
                        size=r"\footnotesize")


def fig_growth(problem, algo="KPSH", runs=("gen_N64", "gen_N128", "gen_N256", "gen_N512"),
               ids=(0, 1, 2), out=None):
    r"""APPENDIX: what more particles buy, over several instances.

    Two rows per instance -- the posterior mean and the per-pixel standard deviation -- across
    ensemble sizes. The sd panels share ONE colour scale across all N and all instances shown,
    which is the point: scaling each panel to itself would make every ensemble look equally
    uncertain and hide that the spread grows with N. Several instances because one cannot show
    whether the behaviour is general.
    """

    import matplotlib.pyplot as plt
    style()
    data = []
    for i in ids:
        ens = []
        for r in runs:
            try:
                x, t, _ = load(problem, algo, r, i)
            except Exception:
                continue
            v = x.reshape(x.shape[0], -1).numpy()
            if not np.isfinite(v).all() or v.std() > 50:
                ens.append((r, None, None))          # a diverged arm is left blank, not hidden
            else:
                ens.append((r, x.mean(0), x.std(0)))
        if ens:
            data.append((i, t[0], ens))
    if not data:
        return None

    fields = np.concatenate([np.stack([d[1].numpy()] +
                                      [m.numpy() for _, m, _ in d[2] if m is not None])
                             for d in data])
    sds = np.concatenate([np.stack([s.numpy() for _, _, s in d[2] if s is not None])
                          for d in data])
    fs, ss = _scale(fields, problem), _scale(sds, problem, "mag")

    nc = len(runs) + 1
    fig, ax = plt.subplots(2 * len(data), nc, figsize=(1.22 * nc, 1.25 * 2 * len(data)),
                           squeeze=False)
    for b, (i, truth, ens) in enumerate(data):
        rm, rs = 2 * b, 2 * b + 1
        _panel(ax[rm][0], truth, "truth" if b == 0 else "", **fs)
        ax[rs][0].axis("off")
        for j, (r, m, sdev) in enumerate(ens):
            N = r.split("_N")[-1]
            if m is None:
                ax[rm][j + 1].axis("off"); ax[rs][j + 1].axis("off")
                ax[rm][j + 1].text(.5, .5, "diverged", ha="center", va="center", fontsize=6,
                                   color="#8a8a84")
                continue
            _panel(ax[rm][j + 1], m, f"$N={N}$" if b == 0 else "", **fs)
            _panel(ax[rs][j + 1], sdev, "", **ss)
        for a, lab in ((ax[rm][0], "mean"), (ax[rs][1], "sd")):
            a.set_ylabel(lab, fontsize=6.5)
            a.yaxis.set_visible(True); a.set_yticks([])
    fig.suptitle(f"{PLAIN[problem]} — {LBL[algo]}, three test instances", fontsize=8,
                 x=0.005, ha="left")
    if out:
        fig.savefig(out, bbox_inches="tight")
    return fig


_FWD = {}


def blackhole_op():
    r"""The blackhole forward model, instantiated once. Needed only to decompress y and to
    push the ground truth through the measurement, which the stored results do not contain."""

    if "blackhole" not in _FWD:
        from hydra import compose, initialize_config_dir
        from hydra.utils import instantiate
        with initialize_config_dir(version_base="1.3", config_dir=str(ROOT / "configs")):
            cfg = compose(config_name="config",
                          overrides=["problem=blackhole", "pretrain=blackhole"])
        _FWD["blackhole"] = instantiate(cfg.problem.model, device="cpu")
    return _FWD["blackhole"]


def fig_multimodal_row(idx=68, run="gen_N512", out=None, seed=0, show_cp=False):
    r"""MAIN TEXT: two modes, and the observation each one predicts, in a single row.

        truth | y(truth) | mode A | y(mode A) | mode B | y(mode B)

    The claim is that the modes are alternative explanations of the SAME data, so the y panels
    have to show it rather than be asserted. They plot predicted against measured visibility
    AMPLITUDE on log axes: points on the diagonal mean the data is reproduced, and the title
    gives the relative RMSE. The truth's own panel is the reference -- the measurement is
    stochastic, so even the ground truth does not reproduce y exactly, and a mode that matches
    the truth's RMSE is fitting as well as the truth does.

    THREE THINGS THAT HAD TO BE GOT RIGHT, each of which produced a wrong figure first:

    * Amplitude, not closure phase. Closure phase is an ANGLE, so a plain RMSE is meaningless
      -- a phase near +pi and one near -pi are physically close but differ by 2pi -- and it
      scored the TRUTH at 2.28 against a mode's 0.92. The problem's own chi2_cphase handles
      the wrap, but its scatter is a noisy cloud at chi2 ~ 2-5; amplitude is positive, needs
      no wrap, and reads as a clean diagonal.
    * The saved target is ALREADY unnormalised (main.py stores unnormalize(target)) while
      `recon_obs` came from the raw, still-normalised recon and `__call__` unnormalises what
      it is given. Feeding the saved target straight in unnormalises twice and put the truth's
      chi2 at 116, worse than either mode.
    * The MEDOID of each mode, not the extreme member along PC1. The extremes are the most
      atypical members of each cluster.

    INSTANCE 68 IS PINNED, AND THE CHOICE IS THE POINT. Scanning the bimodal instances for
    ones whose two modes BOTH explain the data (median chi2_cp per mode against the truth's):

        id   bimodality   |A|/|B|   chi2 truth   chi2 A   chi2 B   ratio
        46         77.4   430/82          2.03     2.23    14.37    6.45
        68         77.3   154/358         2.12     5.11     3.80    1.35
        86         66.2   299/213         2.58     3.89    23.55    6.06
        38         11.3   277/235         2.22     2.71     3.20    1.18

    Instance 46 is the visually dramatic one -- a crescent that flips left to right -- but its
    minority mode fits SIX TIMES worse than the truth, so drawing it here would illustrate a
    claim that is false on that instance. 68 keeps the bimodality and has modes that agree
    with the data to within 15% of each other on amplitude (0.084 and 0.087 against the
    truth's 0.076). The trade is real: 68's modes are less visually distinct.
    """

    import matplotlib.pyplot as plt
    style()
    fwd = blackhole_op()
    f = EXP / "blackhole" / "KPSH" / run / f"result_{idx}.pt"
    d = torch.load(f, map_location="cpu", weights_only=False)
    x = d["recon"].float()[:, 0]
    truth = d["target"].float().reshape(64, 64)
    obs, robs = d["observation"].float(), d["recon_obs"].float()

    v = x.reshape(x.shape[0], -1).numpy().astype(np.float64)
    U, S, _ = np.linalg.svd(v - v.mean(0), full_matrices=False)
    z = U[:, 0] * S[0]

    def medoid(mask):
        i = np.flatnonzero(mask)
        c = v[i].mean(0)
        return int(i[np.argmin(((v[i] - c) ** 2).sum(1))])

    iA, iB = medoid(z < z.mean()), medoid(z >= z.mean())
    amp = lambda y: fwd.decompress(y.reshape(1, 1, -1, 1))[0].reshape(-1).numpy()
    am = amp(obs)
    rel = lambda y: float(np.sqrt(np.mean((amp(y) - am) ** 2)) / np.sqrt(np.mean(am ** 2)))
    # SEED THE MEASUREMENT. `fwd` draws fresh noise on every call, so the truth's reference
    # RMSE is a random number -- it read 0.076 on one run and 0.083 on the next, for the same
    # instance. Seeding makes the figure reproducible.
    torch.manual_seed(seed)
    y_truth = fwd({"target": fwd.normalize(truth.reshape(1, 1, 64, 64).double())}).float()

    fig, ax = plt.subplots(1, 6, figsize=(FIGW, FIGW / 6 + 0.35))
    fs = _scale(np.stack([truth.numpy(), x[iA].numpy(), x[iB].numpy()]), "blackhole")
    lo = max(np.percentile(am[am > 0], 1), 1e-4)
    hi = np.percentile(am, 99.9)

    def sca(a, pred, title, color):
        m = am > 0
        a.scatter(am[m], np.abs(pred)[m], s=0.6, color=color, alpha=.35, lw=0)
        a.plot([lo, hi], [lo, hi], color="#8a8a84", lw=0.7, ls=(0, (3, 3)))
        a.set_xscale("log"); a.set_yscale("log")
        a.set_xlim(lo, hi); a.set_ylim(lo, hi); a.set_aspect("equal")
        a.set_xticks([]); a.set_yticks([])
        a.tick_params(which="both", length=0)
        for sp in a.spines.values():
            sp.set_color("#c9c9c4")
        a.set_title(title, fontsize=6.5, pad=2)

    cpm, sig = fwd.decompress(obs.reshape(1, 1, -1, 1))[2], \
        fwd.decompress(obs.reshape(1, 1, -1, 1))[3]

    def lab(name, y):
        t = f"{name}\nRMSE {rel(y):.3f}"
        if show_cp:      # amplitude and closure phase can disagree -- see the docstring
            c = float(fwd.chi2_cphase_from_meas(
                fwd.decompress(y.reshape(1, 1, -1, 1))[2], cpm, sig).item())
            t += f", $\\chi^2_{{cp}}$ {c:.1f}"
        return t

    # BOTH the amplitude RMSE and chi2_cp are OBSERVATION-space scores -- chi2_cp comes from
    # closure phases, which are derived from y -- so both belong on the scatter panels. They
    # were briefly split across the image and scatter titles, which mislabelled chi2_cp as an
    # x-space quantity and made the titles collide at this panel width.
    _panel(ax[0], truth, r"truth $x^\star$", **fs)
    sca(ax[1], amp(y_truth), lab(r"$y(x^\star)$", y_truth), C["KPSH"])
    _panel(ax[2], x[iA], "mode A", **fs)
    sca(ax[3], amp(robs[iA]), lab("$y$(mode A)", robs[iA]), C["KPSH"])
    _panel(ax[4], x[iB], "mode B", **fs)
    sca(ax[5], amp(robs[iB]), lab("$y$(mode B)", robs[iB]), C["KPSH"])
    if out:
        fig.savefig(out, bbox_inches="tight")
    return fig


# ----------------------------------------------------------------- 9. the method figure


# A 2D MOCK WORLD for the method figure. Small enough that the prior's denoiser is closed
# form, so every step drawn in the figure is the REAL operation applied to a toy problem
# rather than a hand-drawn arrow: the draw is an exact sample of p(x | x_t) for this prior,
# and the analysis is the KPS update itself. The problem is invented; the operations are not.
_MU = np.array([[-1.12, -0.56], [1.08, -0.50]])       # two prior modes, CROSSING the band
#   THREE THINGS HAVE TO HOLD AT ONCE, AND THEY PULL AGAINST EACH OTHER.
#     (a) the band must CROSS the modes, so the posterior reads as prior-meets-likelihood;
#     (b) the analysis must visibly MOVE the cloud, which wants the modes off the band;
#     (c) the two modes must stay DISTINCT, which wants them far apart in x.
#   The gain is V / (V + s2 + sigma_y^2) with s2 the residual a linearisation leaves on a
#   curved h, ~ (2 c xbar sd_x)^2. Since V ~ sd_x^2, spread cancels out of the ratio and the
#   gain is stuck near 0.5 however wide the cloud is -- which is why early versions looked
#   static no matter what was tuned. Only c and xbar are free, and (c) spends xbar. So the
#   CURVATURE pays for it: at c = 0.26 the gain is ~0.67, which is enough to land the cloud on
#   the band from an offset of only ~1.2 sd_x -- close enough that the band still slices
#   through the upper half of each mode. c is still large enough to fan the slopes visibly in
#   the fork panels, which is the other thing it has to do.
_SD, _YOBS, _SIGY = 0.55, 0.42, 0.15
_CURV = 0.26


def _h(X):
    r"""The mock observation. CURVED, which is the whole reason a linearisation has to be
    chosen, and placed so that BOTH prior modes are consistent with y: the mock posterior is
    bimodal, which is the case a point estimate cannot represent."""
    return X[:, 1] + _CURV * X[:, 0] ** 2


def _prior_pdf(G):
    d = np.stack([np.exp(-((G - m) ** 2).sum(-1) / (2 * _SD ** 2)) for m in _MU])
    return d.mean(0) / (2 * np.pi * _SD ** 2)


def _draw(x_t, sigma, rng):
    r"""An EXACT sample of p(x | x_t) for a Gaussian-mixture prior: each component's posterior
    is Gaussian, and the mixture weights are the component evidences. This is the closed form
    of what the denoiser's ODE approximates on a real prior."""
    v = _SD ** 2 * sigma ** 2 / (_SD ** 2 + sigma ** 2)
    mus = np.stack([(_SD ** 2 * x_t + sigma ** 2 * m) / (_SD ** 2 + sigma ** 2) for m in _MU])
    lw = np.stack([-((x_t - m) ** 2).sum(-1) / (2 * (_SD ** 2 + sigma ** 2)) for m in _MU])
    w = np.exp(lw - lw.max(0)); w /= w.sum(0)
    pick = (rng.random(len(x_t)) > w[0]).astype(int)
    return mus[pick, np.arange(len(x_t))] + rng.normal(0, np.sqrt(v), x_t.shape), v


def _slr(X, Y):
    r"""The KPS-H slope: statistical linear regression of the simulator output on the cloud,
    with the residual variance it leaves behind. That residual IS the Sigma_y the update uses,
    which is why the figure draws it as a band and not just a line."""
    dX, dY = X - X.mean(0), Y - Y.mean()
    A = np.linalg.solve(dX.T @ dX, dX.T @ dY)
    return A, float(((dY - dX @ A) ** 2).mean())


def fig_method(out=None, scale=1.0, seed=5, n=60, height=2.45):
    r"""METHOD ILLUSTRATION on a 2D mock problem: the steps of one sweep, then the fork.

    LAID OUT FOR A PAPER: the four steps are a 2x2 block on the left and the two variants stack
    beside them, so the whole thing is 2.45in tall instead of 4in and drops into a column
    without a full-page figure. Panel names sit above, the distribution each step targets sits
    inside the panel, and the fork captions are inside too -- at this size a caption under
    every panel costs more height than the panels.

    The mock prior and observation are invented. The OPERATIONS are not: the draw is an exact
    sample of p(x | x_t) for a Gaussian-mixture prior, and the analysis is the KPS update with
    V_t the conditional covariance of that draw. It is still an ILLUSTRATION and no number in
    it is a measurement.

    THE KPS-G PANEL DRAWS THE AVERAGED SLOPE ON PURPOSE, AND THE CODE DOES NOT AVERAGE. A
    deliberate divergence, chosen by the author: the METHOD is formulated with one averaged
    slope A = mean_i grad h(x_i), which is what the panel shows (each particle's exact slope
    faint, the single map they average into bold), while the IMPLEMENTATION uses each
    particle's own Jacobian, which the paper text explains as an efficient approximation of
    that average. Verified: with h_i = x_i0^2 and particles at x0 = 1, 2, 5, A(e_0) returns
    2, 4, 10 -- per-particle, not their mean 5.33. So this figure is the formulation and
    `tab_sampler.tex` is the implementation; if you change one, say which you meant.
    """

    import matplotlib.patheffects as pe
    import matplotlib.pyplot as plt

    style()
    rng = np.random.default_rng(seed)
    INK, GREY, PALE, LINE = "#2b2b28", "#6f6f69", "#bdbdb7", "#4a4a45"
    halo = [pe.withStroke(linewidth=3.2, foreground="white")]
    band_k = 2.0                     # band drawn at 2 sigma_y; at 1 it is a hairline once
    XL = (-3.25, 3.25)               # sigma_y is small enough for the analysis to bite
    # FXL is chosen so the curve RISES ACROSS THE WHOLE PANEL at this aspect: too wide a
    # window and it flattens, too narrow and it runs off the top. At c = 0.26 and an aspect
    # near 1.9 these limits make the rise and the panel height match to a few percent.
    FXL, ycf = (-2.10, 0.30), -0.165

    fig = plt.figure(figsize=(FIGW, height))
    fig.set_layout_engine("none")    # style() enables constrained layout, which scrambles
    gs1 = fig.add_gridspec(2, 2, left=0.012, right=0.612, top=0.875,   # hand-placed grids
                           bottom=0.125, wspace=0.07, hspace=0.30)
    gs2 = fig.add_gridspec(2, 1, left=0.664, right=0.988, top=0.875,
                           bottom=0.125, hspace=0.18)   # no titles here, so less gap

    def panel(spec, yc=-0.30, band=True, xl=None):
        xl = xl or XL
        a = fig.add_subplot(spec)
        bb = a.get_position()
        yr = (xl[1] - xl[0]) * (bb.height * height) / (bb.width * FIGW)
        a.set_xlim(*xl); a.set_ylim(yc - yr / 2, yc + yr / 2)
        a.set_xticks([]); a.set_yticks([])
        for sp in a.spines.values():
            sp.set_visible(True); sp.set_color("#d8d8d3"); sp.set_linewidth(0.7)
        gx = np.linspace(*xl, 240)
        gy = np.linspace(*a.get_ylim(), 240)
        GX, GY = np.meshgrid(gx, gy)
        a.contour(GX, GY, _prior_pdf(np.stack([GX, GY], -1)), levels=4,
                  colors="#c6c6bf", linewidths=0.6, zorder=1)
        if band:
            # GREY, not blue: blue is KPS-H's colour, and in the fork panels a blue band is
            # already the Sigma_y it estimates. Two blue bands meaning different things in one
            # figure is worse than a neutral one here.
            a.fill_between(gx, _YOBS - band_k * _SIGY - _CURV * gx ** 2,
                           _YOBS + band_k * _SIGY - _CURV * gx ** 2,
                           color="#d9d9d2", alpha=0.95, lw=0, zorder=2)
            a.plot(gx, _YOBS - _CURV * gx ** 2, color=LINE, lw=0.9, zorder=3)
        return a

    def dots(a, P, col, s=5.0, z=8):
        a.scatter(*P.T, s=s * scale, color=col, lw=0, zorder=z)

    # ---------------- one sweep, as a 2x2 block
    sig0, sig1 = 1.15, 0.52
    x_t = _draw(rng.normal(0, 1, (n, 2)) * 2.0, 2.6, rng)[0] + rng.normal(0, sig0, (n, 2))
    xd, vcond = _draw(x_t, sig0, rng)                       # 1. prior draw
    Yd = _h(xd)                                             # 2. simulate
    A, s2 = _slr(xd, Yd)                                    # 3. linearise
    V = vcond * np.eye(2)
    K = (V @ A) / (A @ V @ A + s2 + _SIGY ** 2)
    z = xd + np.outer(_YOBS - Yd, K)                        # 4. the Kalman analysis
    x_t2 = z + rng.normal(0, sig1, z.shape)                 # 5. re-noise at the next level

    for k, (ttl, mth, src, dst) in enumerate((
            ("noisy cloud", r"$p(x_t)$", None, x_t),
            ("prior draw", r"$p(x \mid x_t)$", x_t, xd),
            ("Kalman update", r"$p(x \mid x_t,\, y)$", xd, z),
            ("re-noise", r"$p(x_{t'} \mid x)$", z, x_t2))):
        a = panel(gs1[k // 2, k % 2])
        if src is not None:
            dots(a, src, PALE, s=3.6, z=4)
        dots(a, dst, C["KPSH"])
        a.set_title(f"{k + 1}.  {ttl}", fontsize=6.2 * scale, color=INK, pad=2.5)
        a.text(0.035, 0.05, mth, transform=a.transAxes, ha="left", va="bottom",
               fontsize=5.8 * scale, color=GREY, zorder=9,
               bbox=dict(fc="white", ec="none", pad=1.0))

    # BOTH KEYS SIT OVER THE STEP BLOCK, not the figure. Centred on the figure the line spreads
    # until its tail lands above the fork panels, which it does not describe.
    xs = 0.5 * (0.012 + 0.612)
    fig.text(xs, 0.985, "contours: the prior        grey band: the states that fit the data",
             ha="center", va="top", fontsize=6.2 * scale, color=INK)
    fig.text(xs, 0.012, "1 to 4, then repeat with smaller $\\sigma$        "
             "blue: after the step, grey: before",
             ha="center", va="bottom", fontsize=5.8 * scale, color=GREY)

    # ---------------- the fork, beside the steps
    fx = np.linspace(-1.80, -0.25, 6)
    sub = np.stack([fx, _YOBS - _CURV * fx ** 2 + rng.normal(0, 0.11, 6)], 1)

    for j, (lab, col, cap) in enumerate((
            ("KPS-H", C["KPSH"], "SLR over the cloud: one fitted slope"),
            ("KPS-G", C["KPSG"], "every particle's exact slope, averaged"))):
        a = panel(gs2[j, 0], yc=ycf, band=False, xl=FXL)
        gx = np.linspace(*FXL, 240)
        a.plot(gx, _YOBS - _CURV * gx ** 2, color=LINE, lw=0.9, zorder=3)

        if j == 0:
            Asub, s2sub = _slr(sub, _h(sub))
            xb, yb = sub.mean(0), _h(sub).mean()
            ly = (_YOBS - yb + Asub @ xb - Asub[0] * gx) / Asub[1]
            hw = np.sqrt(s2sub + _SIGY ** 2) / abs(Asub[1])
            a.fill_between(gx, ly - hw, ly + hw, color=col, alpha=0.15, lw=0, zorder=3)
            a.plot(gx, ly, color=col, lw=1.7, zorder=6, path_effects=halo)
        else:
            for qx, qy in sub:                 # short sticks: a slope HERE, not a whole map
                d, g = 0.40, 2 * _CURV * qx
                a.plot([qx - d, qx + d], [qy + d * g, qy - d * g], color=col, lw=1.2,
                       alpha=0.60, zorder=4, solid_capstyle="round")
            Ag = np.array([2 * _CURV * sub[:, 0].mean(), 1.0])
            xb, yb = sub.mean(0), _h(sub).mean()
            s2g = float(((_h(sub) - (yb + (sub - xb) @ Ag)) ** 2).mean())
            lyg = (_YOBS - yb + Ag @ xb - Ag[0] * gx) / Ag[1]
            hwg = np.sqrt(s2g + _SIGY ** 2) / abs(Ag[1])
            a.fill_between(gx, lyg - hwg, lyg + hwg, color=col, alpha=0.15, lw=0, zorder=3)
            a.plot(gx, lyg, color=col, lw=1.7, zorder=6, path_effects=halo)

        dots(a, sub, col, s=9)
        a.text(0.025, 0.955, lab, transform=a.transAxes, ha="left", va="top",
               fontsize=6.6 * scale, color=col, fontweight="bold", zorder=9,
               bbox=dict(fc="white", ec="none", pad=1.0))
        a.text(0.025, 0.045, cap, transform=a.transAxes, ha="left", va="bottom",
               fontsize=5.8 * scale, color=INK, zorder=9,
               bbox=dict(fc="white", ec="none", pad=1.0))

    fig.text(0.826, 0.012, "both then take the same step", ha="center", va="bottom",
             fontsize=5.8 * scale, color=GREY)

    if out:
        fig.savefig(out, bbox_inches="tight")
    return fig


# ----------------------------------------------------------------- 10. the main-text table

# ONE headline metric per problem, named in that problem's block header. The three problems
# are scored on different quantities and there is no way to share a column across them; naming
# it per block costs one row and keeps the table four numeric columns wide instead of nine.
MAIN_HEAD = {
    "inv-scatter":   ("ib_psnr",              "ib_psnr",        "PSNR $\\uparrow$", +1, 1),
    "blackhole":     ("ib_blur_psnr (f=15)",  "blur",           "blur PSNR $\\uparrow$", +1, 1),
    "navier-stokes": ("ib_relative l2",       "ib_relative l2", "rel. $L_2$ $\\downarrow$", -1, 3),
}
MAIN_OURS = (("KPSG", 16), ("KPSH", 512))


def main_table(df, top=2, stat="mean"):
    r"""MAIN TEXT: wide and four rows, our two arms against the benchmark's best.

    PROBLEMS ARE COLUMN GROUPS, NOT ROW BLOCKS. As row blocks (three sections of five rows)
    this ran half a page; the same content transposed is four rows and about 400pt wide, which
    is the shape a main-text table wants. The cost is that a baseline row can no longer name
    one method -- the best method differs per problem -- so the rows are "best published" and
    "2nd published" and `main_caption` names them. That is the only thing the caption has to
    carry that the table cannot.

    TRIMMED ON PURPOSE for the main text: no N column (it rides in the row label), no standard
    deviations, no footnote block, no skill column. All four are in `tab_bench_<problem>.tex`
    and `tab_posterior.tex`. What survives is one benchmark metric per problem and the two
    posterior columns no published method reports -- the em-dashes in those columns are the
    argument the table exists to make.
    """

    def first(v):
        return v[0] if isinstance(v, (list, tuple)) else v

    def fmt(x, nd):
        return "--" if x is None or not np.isfinite(x) else f"{x:.{nd}f}"

    probs = list(MAIN_HEAD)
    bl, mine, best = {}, {}, {}
    for prob in probs:
        ours_col, bl_key, lab, sign, nd = MAIN_HEAD[prob]
        b = [(n, first(v.get(bl_key))) for n, v in BASELINES[prob] if v.get(bl_key) is not None]
        b.sort(key=lambda t: -sign * t[1])
        bl[prob] = b[:top]
        d = df[df.problem == prob]
        mine[prob] = {}
        for algo, N in MAIN_OURS:
            g = d[(d.algo == algo) & (d.n == N)]
            f = (lambda c: float(g[c].median())) if stat == "median" else \
                (lambda c: float(g[c].mean()))
            mine[prob][algo] = None if not len(g) else {
                "head": f(ours_col) if ours_col in g else None,
                "crps": f("x_crps_n"), "ssr": f("x_ssr")}
        cand = [v for _, v in bl[prob]] + [m["head"] for m in mine[prob].values()
                                           if m and m["head"] is not None]
        best[prob] = max(cand, key=lambda v: sign * v)

    # WIDTH IS THE BINDING CONSTRAINT AT TEN COLUMNS. Natural width is 519pt against a
    # 397pt column, and the two things that fix it are not the font: nine inter-column gaps
    # at the default 6pt tabcolsep cost 108pt on their own, and "spread-skill" set on one
    # line is the widest repeated token in the header. Halving the first and breaking the
    # second save ~130pt between them; \small then does the rest.
    H = [r"\small", r"\setlength{\tabcolsep}{2pt}",
         r"\begin{tabular}{@{}l" + r"@{\hspace{7pt}}rrr" * len(probs) + r"@{}}",
         r"\toprule",
         "& " + " & ".join(rf"\multicolumn{{3}}{{c}}{{{NICE[p]}}}" for p in probs) + r" \\",
         "".join(rf"\cmidrule(lr){{{2 + 3 * k}-{4 + 3 * k}}}" for k in range(len(probs)))]
    sub = []
    for p in probs:
        lab = MAIN_HEAD[p][2].replace(" $\\uparrow$", "").replace(" $\\downarrow$", "")
        arr = "$\\uparrow$" if MAIN_HEAD[p][3] > 0 else "$\\downarrow$"
        # the FIRST line of a shortstack sets the column width, so "blur PSNR" on one line
        # widens the black-hole group enough to collide with its neighbour: break at the space
        a, _, b = lab.partition(" ")
        top_l, bot_l = (a, f"{b} {arr}") if b else (lab, arr)
        sub += [rf"\shortstack{{{top_l}\\{bot_l}}}", r"\shortstack{CRPS\\$\downarrow$}",
                r"\shortstack{spread-\\skill $\to 1$}"]
    H.append("Method & " + " & ".join(sub) + r" \\")
    H.append(r"\midrule")

    def cell(prob, v):
        nd = MAIN_HEAD[prob][4]
        t = fmt(v, nd)
        return rf"\textbf{{{t}}}" if v is not None and v == best[prob] else t

    for r in range(top):
        name = "Best published" if r == 0 else f"{r + 1}nd published" if r == 1 \
            else f"{r + 1}th published"
        cs = []
        for p in probs:
            cs += [cell(p, bl[p][r][1]) if r < len(bl[p]) else "--", "--", "--"]
        H.append(f"{name} & " + " & ".join(cs) + r" \\")

    H.append(r"\midrule")
    for algo, N in MAIN_OURS:
        cs = []
        for p in probs:
            m = mine[p][algo]
            cs += ["--", "--", "--"] if m is None else \
                [cell(p, m["head"]), fmt(m["crps"], 3), fmt(m["ssr"], 2)]
        H.append(rf"{LBL[algo]} ($N{{=}}{N}$) & " + " & ".join(cs) + r" \\")

    H += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(H)


def main_caption(df, top=2):
    r"""The caption for `main_table`, carrying what the trimmed table cannot: which methods the
    baseline rows are, that the black-hole measurement model is ours and not the benchmark's,
    and that EnKG's Navier-Stokes number is bought with about 10x the simulator calls."""

    def first(v):
        return v[0] if isinstance(v, (list, tuple)) else v

    named = []
    for prob in MAIN_HEAD:
        _, bl_key, _, sign, _ = MAIN_HEAD[prob]
        b = [(n, first(v.get(bl_key))) for n, v in BASELINES[prob] if v.get(bl_key) is not None]
        b.sort(key=lambda t: -sign * t[1])
        ms = [n for n, _ in b[:top]]
        lead = " and ".join([", ".join(ms[:-1]), ms[-1]] if len(ms) > 2 else ms)
        low = NICE[prob] if prob == "navier-stokes" else NICE[prob][0].lower() + NICE[prob][1:]
        named.append(f"{lead} ({low})")
    return (
        r"\caption{KPS against the best published methods on three InverseBench problems. "
        r"For each problem we report its benchmark metric, then fair CRPS and the spread-skill "
        r"ratio (ensemble spread over posterior-mean error; $1$ when calibrated), neither of "
        r"which any published method reports. Baseline rows are " + "; ".join(named) + r". "
        r"KPS-G runs at $N{=}16$ and KPS-H at $N{=}512$, against EnKG's $N{=}2048$; our "
        r"black-hole noise model differs from the benchmark's. Means over 100 instances (10 "
        r"for Navier--Stokes). For a more thorough comparison, see "
        r"Appendix~\ref{app:full}.}")
