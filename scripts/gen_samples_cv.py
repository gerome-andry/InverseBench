#!/usr/bin/env python3
r"""Generate the ridge-SLR sample sets, as one throttled Slurm array.

Sibling of `gen_samples.py`, same shape, same discipline, different arms. It writes its own
manifest and array script so the original campaign's provenance is untouched.

    python3 scripts/gen_samples_cv.py --probe       # measure the rates (a few minutes, 1 GPU)
    python3 scripts/gen_samples_cv.py               # plan: the task table and the estimate
    python3 scripts/gen_samples_cv.py --write       # write the manifest + array script
    sbatch scripts/samples_cv/array.sbatch          # YOU submit this (the script never does)

WHAT IT COVERS, mirroring the kpsh/kpsg campaign so the comparison is paired arm for arm:

    KPSCV  navier-stokes, inv-scatter, blackhole     N in {64, 128, 256, 512}
    KPSGF  inv-scatter, blackhole                    N from configs/algorithm/kpsgf.yaml

Navier-Stokes is H-only for the same reason as before: G needs the simulator's Jacobian, and on
NS no linear map represents the dynamics.

WHY --probe IS NOT OPTIONAL. `gen_samples.py` records rates carried over from older runs being
wrong by 1.8x / 3.0x / 1.7x IN BOTH DIRECTIONS, and says to re-probe after any change to
draw_steps, rank or solve_iter. These arms changed more than that: `kpscv` replaced the rank-32
direct solve with a matrix-free Krylov one, so its cost no longer scales with the rank at all,
and `kpsgf` adds one jvp per sweep to G. Measured on blackhole, kpscv runs at 0.0137 s per
(particle x sweep) against kpsh's 0.0303 -- 2.2x faster -- but that ratio is NOT transferable:
the saving depends on k = N - 1 against rank 32, on the QR of a D_y x (k + N) matrix, and D_y
is 2004 on blackhole, 4096 on NS and 14400 on inverse scattering. So blackhole is pre-filled
from real runs and everything else must be probed. A wrong rate mis-sizes the tasks and times
them out; it is not a cosmetic error.

IT IS IDEMPOTENT, like its sibling. A task re-filters its own id list against the
`result_<id>.pt` files on disk immediately before running, so a timed-out or half-finished
array can be resubmitted unchanged. Note `KPSCV/gen_cv_N64` may already hold ids 0-31 from an
interrupted run; those are skipped automatically.
"""

import argparse
import json
import os
import subprocess
import sys
import time

from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

from gen_samples import STARTUP, compact, hms, parse_ids      # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "scripts" / "samples_cv"
MANIFEST = OUT / "manifest.tsv"
SBATCH = OUT / "array.sbatch"
RATEFILE = OUT / "rates.json"

LOGROOT = Path("/mnt/home/gandry/ceph/ibench/ibench_exp/_gen_cv")

PARTICLES = (64, 128, 256, 512)
DRAW_BATCH = 64                    # held fixed across N, exactly as in the original campaign
SOLVE_BATCH = 64                   # likewise, and for the same reason -- see below

# BOTH BATCHES ARE PART OF THE CONFIGURATION, NOT FREE OPTIMISATIONS. Chunking either the
# prior-factor draw or the Krylov solve over particles is exact in exact arithmetic -- the
# per-particle systems are independent given the cloud statistics -- but NOT in floating point:
# cuDNN picks different kernels per batch shape. Measured on one analysis call, a chunked solve
# differs from an unchunked one by 1.9e-4 relative and the figure does NOT grow as the chunk
# shrinks (64 / 32 / 16 give 1.93e-4 / 2.05e-4 / 2.01e-4), so it is a constant kernel offset
# and not an accumulating error. Over the chaotic ladder it amplifies: a 4-level chain at
# solve_batch 128 against 32 ends up 4.0e-2 apart, matching the 2.4e-4 -> 2e-2 that
# `gen_samples.py` records for draw_batch. So both are pinned for EVERY N, or the particle
# sweep would confound N with a change of kernel realisation.

GRID = (
    ("navier-stokes", "kpscv"),
    ("inv-scatter", "kpscv"),
    ("blackhole", "kpscv"),
    ("inv-scatter", "kpsgf"),
    ("blackhole", "kpsgf"),
)

ALGO_DIR = {"kpscv": "KPSCV", "kpsgf": "KPSGF"}
SWEEP_ALGO = "kpscv"               # the arm that gets the PARTICLES sweep; read by progress.py
JOB_NAME = "kps-gen-cv"            # matches array.sbatch, so progress.py can scope squeue

# Seconds per (particle x sweep). ONLY entries measured on this code are listed; the rest come
# from --probe. blackhole/kpscv is from four production runs at N = 16/32/64/128 (21/31/56/112 s
# for 64 sweeps), whose implied rate is flat at 0.0137 from N = 64 up.
RATE_SEED = {
    ("blackhole", "kpscv"): 0.0137,
}

# --assume: size the tasks from the ORIGINAL campaign's measured rates, scaled by the one ratio
# that has been measured on this code. It is an estimate and is labelled as one everywhere it
# is used, but the cost of it being wrong is bounded: a mis-sized task only wastes wall time,
# and `--requeue` plus the per-task id re-filter means an oversized task that times out is
# resubmitted and picks up exactly where it stopped. Nothing is ever recomputed.
#
#   kpscv   0.0137 / 0.0303 = 0.45 of kpsh, measured on blackhole at N = 64..128. The saving is
#           the matrix-free solve, whose cost no longer scales with the rank; how much of it
#           survives at D_y = 4096 (NS) and 14400 (inv-scatter) is exactly what --probe settles.
#   kpsgf   6.5 of kpsg with slope='mean' (the default since 2026-09-25). The averaged Jacobian
#           needs every particle's J applied to the SAME vector, so one solver application is N
#           jvps and N vjps instead of one each. MEASURED 6.1x the per-particle arm on blackhole
#           id 2 (116 s vs 19 s at N = 16), times the 1.06 that arm cost over kpsg. It is a
#           BLACKHOLE ratio: inv-scatter has D_y = 14400 and a heavier forward, so --probe is
#           worth more here than anywhere else in this file. With slope='particle' use 1.06.
ASSUME = {"kpscv": ("kpsh", 0.45), "kpsgf": ("kpsg", 6.5)}


def load_cfg(problem, algo):
    p = yaml.safe_load((ROOT / "configs" / "problem" / f"{problem}.yaml").read_text())
    a = yaml.safe_load((ROOT / "configs" / "algorithm" / f"{algo}.yaml").read_text())["method"]

    return p, a


def exp_name(algo, n):
    r"""Run directory. G's carries the SLOPE, because 'mean' and 'particle' are different
    estimators and `remaining()` skips ids by filename -- one shared directory would let a
    re-plan silently mix them, or silently do nothing. `gen_gf` holds the per-particle runs of
    2026-09-23; `gen_gfm` is the averaged Jacobian."""

    if algo == "kpscv":
        return f"gen_cv_N{n}"

    _, a = load_cfg("blackhole", "kpsgf")            # the slope is not problem-dependent

    return "gen_gfm" if a.get("slope", "particle") == "mean" else "gen_gf"


def exp_dir(problem, algo, n):
    p, _ = load_cfg(problem, algo)

    return Path(p["exp_dir"]) / ALGO_DIR[algo] / exp_name(algo, n)


def remaining(problem, algo, n, ids):
    d = exp_dir(problem, algo, n)

    return [i for i in ids if not (d / f"result_{i}.pt").exists()]


def rates(assume=False):
    r"""Measured rates: the seed table, overridden by anything --probe has written.

    `assume=True` fills the gaps from the original campaign, scaled -- see ASSUME. The returned
    dict is paired with a set naming which entries are estimates, so every caller can say so.
    """

    out = dict(RATE_SEED)
    if RATEFILE.exists():
        out.update({tuple(k.split("|")): v for k, v in json.loads(RATEFILE.read_text()).items()})

    guessed = set()
    if assume:
        import gen_samples as orig

        for problem, algo in GRID:
            if (problem, algo) in out:
                continue
            base, ratio = ASSUME[algo]
            if (problem, base) in orig.RATE:
                out[(problem, algo)] = orig.RATE[(problem, base)] * ratio
                guessed.add((problem, algo))

    return out, guessed


def probe(only=None):
    r"""Time a levels=4 / sweeps=1 run of one id at each arm's top N, and cache the rate.

    Same probe the original campaign used. The startup cost is subtracted before dividing, so
    what is cached is the marginal cost of one (particle x sweep) and nothing else.
    """

    OUT.mkdir(parents=True, exist_ok=True)
    have = {tuple(k.split("|")): v
            for k, v in (json.loads(RATEFILE.read_text()) if RATEFILE.exists() else {}).items()}

    for problem, algo in GRID:
        if only and only not in (problem, algo):
            continue
        if (problem, algo) in have or (problem, algo) in RATE_SEED:
            print(f"{problem}/{algo}: already have a rate, skipping")
            continue

        p, a = load_cfg(problem, algo)
        n = max(PARTICLES) if algo == "kpscv" else int(a["num_particles"])
        first = parse_ids(p["data"]["id_list"])[0]

        cmd = [sys.executable, "main.py", f"problem={problem}", f"pretrain={problem}",
               f"algorithm={algo}", f'++problem.data.id_list="{first}-{first}"',
               "++algorithm.method.levels=4", "++algorithm.method.sweeps=1",
               "++algorithm.method.progress=False", "++exp_name=_probe",
               f"++hydra.run.dir={LOGROOT}/hydra/_probe_{problem}_{algo}"]
        if algo == "kpscv":
            cmd += [f"++algorithm.method.num_particles={n}",
                    f"++algorithm.method.draw_batch={DRAW_BATCH}",
                    f"++algorithm.method.solve_batch={SOLVE_BATCH}"]

        print(f"\nprobing {problem}/{algo} at N={n} (levels=4, sweeps=1)...", flush=True)
        t0 = time.time()
        rc = subprocess.call(cmd, cwd=ROOT, env={**os.environ, "PYTHONUNBUFFERED": "1"})
        wall = time.time() - t0

        if rc != 0:
            print(f"  FAILED (rc {rc}) -- not caching a rate for {problem}/{algo}")
            continue

        rate = max(wall - STARTUP[problem], 1.0) / (4 * n)
        have[(problem, algo)] = rate
        print(f"  {wall:.0f} s wall, {rate:.4f} s per (particle x sweep)")

    RATEFILE.write_text(json.dumps({"|".join(k): v for k, v in have.items()}, indent=2))
    print(f"\nwrote {RATEFILE}")


# The arm each new run is paired against, for the ladder-parameter check below.
BASELINE = {"kpscv": ("KPSH", lambda n: f"gen_N{n}"), "kpsgf": ("KPSG", lambda n: "gen")}

# Everything that defines the sampling schedule. `rank` is deliberately absent from kpscv and
# `cv_folds`/`mf_iter` are deliberately new, so those are reported as the INTENDED delta rather
# than flagged; anything else differing is drift and is flagged loudly.
LADDER = ("levels", "sweeps", "draw_steps", "sigma_max", "sigma_min", "solve_iter", "mode")
INTENDED = {"rank", "cv_folds", "mf_iter", "sigma_mode", "num_particles", "draw_batch",
            "progress", "_target_"}


def check_ladder():
    r"""Assert the new arms sample on exactly the ladder the baselines did.

    `KPS/report/KPSTEST_PROMPT.md`: read the STORED config.yaml of a run before quoting its
    numbers, not the config file as it reads today. The same applies before generating
    something meant to be compared against it -- a silent drift in levels, sweeps or draw_steps
    would make the whole campaign incomparable, and would not be visible in any result file.
    """

    bad = []

    for problem, algo in GRID:
        p, cur = load_cfg(problem, algo)
        sub, name = BASELINE[algo]
        n0 = PARTICLES[0] if algo == "kpscv" else int(cur["num_particles"])
        cfg = Path(p["exp_dir"]) / sub / name(n0) / "config.yaml"

        if not cfg.exists():
            print(f"  {problem}/{algo}: no baseline at {cfg.parent.name}, cannot check")
            continue

        old = yaml.safe_load(cfg.read_text())["algorithm"]["method"]
        diff = [(k, old.get(k), cur.get(k)) for k in LADDER
                if k in old and k in cur and old[k] != cur[k]]

        if diff:
            bad.append((problem, algo, diff))
        else:
            print(f"  {problem}/{algo}: ladder matches {sub}/{name(n0)} "
                  f"(levels {old['levels']}, sweeps {old['sweeps']}, "
                  f"draw_steps {old['draw_steps']}, sigma {old['sigma_max']}-{old['sigma_min']})")

    for problem, algo, diff in bad:
        print(f"\n  *** {problem}/{algo} DIFFERS FROM ITS BASELINE ON THE LADDER ***")
        for k, a, b in diff:
            print(f"        {k}: baseline {a}  ->  new {b}")

    return not bad


def plan(target, assume=False):
    tasks, skipped, missing = [], 0, []
    rate, guessed = rates(assume)

    for problem, algo in GRID:
        p, a = load_cfg(problem, algo)
        ids_all = parse_ids(p["data"]["id_list"])
        sweeps = int(a["levels"]) * int(a["sweeps"])
        ns = PARTICLES if algo == "kpscv" else (int(a["num_particles"]),)

        if (problem, algo) not in rate:
            missing.append(f"{problem}/{algo}")
            continue

        for n in ns:
            ids = remaining(problem, algo, n, ids_all)
            skipped += len(ids_all) - len(ids)
            if not ids:
                continue

            per_id = rate[(problem, algo)] * sweeps * n
            size = max(1, int(target // per_id))

            for k in range(0, len(ids), size):
                chunk = ids[k:k + size]
                tasks.append(dict(problem=problem, algo=algo, n=n, exp=exp_name(algo, n),
                                  ids=compact(chunk), n_ids=len(chunk),
                                  est=STARTUP[problem] + per_id * len(chunk), per_id=per_id,
                                  guessed=(problem, algo) in guessed))

    return tasks, skipped, missing, guessed


def report(tasks, skipped, missing, guessed, throttle):
    print(f"{'problem':<14}{'algo':<7}{'N':>5}{'ids':>5}{'tasks':>7}{'s/id':>9}"
          f"{'GPU-h':>9}{'per task':>10}")
    print("-" * 66)

    groups, total = {}, 0.0
    for t in tasks:
        g = groups.setdefault((t["problem"], t["algo"], t["n"]),
                              dict(tasks=0, ids=0, est=0.0, per_id=t["per_id"],
                                   guessed=t["guessed"]))
        g["tasks"] += 1
        g["ids"] += t["n_ids"]
        g["est"] += t["est"]

    for (problem, algo, n), g in groups.items():
        total += g["est"]
        print(f"{problem:<14}{algo:<7}{n:>5}{g['ids']:>5}{g['tasks']:>7}{g['per_id']:>9.0f}"
              f"{g['est'] / 3600:>9.1f}{hms(g['est'] / g['tasks']):>10}"
              f"{'  est' if g['guessed'] else ''}")

    longest = max((t["est"] for t in tasks), default=0.0)
    print("-" * 66)
    print(f"{'TOTAL':<26}{sum(t['n_ids'] for t in tasks):>5}{len(tasks):>7}{'':>9}"
          f"{total / 3600:>9.1f}")

    if skipped:
        print(f"\n{skipped} (arm, id) pairs already have samples on disk and were dropped.")
    if guessed:
        print(f"\n'est' marks {len(guessed)} arm(s) sized from the original campaign's rates "
              f"scaled by the\none ratio measured on this code (see ASSUME). Task sizes and the "
              f"GPU-h figure\nabove are therefore estimates; --probe replaces them with "
              f"measurements. A wrong\nsize only wastes wall time -- requeue plus the per-task "
              f"id re-filter makes an\noversized task resume rather than restart.")
    if missing:
        print(f"\nNO RATE for: {', '.join(missing)} -- these arms are NOT in the plan.")
        print("Run `--probe` first. Guessing a rate mis-sizes the tasks and times them out;")
        print("gen_samples.py records carried-over rates being wrong by 1.8-3.0x both ways.")

    print(f"\nlongest task {hms(longest)}; at throttle {throttle} the array finishes in about "
          f"{hms(max(total / throttle, longest))} of wall time.")

    return total, longest


def write(tasks, throttle, walltime, partition, constraint):
    cons = f'"{constraint}"' if "|" in constraint or "&" in constraint else constraint

    OUT.mkdir(parents=True, exist_ok=True)
    (LOGROOT / "logs").mkdir(parents=True, exist_ok=True)

    with MANIFEST.open("w") as f:
        for i, t in enumerate(tasks, 1):
            f.write(f"{i}\t{t['problem']}\t{t['algo']}\t{t['n']}\t{t['exp']}\t{t['ids']}\n")

    SBATCH.write_text(f"""#!/bin/bash
#SBATCH --job-name=kps-gen-cv
#SBATCH --partition={partition}
#SBATCH --gres=gpu:1
#SBATCH --constraint={cons}
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time={walltime}
#SBATCH --array=1-{len(tasks)}%{throttle}
#SBATCH --output={LOGROOT}/logs/%A_%a.log
#SBATCH --requeue

# One task = one (problem, algorithm, N) arm over a chunk of test ids, in a single process so
# the checkpoint loads once per chunk. The constraint pins the GPU model: the rate table is
# calibrated on one, and a mixed pool would vary both the wall time and the floating-point
# realisation. The venv interpreter is absolute and this is not a login shell, so nothing
# re-sources modules over the torch wheels.

set -euo pipefail
cd {ROOT}
exec {ROOT}/.venv/bin/python scripts/gen_samples_cv.py --run "$SLURM_ARRAY_TASK_ID"
""")
    SBATCH.chmod(0o755)

    print(f"\nwrote {MANIFEST} ({len(tasks)} tasks)")
    print(f"wrote {SBATCH}")
    print(f"\n    sbatch {SBATCH}\n")
    print("Submit that yourself -- this script does not, and neither will I.")
    print("Resubmitting after a failure is safe: each task re-checks which of its ids already")
    print("have a result file and runs only the rest.")


def run(task_id):
    line = MANIFEST.read_text().splitlines()[task_id - 1]
    _, problem, algo, n, exp, ids = line.split("\t")
    n = int(n)

    todo = remaining(problem, algo, n, parse_ids(ids))
    if not todo:
        print(f"task {task_id}: {problem}/{algo}/N={n} ids {ids} already complete", flush=True)
        return 0

    over = [
        f"problem={problem}", f"pretrain={problem}", f"algorithm={algo}",
        f'++problem.data.id_list="{compact(todo)}"', f"++exp_name={exp}",
        "++algorithm.method.progress=False",
        f"++hydra.run.dir={LOGROOT}/hydra/{problem}_{algo}_N{n}_{task_id}",
    ]
    if algo == "kpscv":
        over += [f"++algorithm.method.num_particles={n}",
                 f"++algorithm.method.draw_batch={DRAW_BATCH}",
                 f"++algorithm.method.solve_batch={SOLVE_BATCH}"]

    cmd = [sys.executable, "main.py"] + over
    print(f"task {task_id}: {' '.join(cmd)}", flush=True)

    return subprocess.call(cmd, cwd=ROOT, env={**os.environ, "PYTHONUNBUFFERED": "1"})


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--probe", action="store_true", help="measure the missing rates, then exit")
    ap.add_argument("--assume", action="store_true",
                    help="size unprobed arms from the original campaign's rates, scaled. Lets "
                         "the array go out now; --probe later replaces the estimates")
    ap.add_argument("--only", help="restrict --probe to one problem or algorithm")
    ap.add_argument("--write", action="store_true", help="write the manifest and array script")
    ap.add_argument("--run", type=int, metavar="TASKID", help="run one array task (Slurm)")
    ap.add_argument("--throttle", type=int, default=20,
                    help="max concurrent tasks; the gpu QOS caps a user at 24 (default: 20)")
    ap.add_argument("--target", type=float, default=7200.0,
                    help="seconds of work per task (default: 7200)")
    ap.add_argument("--walltime", default="5:00:00")
    ap.add_argument("--partition", default="gpu")
    ap.add_argument("--constraint", default="rtxblackwell")
    args = ap.parse_args()

    if args.run is not None:
        sys.exit(run(args.run))

    if args.probe:
        probe(args.only)
        return

    print("ladder check (new arms against the STORED config of the run they pair with):")
    ok = check_ladder()
    print()

    tasks, skipped, missing, guessed = plan(args.target, args.assume)
    report(tasks, skipped, missing, guessed, args.throttle)

    if args.write:
        if not ok:
            sys.exit("\nrefusing to write: the ladder drifted from the baseline (see above).")
        if missing:
            sys.exit("\nrefusing to write: probe the missing arms first (see above).")
        if not tasks:
            sys.exit("\nnothing to do -- every arm already has its samples on disk.")
        write(tasks, args.throttle, args.walltime, args.partition, args.constraint)


if __name__ == "__main__":
    main()
