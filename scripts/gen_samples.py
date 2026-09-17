#!/usr/bin/env python3
r"""Generate the KPS sample sets for the paper, as one throttled Slurm array.

    python3 scripts/gen_samples.py                  # plan: the task table and the estimate
    python3 scripts/gen_samples.py --write          # write the manifest + array script
    sbatch scripts/samples/array.sbatch             # YOU submit this (the script never does)

WHAT IT COVERS. Every id of every test set, once:

    KPSH   navier-stokes, inv-scatter, blackhole      N in {64, 128, 256, 512}
    KPSG   inv-scatter, blackhole                     N from configs/algorithm/kpsg.yaml

Navier-Stokes is KPSH-only: KPSG needs the simulator's Jacobian, and on NS no linear map
represents the dynamics (measured 6.8x worse than the fitted slope), so it is not a variant
worth 10 more runs. Everything else -- levels, sweeps, rank, draw_steps, sigma range -- comes
from the algorithm yamls, so this script has no opinion about the method: change the yaml and
re-plan.

IT IS IDEMPOTENT. A task filters its own id list against the `result_<id>.pt` files already on
disk immediately before running, so a timed-out, failed or half-finished array can be
resubmitted unchanged and will only do the work that is missing. That also means the manifest
is safe to reuse: nothing is ever recomputed.

ONE ARRAY, ONE THROTTLE. `--array=1-K%<throttle>` is the only concurrency control, so the cap
is exact however the tasks are sized. The `gpu` partition carries QOS `gpu`, which limits a
user to 24 running jobs and 24 GPUs, and an interactive session already holds one of them, so
the default throttle is 20.

THE DENOISER BATCH IS PART OF THE CONFIGURATION, NOT A FREE OPTIMISATION. The prior-factor
draw is what scales in memory -- 39.6 GB at N=512 on inv-scatter, against 10.2 GB for the
simulator -- so it is chunked over particles (`draw_batch`). The chunking is exact in exact
arithmetic (the net has no batch-coupling layers, only per-sample GroupNorm, and the ODE is
deterministic at churn=0), but NOT in floating point: cuDNN picks different kernels per batch
shape, which moves one denoise call by 2.4e-4 relative, and the ODE's (x - D)/t tail amplifies
that to ~2e-2 by sigma_eps. So `draw_batch` is held at 64 for EVERY N, and N=64 runs
unchunked-but-equal (one batch of 64). Otherwise the particle sweep would confound N with a
change of kernel realisation.

The simulator is deliberately NOT chunked. It is small enough not to need it (0.8 GB on NS,
10.2 GB on inv-scatter at N=512), and on Navier-Stokes chunking it would not be exact even in
exact arithmetic: `time_step` takes torch.max over the whole batch, so each chunk would
integrate with its own adaptive dt.
"""

import argparse
import os
import subprocess
import sys

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent            # the InverseBench checkout
OUT = ROOT / "scripts" / "samples"                       # provenance: manifest + array script
MANIFEST = OUT / "manifest.tsv"
SBATCH = OUT / "array.sbatch"

# Slurm logs and per-task hydra dirs go to ceph, not into the checkout: 200+ task directories
# in a git tree is noise, and home enforces an inode quota.
LOGROOT = Path("/mnt/home/gandry/ceph/ibench/ibench_exp/_gen")

PARTICLES = (64, 128, 256, 512)                          # the KPSH sweep
DRAW_BATCH = 64                                          # fixed across the sweep, see above

GRID = (                                                 # (problem, algorithm)
    ("navier-stokes", "kpsh"),
    ("inv-scatter", "kpsh"),
    ("blackhole", "kpsh"),
    ("inv-scatter", "kpsg"),
    ("blackhole", "kpsg"),
)

# Seconds per (particle x sweep) on an RTX PRO 6000 Blackwell, each measured 2026-09-17 by a
# levels=4/sweeps=1 probe of the production config at the top of its particle range (N=512 for
# H, N=16 for G). A sweep is one prior draw, one simulator call and the rank-r Jacobian pass,
# and the cost is linear in N to within a few percent over 64..1024, so one constant per arm
# predicts a run to about +/-15%.
#
# DO NOT GUESS THESE FROM OLD LOGS. Estimates carried over from earlier runs were wrong by
# 1.8x (inv-scatter H), 3.0x (inv-scatter G) and 1.7x (blackhole G), in both directions,
# because those runs had different levels/sweeps/draw_steps and, for inv-scatter, paid a
# one-off SVD build. Re-probe after any change to draw_steps, rank or solve_iter.
#
# CONFIRMED AGAINST THE LIVE ARRAY (job 7057547, 20 concurrent tasks, tf32 on): inv-scatter H
# N=64 ran at 332 s/id against 334 predicted, navier-stokes H N=64 at 363 against 350. So the
# table is not sensitive to tf32, and sharing a node with other tasks of the same array does
# not measurably slow a task down.
#
# Peak GPU memory at those same points, for the --constraint choice: inv-scatter H 42.4 GB,
# navier-stokes H 35.5 GB, blackhole H 15.0 GB, inv-scatter G 14.6 GB, blackhole G 2.6 GB.
RATE = {
    ("navier-stokes", "kpsh"): 0.0855,
    ("inv-scatter", "kpsh"): 0.0816,
    ("blackhole", "kpsh"): 0.0303,
    ("inv-scatter", "kpsg"): 0.1110,
    ("blackhole", "kpsg"): 0.0326,
}

# Fixed cost paid once per TASK, not per id: interpreter, torch, the checkpoint, and for
# inv-scatter the cached 7200 x 16384 SVD.
STARTUP = {"inv-scatter": 90.0, "navier-stokes": 45.0, "blackhole": 45.0}

ALGO_DIR = {"kpsh": "KPSH", "kpsg": "KPSG"}              # config.algorithm.name -> exp subdir


def parse_ids(spec):
    r"""InverseBench's own id syntax: '0-9', '3', '0-4,7,9-11'."""

    ids = []
    for part in str(spec).split(","):
        if "-" in part[1:]:
            lo, hi = part.split("-", 1)
            ids.extend(range(int(lo), int(hi) + 1))
        else:
            ids.append(int(part))

    return ids


def compact(ids):
    r"""The inverse: [0,1,2,5] -> '0-2,5'. Keeps the override short enough to read in a log.

    A chunk left over after filtering finished ids need not be contiguous, so this can emit a
    comma -- and hydra reads an unquoted comma in an override as a LIST separator, which fails
    with "Ambiguous value for argument". The override is therefore always quoted, which hydra
    resolves to a plain string and `utils.helper.parse_int_list` then expands.
    """

    out, i = [], 0
    while i < len(ids):
        j = i
        while j + 1 < len(ids) and ids[j + 1] == ids[j] + 1:
            j += 1
        out.append(f"{ids[i]}-{ids[j]}" if j > i else f"{ids[i]}")
        i = j + 1

    return ",".join(out)


def load_cfg(problem, algo):
    p = yaml.safe_load((ROOT / "configs" / "problem" / f"{problem}.yaml").read_text())
    a = yaml.safe_load((ROOT / "configs" / "algorithm" / f"{algo}.yaml").read_text())["method"]

    return p, a


def exp_dir(problem, algo, n):
    p, _ = load_cfg(problem, algo)

    return Path(p["exp_dir"]) / ALGO_DIR[algo] / exp_name(algo, n)


def exp_name(algo, n):
    r"""One directory per (algorithm, N) so the sweep points never share a result file."""

    return f"gen_N{n}" if algo == "kpsh" else "gen"


def remaining(problem, algo, n, ids):
    d = exp_dir(problem, algo, n)

    return [i for i in ids if not (d / f"result_{i}.pt").exists()]


def plan(target):
    r"""Enumerate (problem, algo, N) arms, size the id chunks by cost, return the task list."""

    tasks, skipped = [], 0

    for problem, algo in GRID:
        p, a = load_cfg(problem, algo)
        ids_all = parse_ids(p["data"]["id_list"])
        sweeps = int(a["levels"]) * int(a["sweeps"])
        ns = PARTICLES if algo == "kpsh" else (int(a["num_particles"]),)

        for n in ns:
            ids = remaining(problem, algo, n, ids_all)
            skipped += len(ids_all) - len(ids)
            if not ids:
                continue

            per_id = RATE[(problem, algo)] * sweeps * n
            size = max(1, int(target // per_id))

            for k in range(0, len(ids), size):
                chunk = ids[k:k + size]
                tasks.append(dict(problem=problem, algo=algo, n=n, exp=exp_name(algo, n),
                                  ids=compact(chunk), n_ids=len(chunk),
                                  est=STARTUP[problem] + per_id * len(chunk), per_id=per_id))

    return tasks, skipped


def hms(s):
    return f"{int(s) // 3600:d}:{int(s) % 3600 // 60:02d}:{int(s) % 60:02d}"


def report(tasks, skipped, throttle):
    print(f"{'problem':<14}{'algo':<6}{'N':>5}{'ids':>5}{'tasks':>7}{'s/id':>9}"
          f"{'GPU-h':>9}{'per task':>10}")
    print("-" * 65)

    groups, total = {}, 0.0
    for t in tasks:
        g = groups.setdefault((t["problem"], t["algo"], t["n"]), dict(tasks=0, ids=0, est=0.0,
                                                                     per_id=t["per_id"]))
        g["tasks"] += 1
        g["ids"] += t["n_ids"]
        g["est"] += t["est"]

    for (problem, algo, n), g in groups.items():
        total += g["est"]
        print(f"{problem:<14}{algo:<6}{n:>5}{g['ids']:>5}{g['tasks']:>7}{g['per_id']:>9.0f}"
              f"{g['est'] / 3600:>9.1f}{hms(g['est'] / g['tasks']):>10}")

    longest = max((t["est"] for t in tasks), default=0.0)
    print("-" * 65)
    print(f"{'TOTAL':<25}{sum(t['n_ids'] for t in tasks):>5}{len(tasks):>7}{'':>9}"
          f"{total / 3600:>9.1f}")
    if skipped:
        print(f"\n{skipped} (arm, id) pairs already have samples on disk and were dropped.")
    print(f"\nlongest task {hms(longest)}; at throttle {throttle} the array finishes in about "
          f"{hms(max(total / throttle, longest))} of wall time.")

    return total, longest


def write(tasks, throttle, walltime, det, partition, constraint):
    # Quote only an OR expression: Slurm strips the quotes, but an unquoted '|' is a pipe to
    # any shell that ever re-reads this file, and a quoted single feature risks being taken
    # literally by older Slurm argument parsers.
    cons = f'"{constraint}"' if "|" in constraint or "&" in constraint else constraint

    OUT.mkdir(parents=True, exist_ok=True)
    (LOGROOT / "logs").mkdir(parents=True, exist_ok=True)

    with MANIFEST.open("w") as f:
        for i, t in enumerate(tasks, 1):
            f.write(f"{i}\t{t['problem']}\t{t['algo']}\t{t['n']}\t{t['exp']}\t{t['ids']}\n")

    SBATCH.write_text(f"""#!/bin/bash
#SBATCH --job-name=kps-gen
#SBATCH --partition={partition}
#SBATCH --gres=gpu:1
#SBATCH --constraint={cons}
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time={walltime}
#SBATCH --array=1-{len(tasks)}%{throttle}
#SBATCH --output={LOGROOT}/logs/%A_%a.log
#SBATCH --requeue

# The constraint pins the GPU model. Any 80 GB card fits the N=512 cloud (peak 42.4 GB on
# inv-scatter, 35.5 GB on Navier-Stokes; the a100_2g.20gb MIG slices do not), but a mixed pool
# would vary both the wall time -- the RATE table is calibrated on one model -- and the
# floating-point realisation of the run.
#
# One task = one (problem, algorithm, N) arm over a chunk of test ids, run in a single process
# so the checkpoint is loaded once for the whole chunk. The venv interpreter is called by
# absolute path and this is NOT a login shell, so nothing re-sources modules on top of the
# torch wheels, which carry their own CUDA.

set -euo pipefail
cd {ROOT}
exec {ROOT}/.venv/bin/python scripts/gen_samples.py --run "$SLURM_ARRAY_TASK_ID"{' --det' if det else ''}
""")
    SBATCH.chmod(0o755)

    print(f"\nwrote {MANIFEST} ({len(tasks)} tasks)")
    print(f"wrote {SBATCH}")
    print(f"\n    sbatch {SBATCH}\n")
    print("Submit that yourself -- this script does not, and neither will I.")
    print("Resubmitting the same array after a failure is safe: each task re-checks which of")
    print("its ids already have a result file and runs only the rest.")


def run(task_id, det):
    r"""Inside the job: re-filter the id list, then hand the whole chunk to main.py."""

    line = MANIFEST.read_text().splitlines()[task_id - 1]
    _, problem, algo, n, exp, ids = line.split("\t")
    n = int(n)

    todo = remaining(problem, algo, n, parse_ids(ids))
    if not todo:
        print(f"task {task_id}: {problem}/{algo}/N={n} ids {ids} already complete", flush=True)
        return 0

    over = [
        f"problem={problem}", f"pretrain={problem}", f"algorithm={algo}",
        f'++problem.data.id_list="{compact(todo)}"',      # quoted: see compact()
        f"++exp_name={exp}",
        "++algorithm.method.progress=False",
        f"++hydra.run.dir={LOGROOT}/hydra/{problem}_{algo}_N{n}_{task_id}",
    ]
    if algo == "kpsh":
        over += [f"++algorithm.method.num_particles={n}",
                 f"++algorithm.method.draw_batch={DRAW_BATCH}"]

    entry = "run_det.py" if det else "main.py"
    cmd = [sys.executable, entry] + over

    print(f"task {task_id}: {' '.join(cmd)}", flush=True)

    return subprocess.call(cmd, cwd=ROOT, env={**os.environ, "PYTHONUNBUFFERED": "1"})


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--write", action="store_true", help="write the manifest and array script")
    ap.add_argument("--run", type=int, metavar="TASKID", help="run one array task (used by Slurm)")
    ap.add_argument("--throttle", type=int, default=20,
                    help="max concurrent tasks; the gpu QOS caps a user at 24 (default: 20)")
    ap.add_argument("--target", type=float, default=7200.0,
                    help="seconds of work to put in one task (default: 7200)")
    ap.add_argument("--partition", default="gpu",
                    help="gpu (QOS gpu: 24 GPUs/user, no preemption) or gpupreempt (no cap, "
                         "preemptible -- safe here, a task loses at most the id in flight)")
    ap.add_argument("--constraint", default="rtxblackwell",
                    help="GPU feature. The default pins ONE model, so every arm of the sweep "
                         "runs on the same kernels; widening it (e.g. "
                         "'a100-80gb|h100|rtxblackwell') schedules sooner but mixes hardware, "
                         "and the RATE table below was calibrated on the default")
    ap.add_argument("--det", action="store_true",
                    help="run through run_det.py (bitwise reproducible, verified on NS only)")
    args = ap.parse_args()

    if args.run is not None:
        return run(args.run, args.det)

    tasks, skipped = plan(args.target)
    if not tasks:
        print("nothing to do: every arm already has a result file for every id.")
        return 0

    _, longest = report(tasks, skipped, args.throttle)

    if args.write:
        hours = max(2, int(longest * 2 // 3600) + 1)     # 2x the longest task, floor 2 h
        write(tasks, args.throttle, f"{hours}:00:00", args.det, args.partition,
              args.constraint)
    else:
        print("\nre-run with --write to emit the manifest and the array script.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
