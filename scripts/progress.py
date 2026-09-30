#!/usr/bin/env python3
r"""How far along is the sample-generation array?

    python3 scripts/progress.py                  # the kpsh/kpsg campaign
    python3 scripts/progress.py --campaign cv    # the ridge-SLR campaign (kpscv/kpsgf)

COUNTING SAMPLES IS MISLEADING, so this weights them. One id costs 33 s (blackhole KPSG) to
47 min (navier-stokes KPSH at N=512), a factor of 85, and the array works through the manifest
in order -- cheap arms first -- so the fraction of RESULT FILES written runs far ahead of the
fraction of WORK done early on and falls behind it later. Every figure below is in GPU-hours,
using the same measured cost table the plan was built from (`gen_samples.RATE`).

THE ETA DIVIDES THE WORK LEFT BY THE SLOTS ACTUALLY RUNNING, which squeue reports directly.
The obvious alternative -- GPU-hours completed over wall time elapsed -- is badly biased while
the array is young, because every running task holds a partly finished id that no result file
accounts for yet: with 20 tasks in flight on hour-long ids it can read 10x low. That figure is
still printed, as a cross-check that becomes trustworthy once most arms have landed something,
but it is deliberately not what the ETA uses.

RUN IT ON DEMAND, NOT ON A TIMER. It calls squeue once, scoped to this user. Slurm here is a
shared service for hundreds of people and repeated polling degrades it for everyone, so this
belongs in `watch` or a cron loop under no circumstances.
"""

import argparse
import importlib
import os
import re
import subprocess
import sys
import time

from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

# Which campaign to report on. Both planners expose the same surface -- GRID, PARTICLES,
# load_cfg, parse_ids, exp_dir, SBATCH -- so this module needs no other change. The ridge-SLR
# one keeps its rates in a cache written by `--probe` rather than a static table, because its
# solve cost no longer scales with the rank and the old numbers do not transfer.
MODULE = {"kpsh": "gen_samples", "cv": "gen_samples_cv"}

g = None            # the selected planner module
RATE = {}           # (problem, algo) -> seconds per (particle x sweep)
SBATCH = None
JOB = "kps-gen"


def select(campaign):
    global g, RATE, SBATCH, JOB

    g = importlib.import_module(MODULE[campaign])
    RATE = g.rates(assume=True)[0] if hasattr(g, "rates") else g.RATE
    SBATCH = g.SBATCH
    JOB = getattr(g, "JOB_NAME", "kps-gen")


def cost(problem, algo, n):
    r"""Predicted GPU-seconds for one id of this arm."""

    _, a = g.load_cfg(problem, algo)

    return RATE[(problem, algo)] * int(a["levels"]) * int(a["sweeps"]) * n


def arms():
    sweep_algo = getattr(g, "SWEEP_ALGO", "kpsh")

    for problem, algo in g.GRID:
        p, a = g.load_cfg(problem, algo)
        ids = g.parse_ids(p["data"]["id_list"])

        if (problem, algo) not in RATE:
            continue                      # unprobed arm: no cost model, so nothing to weight

        for n in (g.PARTICLES if algo == sweep_algo else (int(a["num_particles"]),)):
            yield problem, algo, n, ids


def squeue_state():
    r"""(running, pending, earliest start) for the generation array, or None if squeue fails.

    One call, scoped to this user and this job name.
    """

    try:
        out = subprocess.run(["squeue", "-u", os.environ.get("USER", ""), "-h", "-r",
                              "-n", JOB, "-o", "%T|%S"], capture_output=True,
                             text=True, timeout=60).stdout.splitlines()
    except Exception:
        return None

    states = [ln.split("|")[0] for ln in out if "|" in ln]
    starts = []
    for ln in out:
        state, _, start = ln.partition("|")
        if state == "RUNNING":
            try:
                starts.append(time.mktime(time.strptime(start, "%Y-%m-%dT%H:%M:%S")))
            except ValueError:
                pass

    return states.count("RUNNING"), states.count("PENDING"), (min(starts) if starts else None)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--campaign", choices=tuple(MODULE), default="kpsh",
                    help="which generation campaign to report on (default: kpsh)")
    select(ap.parse_args().campaign)

    rows, done_h, todo_h, stamped = [], 0.0, 0.0, []

    if not RATE:
        print("no measured rates for this campaign yet -- run the planner's --probe first")
        return 1

    for problem, algo, n, ids in arms():
        d = g.exp_dir(problem, algo, n)
        per = cost(problem, algo, n)
        have = [i for i in ids if (d / f"result_{i}.pt").exists()]
        stamped += [((d / f"result_{i}.pt").stat().st_mtime, per) for i in have]
        done_h += per * len(have) / 3600
        todo_h += per * (len(ids) - len(have)) / 3600
        rows.append((problem, algo, n, len(have), len(ids), per * len(ids) / 3600))

    print(f"{'problem':<14}{'algo':<6}{'N':>5}{'done':>6}{'of':>5}{'%':>5}"
          f"{'GPU-h':>8}{'':>3}")
    print("-" * 49)
    for problem, algo, n, have, tot, arm_h in rows:
        bar = "#" * int(10 * have / tot) + "." * (10 - int(10 * have / tot))
        print(f"{problem:<14}{algo:<6}{n:>5}{have:>6}{tot:>5}{100 * have / tot:>4.0f}%"
              f"{arm_h:>8.1f}  {bar}")

    n_done = sum(r[3] for r in rows)
    n_tot = sum(r[4] for r in rows)
    print("-" * 49)
    print(f"{'TOTAL':<25}{n_done:>6}{n_tot:>5}{100 * n_done / n_tot:>4.0f}%"
          f"{done_h + todo_h:>8.1f}")

    print(f"\nwork    {done_h:.1f} of {done_h + todo_h:.1f} GPU-h done "
          f"({100 * done_h / (done_h + todo_h):.0f}%), {todo_h:.1f} to go")

    state = squeue_state()
    running, pending, started = state if state else (0, 0, None)

    if state:
        print(f"slurm   {running} running, {pending} pending (throttle caps the rest)")
    if state and not running and not pending:
        print(f"        no {JOB} tasks left -- the array is finished or was never submitted")

    # Throughput of the CURRENT array only: work finished since its earliest running task
    # started, over that same window. Dividing all historical work by this window instead --
    # which an earlier version did -- reads absurdly high right after an array transition,
    # when the window is minutes old but the completed work is hours of it.
    if started:
        elapsed = (time.time() - started) / 3600
        recent = sum(per for ts, per in stamped if ts >= started) / 3600
        if elapsed > 0.25:
            print(f"rate    {recent / elapsed:.1f} GPU-h per wall hour over the last "
                  f"{elapsed:.1f} h (a floor: it misses the {running} ids in flight)")
        else:
            print(f"rate    too early to measure -- this array has been up {60 * elapsed:.0f} min")

    # Slots, not tasks. While a dependency or the throttle holds the queue, the number
    # RUNNING understates what the array will get the moment it is released -- dividing by it
    # gave 45 h for a 7 h job during the hand-over between two arrays. Use the throttle as the
    # ceiling whenever there is enough queued work to fill it.
    throttle = 20
    try:
        m = re.search(r"--array=\S*?%(\d+)", SBATCH.read_text())
        throttle = int(m.group(1)) if m else throttle
    except OSError:
        pass

    slots = max(running, min(throttle, running + pending)) if (running or pending) else 0

    if slots:
        eta = todo_h / slots
        note = f"at {slots} slots" if slots == running else (
            f"at {slots} slots once the queue releases ({running} busy now)")
        print(f"ETA     {eta:.1f} h, about "
              f"{time.strftime('%a %H:%M', time.localtime(time.time() + 3600 * eta))}, {note}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
