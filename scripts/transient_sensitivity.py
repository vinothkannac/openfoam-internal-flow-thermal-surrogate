#!/usr/bin/env python3
"""Sensitivity of the transient time averages to the time step and to the run length (one case).

The README reports time-averaged T_max for the four unsteady cases from ONE setting each
(Courant limit 0.8, 100 s of flow time, average over the second half). I do not know how much of
the reported mean is the setting. Here I take the cheapest unsteady case, u0.2_q100, and repeat it:

  co0.8_100s  the setting used in the README (reproduction check)
  co0.4_100s  half the time step (maxCo 0.4)
  co1.6_100s  twice the time step (maxCo 1.6)
  co0.8_200s  twice the run length

All start from the same saved steady state. For every run I report the mean, standard deviation and
a standard error of the mean over (a) the second half of the run (the README definition) and
(b) fixed windows of flow time, so that runs of different length are compared on the same
seconds. The averages are TIME averages (the series is first resampled to a uniform time grid,
because a variable time step would otherwise weight some instants more). The standard error uses
batch means with 5 blocks; it is only a rough guide for a signal that oscillates with a period of
seconds, and the 5 blocks of a 50 s window are only a few oscillation periods each.

    python3 scripts/transient_sensitivity.py start            # steady start state (8000 + 2000 iterations)
    python3 scripts/transient_sensitivity.py run --jobs 4     # the four transient runs
    python3 scripts/transient_sensitivity.py collect          # statistics -> results/transient_sensitivity.csv
    python3 scripts/transient_sensitivity.py run --variants co0.8_100s --end-time 3 --out-dir smoke
"""
import argparse
import csv
import re
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import run_sweep  # noqa: E402
from continue_unconverged import lower_relaxation  # noqa: E402
from parse_foam import run_time, time_dirs  # noqa: E402
from run_transient import load_series  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CASE = (0.2, 100)  # u_in, q_ref
BASE = ROOT / "cabinet_case" / "runs_tsens"
TEMPLATE = ROOT / "cabinet_case" / "transient_template"
OUT = ROOT / "results" / "transient_sensitivity.csv"
VARIANTS = {  # name: (maxCo, flow time in s)
    "co0.8_100s": (0.8, 100.0),
    "co0.4_100s": (0.4, 100.0),
    "co1.6_100s": (1.6, 100.0),
    "co0.8_200s": (0.8, 200.0),
}
GRID_DT = 0.02       # uniform resampling step in s (finer than the smallest time step used)
WINDOWS = [(50, 100), (100, 150), (150, 200), (100, 200)]


def start_state():
    """Steady fields of the case: main sweep settings for 8000 iterations, then the same
    lower-relaxation continuation as continue_unconverged.py for 2000 more."""
    u, q = CASE
    d = run_sweep.prepare(u, q, root=BASE / "steady")
    shutil.copytree(run_sweep.TEMPLATE / "constant" / "polyMesh", d / "constant" / "polyMesh",
                    dirs_exist_ok=True)
    run_sweep.execute(d)
    lower_relaxation(d, 10000, 0.1, 0.2)
    with open(d / "log.solver_continue", "w") as fh:
        subprocess.run(["buoyantSimpleFoam"], cwd=d, stdout=fh, stderr=subprocess.STDOUT, check=True)
    print("steady start state in", d, "at", time_dirs(d)[-1].name)


def steady_dir():
    return BASE / "steady" / run_sweep.case_name(*CASE)


def prepare(name, out_dir, end_time=None):
    max_co, flow_time = VARIANTS[name]
    d = BASE / out_dir / name
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)
    src = steady_dir()
    shutil.copytree(src / "constant", d / "constant")
    shutil.copytree(TEMPLATE / "system", d / "system")
    cd = d / "system" / "controlDict"
    t = cd.read_text().replace("@ENDTIME@", f"{end_time or flow_time:g}")
    t = re.sub(r"(?m)^maxCo\s+\S+;", f"maxCo           {max_co:g};", t)
    t = re.sub(r"(?m)^writeInterval\s+\S+;", "writeInterval   20;", t, count=1)  # fewer field dumps
    cd.write_text(t)
    shutil.copytree(time_dirs(src)[-1], d / "0", ignore=shutil.ignore_patterns("uniform"))
    return d


def run_one(task):
    name, out_dir, end_time = task
    d = prepare(name, out_dir, end_time)
    with open(d / "log.solver", "w") as fh:
        r = subprocess.run(["buoyantPimpleFoam"], cwd=d, stdout=fh, stderr=subprocess.STDOUT)
    print(f"{'finished' if r.returncode == 0 else 'FAILED'} {name}", flush=True)


def uniform(t, y, t0, t1):
    """Resample onto a uniform grid in [t0, t1]; None if the run does not cover the window."""
    if t[-1] < t1 - 1e-6 or t[0] > t0 + 1e-6:
        return None
    g = np.arange(t0, t1 + 1e-9, GRID_DT)
    return np.interp(g, t, y)


def batch_sem(y, blocks=5):
    n = len(y) // blocks
    if n < 2:
        return float("nan")
    means = y[: n * blocks].reshape(blocks, n).mean(axis=1)
    return float(means.std(ddof=1) / np.sqrt(blocks))


def stats(t, tmax, dp, t0, t1):
    a, b = uniform(t, tmax, t0, t1), uniform(t, dp, t0, t1)
    if a is None:
        return None
    return {"window_s": f"{t0:g}-{t1:g}", "T_max_mean_K": a.mean(), "T_max_std_K": a.std(),
            "T_max_sem_K": batch_sem(a), "dp_mean_Pa": b.mean(), "dp_std_Pa": b.std(),
            "dp_sem_Pa": batch_sem(b)}


def collect(out_dir):
    rows = []
    for name, (max_co, flow_time) in VARIANTS.items():
        d = BASE / out_dir / name
        if not (d / "log.solver").exists():
            continue
        try:
            t, tmax, dp = load_series(d)
        except Exception as exc:
            print(f"cannot read {name}: {exc}")
            continue
        info = {"run": name, "max_co": max_co, "flow_time_s": t[-1], "steps": len(t),
                "mean_dt_s": t[-1] / len(t), "wall_time_s": run_time(d / "log.solver")}
        # the README definition: step average over the second half of THIS run
        half = t >= t[0] + 0.5 * (t[-1] - t[0])
        wins = [(t[0] + 0.5 * (t[-1] - t[0]), t[-1], "second half")] + [(a, b, None) for a, b in WINDOWS]
        for t0, t1, label in wins:
            s = stats(t, tmax, dp, t0, t1)
            if s is None:
                continue
            if label:
                s["window_s"] = f"{label} ({t0:g}-{t1:g})"
                s["T_max_mean_stepavg_K"] = float(tmax[half].mean())  # comparable with transient_results.csv
            rows.append({**info, **s})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["start", "run", "collect"])
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS))
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--out-dir", default="variants")
    ap.add_argument("--end-time", type=float, default=None, help="override flow time (smoke test)")
    args = ap.parse_args()

    if args.stage == "start":
        start_state()
    elif args.stage == "run":
        if not steady_dir().exists():
            sys.exit("run the 'start' stage first")
        with ThreadPoolExecutor(max_workers=args.jobs) as pool:
            list(pool.map(run_one, [(v, args.out_dir, args.end_time) for v in args.variants]))
    else:
        rows = collect(args.out_dir)
        if not rows:
            sys.exit("no finished runs")
        keys = []
        for r in rows:
            keys += [k for k in r if k not in keys]
        with open(OUT, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
        for r in rows:
            print(f"{r['run']:11s} {r['window_s']:22s} T_max {r['T_max_mean_K']:.2f} +- {r['T_max_sem_K']:.2f} K "
                  f"(std {r['T_max_std_K']:.2f}), dp {r['dp_mean_Pa']:.4f} Pa")
        print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
