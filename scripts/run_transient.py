#!/usr/bin/env python3
"""Transient (buoyantPimpleFoam) runs for the sweep cases a steady solver could not converge.

Why I do this: for four low-velocity cases T_max kept swinging by 1-5 K between saved
iterations even at low relaxation. That is what a flow looks like when it is truly
unsteady (separated, oscillating), and a steady solver then has no fixed point to reach.
A transient solver can follow the oscillation, and the time-average of a statistically
steady flow is a well-defined answer.

For each case that has converged = False in results/sweep_results.csv:
  1. copy the case's constant/ and the last steady fields as the start (time 0),
  2. use the transient dictionaries in cabinet_case/transient_template/system,
  3. run buoyantPimpleFoam for --end-time seconds of flow time (Courant-limited steps),
  4. read T_max (hottest component face) and dp (p_rgh inlet - outlet) at every step,
     and report mean / std / min / max over the second half of the run.

Results: results/transient_results.csv. The steady sweep_results.csv is not touched.

    python3 scripts/run_transient.py --jobs 2
    python3 scripts/run_transient.py --cases u0.2_q100 --end-time 20   # quick test
    python3 scripts/run_transient.py --collect-only
"""
import argparse
import re
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from parse_foam import time_dirs, run_time  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
STEADY = ROOT / "cabinet_case" / "runs"
TEMPLATE = ROOT / "cabinet_case" / "transient_template"
RUNS = ROOT / "cabinet_case" / "runs_transient"
CSV = ROOT / "results" / "sweep_results.csv"
OUT = ROOT / "results" / "transient_results.csv"


def prepare(case, end_time):
    d = RUNS / case
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)
    shutil.copytree(STEADY / case / "constant", d / "constant")
    shutil.copytree(TEMPLATE / "system", d / "system")
    cd = d / "system" / "controlDict"
    cd.write_text(cd.read_text().replace("@ENDTIME@", f"{end_time:g}"))
    # I start from the last steady iterate: it is close to the flow I want to follow,
    # so the transient spends its time on the oscillation and not on the start-up.
    last = time_dirs(STEADY / case)[-1]
    shutil.copytree(last, d / "0", ignore=shutil.ignore_patterns("uniform"))
    return d


def run_case(case, end_time):
    d = prepare(case, end_time)
    with open(d / "log.solver", "w") as fh:
        subprocess.run(["buoyantPimpleFoam"], cwd=d, stdout=fh, stderr=subprocess.STDOUT, check=True)
    return case


def series(run_dir, name):
    """Time and value columns of a surfaceFieldValue .dat file (all restarts joined)."""
    files = sorted((run_dir / "postProcessing" / name).glob("*/surfaceFieldValue.dat"))
    rows = []
    for f in files:
        for line in f.read_text().splitlines():
            if line.strip() and not line.lstrip().startswith("#"):
                rows.append([float(x) for x in line.split()[:2]])
    a = np.array(rows)
    return a[:, 0], a[:, 1]


def load_series(run_dir):
    t, t1 = series(run_dir, "Tmax_comp1")
    _, t2 = series(run_dir, "Tmax_comp2")
    _, t3 = series(run_dir, "Tmax_comp3")
    _, pin = series(run_dir, "prgh_inlet")
    _, pout = series(run_dir, "prgh_outlet")
    return t, np.maximum(np.maximum(t1, t2), t3), pin - pout


def summarise(case, steady_row):
    d = RUNS / case
    t, tmax, dp = load_series(d)
    half = t >= t[0] + 0.5 * (t[-1] - t[0])      # I discard the first half as start-up
    a, b = half & (t < t[0] + 0.75 * (t[-1] - t[0])), t >= t[0] + 0.75 * (t[-1] - t[0])
    return {
        "case": case,
        "T_max_steady_K": steady_row["T_max_K"],
        "T_max_mean_K": tmax[half].mean(), "T_max_std_K": tmax[half].std(),
        "T_max_min_K": tmax[half].min(), "T_max_max_K": tmax[half].max(),
        # if the two quarters of the averaging window agree, the mean is stationary
        "T_max_mean_drift_K": abs(tmax[a].mean() - tmax[b].mean()),
        "dp_steady_Pa": steady_row["dp_Pa"],
        "dp_mean_Pa": dp[half].mean(), "dp_std_Pa": dp[half].std(),
        "flow_time_s": t[-1],
        "wall_time_s": run_time(d / "log.solver"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--end-time", type=float, default=200.0)
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--cases", nargs="*")
    ap.add_argument("--collect-only", action="store_true")
    args = ap.parse_args()

    df = pd.read_csv(CSV).set_index("case")
    cases = args.cases or list(df.index[~df["converged"].fillna(False).astype(bool)])
    if not args.collect_only:
        with ThreadPoolExecutor(max_workers=args.jobs) as ex:
            for c in ex.map(lambda c: run_case(c, args.end_time), cases):
                print("finished", c, flush=True)
    rows = [summarise(c, df.loc[c]) for c in cases]
    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
