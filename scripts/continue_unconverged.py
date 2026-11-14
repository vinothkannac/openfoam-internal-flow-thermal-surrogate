#!/usr/bin/env python3
"""Continuation runs for the sweep cases that did not meet the convergence flag.

For every case with converged = False in results/sweep_results.csv, one at a time:
  1. lower the relaxation factors of THIS case only (U, h: 0.1; k, omega: 0.2),
  2. restart buoyantSimpleFoam from the latest saved time,
  3. run on until the iteration limit (default 10000, i.e. 2000 more than the sweep),
  4. re-read T_max, dp and the convergence flag and update the row in the CSV.

I do this one case at a time and on a copy of the settings inside the run directory,
so the shared template in cabinet_case/system is never changed and each case can be
judged on its own. The state after the main sweep stays in the CSV columns
T_max_8000it_K, dp_8000it_Pa and converged_8000it, next to the final values.

    python3 scripts/continue_unconverged.py                  # run the solver, update the CSV
    python3 scripts/continue_unconverged.py --collect-only   # solver already ran, only update the CSV
"""
import argparse
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from parse_foam import case_metrics, run_time  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "cabinet_case" / "runs"
CSV = ROOT / "results" / "sweep_results.csv"
CONTINUE_LOG = "log.solver_continue"


def lower_relaxation(run_dir, end_time, uh, turb):
    fv = run_dir / "system" / "fvSolution"
    text = fv.read_text()
    # I match the whole line, so this works whatever factor the sweep used before
    text = re.sub(r'(?m)^(\s*)U\s+[0-9.]+;', rf"\g<1>U               {uh};", text)
    text = re.sub(r'(?m)^(\s*)"\(h\|e\)"\s+[0-9.]+;', rf'\g<1>"(h|e)"         {uh};', text)
    text = re.sub(r'(?m)^(\s*)"\(k\|omega\)"\s+[0-9.]+;', rf'\g<1>"(k|omega)"     {turb};', text)
    fv.write_text(text)
    cd = run_dir / "system" / "controlDict"
    text = cd.read_text()
    text = re.sub(r"(?m)^startFrom\s+\S+;", "startFrom       latestTime;", text)
    text = re.sub(r"(?m)^endTime\s+\S+;", f"endTime         {end_time};", text)
    cd.write_text(text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--end-time", type=int, default=10000)
    ap.add_argument("--relax-uh", type=float, default=0.1)
    ap.add_argument("--relax-turb", type=float, default=0.2)
    ap.add_argument("--collect-only", action="store_true")
    args = ap.parse_args()

    df = pd.read_csv(CSV)
    # I keep the main-sweep result in its own columns, so the improvement from the
    # continuation can be read from one table. Only written once, on the first call.
    if "converged_8000it" not in df.columns:
        df["T_max_8000it_K"], df["dp_8000it_Pa"], df["converged_8000it"] = df["T_max_K"], df["dp_Pa"], df["converged"]
    todo = df.index[~df["converged"].fillna(False).astype(bool)]
    for i in todo:
        case = df.at[i, "case"]
        run_dir = RUNS / case
        if not args.collect_only:
            lower_relaxation(run_dir, args.end_time, args.relax_uh, args.relax_turb)
            with open(run_dir / CONTINUE_LOG, "w") as fh:
                subprocess.run(["buoyantSimpleFoam"], cwd=run_dir, stdout=fh, stderr=subprocess.STDOUT, check=True)
        m = case_metrics(run_dir, log_name=CONTINUE_LOG)
        # total CFD cost of the case = main sweep + continuation
        m["wall_time_s"] = (df.at[i, "wall_time_s"] or 0) + (run_time(run_dir / CONTINUE_LOG) or 0)
        for k, v in m.items():
            df.at[i, k] = v
        print(f"{case}: T_max = {m['T_max_K']:.2f} K, dp = {m['dp_Pa']:.3f} Pa, "
              f"end = {m['end_time_s']:.0f}, converged = {m['converged']}", flush=True)
    df.to_csv(CSV, index=False)
    print(f"{int(df['converged'].sum())} of {len(df)} cases now meet the convergence flag; wrote {CSV}")


if __name__ == "__main__":
    main()
