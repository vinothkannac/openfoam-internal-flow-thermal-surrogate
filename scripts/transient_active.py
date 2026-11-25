#!/usr/bin/env python3
"""Transient runs of the four unsteady active-learning cases (roadmap item 11, data for the transient surrogate).

The active-learning study (active_learning.py) found four proposals that stay unsteady. Their run
directories were not kept, so this script rebuilds each steady start state exactly as
active_learning.run_cfd does (8000 upwind iterations, then the 2000-iteration lower-relaxation
continuation), and then follows the flow with buoyantPimpleFoam exactly as run_transient.py does
(maxCo 0.8, 100 s of flow time, statistics over the second half). That gives 8 unsteady cases
with the same method: the 4 of the sweep and these 4.

    python3 scripts/transient_active.py run --jobs 4
    python3 scripts/transient_active.py collect      # -> results/transient_active_results.csv
"""
import argparse
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import run_sweep  # noqa: E402
import run_transient as rt  # noqa: E402
from continue_unconverged import lower_relaxation  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "cabinet_case" / "runs_transient_active"
CASES_CSV = ROOT / "results" / "active_learning_cases.csv"
OUT = ROOT / "results" / "transient_active_results.csv"
END_TIME = 100.0


def unsteady_cases():
    df = pd.read_csv(CASES_CSV)
    return df[~df["accepted"].astype(bool)]


def run_one(row):
    u, q = float(row["u_in"]), float(row["q_ref"])
    steady = run_sweep.prepare(u, q, root=BASE / "steady")
    shutil.copytree(run_sweep.TEMPLATE / "constant" / "polyMesh", steady / "constant" / "polyMesh",
                    dirs_exist_ok=True)
    run_sweep.execute(steady)
    lower_relaxation(steady, 10000, 0.1, 0.2)
    with open(steady / "log.solver_continue", "w") as fh:
        subprocess.run(["buoyantSimpleFoam"], cwd=steady, stdout=fh, stderr=subprocess.STDOUT, check=True)
    rt.STEADY, rt.RUNS = BASE / "steady", BASE / "transient"   # run_transient.prepare reads these globals
    rt.run_case(row["case"], END_TIME)
    print("finished", row["case"], flush=True)


def collect():
    rt.RUNS = BASE / "transient"
    rows = []
    for _, r in unsteady_cases().iterrows():
        rows.append(rt.summarise(r["case"], {"T_max_K": r["T_max_K"], "dp_Pa": r["dp_Pa"]}))
    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False)
    print(out.to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["run", "collect"])
    ap.add_argument("--jobs", type=int, default=4)
    a = ap.parse_args()
    if a.action == "run":
        rows = [r for _, r in unsteady_cases().iterrows()]
        with ThreadPoolExecutor(max_workers=a.jobs) as ex:
            list(ex.map(run_one, rows))
    collect()


if __name__ == "__main__":
    main()
