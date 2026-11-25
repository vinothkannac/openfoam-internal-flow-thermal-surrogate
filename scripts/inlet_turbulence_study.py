#!/usr/bin/env python3
"""Inlet-turbulence sensitivity (roadmap item 10, the cheap part).

The sweep fixes the inlet at k = 1e-3 m2/s2 and omega = 4 1/s for every velocity, which is a turbulence
intensity of about 13 % at 0.2 m/s and 2.6 % at 1 m/s. I never tested how much T_max and dp depend on it.
Here I re-run three converged cases (same steady settings as the sweep, 8000 iterations) with the inlet k
multiplied by 0.1, 1, 10 and 100. omega is scaled with sqrt(k), so the turbulent length scale
l = sqrt(k) / (Cmu^0.25 omega) stays the same as in the sweep and only the intensity changes.
The factor 1 run reproduces the sweep and is the control.

    python3 scripts/inlet_turbulence_study.py --jobs 6
    python3 scripts/inlet_turbulence_study.py --collect-only
"""
import argparse
import csv
import math
import re
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import run_sweep  # noqa: E402
from parse_foam import case_metrics  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "cabinet_case" / "runs_inlet_turb"
OUT = ROOT / "results" / "inlet_turbulence_study.csv"
SWEEP = ROOT / "results" / "sweep_results.csv"
T_INLET = 293.15
K0, OMEGA0 = 1e-3, 4.0
CASES = [(0.5, 400), (1.0, 100), (1.0, 600)]
K_FACTORS = [0.1, 1.0, 10.0, 100.0]


def set_inlet(d, kf):
    """Replace every `uniform 1e-3` in 0/k and `uniform 4` in 0/omega (inlet, internal field,
    wall and outlet values alike, so nothing starts inconsistent)."""
    k, om = K0 * kf, OMEGA0 * math.sqrt(kf)
    for name, old, new in (("k", "1e-3", f"{k:g}"), ("omega", "4", f"{om:g}")):
        f = d / "0" / name
        t, n = re.subn(rf"uniform {re.escape(old)};", f"uniform {new};", f.read_text())
        if n == 0:
            raise RuntimeError(f"no 'uniform {old};' in {f}")
        f.write_text(t)


def run_one(task):
    (u, q), kf = task
    name = f"{run_sweep.case_name(u, q)}_kx{kf:g}"
    # prepare() names the directory after (u, q) and deletes an existing one, so each variant
    # gets its own parent directory; otherwise parallel variants of one case would delete each other.
    d = run_sweep.prepare(u, q, root=RUNS / name)
    shutil.copytree(run_sweep.TEMPLATE / "constant" / "polyMesh", d / "constant" / "polyMesh", dirs_exist_ok=True)
    set_inlet(d, kf)
    run_sweep.execute(d)
    print("finished", name, flush=True)


def collect():
    sw = pd.read_csv(SWEEP).set_index("case")
    rows = []
    for (u, q) in CASES:
        base = sw.loc[run_sweep.case_name(u, q)]
        for kf in K_FACTORS:
            n = f"{run_sweep.case_name(u, q)}_kx{kf:g}"
            d = RUNS / n / run_sweep.case_name(u, q)
            if not d.exists():
                continue
            m = case_metrics(d)
            k = K0 * kf
            rows.append({"case": run_sweep.case_name(u, q), "u_in": u, "q_ref": q, "k_factor": kf, "k_in": k,
                         "omega_in": OMEGA0 * math.sqrt(kf), "intensity_pct": 100 * math.sqrt(2 * k / 3) / u,
                         "T_max_K": m["T_max_K"], "dT_max_K": m["T_max_K"] - T_INLET, "dp_Pa": m["dp_Pa"],
                         "converged": m["converged"], "sweep_T_max_K": base["T_max_K"], "sweep_dp_Pa": base["dp_Pa"],
                         "dT_vs_sweep_K": m["T_max_K"] - base["T_max_K"],
                         "dp_vs_sweep_pct": 100 * (m["dp_Pa"] / base["dp_Pa"] - 1)})
    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)
    print(df.to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--collect-only", action="store_true")
    a = ap.parse_args()
    if not a.collect_only:
        tasks = [(c, kf) for c in CASES for kf in K_FACTORS]
        with ThreadPoolExecutor(max_workers=a.jobs) as ex:
            list(ex.map(run_one, tasks))
    collect()


if __name__ == "__main__":
    main()
