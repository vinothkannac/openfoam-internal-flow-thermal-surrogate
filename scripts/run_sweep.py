#!/usr/bin/env python3
"""Parametric sweep of the cabinet case: inlet velocity x heat flux.

I mesh once (cabinet_case/Allmesh) because the geometry is the same in every case;
only the boundary values change. For every (u_in, q_ref) I copy the template, fill
the @placeholders@ in 0.orig, run buoyantSimpleFoam and collect T_max and dp in
results/sweep_results.csv. Keeping one template means a fix to a dictionary
reaches all 20 cases.

Heat flux on the three components: q1 = q_ref, q2 = 0.6 q_ref, q3 = 0.3 q_ref.
I gave the components different loads because real electronics do not dissipate
the same power in every part.

    python3 scripts/run_sweep.py                      # 5 x 4 = 20 cases
    python3 scripts/run_sweep.py --velocities 0.5 --loads 300     # one case
    python3 scripts/run_sweep.py --dry-run            # only prepare the cases
    python3 scripts/run_sweep.py --jobs 4             # 4 cases in parallel
"""
import argparse
import csv
import re
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from parse_foam import case_metrics  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "cabinet_case"
RUNS = TEMPLATE / "runs"
RESULTS = ROOT / "results"
Q_FACTORS = (1.0, 0.6, 0.3)
T_INLET = 293.15


def case_name(u, q):
    return f"u{u:g}_q{q:g}"


END_TIME = None  # set by --end-time; I use it for short smoke tests and longer convergence tests


def prepare(u, q):
    d = RUNS / case_name(u, q)
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)
    shutil.copytree(TEMPLATE / "constant", d / "constant")
    shutil.copytree(TEMPLATE / "system", d / "system")
    if END_TIME is not None:
        cd = d / "system" / "controlDict"
        cd.write_text(re.sub(r"(?m)^endTime\s+\S+;", f"endTime         {END_TIME};", cd.read_text()))
    (d / "0").mkdir()
    subs = {"@UIN@": f"{u:g}", "@Q1@": f"{q * Q_FACTORS[0]:g}",
            "@Q2@": f"{q * Q_FACTORS[1]:g}", "@Q3@": f"{q * Q_FACTORS[2]:g}"}
    for f in (TEMPLATE / "0.orig").iterdir():
        text = f.read_text()
        for k, v in subs.items():
            text = text.replace(k, v)
        (d / "0" / f.name).write_text(text)
    return d


def execute(d):
    with open(d / "log.solver", "w") as fh:
        subprocess.run(["buoyantSimpleFoam"], cwd=d, stdout=fh, stderr=subprocess.STDOUT, check=True)


def run_case(task):
    u, q, dry = task
    d = prepare(u, q)
    shutil.copytree(TEMPLATE / "constant" / "polyMesh", d / "constant" / "polyMesh",
                    dirs_exist_ok=True)
    if dry:
        return None
    try:
        execute(d)
        row = case_metrics(d)
    except Exception as exc:  # one failed case must not kill the other 19; I print it and go on
        print(f"FAILED {case_name(u, q)}: {exc}", flush=True)
        return None
    row.update({"case": case_name(u, q), "u_in": u, "q_ref": q})
    print(f"done {case_name(u, q)}: T_max = {row['T_max_K']:.2f} K, "
          f"dp = {row['dp_Pa']:.3f} Pa, converged = {row['converged']}", flush=True)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--velocities", type=float, nargs="+", default=[0.2, 0.35, 0.5, 0.75, 1.0],
                    help="inlet velocity in m/s")
    ap.add_argument("--loads", type=float, nargs="+", default=[100, 200, 400, 600],
                    help="reference heat flux q_ref on comp1 in W/m2")
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--end-time", type=int, default=None,
                    help="override the iteration limit (smoke test, e.g. 200)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    global END_TIME
    END_TIME = args.end_time

    if not (TEMPLATE / "constant" / "polyMesh").exists():
        sys.exit("No mesh found. Run cabinet_case/Allmesh first.")

    tasks = [(u, q, args.dry_run) for u in args.velocities for q in args.loads]
    RUNS.mkdir(exist_ok=True)
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        rows = [r for r in pool.map(run_case, tasks) if r]

    if args.dry_run:
        print(f"prepared {len(tasks)} cases in {RUNS}")
        return
    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / "sweep_results.csv"
    fields = ["case", "u_in", "q_ref", "T_max_K", "dp_Pa", "end_time_s", "converged", "wall_time_s"]
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)}/{len(tasks)} cases succeeded, results in {out}")


if __name__ == "__main__":
    main()
