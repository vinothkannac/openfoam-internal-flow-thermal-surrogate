#!/usr/bin/env python3
"""Check y+ on the cabinet walls and components for a few representative cases.

I use wall functions, so y+ must sit in a sensible range (roughly 30 to 300 for the
log-law; lower values are not wrong with the blended wall functions of OpenFOAM, but
I want to know where I am). I do not run all 20 cases: I pick the corners and the
middle of the sweep, which bracket the y+ range because y+ scales with the wall
shear, and that scales with the inlet velocity.

    python3 scripts/check_yplus.py                       # default 4 cases, writes results/yplus.csv
    python3 scripts/check_yplus.py --cases u0.2_q400     # one case

It reuses run_sweep.prepare(), so the cases are identical to the sweep ones. It never
touches results/sweep_results.csv.
"""
import argparse
import csv
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import run_sweep  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ["u0.2_q400", "u0.5_q400", "u1_q100", "u1_q600"]
LINE = re.compile(r"patch\s+(\S+)\s+y\+\s*:\s*min\s*=\s*(\S+),\s*max\s*=\s*(\S+),\s*average\s*=\s*(\S+)")


def parse_case(name):
    m = re.fullmatch(r"u([\d.]+)_q([\d.]+)", name)
    if not m:
        raise ValueError(f"bad case name {name}")
    return float(m.group(1)), float(m.group(2))


def run_one(name):
    u, q = parse_case(name)
    d = run_sweep.prepare(u, q)
    # run_sweep.prepare() does not copy the mesh; run_sweep.run_case() does it, so I do the same
    import shutil
    shutil.copytree(run_sweep.TEMPLATE / "constant" / "polyMesh", d / "constant" / "polyMesh",
                    dirs_exist_ok=True)
    run_sweep.execute(d)
    with open(d / "log.yPlus", "w") as fh:
        subprocess.run(["buoyantSimpleFoam", "-postProcess", "-func", "yPlus", "-latestTime"],
                       cwd=d, stdout=fh, stderr=subprocess.STDOUT, check=True)
    rows = []
    for patch, lo, hi, avg in LINE.findall((d / "log.yPlus").read_text()):
        rows.append({"case": name, "patch": patch, "yplus_min": float(lo),
                     "yplus_max": float(hi), "yplus_avg": float(avg)})
    print(f"done {name}: {len(rows)} patches", flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", nargs="+", default=DEFAULT)
    ap.add_argument("--jobs", type=int, default=2)
    args = ap.parse_args()
    if not (run_sweep.TEMPLATE / "constant" / "polyMesh").exists():
        sys.exit("No mesh found. Run cabinet_case/Allmesh first.")
    run_sweep.RUNS.mkdir(exist_ok=True)
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        rows = [r for rs in pool.map(run_one, args.cases) for r in rs]
    out = ROOT / "results" / "yplus.csv"
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["case", "patch", "yplus_min", "yplus_max", "yplus_avg"])
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
