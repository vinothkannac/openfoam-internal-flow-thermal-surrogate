#!/usr/bin/env python3
"""Run the 3-mesh study of the validation cavity and report Nu, error and GCI.

I run three meshes (32, 64, 128 cells per side) with a constant refinement ratio
of 2, which is the cleanest case for the GCI procedure. Each mesh runs in its own
copy under validation_cavity/runs/N<n>, so the base case is never modified and the
study can be repeated from scratch. Figures are made by scripts/make_figures.py.

Needs an OpenFOAM environment (blockMesh, buoyantSimpleFoam, ...) on PATH.
    python3 scripts/mesh_study.py                 # N = 32 64 128
    python3 scripts/mesh_study.py --cells 24 48 96
    python3 scripts/mesh_study.py --reuse         # only re-read finished runs
"""
import argparse
import csv
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from extract_nusselt import nusselt, read_integrals  # noqa: E402
from gci import gci  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CASE = ROOT / "validation_cavity"
RESULTS = ROOT / "results"


def run(cmd, cwd, log):
    with open(cwd / log, "w") as fh:
        subprocess.run(cmd, cwd=cwd, stdout=fh, stderr=subprocess.STDOUT, check=True)


def prepare(n):
    run_dir = CASE / "runs" / f"N{n}"
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)
    for sub in ("0", "constant", "system"):
        shutil.copytree(CASE / sub, run_dir / sub, ignore=shutil.ignore_patterns("polyMesh"))
    shutil.copy(CASE / "case_params.json", run_dir)
    mp = run_dir / "system" / "meshParams"
    mp.write_text(re.sub(r"^N\s+\d+\s*;", f"N       {n};", mp.read_text(), flags=re.M))
    return run_dir


def execute(run_dir):
    run(["blockMesh"], run_dir, "log.blockMesh")
    run(["checkMesh"], run_dir, "log.checkMesh")
    run(["buoyantSimpleFoam"], run_dir, "log.solver")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", type=int, nargs=3, default=[32, 64, 128],
                    help="cells per side, coarse to fine")
    ap.add_argument("--reuse", action="store_true", help="skip the runs, only evaluate")
    args = ap.parse_args()

    params = json.loads((CASE / "case_params.json").read_text())
    rows = []
    for n in args.cells:
        run_dir = CASE / "runs" / f"N{n}"
        if not args.reuse:
            run_dir = prepare(n)
            print(f"running N = {n} ...", flush=True)
            execute(run_dir)
        nu = nusselt(read_integrals(run_dir)["hot"], params)
        err = 100 * (nu - params["Nu_ref"]) / params["Nu_ref"]
        rows.append({"N": n, "cells_2d": n * n, "Nu": nu, "error_pct": err})
        print(f"N = {n:4d}  Nu = {nu:.4f}  error vs ref = {err:+.2f} %")

    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / "cavity_mesh_study.csv"
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["N", "cells_2d", "Nu", "error_pct"])
        w.writeheader()
        w.writerows(rows)

    # I keep the rows coarse to fine; the GCI formulas want f1 = finest
    c, m, f = rows
    g = gci(f["Nu"], m["Nu"], c["Nu"], 1 / f["N"], 1 / m["N"], 1 / c["N"])
    print("\nGrid convergence (Celik et al. 2008):")
    for k, v in g.items():
        print(f"  {k:13s} {v}")
    (RESULTS / "cavity_gci.json").write_text(json.dumps(g, indent=2))


if __name__ == "__main__":
    main()
