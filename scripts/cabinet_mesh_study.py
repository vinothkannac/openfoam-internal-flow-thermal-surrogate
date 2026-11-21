#!/usr/bin/env python3
"""Mesh study for the cabinet: a coarser and a finer mesh for a few representative cases, with a GCI.

The cavity has a three-mesh study and a GCI; the cabinet had one mesh only. Here I mesh the
same geometry at three sizes (make_cabinet_mesh.py --scale 1.5 / 1.0 / 0.7, so roughly 10k, 23k
and 46k cells), run the same four cases on each with the unchanged sweep settings (8000
iterations, the template in cabinet_case/system), and apply the Celik et al. (2008) procedure
of gci.py to T_max and dp.

Choices I made, and why:
  * The representative cases are four that met the convergence flag in the main sweep without
    the lower-relaxation continuation: slow/cool, medium/hot, fast/cool, fast/hot.
  * The GCI is applied to the temperature RISE T_max - 293.15 K, not to T_max in kelvin: the
    surrogate predicts the rise, and a percentage of an absolute temperature hides the error.
  * Representative cell size h = N_cells^(-1/2), because the mesh is 2D (one cell thick) and
    graded, so there is no single N per side as in the cavity.
  * The medium mesh is re-run here with the same code path as the other two, so the three
    results are like for like. The value from results/sweep_results.csv is printed next to it
    as a reproducibility check.
  * T_max is the maximum face value on the components, so it is a point-like quantity. It can
    be noisier under refinement than an integral one; I say so in the README.

    python3 scripts/cabinet_mesh_study.py --jobs 4
    python3 scripts/cabinet_mesh_study.py --collect-only      # runs already done, only evaluate
    python3 scripts/cabinet_mesh_study.py --end-time 200 --scales 1.5 --cases u0.5_q400   # smoke test
"""
import argparse
import csv
import json
import re
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import gci as gci_mod  # noqa: E402
import run_sweep  # noqa: E402
from parse_foam import case_metrics, monitor_drift  # noqa: E402
from check_yplus import parse_case  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CASE_DIR = ROOT / "cabinet_case"
STUDY = CASE_DIR / "mesh_study"          # one mesh per scale
RUNS = CASE_DIR / "runs_meshstudy"       # one run directory per (scale, case)
RESULTS = ROOT / "results"
SCALES = {"coarse": 1.5, "medium": 1.0, "fine": 0.7}
DEFAULT_CASES = ["u0.35_q200", "u0.5_q400", "u1_q100", "u1_q600"]
T_INLET = run_sweep.T_INLET


def mesh_dir(scale):
    return STUDY / f"scale_{scale:g}"


def build_mesh(scale):
    """Gmsh -> gmshToFoam -> patch types -> checkMesh in its own directory. Returns cell count."""
    import make_cabinet_mesh
    d = mesh_dir(scale)
    if (d / "constant" / "polyMesh" / "boundary").exists() and (d / "log.checkMesh").exists():
        return cell_count(d)
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)
    shutil.copytree(CASE_DIR / "system", d / "system")
    make_cabinet_mesh.build(str(d / "cabinet.msh"), scale)
    for cmd, log in ((["gmshToFoam", "cabinet.msh"], "log.gmshToFoam"),
                     ([sys.executable, str(ROOT / "scripts" / "fix_boundary.py"),
                       "constant/polyMesh/boundary"], "log.fix_boundary"),
                     (["checkMesh"], "log.checkMesh")):
        with open(d / log, "w") as fh:
            r = subprocess.run(cmd, cwd=d, stdout=fh, stderr=subprocess.STDOUT)
        if r.returncode and cmd[0] != "checkMesh":
            raise RuntimeError(f"{cmd[0]} failed for scale {scale}, see {d / log}")
    return cell_count(d)


def cell_count(d):
    m = re.search(r"cells:\s+(\d+)", (Path(d) / "log.checkMesh").read_text())
    return int(m.group(1))


def run_one(task):
    label, scale, case, end_time = task
    u, q = parse_case(case)
    root = RUNS / label
    d = run_sweep.prepare(u, q, root=root)
    if end_time:
        cd = d / "system" / "controlDict"
        cd.write_text(re.sub(r"(?m)^endTime\s+\S+;", f"endTime         {end_time};", cd.read_text()))
    shutil.copytree(mesh_dir(scale) / "constant" / "polyMesh", d / "constant" / "polyMesh",
                    dirs_exist_ok=True)
    try:
        run_sweep.execute(d)
    except Exception as exc:
        print(f"FAILED {label}/{case}: {exc}", flush=True)
        return
    print(f"done {label}/{case}", flush=True)


def collect(labels, cases, cells):
    rows = []
    for label in labels:
        for case in cases:
            d = RUNS / label / case
            if not (d / "log.solver").exists():
                continue
            try:
                m = case_metrics(d)
            except Exception as exc:
                print(f"cannot read {label}/{case}: {exc}")
                continue
            drift = monitor_drift(d, window=500) or {}
            u, q = parse_case(case)
            rows.append({"case": case, "mesh": label, "scale": SCALES[label], "cells": cells[label],
                         "T_max_K": m["T_max_K"], "dT_max_K": m["T_max_K"] - T_INLET,
                         "dp_Pa": m["dp_Pa"], "converged": m["converged"],
                         "end_time_s": m["end_time_s"], "wall_time_s": m["wall_time_s"],
                         "T_max_spread_last500_K": drift.get("T_max_spread_K"),
                         "dp_spread_last500_Pa": drift.get("dp_spread_Pa")})
    return rows


def gci_for(rows, case, key):
    """GCI of one quantity of one case; returns a dict that always says why when it fails."""
    by = {r["mesh"]: r for r in rows if r["case"] == case}
    if not all(k in by for k in SCALES):
        return {"status": "missing a mesh"}
    f = [by[k][key] for k in ("fine", "medium", "coarse")]
    h = [by[k]["cells"] ** -0.5 for k in ("fine", "medium", "coarse")]
    out = {"fine": f[0], "medium": f[1], "coarse": f[2],
           "diff_fine_medium_pct": 100 * (f[0] - f[1]) / f[1],
           "diff_medium_coarse_pct": 100 * (f[1] - f[2]) / f[2],
           "r21": h[1] / h[0], "r32": h[2] / h[1]}
    try:
        g = gci_mod.gci(*f, *h)
    except (ValueError, ZeroDivisionError) as exc:
        out["status"] = f"GCI undefined: {exc}"
        return out
    out.update(g)
    # Celik et al. ask for 0.5 < p < 2-ish sanity; a very large or tiny order means three
    # points do not show an asymptotic trend (noise or oscillation), so I flag it
    out["status"] = "ok" if g["monotonic"] and 0.5 <= g["p"] <= 4 else \
        "not asymptotic: oscillatory or order outside 0.5-4, do not trust the GCI"
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", nargs="+", default=DEFAULT_CASES)
    ap.add_argument("--scales", nargs="+", type=float, default=list(SCALES.values()))
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--end-time", type=int, default=None, help="iteration limit for smoke tests")
    ap.add_argument("--collect-only", action="store_true")
    args = ap.parse_args()

    labels = [k for k, v in SCALES.items() if v in args.scales]
    cells = {}
    for label in labels:
        cells[label] = build_mesh(SCALES[label])
        print(f"mesh {label} (scale {SCALES[label]}): {cells[label]} cells", flush=True)
    if not args.collect_only:
        # fine-mesh cases first: they are the slowest, so they should not be the last to start
        tasks = [(lab, SCALES[lab], c, args.end_time) for lab in reversed(labels) for c in args.cases]
        with ThreadPoolExecutor(max_workers=args.jobs) as pool:
            list(pool.map(run_one, tasks))
    rows = collect(labels, args.cases, cells)
    RESULTS.mkdir(exist_ok=True)
    if args.end_time:
        print("smoke test: not writing results files")
        for r in rows:
            print(r)
        return
    fields = list(rows[0].keys())
    with open(RESULTS / "cabinet_mesh_study.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    # reproducibility check of the medium mesh against the sweep
    sweep = {}
    with open(RESULTS / "sweep_results.csv") as fh:
        for r in csv.DictReader(fh):
            sweep[r["case"]] = float(r["T_max_K"])
    out = {"cells": cells, "scales": SCALES, "cases": {}}
    for case in args.cases:
        entry = {key: gci_for(rows, case, key) for key in ("dT_max_K", "dp_Pa")}
        med = [r for r in rows if r["case"] == case and r["mesh"] == "medium"]
        if med and case in sweep:
            entry["medium_vs_sweep_T_max_diff_K"] = med[0]["T_max_K"] - sweep[case]
        out["cases"][case] = entry
    (RESULTS / "cabinet_gci.json").write_text(json.dumps(out, indent=2, default=float))
    for case, e in out["cases"].items():
        for key in ("dT_max_K", "dp_Pa"):
            g = e[key]
            print(case, key, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in g.items()})
    print(f"wrote {RESULTS / 'cabinet_mesh_study.csv'} and {RESULTS / 'cabinet_gci.json'}")


if __name__ == "__main__":
    main()
