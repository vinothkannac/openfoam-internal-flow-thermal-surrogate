#!/usr/bin/env python3
"""3D slice of the cabinet (roadmap item 10, 3D part).

The sweep is 2D: one cell through the 0.1 m depth and 'empty' front and back, so nothing can happen in the
third direction. Here I keep the same cross-section, the same in-plane mesh, the same depth (so the inlet
area and the total heat are the same as in the sweep) and split the depth into 12 cells, graded from 3 mm at
the side panels to 16 mm in the middle. The front and back become no-slip, adiabatic walls with wall
functions, because a real cabinet has side panels. The components still span the full depth.

So the only difference to the sweep is the third dimension: the friction of the side panels and the flow in
the depth direction. I run three converged sweep cases with the sweep's steady settings (8000 iterations,
upwind) in parallel on 6 cores, one case after the other, and compare T_max and dp with the sweep.

    python3 scripts/slice3d_study.py --np 6
    python3 scripts/slice3d_study.py --end-time 20 --cases 1.0:100 --np 6     # smoke test
    python3 scripts/slice3d_study.py --collect-only
"""
import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import run_sweep  # noqa: E402
from parse_foam import case_metrics, patch_values, time_dirs  # noqa: E402
from radiation_study import face_areas  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
MESH = ROOT / "cabinet_case" / "mesh3d"
RUNS = ROOT / "cabinet_case" / "runs_slice3d"
OUT = ROOT / "results" / "slice3d_study.csv"
SWEEP = ROOT / "results" / "sweep_results.csv"
T_INLET = 293.15
CASES = [(0.5, 400), (1.0, 100), (1.0, 600)]
LAYERS = 12

DECOMPOSE = """FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      decomposeParDict;
}
numberOfSubdomains  %d;
method              scotch;
"""


def sh(cmd, cwd, log=None):
    """Run a command in `cwd`; stop with an error if it fails. The output goes to a log file."""
    out = open(cwd / log, "w") if log else subprocess.DEVNULL
    subprocess.run(cmd, cwd=cwd, stdout=out, stderr=subprocess.STDOUT, check=True)


def build_mesh():
    """Gmsh -> OpenFOAM, with front and back as walls. Returns the cell count."""
    if MESH.exists():
        shutil.rmtree(MESH)
    MESH.mkdir(parents=True)
    shutil.copytree(run_sweep.TEMPLATE / "system", MESH / "system")  # checkMesh needs a complete system folder
    sh([sys.executable, str(ROOT / "scripts" / "make_cabinet_mesh.py"), "--out", "cabinet3d.msh",
        "--layers", str(LAYERS)], MESH, "log.gmsh")
    sh(["gmshToFoam", "cabinet3d.msh"], MESH, "log.gmshToFoam")
    sh([sys.executable, str(ROOT / "scripts" / "fix_boundary.py"),
        "constant/polyMesh/boundary", "--3d"], MESH, "log.fixBoundary")
    sh(["checkMesh"], MESH, "log.checkMesh")
    return cell_count()


def cell_count():
    m = re.search(r"cells:\s+(\d+)", (MESH / "log.checkMesh").read_text())
    return int(m.group(1))


def sides_as_walls(d):
    """In every field file of 0/ the front and back were 'empty'. I add them to the patches that
    already have the wall conditions and delete the empty block."""
    for f in (d / "0").iterdir():
        t = f.read_text()
        if f.name == "T":
            # T lists the shell on its own (adiabatic, zeroGradient): the side panels get the same condition
            t, n = re.subn(r"(\n\s*)walls(\s*\{)", r'\1"(walls|front|back)"\2', t, count=1)
        else:
            # the other fields use groups like "(walls|comp1|comp2|comp3)"; I add front and back to it
            t, n = re.subn(r'"\((inlet\|outlet\|)?walls\|', lambda m: '"(' + (m.group(1) or "") + "front|back|walls|", t, count=1)
        if n != 1:
            raise RuntimeError(f"did not find the wall entry in {f}")
        t, n = re.subn(r'\n\s*"\(front\|back\)"\s*\{\s*type\s+empty;\s*\}', "", t)
        if n != 1:
            raise RuntimeError(f"did not find the empty entry in {f}")
        f.write_text(t)


def run_one(u, q, n_procs):
    d = run_sweep.prepare(u, q, root=RUNS)
    shutil.copytree(MESH / "constant" / "polyMesh", d / "constant" / "polyMesh", dirs_exist_ok=True)
    sides_as_walls(d)
    (d / "system" / "decomposeParDict").write_text(DECOMPOSE % n_procs)
    sh(["decomposePar", "-force"], d, "log.decomposePar")
    with open(d / "log.solver", "w") as fh:
        subprocess.run(["mpirun", "--oversubscribe", "-np", str(n_procs), "buoyantSimpleFoam", "-parallel"],
                       cwd=d, stdout=fh, stderr=subprocess.STDOUT, check=True)
    sh(["reconstructPar", "-newTimes"], d, "log.reconstructPar")
    for proc in d.glob("processor*"):  # the reconstructed result is enough, the pieces take a lot of disk
        shutil.rmtree(proc)
    print("finished", run_sweep.case_name(u, q), flush=True)


def area_weighted_dp(d):
    """The faces on the inlet and outlet have different thicknesses (graded layers), so I weight the
    p_rgh values by face area. run_sweep's plain average would give the thin wall layers too much weight."""
    last = time_dirs(d)[-1]
    p = {}
    for patch in ("inlet", "outlet"):
        a = face_areas(MESH / "constant" / "polyMesh", patch)
        v = patch_values(last / "p_rgh", patch)
        p[patch] = sum(x * w for x, w in zip(v, a)) / a.sum()
    return p["inlet"] - p["outlet"]


def collect():
    sw = pd.read_csv(SWEEP).set_index("case")
    cells = cell_count() if (MESH / "log.checkMesh").exists() else None
    rows = []
    for (u, q) in CASES:
        d = RUNS / run_sweep.case_name(u, q)
        if not d.exists() or not time_dirs(d):
            continue
        m = case_metrics(d)
        base = sw.loc[run_sweep.case_name(u, q)]
        dp = area_weighted_dp(d)
        rows.append({"case": run_sweep.case_name(u, q), "u_in": u, "q_ref": q, "cells": cells,
                     "T_max_K": m["T_max_K"], "dT_max_K": m["T_max_K"] - T_INLET, "dp_Pa": dp,
                     "converged": m["converged"], "wall_time_s": m["wall_time_s"],
                     "sweep_T_max_K": base["T_max_K"], "sweep_dp_Pa": base["dp_Pa"],
                     "dT_vs_sweep_K": m["T_max_K"] - base["T_max_K"],
                     "dp_vs_sweep_pct": 100 * (dp / base["dp_Pa"] - 1)})
    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)
    print(df.to_string(index=False))


def main():
    global CASES
    ap = argparse.ArgumentParser()
    ap.add_argument("--np", type=int, default=6, help="processors per case")
    ap.add_argument("--end-time", type=int, default=None, help="iteration limit, for smoke tests")
    ap.add_argument("--cases", default=None, help="comma list of u:q, e.g. 1.0:100,0.5:400")
    ap.add_argument("--collect-only", action="store_true")
    ap.add_argument("--keep-mesh", action="store_true", help="do not rebuild the 3D mesh")
    a = ap.parse_args()
    if a.cases:
        CASES = [(float(c.split(":")[0]), int(c.split(":")[1])) for c in a.cases.split(",")]
    run_sweep.END_TIME = a.end_time
    if not a.collect_only:
        if not (a.keep_mesh and MESH.exists()):
            print("3D mesh:", build_mesh(), "cells", flush=True)
        for (u, q) in CASES:
            run_one(u, q, a.np)
    if a.end_time is None:
        collect()


if __name__ == "__main__":
    main()
