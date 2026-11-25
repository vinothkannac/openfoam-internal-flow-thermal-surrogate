#!/usr/bin/env python3
"""Radiation study (roadmap item 10, radiation part).

The sweep has no radiation: the components give off a fixed heat flux to the air and the shell is
adiabatic. In a real cabinet part of that heat leaves the components as thermal radiation, hits the
shell and is carried away by the air from there. Here I add the finite-volume discrete ordinates
model (fvDOM) of buoyantSimpleFoam to the same 2D case and see how T_max and dp change.

What I set up (all of it is my own assumption, not data of a real product):
  - air is transparent (absorptionEmissionModel none), so only surfaces exchange radiation;
  - the components and the shell have the same emissivity, 0.9 (painted metal) or 0.3 (bare metal);
  - the inlet and outlet openings are black (emissivity 1) and sit at the temperature of the boundary
    condition, so they act as a sink for the radiation that goes through them;
  - the shell stays adiabatic towards the outside, but it can now take up radiation and give the heat
    to the air by convection. For that I changed its T condition from zeroGradient to
    externalWallHeatFluxTemperature with q = 0 and the radiative flux qr switched on.

I use the same steady settings as the sweep (8000 iterations, upwind), so the result without radiation
is the sweep itself and serves as the control.

    python3 scripts/radiation_study.py --jobs 6
    python3 scripts/radiation_study.py --end-time 50 --cases 1.0:100 --emissivities 0.9   # smoke test
    python3 scripts/radiation_study.py --collect-only
"""
import argparse
import re
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import run_sweep  # noqa: E402
from parse_foam import case_metrics, patch_values, time_dirs  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "cabinet_case" / "runs_radiation"
OUT = ROOT / "results" / "radiation_study.csv"
SWEEP = ROOT / "results" / "sweep_results.csv"
T_INLET = 293.15
CASES = [(0.5, 400), (1.0, 100), (1.0, 600)]
EMISSIVITIES = [0.9, 0.3]

HEADER = """FoamFile
{
    version     2.0;
    format      ascii;
    class       %s;
    object      %s;
}
"""

RADIATION_PROPERTIES = HEADER % ("dictionary", "radiationProperties") + """
radiation       on;
radiationModel  fvDOM;

fvDOMCoeffs
{
    nPhi        3;          // 4 * 3 = 12 directions in the 2D plane
    nTheta      0;          // not used in 2D
    tolerance   1e-3;
    maxIter     10;
}

solverFreq      10;         // solve the radiation every 10th flow iteration

absorptionEmissionModel none;   // transparent air
scatterModel    none;
sootModel       none;
"""

BOUNDARY_RADIATION = HEADER % ("dictionary", "boundaryRadiationProperties") + """
"(walls|comp1|comp2|comp3)"
{
    type            lookup;
    emissivity      %g;
    absorptivity    %g;
    transmissivity  0;
}

"(inlet|outlet)"
{
    type            lookup;
    emissivity      1;
    absorptivity    1;
    transmissivity  0;
}
"""

G_FIELD = HEADER % ("volScalarField", "G") + """
dimensions      [1 0 -3 0 0 0 0];
internalField   uniform 0;
boundaryField
{
    "(walls|inlet|outlet|comp1|comp2|comp3)"
    {
        type            MarshakRadiation;
        emissivityMode  lookup;
        emissivity      uniform 1;
        value           uniform 0;
    }
    "(front|back)"
    {
        type            empty;
    }
}
"""

IDEFAULT_FIELD = HEADER % ("volScalarField", "IDefault") + """
dimensions      [1 0 -3 0 0 0 0];
internalField   uniform 0;
boundaryField
{
    "(walls|inlet|outlet|comp1|comp2|comp3)"
    {
        type            greyDiffusiveRadiation;
        value           uniform 0;
    }
    "(front|back)"
    {
        type            empty;
    }
}
"""

# the shell was zeroGradient (adiabatic); with radiation it has to be able to take up the radiative flux
WALL_ADIABATIC = "type            zeroGradient;       // adiabatic cabinet shell"
WALL_WITH_QR = """type            externalWallHeatFluxTemperature;
        mode            flux;
        q               uniform 0;          // still adiabatic towards the outside
        kappaMethod     fluidThermo;
        qr              qr;
        value           uniform 293.15;"""


def name_of(u, q, eps):
    return f"{run_sweep.case_name(u, q)}_eps{eps:g}"


def add_radiation(d, eps):
    """Write the radiation files into the case directory d and switch on qr in the T boundary conditions."""
    (d / "constant" / "radiationProperties").write_text(RADIATION_PROPERTIES)
    (d / "constant" / "boundaryRadiationProperties").write_text(BOUNDARY_RADIATION % (eps, eps))
    (d / "0" / "G").write_text(G_FIELD)
    (d / "0" / "IDefault").write_text(IDEFAULT_FIELD)

    t = (d / "0" / "T").read_text()
    # the three components already use externalWallHeatFluxTemperature, so I only add the qr line.
    # I do this first, because the new wall condition below contains the same text.
    t, n = re.subn(r"(mode\s+flux;\n)(\s+q\s+uniform)", r"\1        qr              qr;\n\2", t)
    if n != 3:
        raise RuntimeError(f"expected 3 component conditions in 0/T, found {n}")
    if WALL_ADIABATIC not in t:
        raise RuntimeError("did not find the adiabatic wall condition in 0/T")
    t = t.replace(WALL_ADIABATIC, WALL_WITH_QR)
    (d / "0" / "T").write_text(t)

    # solver settings for the radiation intensities (same as the OpenFOAM fvDOM tutorial)
    sol = d / "system" / "fvSolution"
    s = sol.read_text()
    s = s.replace("solvers\n{", """solvers
{
    Ii
    {
        solver              GAMG;
        tolerance           1e-4;
        relTol              0;
        smoother            symGaussSeidel;
        maxIter             5;
        nPostSweeps         1;
    }
""", 1)
    s = s.replace('        ".*"            0.3;', '        ".*"            0.3;\n        "ILambda.*"     0.7;')
    sol.write_text(s)

    sch = d / "system" / "fvSchemes"
    c = sch.read_text()
    c = c.replace("    div(phi,k)", "    div(Ji,Ii_h)    bounded Gauss linearUpwind grad(Ii_h);\n    div(phi,k)", 1)
    sch.write_text(c)


def run_one(task):
    (u, q), eps = task
    name = name_of(u, q, eps)
    d = run_sweep.prepare(u, q, root=RUNS / name)
    shutil.copytree(run_sweep.TEMPLATE / "constant" / "polyMesh", d / "constant" / "polyMesh", dirs_exist_ok=True)
    add_radiation(d, eps)
    run_sweep.execute(d)
    print("finished", name, flush=True)


def face_areas(mesh_dir, patch):
    """Areas of the faces of one boundary patch, computed from constant/polyMesh (points, faces, boundary)."""
    def read(name):
        return re.sub(r"//.*", "", (mesh_dir / name).read_text())
    pts_text = read("points")
    n = int(re.search(r"(?m)^(\d+)\s*\n\(", pts_text).group(1))
    pts = np.array(re.findall(r"\(([-\d.eE+]+) ([-\d.eE+]+) ([-\d.eE+]+)\)", pts_text.split("(", 1)[1]), float)[:n]
    faces = [[int(i) for i in f.split()] for f in re.findall(r"\d+\(([\d ]+)\)", read("faces"))]
    m = re.search(patch + r"\s*\{[^}]*?nFaces\s+(\d+);\s*startFace\s+(\d+);", read("boundary"))
    n_faces, start = int(m.group(1)), int(m.group(2))
    areas = []
    for f in faces[start:start + n_faces]:
        # a boundary face is a flat polygon: I sum the triangles from its first point
        vec = sum(np.cross(pts[f[i]] - pts[f[0]], pts[f[i + 1]] - pts[f[0]]) for i in range(1, len(f) - 1)) / 2
        areas.append(np.linalg.norm(vec))
    return np.array(areas)


def radiation_balance(d, q):
    """Net radiative heat (W) of each group of patches, from the qr field of the last time step.
    qr is the radiative flux into the air; it is negative where a patch loses heat by radiation.
    I integrate it over each patch myself, because postProcess gave me empty values for qr."""
    mesh_dir = run_sweep.TEMPLATE / "constant" / "polyMesh"
    last = max(time_dirs(d))
    watts = {}
    for patch in ["comp1", "comp2", "comp3", "walls", "inlet", "outlet"]:
        qr = np.array(patch_values(last / "qr", patch), float)
        watts[patch] = float((qr * face_areas(mesh_dir, patch)).sum())
    q_in = sum(q * f * face_areas(mesh_dir, c).sum() for c, f in zip(["comp1", "comp2", "comp3"], run_sweep.Q_FACTORS))
    return {"q_in_W": q_in,
            "rad_components_W": watts["comp1"] + watts["comp2"] + watts["comp3"],
            "rad_walls_W": watts["walls"],
            "rad_openings_W": watts["inlet"] + watts["outlet"],
            "rad_sum_W": sum(watts.values())}


def collect():
    sw = pd.read_csv(SWEEP).set_index("case")
    rows = []
    for (u, q) in CASES:
        base = sw.loc[run_sweep.case_name(u, q)]
        for eps in EMISSIVITIES:
            d = RUNS / name_of(u, q, eps) / run_sweep.case_name(u, q)
            if not d.exists():
                continue
            m = case_metrics(d)
            rows.append({"case": run_sweep.case_name(u, q), "u_in": u, "q_ref": q, "emissivity": eps,
                         "T_max_K": m["T_max_K"], "dT_max_K": m["T_max_K"] - T_INLET, "dp_Pa": m["dp_Pa"],
                         "converged": m["converged"], "sweep_T_max_K": base["T_max_K"], "sweep_dp_Pa": base["dp_Pa"],
                         "dT_vs_sweep_K": m["T_max_K"] - base["T_max_K"],
                         "dp_vs_sweep_pct": 100 * (m["dp_Pa"] / base["dp_Pa"] - 1),
                         **radiation_balance(d, q)})
    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)
    print(df.to_string(index=False))


def main():
    global CASES, EMISSIVITIES
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--end-time", type=int, default=None, help="iteration limit, for smoke tests")
    ap.add_argument("--cases", default=None, help="comma list of u:q, e.g. 1.0:100,0.5:400")
    ap.add_argument("--emissivities", default=None, help="comma list, e.g. 0.9,0.3")
    ap.add_argument("--collect-only", action="store_true")
    a = ap.parse_args()
    if a.cases:
        CASES = [tuple(float(x) for x in c.split(":")) for c in a.cases.split(",")]
        CASES = [(u, int(q)) for u, q in CASES]
    if a.emissivities:
        EMISSIVITIES = [float(x) for x in a.emissivities.split(",")]
    run_sweep.END_TIME = a.end_time
    if not a.collect_only:
        tasks = [(c, e) for c in CASES for e in EMISSIVITIES]
        with ThreadPoolExecutor(max_workers=a.jobs) as ex:
            list(ex.map(run_one, tasks))
    if a.end_time is None:
        collect()


if __name__ == "__main__":
    main()
