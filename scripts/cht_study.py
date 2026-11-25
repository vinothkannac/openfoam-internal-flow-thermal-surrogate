#!/usr/bin/env python3
"""Conjugate heat transfer in the components (roadmap item 10, last part).

In the sweep the three components are holes in the mesh with a fixed heat flux on their surface. Here I
mesh them as solid blocks, give each block a uniform heat source inside and let the heat conduct through
the solid to the air (chtMultiRegionSimpleFoam: one air region and three solid regions, coupled at the
interfaces by the temperature and the heat flux). I compare T_max and dp with the sweep for three cases.

Assumptions (all mine, none of them is data of a real product):
  1. Total power is the same as in the sweep. Each component gets P = q * perimeter * depth, so for
     component 1 P = q_ref * 0.7 m * 0.1 m, and 0.6 and 0.3 times that for components 2 and 3. In the sweep
     this power leaves through the surface; here it is generated evenly over the volume of the block.
  2. Solid material: constant conductivity kappa. I run two values: 200 W/(m K), a metal block such as
     aluminium (the block is almost one temperature), and 5 W/(m K), a poor conductor such as potted
     electronics (large temperature differences inside the block). Density 2719 kg/m3 and Cp 871 J/(kg K)
     are copied from the OpenFOAM tutorial; they do not matter in a steady state.
  3. Perfect thermal contact between solid and air: no contact resistance, no coating.
  4. The same air region as in the sweep: same geometry, in-plane mesh (22 636 cells in the air), wall
     functions, k-omega SST, inlet k and omega, steady settings, upwind, relaxation, 8000 iterations.
     The shell is still adiabatic, there is no radiation, and the case is still 2D (empty front and back).
  5. T_max is the maximum temperature on the air side of the interfaces (the same kind of quantity as in
     the sweep, a face value). I also report the maximum temperature inside the solids.
  6. dp is the same p_rgh difference inlet minus outlet as in the sweep.
  7. The interface condition uses useImplicit false. The OpenFOAM tutorial uses true, but with true only
     about 86 % of the input power reached the air in my case (the enthalpy flux through inlet and outlet
     was 86 % of the power), the same share in every case and for both conductivities. With false it is
     99 %, which is as close as the sweep (99.6 to 99.7 % for the two well-behaved cases). I found this
     with a test on one case (see README), so my first set of CHT results with true was wrong and I discarded it.

    python3 scripts/cht_study.py --jobs 6
    python3 scripts/cht_study.py --end-time 30 --cases 1.0:100 --kappas 200 --jobs 1     # smoke test
    python3 scripts/cht_study.py --collect-only
"""
import argparse
import re
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import run_sweep  # noqa: E402
from parse_foam import patch_values, run_time, time_dirs  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "cabinet_case"
MESH = TEMPLATE / "mesh_cht"
RUNS = TEMPLATE / "runs_cht"
OUT = ROOT / "results" / "cht_study.csv"
SWEEP = ROOT / "results" / "sweep_results.csv"
T_INLET = 293.15
CASES = [(0.5, 400), (1.0, 100), (1.0, 600)]
KAPPAS = [200.0, 5.0]
COMPS = ["comp1", "comp2", "comp3"]
PERIMETER, DEPTH = 0.7, 0.1      # of every component: 2 * (0.2 + 0.15) m, and the depth of the 2D section
SOLID_CP, SOLID_RHO = 871, 2719  # from the OpenFOAM tutorial; only kappa matters in a steady state

HEADER = """FoamFile
{
    version     2.0;
    format      ascii;
    class       %s;
    object      %s;
}
"""

REGION_PROPERTIES = HEADER % ("dictionary", "regionProperties") + """
regions
(
    fluid ( fluid )
    solid ( comp1 comp2 comp3 )
);
"""

CONTROL_DICT = HEADER % ("dictionary", "controlDict") + """
application     chtMultiRegionSimpleFoam;
startFrom       startTime;
startTime       0;
stopAt          endTime;
endTime         %d;
deltaT          1;
writeControl    timeStep;
writeInterval   250;            // the last two write times are compared for convergence
purgeWrite      3;
writeFormat     ascii;          // parse_foam.py reads ASCII fields
writePrecision  8;
writeCompression off;
timeFormat      general;
timePrecision   6;
runTimeModifiable true;
"""

# the top-level fvSchemes and fvSolution of a multi-region case only need to exist
EMPTY_DICT = HEADER % ("dictionary", "%s") + "\n"

SOLID_THERMO = HEADER % ("dictionary", "thermophysicalProperties") + """
thermoType
{
    type            heSolidThermo;
    mixture         pureMixture;
    transport       constIso;
    thermo          hConst;
    equationOfState rhoConst;
    specie          specie;
    energy          sensibleEnthalpy;
}

mixture
{
    specie { nMoles 1; molWeight 26.98; }
    thermodynamics { Hf 0; Sf 0; Cp %g; }
    transport { kappa %g; }
    equationOfState { rho %g; }
}
"""

SOLID_SCHEMES = HEADER % ("dictionary", "fvSchemes") + """
ddtSchemes          { default steadyState; }
divSchemes          { default Gauss linear; }
laplacianSchemes    { default Gauss linear limited 0.33; }
interpolationSchemes { default linear; }
snGradSchemes       { default limited 0.33; }
gradSchemes         { default Gauss linear; }
"""

SOLID_SOLUTION = HEADER % ("dictionary", "fvSolution") + """
solvers
{
    h
    {
        solver          PCG;
        preconditioner  DIC;
        tolerance       1e-8;
        relTol          0.05;
    }
}
SIMPLE { residualControl { default 1e-20; } }
relaxationFactors { equations { h 1; } }
"""

SOLID_SOURCE = HEADER % ("dictionary", "fvOptions") + """
heat_source
{
    type            scalarSemiImplicitSource;
    active          true;
    selectionMode   all;
    volumeMode      absolute;       // the number below is the total power of the block in W
    sources
    {
        h           ( %g 0 );
    }
}
"""

SOLID_T = HEADER % ("volScalarField", "T") + """
dimensions      [0 0 0 1 0 0 0];
internalField   uniform 293.15;
boundaryField
{
    #includeEtc "caseDicts/setConstraintTypes"
    %s_to_fluid
    {
        type            compressible::turbulentTemperatureRadCoupledMixed;
        qrNbr           none;
        qr              none;
        Tnbr            T;
        kappaMethod     solidThermo;
        useImplicit     false;
        value           $internalField;
    }
}
"""

SOLID_P = HEADER % ("volScalarField", "p") + """
dimensions      [1 -1 -2 0 0 0 0];
internalField   uniform 100000;
boundaryField
{
    #includeEtc "caseDicts/setConstraintTypes"
    %s_to_fluid
    {
        type            calculated;
        value           $internalField;
    }
}
"""

FLUID_T_INTERFACE = """    fluid_to_%s
    {
        type            compressible::turbulentTemperatureRadCoupledMixed;
        qrNbr           none;
        qr              none;
        Tnbr            T;
        kappaMethod     fluidThermo;
        useImplicit     false;
        value           uniform 293.15;
    }
"""


def sh(cmd, cwd, log=None):
    """Run a command in `cwd`; stop with an error if it fails. The output goes to a log file."""
    out = open(Path(cwd) / log, "w") if log else subprocess.DEVNULL
    subprocess.run(cmd, cwd=cwd, stdout=out, stderr=subprocess.STDOUT, check=True)


def build_mesh():
    """Gmsh (components as solid volumes) -> OpenFOAM -> one mesh per region. Returns the cell count of the air."""
    if MESH.exists():
        shutil.rmtree(MESH)
    MESH.mkdir(parents=True)
    shutil.copytree(TEMPLATE / "system", MESH / "system")  # checkMesh needs a complete system folder
    sh([sys.executable, str(ROOT / "scripts" / "make_cabinet_mesh.py"), "--out", "cht.msh", "--solids"], MESH, "log.gmsh")
    sh(["gmshToFoam", "cht.msh"], MESH, "log.gmshToFoam")
    sh([sys.executable, str(ROOT / "scripts" / "fix_boundary.py"), "constant/polyMesh/boundary", "--cht"], MESH, "log.fixBoundary")
    sh(["checkMesh"], MESH, "log.checkMesh")
    # every cell zone (fluid, comp1..3) becomes its own region with interface patches 'fluid_to_comp1' etc.
    sh(["splitMeshRegions", "-cellZones", "-overwrite"], MESH, "log.splitMeshRegions")
    return int(re.search(r"nCells:(\d+)", (MESH / "constant" / "fluid" / "polyMesh" / "owner").read_text()).group(1))


def fluid_field(name, text, u):
    """Turn a field file of the sweep (0.orig) into the air-region version: the three component patches
    become the interface patches fluid_to_comp1..3."""
    text = text.replace("@UIN@", f"{u:g}")
    if name == "T":
        # the sweep prescribes a heat flux on comp1..3; here these are coupled interfaces
        text, n = re.subn(r"\n    comp\d\n    \{.*?\n    \}", "", text, flags=re.S)
        if n != 3:
            raise RuntimeError(f"expected 3 component conditions in 0/T, found {n}")
        interfaces = "".join(FLUID_T_INTERFACE % c for c in COMPS)
        text, n = re.subn(r'(\n    "\(front\|back\)")', "\n" + interfaces.rstrip("\n") + r"\1", text)
        if n != 1:
            raise RuntimeError("did not find the front/back entry in 0/T")
    else:
        text = re.sub(r"\bcomp(\d)\b", r"fluid_to_comp\1", text)
    return text


def component_power(q, factor):
    """Power (W) of one component for the sweep's heat flux q (W/m2): the same heat the flux would carry out
    through the four sides of the block (perimeter x depth)."""
    return q * factor * PERIMETER * DEPTH


def prepare(u, q, kappa, end_time):
    name = f"{run_sweep.case_name(u, q)}_k{kappa:g}"
    d = RUNS / name
    if d.exists():
        shutil.rmtree(d)
    (d / "constant").mkdir(parents=True)
    (d / "system").mkdir()
    (d / "constant" / "regionProperties").write_text(REGION_PROPERTIES)
    (d / "constant" / "g").write_text((TEMPLATE / "constant" / "g").read_text())
    (d / "constant" / "hRef").write_text((TEMPLATE / "constant" / "hRef").read_text())
    (d / "system" / "controlDict").write_text(CONTROL_DICT % (end_time or 8000))
    (d / "system" / "fvSchemes").write_text(EMPTY_DICT % "fvSchemes")
    (d / "system" / "fvSolution").write_text(EMPTY_DICT % "fvSolution")

    # air region: the settings of the sweep
    for region in ["fluid"] + COMPS:
        shutil.copytree(MESH / "constant" / region / "polyMesh", d / "constant" / region / "polyMesh")
        (d / "system" / region).mkdir()
        (d / "0" / region).mkdir(parents=True, exist_ok=True)
    for f in ("thermophysicalProperties", "turbulenceProperties"):
        shutil.copy(TEMPLATE / "constant" / f, d / "constant" / "fluid" / f)
    for f in ("fvSchemes", "fvSolution"):
        shutil.copy(TEMPLATE / "system" / f, d / "system" / "fluid" / f)
    for f in (TEMPLATE / "0.orig").iterdir():
        (d / "0" / "fluid" / f.name).write_text(fluid_field(f.name, f.read_text(), u))
    fs = d / "system" / "fluid" / "fvSolution"
    t = fs.read_text()
    # the multi-region solver also solves for the density
    # (and the energy equation of air and solids is solved as one coupled system; the smoothSolver of the
    # sweep hit its 1000-iteration limit on it in every step, so I use the solver of the OpenFOAM tutorial for h)
    t = t.replace("solvers\n{", """solvers
{
    rho
    {
        solver          PCG;
        preconditioner  DIC;
        tolerance       1e-7;
        relTol          0;
    }
    h
    {
        solver          PBiCGStab;
        preconditioner  DILU;
        tolerance       1e-8;
        relTol          0.05;
    }
""", 1)
    fs.write_text(t)

    # solid regions: heat source and conductivity
    for c, factor in zip(COMPS, run_sweep.Q_FACTORS):
        power = component_power(q, factor)
        (d / "constant" / c / "thermophysicalProperties").write_text(SOLID_THERMO % (SOLID_CP, kappa, SOLID_RHO))
        (d / "system" / c / "fvSchemes").write_text(SOLID_SCHEMES)
        (d / "system" / c / "fvSolution").write_text(SOLID_SOLUTION)
        (d / "system" / c / "fvOptions").write_text(SOLID_SOURCE % power)
        (d / "0" / c / "T").write_text(SOLID_T % c)
        (d / "0" / c / "p").write_text(SOLID_P % c)
    return name, d


def run_one(task):
    (u, q), kappa, end_time = task
    name, d = prepare(u, q, kappa, end_time)
    with open(d / "log.solver", "w") as fh:
        subprocess.run(["chtMultiRegionSimpleFoam"], cwd=d, stdout=fh, stderr=subprocess.STDOUT, check=True)
    print("finished", name, flush=True)


def internal_max(field_file):
    """Largest value in the internalField of a scalar field file."""
    text = Path(field_file).read_text()
    m = re.search(r"internalField\s+nonuniform\s+List<scalar>\s*(\d+)\s*\(([^)]*)\)", text, re.S)
    if m:
        return max(float(v) for v in m.group(2).split())
    return float(re.search(r"internalField\s+uniform\s+([^;\s]+);", text).group(1))


def t_max_of(time_dir):
    """(max air-side interface temperature, max temperature inside the solids) at one write time."""
    face = max(max(patch_values(time_dir / "fluid" / "T", f"fluid_to_{c}")) for c in COMPS)
    inside = max(internal_max(time_dir / c / "T") for c in COMPS)
    return face, inside


AIR_CP = 1005  # J/(kg K), the Cp of the air in constant/thermophysicalProperties


def heat_to_air(time_dir):
    """Heat (W) carried out of the cabinet by the air: Cp * sum of (mass flux * (T - T_inlet)) over the inlet
    and outlet faces. In a steady state this must equal the total power of the three components, so it is my
    check that the coupling between the solids and the air does not lose or create heat."""
    total = 0.0
    for patch in ("inlet", "outlet"):
        phi = patch_values(time_dir / "fluid" / "phi", patch)
        temp = patch_values(time_dir / "fluid" / "T", patch)
        if len(temp) == 1:                      # a uniform patch value is stored once
            temp = temp * len(phi)
        total += AIR_CP * sum(f * (x - T_INLET) for f, x in zip(phi, temp))
    return total


def mean_face_t_cht(time_dir):
    """Mean temperature over all air-side interface faces (a plain average over faces, not area-weighted)."""
    vals = [v for c in COMPS for v in patch_values(time_dir / "fluid" / "T", f"fluid_to_{c}")]
    return sum(vals) / len(vals)


def mean_face_t_sweep(case):
    """The same average over the component faces of the sweep case, or None if its run folder is not there."""
    d = run_sweep.RUNS / case
    if not d.exists() or not time_dirs(d):
        return None
    vals = [v for c in COMPS for v in patch_values(time_dirs(d)[-1] / "T", c)]
    return sum(vals) / len(vals)


def collect(cases, kappas):
    sw = pd.read_csv(SWEEP).set_index("case")
    rows = []
    for (u, q) in cases:
        base = sw.loc[run_sweep.case_name(u, q)]
        for kappa in kappas:
            d = RUNS / f"{run_sweep.case_name(u, q)}_k{kappa:g}"
            dirs = time_dirs(d) if d.exists() else []
            if not dirs:
                continue
            face, inside = t_max_of(dirs[-1])
            converged = None
            if len(dirs) >= 2:
                converged = abs(inside - t_max_of(dirs[-2])[1]) < 0.05
            p_in = patch_values(dirs[-1] / "fluid" / "p_rgh", "inlet")
            p_out = patch_values(dirs[-1] / "fluid" / "p_rgh", "outlet")
            dp = sum(p_in) / len(p_in) - sum(p_out) / len(p_out)
            power = sum(component_power(q, f) for f in run_sweep.Q_FACTORS)
            out = heat_to_air(dirs[-1])
            rows.append({"case": run_sweep.case_name(u, q), "u_in": u, "q_ref": q, "kappa": kappa,
                         "T_max_face_K": face, "T_max_solid_K": inside, "T_mean_face_K": mean_face_t_cht(dirs[-1]),
                         "sweep_T_mean_face_K": mean_face_t_sweep(run_sweep.case_name(u, q)), "dp_Pa": dp, "converged": converged,
                         "power_W": power, "heat_to_air_W": out, "heat_balance_pct": 100 * (out / power - 1),
                         "wall_time_s": run_time(d / "log.solver"),
                         "sweep_T_max_K": base["T_max_K"], "sweep_dp_Pa": base["dp_Pa"],
                         "dT_face_vs_sweep_K": face - base["T_max_K"],
                         "dT_solid_vs_sweep_K": inside - base["T_max_K"],
                         "dp_vs_sweep_pct": 100 * (dp / base["dp_Pa"] - 1)})
    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)
    print(df.to_string(index=False))


def main():
    cases, kappas = CASES, KAPPAS
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--end-time", type=int, default=None, help="iteration limit, for smoke tests")
    ap.add_argument("--cases", default=None, help="comma list of u:q, e.g. 1.0:100,0.5:400")
    ap.add_argument("--kappas", default=None, help="comma list of solid conductivities in W/(m K)")
    ap.add_argument("--collect-only", action="store_true")
    ap.add_argument("--keep-mesh", action="store_true", help="do not rebuild the multi-region mesh")
    a = ap.parse_args()
    if a.cases:
        cases = [(float(c.split(":")[0]), int(c.split(":")[1])) for c in a.cases.split(",")]
    if a.kappas:
        kappas = [float(k) for k in a.kappas.split(",")]
    if not a.collect_only:
        if not (a.keep_mesh and MESH.exists()):
            print("air cells:", build_mesh(), flush=True)
        tasks = [(c, k, a.end_time) for c in cases for k in kappas]
        with ThreadPoolExecutor(max_workers=a.jobs) as ex:
            list(ex.map(run_one, tasks))
    if a.end_time is None or a.collect_only:
        collect(cases, kappas)


if __name__ == "__main__":
    main()
