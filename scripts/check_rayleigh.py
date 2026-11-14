#!/usr/bin/env python3
"""Check that the cavity in validation_cavity/ really has Ra = 1e5.

I run this before every cavity simulation. If Ra is wrong, the Nusselt number
cannot match the benchmark no matter how good the mesh is, so I want to rule that
out first and cheaply, without OpenFOAM.

Ra = g * beta * dT * L^3 / (nu * alpha). I take beta = 1/T_mean because the fluid
is an ideal gas, nu = mu / rho, alpha = nu / Pr and rho = p / (R T_mean).
I also check that L in system/meshParams equals L in case_params.json, because
the geometry and the post-processing must use the same length.
"""
import json
import re
import sys
from pathlib import Path

R_UNIVERSAL = 8314.462  # J/(kmol K)


def rayleigh(params):
    t_mean = 0.5 * (params["T_hot"] + params["T_cold"])
    d_t = params["T_hot"] - params["T_cold"]
    r_spec = R_UNIVERSAL / params["molWeight"]
    rho = params["p"] / (r_spec * t_mean)
    nu = params["mu"] / rho
    alpha = nu / params["Pr"]
    beta = 1.0 / t_mean
    return params["g"] * beta * d_t * params["L"] ** 3 / (nu * alpha)


def length_for_ra(params, ra):
    """Cavity edge length that gives the requested Rayleigh number.
    I used this to choose L = 0.10116 m instead of rounding the cavity to 0.1 m."""
    p = dict(params, L=1.0)
    return (ra / rayleigh(p)) ** (1.0 / 3.0)


def main(case_dir="validation_cavity"):
    case = Path(case_dir)
    params = json.loads((case / "case_params.json").read_text())
    ra = rayleigh(params)
    print(f"Ra from case_params.json (L = {params['L']} m): {ra:.1f}")
    print(f"L for Ra = {params['Ra_target']:.0f}: {length_for_ra(params, params['Ra_target']):.5f} m")

    mesh_params = (case / "system" / "meshParams").read_text()
    match = re.search(r"^\s*L\s+([0-9.eE+-]+)\s*;", mesh_params, re.M)
    if not match:
        print("WARNING: could not find L in system/meshParams")
        return 1
    if abs(float(match.group(1)) - params["L"]) > 1e-9:
        print(f"MISMATCH: meshParams L = {match.group(1)}, case_params.json L = {params['L']}")
        return 1
    print("meshParams and case_params.json agree.")
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:2]))
