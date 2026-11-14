#!/usr/bin/env python3
"""Average Nusselt number of the hot wall from the wallHeatFlux function object.

I take the heat flow Q through the wall straight from the solver's own function
object instead of differentiating the temperature field myself, so the number
uses the same discretisation as the simulation.

The function object writes postProcessing/wallHeatFlux1/<time>/wallHeatFlux.dat
with the columns: Time  patch  min  max  integral   (integral in W).
For a 2D cavity with out-of-plane thickness `depth`:

    Q = Nu * kappa * dT / L * (L * depth)   ->   Nu = Q / (kappa * dT * depth)

L cancels, which is why only `depth` appears below.

I also print the hot/cold heat balance. In a closed cavity every watt entering at
the hot wall must leave at the cold wall, so any imbalance tells me the run is
not converged, independent of the residuals.
Usage: extract_nusselt.py <case_dir>
"""
import json
import sys
from pathlib import Path


def read_integrals(case_dir):
    """Return {patch: integral} from the latest wallHeatFlux.dat file.
    I read the last row per patch because it is the converged end of a steady run."""
    files = list(Path(case_dir).glob("postProcessing/wallHeatFlux1/*/wallHeatFlux.dat"))
    if not files:
        raise FileNotFoundError(
            "no postProcessing/wallHeatFlux1/*/wallHeatFlux.dat found; did buoyantSimpleFoam finish and write at least one time directory?"
        )
    latest = max(files, key=lambda f: float(f.parent.name))
    last = {}
    for line in latest.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        tokens = line.split()
        # I look the patch up by name instead of by column index, because I was not sure
        # that every OpenFOAM release writes the columns in the same order.
        name = next((t for t in tokens if t in ("hot", "cold")), None)
        if name is None:
            continue
        last[name] = float(tokens[-1])
    return last


def nusselt(q_integral, params):
    # Thermal conductivity from mu, Cp and Pr, because the thermo model is
    # hConst with a constant Prandtl number (kappa = mu Cp / Pr).
    kappa = params["mu"] * params["Cp"] / params["Pr"]
    d_t = params["T_hot"] - params["T_cold"]
    return abs(q_integral) / (kappa * d_t * params["depth"])


def main(case_dir="."):
    params = json.loads((Path(case_dir) / "case_params.json").read_text())
    integrals = read_integrals(case_dir)
    nu_hot = nusselt(integrals["hot"], params)
    nu_cold = nusselt(integrals["cold"], params)
    balance = abs(abs(integrals["hot"]) - abs(integrals["cold"])) / abs(integrals["hot"])
    err = 100.0 * (nu_hot - params["Nu_ref"]) / params["Nu_ref"]
    print(f"Nu_hot  = {nu_hot:.4f}")
    print(f"Nu_cold = {nu_cold:.4f}")
    print(f"heat balance |Q_hot| vs |Q_cold|: {100 * balance:.3f} %")
    print(f"Nu_ref  = {params['Nu_ref']}  ->  error {err:+.2f} %")
    return nu_hot


if __name__ == "__main__":
    main(*sys.argv[1:2])
