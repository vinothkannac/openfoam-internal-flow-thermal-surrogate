#!/usr/bin/env python3
"""Share of the cabinet cells with downward vertical velocity (Uy < 0), per sweep case.

The cabinet flows upward (inlet slot at the bottom, outlet at the top), so Uy < 0 means
the fluid is moving against the main direction, i.e. recirculating. I use this crude
indicator to check a claim I would otherwise only assume: does low inlet velocity give
more flow separation? Steady cases use the last saved iteration; the four cases that
were re-run transient use the last saved flow time. Results: results/recirculation.csv.
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from parse_foam import time_dirs  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CASES = ROOT / "cabinet_case"


def cell_velocity(field_file):
    text = Path(field_file).read_text()
    m = re.search(r"internalField\s+nonuniform\s+List<vector>\s*\d+\s*\(", text)
    body = text[m.end():]
    body = body[:body.index("\n)\n")]
    return np.array([[float(x) for x in s.split()] for s in re.findall(r"\(([^()]+)\)", body)])


def row(label, case_dir, u_in):
    u = cell_velocity(time_dirs(case_dir)[-1] / "U")
    return {"case": label, "u_in": u_in, "cells_Uy_negative_pct": 100 * (u[:, 1] < 0).mean(),
            "min_Uy_ms": u[:, 1].min(), "max_Uy_ms": u[:, 1].max()}


def main():
    rows = []
    for d in sorted((CASES / "runs").iterdir()):
        rows.append(row(d.name, d, float(re.match(r"u([0-9.]+)_", d.name).group(1))))
    if (CASES / "runs_transient").exists():
        for d in sorted((CASES / "runs_transient").iterdir()):
            rows.append(row(d.name + " (transient)", d, float(re.match(r"u([0-9.]+)_", d.name).group(1))))
    out = pd.DataFrame(rows)
    out.to_csv(ROOT / "results" / "recirculation.csv", index=False)
    print(out.to_string(index=False))
    s = out[~out["case"].str.contains("transient")].groupby("u_in")["cells_Uy_negative_pct"].mean()
    print("\nmean over the 4 heat loads, steady runs:\n", s.round(1).to_string())


main()
