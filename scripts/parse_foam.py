#!/usr/bin/env python3
"""Minimal readers for ASCII OpenFOAM output: patch values and run time.

I wrote this small reader myself instead of using a mesh-processing library,
because the sweep only needs a few numbers: the maximum temperature on the
component patches, the mean p_rgh on inlet and outlet, a convergence flag from the
last two write times, and the wall-clock time from the solver log. Keeping it
dependency-free also lets me unit-test it without an OpenFOAM installation.
It only reads ASCII output, which is why both controlDicts set writeFormat ascii.
"""
import re
from pathlib import Path


def _block(text, name):
    """Body of `name { ... }` including nested braces, or None."""
    m = re.search(r"(?m)^\s*" + re.escape(name) + r"\s*\{", text)
    if not m:
        return None
    depth, i = 1, m.end()
    while depth and i < len(text):
        depth += {"{": 1, "}": -1}.get(text[i], 0)
        i += 1
    return text[m.end(): i - 1]


def patch_values(field_file, patch):
    """Values of a scalar field on `patch` as a list of floats."""
    text = Path(field_file).read_text()
    boundary = _block(text, "boundaryField")
    if boundary is None:
        raise ValueError(f"no boundaryField in {field_file}")
    body = _block(boundary, patch)
    if body is None:
        raise KeyError(f"patch {patch} not in {field_file}")
    m = re.search(r"\bvalue\s+uniform\s+([^;\s]+)\s*;", body)
    if m:
        return [float(m.group(1))]
    m = re.search(r"\bvalue\s+nonuniform\s+List<scalar>\s*(\d+)\s*\(([^)]*)\)", body, re.S)
    if m:
        vals = [float(v) for v in m.group(2).split()]
        if len(vals) != int(m.group(1)):
            raise ValueError("list length does not match its header")
        return vals
    if re.search(r"\bvalue\s+nonuniform\s+0\s*\(\s*\)", body):
        return []
    raise ValueError(f"cannot parse value of patch {patch} in {field_file}")


def time_dirs(case_dir):
    """Numeric time directories (excluding 0), sorted ascending."""
    out = []
    for d in Path(case_dir).iterdir():
        if d.is_dir():
            try:
                t = float(d.name)
            except ValueError:
                continue
            if t > 0:
                out.append((t, d))
    return [d for _, d in sorted(out)]


def run_time(log_file):
    """Last 'ExecutionTime = X s' in the solver log, or None."""
    vals = re.findall(r"ExecutionTime\s*=\s*([0-9.eE+-]+)\s*s", Path(log_file).read_text())
    return float(vals[-1]) if vals else None


def t_max_components(time_dir, comps=("comp1", "comp2", "comp3")):
    return max(max(patch_values(Path(time_dir) / "T", c)) for c in comps)


def case_metrics(case_dir, log_name="log.solver", conv_tol=0.05):
    """Result row of one cabinet case.

    I call a case converged when T_max changed by less than conv_tol (0.05 K)
    between the last two write times. I judge it on the quantity I actually report,
    because a residual that looks small does not guarantee the answer has stopped
    moving. With fewer than two write times the flag stays None (unknown)."""
    case = Path(case_dir)
    dirs = time_dirs(case)
    if not dirs:
        raise RuntimeError(f"no time directories in {case}")
    last = dirs[-1]
    t_max = t_max_components(last)
    converged = None
    if len(dirs) >= 2:
        converged = abs(t_max - t_max_components(dirs[-2])) < conv_tol
    p_in = patch_values(last / "p_rgh", "inlet")
    p_out = patch_values(last / "p_rgh", "outlet")
    # dp uses p_rgh, so the hydrostatic part is already removed and the number shows
    # only the pressure loss caused by the flow.
    dp = sum(p_in) / len(p_in) - sum(p_out) / len(p_out)
    return {
        "T_max_K": t_max,
        "dp_Pa": dp,
        "end_time_s": float(last.name),
        "converged": converged,
        "wall_time_s": run_time(case / log_name) if (case / log_name).exists() else None,
    }
