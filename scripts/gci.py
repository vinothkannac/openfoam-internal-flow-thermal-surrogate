#!/usr/bin/env python3
"""Grid Convergence Index after Celik et al. (2008), J. Fluids Eng. 130(7), 078001.

I use this procedure to put a number on the discretisation error instead of just
saying "the finer mesh is closer". It needs three solutions f1 (finest), f2, f3
(coarsest) with representative cell sizes h1 < h2 < h3. For my 2D cases
h ~ 1/N with N = cells per side, so a refinement ratio of 2 gives r = 2.
"""
import math


def gci(f1, f2, f3, h1, h2, h3, fs=1.25):
    # fs = 1.25 is the safety factor Celik et al. recommend when three meshes are used.
    r21, r32 = h2 / h1, h3 / h2
    e21, e32 = f2 - f1, f3 - f2
    if e21 == 0 or e32 == 0:
        raise ValueError("two solutions are identical, order of convergence is undefined")
    s = math.copysign(1.0, e32 / e21)
    # I solve for the apparent order p by the fixed-point iteration in the paper,
    # starting from the simple log-ratio estimate.
    p = abs(math.log(abs(e32 / e21))) / math.log(r21)
    for _ in range(200):
        q = math.log((r21 ** p - s) / (r32 ** p - s))
        p_new = abs(math.log(abs(e32 / e21)) + q) / math.log(r21)
        if abs(p_new - p) < 1e-10:
            p = p_new
            break
        p = p_new
    f_ext = (r21 ** p * f1 - f2) / (r21 ** p - 1.0)
    e_a = abs((f1 - f2) / f1)
    e_ext = abs((f_ext - f1) / f_ext)
    gci_fine = fs * e_a / (r21 ** p - 1.0)
    return {
        "p": p,
        "f_ext": f_ext,
        "e_a_pct": 100 * e_a,
        "e_ext_pct": 100 * e_ext,
        "gci_fine_pct": 100 * gci_fine,
        "monotonic": s > 0,
    }


if __name__ == "__main__":
    # tiny demo with an artificial second-order sequence
    print(gci(4.5190, 4.5280, 4.5640, 1 / 128, 1 / 64, 1 / 32))
