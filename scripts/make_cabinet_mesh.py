#!/usr/bin/env python3
"""Build the 2D cabinet geometry with Gmsh and write a Gmsh 2.2 .msh file.

I build the geometry in code instead of drawing it in a GUI so that the dimensions
live in one place (the constants below) and the mesh can be regenerated, scaled
coarser or finer, and reused by the tests and by the figure script.

Geometry (metres, 2D section, one prism layer in z):
    cabinet 0.6 x 1.2, inlet slot in the bottom wall, outlet slot in the top
    wall, three rectangular heat-source components inside.
Patches: inlet, outlet, walls, comp1, comp2, comp3, front, back, volume 'fluid'.

    python3 scripts/make_cabinet_mesh.py --out cabinet_case/cabinet.msh [--scale 1.0]
--scale > 1 makes the mesh coarser (faster), < 1 finer.
"""
import argparse
import math
import sys

import gmsh

W, H, DEPTH = 0.6, 1.2, 0.1
# The inlet slot (bottom wall) and outlet slot (top wall) share the same x-range, so
# air enters and leaves along the same vertical line.
SLOT = (0.2, 0.4)
# Three rectangular components, placed alternately on the left, right and left side.
# I treat them as separate patches so each one gets its own heat flux in the sweep.
COMPS = {  # name: (x0, y0, x1, y1)
    "comp1": (0.05, 0.30, 0.25, 0.45),
    "comp2": (0.35, 0.55, 0.55, 0.70),
    "comp3": (0.05, 0.80, 0.25, 0.95),
}


def build(out, scale=1.0, lc_wall=0.004, lc_far=0.025):
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("cabinet")
    occ = gmsh.model.occ
    segments = {}  # I remember each edge's midpoint so I can name the extruded side faces later

    def add_loop(points, names):
        pts = [occ.addPoint(x, y, 0) for x, y in points]
        lines = []
        for i, name in enumerate(names):
            a, b = pts[i], pts[(i + 1) % len(pts)]
            lines.append(occ.addLine(a, b))
            (x0, y0), (x1, y1) = points[i], points[(i + 1) % len(pts)]
            segments[(0.5 * (x0 + x1), 0.5 * (y0 + y1))] = name
        return occ.addCurveLoop(lines)

    outer_pts = [
        (0, 0), (SLOT[0], 0), (SLOT[1], 0), (W, 0),
        (W, H), (SLOT[1], H), (SLOT[0], H), (0, H),
    ]
    outer_names = ["walls", "inlet", "walls", "walls", "walls", "outlet", "walls", "walls"]
    loops = [add_loop(outer_pts, outer_names)]
    for name, (x0, y0, x1, y1) in COMPS.items():
        loops.append(add_loop([(x0, y0), (x1, y0), (x1, y1), (x0, y1)], [name] * 4))

    surf = occ.addPlaneSurface(loops)
    # One cell through the thickness, recombined to prisms/hexes: OpenFOAM is 3D, so a
    # "2D" case is one cell thick with 'empty' front and back patches.
    out_tags = occ.extrude([(2, surf)], 0, 0, DEPTH, numElements=[1], recombine=True)
    occ.synchronize()

    vol_tags = [t for d, t in out_tags if d == 3]
    # I name every surface by its centre of mass: front/back by z, side faces by the
    # nearest remembered edge midpoint. That way the patches come out with the right names.
    groups = {}
    for _, tag in gmsh.model.getEntities(2):
        x, y, z = occ.getCenterOfMass(2, tag)
        if abs(z) < 1e-9:
            name = "front"
        elif abs(z - DEPTH) < 1e-9:
            name = "back"
        else:
            # lateral face: nearest segment midpoint in the x-y plane
            name = min(segments.items(), key=lambda kv: math.hypot(kv[0][0] - x, kv[0][1] - y))[1]
        groups.setdefault(name, []).append(tag)
    for name, tags in groups.items():
        gmsh.model.addPhysicalGroup(2, tags, name=name)
    gmsh.model.addPhysicalGroup(3, vol_tags, name="fluid")

    # Mesh size: fine (4 mm) next to every wall and component, coarse (25 mm) in the free
    # stream. This keeps the near-wall regions resolved while the whole mesh stays small
    # enough for a laptop. --scale multiplies both sizes for coarse smoke tests.
    wall_curves = [t for _, t in gmsh.model.getEntities(1)]
    dist = gmsh.model.mesh.field.add("Distance")
    gmsh.model.mesh.field.setNumbers(dist, "CurvesList", wall_curves)
    gmsh.model.mesh.field.setNumber(dist, "Sampling", 50)
    thr = gmsh.model.mesh.field.add("Threshold")
    gmsh.model.mesh.field.setNumber(thr, "InField", dist)
    gmsh.model.mesh.field.setNumber(thr, "SizeMin", lc_wall * scale)
    gmsh.model.mesh.field.setNumber(thr, "SizeMax", lc_far * scale)
    gmsh.model.mesh.field.setNumber(thr, "DistMin", 0.01)
    gmsh.model.mesh.field.setNumber(thr, "DistMax", 0.12)
    gmsh.model.mesh.field.setAsBackgroundMesh(thr)
    for opt in ("MeshSizeFromPoints", "MeshSizeFromCurvature", "MeshSizeExtendFromBoundary"):
        gmsh.option.setNumber("Mesh." + opt, 0)

    gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
    gmsh.option.setNumber("Mesh.SaveAll", 0)
    gmsh.model.mesh.generate(3)
    n_nodes = len(gmsh.model.mesh.getNodes()[0])
    gmsh.write(out)
    gmsh.finalize()
    return sorted(groups), n_nodes


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="cabinet_case/cabinet.msh")
    ap.add_argument("--scale", type=float, default=1.0)
    a = ap.parse_args()
    names, n = build(a.out, a.scale)
    print(f"wrote {a.out}: patches {names}, {n} nodes")
    sys.exit(0)
