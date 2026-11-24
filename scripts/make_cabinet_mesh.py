#!/usr/bin/env python3
"""Build the 2D cabinet geometry with Gmsh and write a Gmsh 2.2 .msh file.

I build the geometry in code instead of drawing it in a GUI so that the dimensions
live in one place (the constants below) and the mesh can be regenerated, scaled
coarser or finer, and reused by the tests and by the figure script.

Geometry (metres, 2D section, one prism layer in z):
    cabinet 0.6 x 1.2, inlet slot in the bottom wall, outlet slot in the top
    wall, three rectangular heat-source components inside.
Patches: inlet, outlet, walls, comp1, comp2, comp3, front, back, volume 'fluid'.

    python3 scripts/make_cabinet_mesh.py --out cabinet_case/cabinet.msh [--scale 1.0] [--layers 12] [--solids]
--scale > 1 makes the mesh coarser (faster), < 1 finer.
--layers N (default 1) splits the 0.1 m depth into N cells for the 3D slice (slice3d_study.py). The layers are
thin at the front and back wall and thicker in the middle, so the walls are resolved like the other walls.
--solids meshes the three components as solid volumes (cell zones comp1..comp3) for conjugate heat transfer (cht_study.py).
"""
import argparse
import math
import sys

import gmsh

W, H, DEPTH = 0.6, 1.2, 0.1
LAYER_GROWTH = 1.4  # growth ratio of the layer thickness from each side wall towards the middle (3D slice only)
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


def layer_fractions(n, growth=LAYER_GROWTH):
    """Cumulative heights (0..1) of n layers through the depth, thin at both ends and thicker in the middle.
    n = 1 gives [1.0], which is the one-cell-thick 2D case."""
    if n == 1:
        return [1.0]
    half = n // 2
    # thickness of the layers on one side, geometric: t0, t0*g, t0*g^2, ... (an odd n gets one middle layer)
    t = [growth ** i for i in range(half)]
    sizes = t + ([t[-1] * growth] if n % 2 else []) + t[::-1]
    total = sum(sizes)
    out, acc = [], 0.0
    for s in sizes:
        acc += s / total
        out.append(acc)
    out[-1] = 1.0
    return out


def build(out, scale=1.0, lc_wall=0.004, lc_far=0.025, layers=1, solids=False):
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
    if solids:
        # conjugate heat transfer: the components are meshed as solid volumes instead of holes.
        # fragment() cuts the shapes so that the air and each component share the same edges
        # (a conforming mesh), which the temperature coupling between the regions needs.
        comp_surfs = [(2, occ.addRectangle(x0, y0, 0, x1 - x0, y1 - y0)) for (x0, y0, x1, y1) in COMPS.values()]
        occ.fragment([(2, surf)], comp_surfs)
        occ.synchronize()
        all_surfs = gmsh.model.getEntities(2)
    else:
        all_surfs = [(2, surf)]
    # One cell through the thickness, recombined to prisms/hexes: OpenFOAM is 3D, so a
    # "2D" case is one cell thick with 'empty' front and back patches.
    # (layers > 1 is the 3D slice: several cells through the depth, with graded thickness)
    out_tags = occ.extrude(all_surfs, 0, 0, DEPTH, numElements=[1] * layers,
                           heights=layer_fractions(layers), recombine=True)
    occ.synchronize()

    vol_tags = [t for d, t in out_tags if d == 3]
    comp_volumes = {}  # solids mode: component name -> volume tag
    if solids:
        # a component volume is small and its centre lies inside its rectangle; the air volume is large
        for tag in vol_tags:
            x, y, _ = occ.getCenterOfMass(3, tag)
            for name, (x0, y0, x1, y1) in COMPS.items():
                if occ.getMass(3, tag) < 0.01 and x0 < x < x1 and y0 < y < y1:
                    comp_volumes[name] = tag
        if len(comp_volumes) != len(COMPS):
            raise RuntimeError("could not identify the three component volumes")
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
            if solids and name.startswith("comp"):
                continue  # faces between air and a component are interior faces; splitMeshRegions makes the interfaces
        groups.setdefault(name, []).append(tag)
    for name, tags in groups.items():
        gmsh.model.addPhysicalGroup(2, tags, name=name)
    if solids:
        for name, tag in comp_volumes.items():
            gmsh.model.addPhysicalGroup(3, [tag], name=name)
        gmsh.model.addPhysicalGroup(3, [t for t in vol_tags if t not in comp_volumes.values()], name="fluid")
    else:
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
    ap.add_argument("--layers", type=int, default=1, help="cells through the depth (1 = the 2D case)")
    ap.add_argument("--solids", action="store_true", help="mesh the components as solid volumes (conjugate heat transfer)")
    a = ap.parse_args()
    names, n = build(a.out, a.scale, layers=a.layers, solids=a.solids)
    print(f"wrote {a.out}: patches {names}, {n} nodes")
    sys.exit(0)
