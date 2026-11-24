#!/usr/bin/env python3
"""Set the patch types in constant/polyMesh/boundary after gmshToFoam.

gmshToFoam makes every patch type 'patch', but I need real types: the wall
functions only work on 'wall' patches, and the single-cell-thick 2D setup needs
'empty' on front and back. I set them by patch name here so the mesh step stays
one command and I never edit the boundary file by hand.
    python3 scripts/fix_boundary.py cabinet_case/constant/polyMesh/boundary
"""
import re
import sys

# I treat the three components as walls (no flow through them), which is also what
# the wall functions and the heat-flux boundary condition on them require.
# Inlet and outlet stay plain patches because the pressure/velocity conditions live there.
TYPES = {
    "front": "empty",
    "back": "empty",
    "walls": "wall",
    "comp1": "wall",
    "comp2": "wall",
    "comp3": "wall",
    "inlet": "patch",
    "outlet": "patch",
}

# For conjugate heat transfer the components are solid regions, so they have no boundary patch of their own yet
# (splitMeshRegions creates the interface patches later).
TYPES_CHT = {k: v for k, v in TYPES.items() if not k.startswith("comp")}

# For the 3D slice the front and back are real no-slip walls (a cabinet has side panels), not 'empty'.
TYPES_3D = {**TYPES, "front": "wall", "back": "wall"}


def fix(text, types=TYPES):
    for name, new_type in types.items():
        pattern = re.compile(r"(\n\s*" + re.escape(name) + r"\s*\{)(.*?)(\})", re.S)
        m = pattern.search(text)
        if not m:
            raise ValueError(f"patch '{name}' not found in boundary file")
        body = m.group(2)
        body = re.sub(r"\s*\bphysicalType\s+[^;]+;", "", body)
        body = re.sub(r"(\btype\s+)[^;]+;", rf"\g<1>{new_type};", body, count=1)
        if new_type == "empty":
            body = re.sub(r"\s*\binGroups[^;]*;", "", body)
        text = text[: m.start()] + m.group(1) + body + m.group(3) + text[m.end():]
    return text


if __name__ == "__main__":
    path = sys.argv[1]
    types = TYPES_3D if "--3d" in sys.argv[2:] else TYPES_CHT if "--cht" in sys.argv[2:] else TYPES
    with open(path) as fh:
        original = fh.read()
    with open(path, "w") as fh:
        fh.write(fix(original, types))
    print(f"patch types set in {path}")
