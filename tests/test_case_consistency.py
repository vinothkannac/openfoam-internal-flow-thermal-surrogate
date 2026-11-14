"""Static checks of the OpenFOAM case files (no OpenFOAM needed).

Catches the typical first-run errors: patch names in the field files that do not
exist in the mesh, unresolved @placeholders@, missing keys.
"""
import re
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import make_cabinet_mesh  # noqa: E402
import parse_foam  # noqa: E402

CABINET_PATCHES = {"inlet", "outlet", "walls", "comp1", "comp2", "comp3", "front", "back"}
CAVITY_PATCHES = {"hot", "cold", "bottomWall", "topWall", "frontAndBack"}


def covered_patches(field_file):
    """Patch names (expanding "(a|b)" regex groups) defined in boundaryField."""
    text = Path(field_file).read_text()
    body = parse_foam._block(text, "boundaryField")
    names = set()
    for m in re.finditer(r"(?m)^ {4}(\S[^\n]*?)\s*\n {4}\{", body):
        token = m.group(1).strip('"').strip("()")
        names.update(token.split("|"))
    return names


class TestCases(unittest.TestCase):
    def test_cavity_patches_match(self):
        for f in (ROOT / "validation_cavity" / "0").iterdir():
            self.assertEqual(covered_patches(f), CAVITY_PATCHES, f.name)

    def test_cabinet_patches_match_mesh(self):
        for f in (ROOT / "cabinet_case" / "0.orig").iterdir():
            self.assertEqual(covered_patches(f), CABINET_PATCHES, f.name)

    def test_gmsh_patch_names(self):
        try:
            import gmsh  # noqa: F401
        except OSError:
            self.skipTest("gmsh shared libs missing")
        with tempfile.TemporaryDirectory() as d:
            names, _ = make_cabinet_mesh.build(str(Path(d) / "c.msh"), scale=3.0)
            self.assertEqual(set(names), CABINET_PATCHES)

    def test_placeholders_known(self):
        found = set()
        for f in (ROOT / "cabinet_case" / "0.orig").iterdir():
            found |= set(re.findall(r"@(\w+)@", f.read_text()))
        self.assertEqual(found, {"UIN", "Q1", "Q2", "Q3"})

    def test_solver_is_consistent(self):
        for case in ("validation_cavity", "cabinet_case"):
            cd = (ROOT / case / "system" / "controlDict").read_text()
            self.assertIn("buoyantSimpleFoam", cd)
            self.assertIn("steadyState", (ROOT / case / "system" / "fvSchemes").read_text())

    def test_required_files(self):
        need = ["controlDict", "fvSchemes", "fvSolution"]
        for case in ("validation_cavity", "cabinet_case"):
            for n in need:
                self.assertTrue((ROOT / case / "system" / n).exists(), f"{case}/{n}")
            for n in ("thermophysicalProperties", "g", "momentumTransport"):
                self.assertTrue((ROOT / case / "constant" / n).exists(), f"{case}/{n}")


if __name__ == "__main__":
    unittest.main()
