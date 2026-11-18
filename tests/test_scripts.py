"""Tests for the pure-Python helpers. Run: python3 -m unittest discover -s tests"""
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import check_rayleigh  # noqa: E402
import fix_boundary  # noqa: E402
import gci  # noqa: E402
import parse_foam  # noqa: E402

FIELD = """FoamFile { version 2.0; format ascii; class volScalarField; object T; }
dimensions [0 0 0 1 0 0 0];
internalField uniform 293.15;
boundaryField
{
    inlet { type fixedValue; value uniform 293.15; }
    comp1
    {
        type externalWallHeatFluxTemperature;
        value nonuniform List<scalar>
3
(
310.5
312.25
311.0
)
;
    }
    comp2
    {
        type externalWallHeatFluxTemperature;
        value uniform 305;
    }
}
"""

BOUNDARY = """FoamFile { version 2.0; format ascii; class polyBoundaryMesh; object boundary; }
8
(
    walls
    {
        type            patch;
        physicalType    patch;
        nFaces          10;
        startFace       100;
    }
    inlet
    {
        type            patch;
        physicalType    patch;
        nFaces          2;
        startFace       110;
    }
    outlet { type patch; physicalType patch; nFaces 2; startFace 112; }
    comp1 { type patch; physicalType patch; nFaces 4; startFace 114; }
    comp2 { type patch; physicalType patch; nFaces 4; startFace 118; }
    comp3 { type patch; physicalType patch; nFaces 4; startFace 122; }
    front { type patch; physicalType patch; nFaces 50; startFace 126; }
    back { type patch; physicalType patch; nFaces 50; startFace 176; }
)
"""


class TestParse(unittest.TestCase):
    def test_nonuniform_and_uniform(self):
        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / "T"
            f.write_text(FIELD)
            self.assertEqual(parse_foam.patch_values(f, "comp1"), [310.5, 312.25, 311.0])
            self.assertEqual(parse_foam.patch_values(f, "comp2"), [305.0])
            self.assertEqual(parse_foam.patch_values(f, "inlet"), [293.15])
            with self.assertRaises(KeyError):
                parse_foam.patch_values(f, "missing")

    def test_run_time(self):
        with tempfile.TemporaryDirectory() as d:
            log = Path(d) / "log"
            log.write_text("Time = 1\nExecutionTime = 3.5 s  ClockTime = 4 s\n"
                           "Time = 2\nExecutionTime = 7.25 s  ClockTime = 8 s\n")
            self.assertEqual(parse_foam.run_time(log), 7.25)


class TestBoundary(unittest.TestCase):
    def test_types(self):
        out = fix_boundary.fix(BOUNDARY)
        self.assertNotIn("physicalType", out)
        import re
        def type_of(name):
            return re.search(r"\n\s*" + name + r"\s*\{[^}]*?type\s+(\w+);", out).group(1)
        for name, expected in fix_boundary.TYPES.items():
            self.assertEqual(type_of(name), expected, name)


class TestGci(unittest.TestCase):
    def test_second_order(self):
        # f(h) = 4.5 + 2*h^2 -> p = 2, f_ext = 4.5
        f = lambda h: 4.5 + 2.0 * h ** 2
        r = gci.gci(f(1 / 128), f(1 / 64), f(1 / 32), 1 / 128, 1 / 64, 1 / 32)
        self.assertAlmostEqual(r["p"], 2.0, places=6)
        self.assertAlmostEqual(r["f_ext"], 4.5, places=6)
        self.assertTrue(r["monotonic"])

    def test_identical_solutions(self):
        with self.assertRaises(ValueError):
            gci.gci(4.5, 4.5, 4.6, 1 / 128, 1 / 64, 1 / 32)


class TestRayleigh(unittest.TestCase):
    def test_case_is_ra_1e5(self):
        params = json.loads((ROOT / "validation_cavity" / "case_params.json").read_text())
        self.assertAlmostEqual(check_rayleigh.rayleigh(params) / 1e5, 1.0, delta=1e-3)
        self.assertEqual(check_rayleigh.main(str(ROOT / "validation_cavity")), 0)


class TestCompareSurrogates(unittest.TestCase):
    def test_out_of_range_flag(self):
        import numpy as np
        import compare_surrogates as cs
        x = np.array([[0.2, 100.0], [1.0, 600.0]])
        flags = cs.out_of_range([[0.1, 300], [0.5, 300], [0.5, 700]], x)
        self.assertEqual(list(flags), [True, False, True])


if __name__ == "__main__":
    unittest.main()
