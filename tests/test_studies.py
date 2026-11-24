"""Tests for the pure helpers of the Tier-2 study scripts (no OpenFOAM needed)."""
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import active_learning as al  # noqa: E402
import cabinet_mesh_study as ms  # noqa: E402
import inlet_turbulence_study as its  # noqa: E402
import fix_boundary  # noqa: E402
import make_cabinet_mesh as mcm  # noqa: E402
import radiation_study as rs  # noqa: E402
import slice3d_study as s3  # noqa: E402
import cht_study as cht  # noqa: E402
import linear_upwind_study as lu  # noqa: E402
import transient_sensitivity as tsens  # noqa: E402

SCHEMES = """divSchemes
{
    default         none;
    div(phi,U)      bounded Gauss upwind;
    div(phi,h)      bounded Gauss upwind;
    div(phi,k)      bounded Gauss limitedLinear 1;
}
"""


class TestLinearUpwind(unittest.TestCase):
    def test_only_u_and_h_change(self):
        out = lu.set_schemes(SCHEMES, lu.SCHEMES["linear"])
        self.assertIn("div(phi,U)      bounded Gauss linearUpwind grad(U);", out)
        self.assertIn("div(phi,h)      bounded Gauss linearUpwind grad(h);", out)
        self.assertIn("div(phi,k)      bounded Gauss limitedLinear 1;", out)

    def test_control_is_unchanged(self):
        self.assertEqual(lu.set_schemes(SCHEMES, lu.SCHEMES["control"]), SCHEMES)

    def test_missing_key_fails(self):
        with self.assertRaises(KeyError):
            lu.set_schemes("divSchemes { }", {"div(phi,U)": "x"})


class TestMeshStudy(unittest.TestCase):
    def rows(self, f):  # f = (fine, medium, coarse) of the temperature rise
        return [{"case": "c", "mesh": m, "cells": n, "dT_max_K": v}
                for m, n, v in zip(("fine", "medium", "coarse"), (40000, 20000, 10000), f)]

    def test_second_order_sequence(self):
        # an error that falls with h^2 (h ~ N^-1/2, so with 1/N) gives p close to 2
        f = [100 + 1 / 40000, 100 + 1 / 20000, 100 + 1 / 10000]
        g = ms.gci_for(self.rows(f), "c", "dT_max_K")
        self.assertEqual(g["status"], "ok")
        self.assertAlmostEqual(g["p"], 2.0, places=3)

    def test_oscillatory_is_flagged(self):
        g = ms.gci_for(self.rows((100.0, 101.0, 100.5)), "c", "dT_max_K")
        self.assertNotEqual(g["status"], "ok")

    def test_identical_solutions_do_not_crash(self):
        g = ms.gci_for(self.rows((100.0, 100.0, 101.0)), "c", "dT_max_K")
        self.assertIn("undefined", g["status"])

    def test_missing_mesh(self):
        self.assertEqual(ms.gci_for(self.rows((1, 2, 3))[:2], "c", "dT_max_K")["status"], "missing a mesh")


class TestTransientStats(unittest.TestCase):
    def test_time_average_is_not_step_average(self):
        # half of the time at 0 with tiny steps (many samples), half at 10 with big steps
        t = np.concatenate([np.linspace(0, 49.99, 5000), np.linspace(50, 100, 50)])
        y = np.where(t < 50, 0.0, 10.0)
        s = tsens.stats(t, y, y, 0, 100)
        self.assertAlmostEqual(s["T_max_mean_K"], 5.0, delta=0.1)
        self.assertLess(y.mean(), 1.0)  # the plain step average would say about 0.1

    def test_window_outside_run_is_none(self):
        t = np.linspace(0, 100, 1000)
        self.assertIsNone(tsens.stats(t, t, t, 100, 200))

    def test_batch_sem_of_constant_is_zero(self):
        self.assertEqual(tsens.batch_sem(np.ones(100)), 0.0)


class TestActiveLearning(unittest.TestCase):
    def test_proposal_is_far_from_data_and_inside_box(self):
        # training data on the right half of the box only: the proposal must be on the left
        u = np.array([0.6, 0.8, 1.0, 0.6, 0.8, 1.0])
        q = np.array([100, 100, 100, 600, 600, 600.0])
        x = np.column_stack([u, q])
        y = np.column_stack([10 + 100 * q / 600 / u, 0.3 * u ** 2])
        pu, pq, score, _, _ = al.propose(x, y, blocked=[])
        self.assertLess(pu, 0.5)
        self.assertTrue(0.2 <= pu <= 1.0 and 100 <= pq <= 600)

    def test_blocked_point_is_not_proposed_again(self):
        u = np.array([0.6, 0.8, 1.0, 0.6, 0.8, 1.0])
        q = np.array([100, 100, 100, 600, 600, 600.0])
        x = np.column_stack([u, q])
        y = np.column_stack([10 + 100 * q / 600 / u, 0.3 * u ** 2])
        pu, pq, *_ = al.propose(x, y, blocked=[])
        pu2, pq2, *_ = al.propose(x, y, blocked=[[pu, pq]])
        d = np.linalg.norm(al.norm([pu, pq]) - al.norm([pu2, pq2]))
        self.assertGreater(d, al.BLOCK_RADIUS)


class TestInletTurbulence(unittest.TestCase):
    def make_case(self, tmp):
        # a tiny fake 0/ folder with the same "uniform" entries as the real k and omega files
        (tmp / "0").mkdir()
        (tmp / "0" / "k").write_text("internalField uniform 1e-3;\ninlet { value uniform 1e-3; }\n")
        (tmp / "0" / "omega").write_text("internalField uniform 4;\ninlet { value uniform 4; }\n")

    def test_k_scales_and_omega_scales_with_sqrt(self):
        with tempfile.TemporaryDirectory() as t:
            d = Path(t)
            self.make_case(d)
            its.set_inlet(d, 100.0)
            self.assertEqual((d / "0" / "k").read_text().count("uniform 0.1;"), 2)
            self.assertEqual((d / "0" / "omega").read_text().count("uniform 40;"), 2)

    def test_factor_one_keeps_the_values(self):
        # the text changes from 1e-3 to 0.001, but the number stays the same
        with tempfile.TemporaryDirectory() as t:
            d = Path(t)
            self.make_case(d)
            its.set_inlet(d, 1.0)
            self.assertEqual((d / "0" / "k").read_text().count("uniform 0.001;"), 2)
            self.assertEqual((d / "0" / "omega").read_text().count("uniform 4;"), 2)

    def test_missing_entry_fails(self):
        with tempfile.TemporaryDirectory() as t:
            d = Path(t)
            (d / "0").mkdir()
            (d / "0" / "k").write_text("internalField uniform 5;\n")
            (d / "0" / "omega").write_text("internalField uniform 4;\n")
            with self.assertRaises(RuntimeError):
                its.set_inlet(d, 10.0)


class TestRadiation(unittest.TestCase):
    def make_case(self, tmp):
        # a copy of the real template with the heat flux placeholders filled in, as run_sweep.prepare() does
        template = ROOT / "cabinet_case"
        (tmp / "0").mkdir()
        for f in (template / "0.orig").iterdir():
            text = f.read_text()
            for k, v in {"@UIN@": "1", "@Q1@": "100", "@Q2@": "60", "@Q3@": "30"}.items():
                text = text.replace(k, v)
            (tmp / "0" / f.name).write_text(text)
        shutil.copytree(template / "system", tmp / "system")
        (tmp / "constant").mkdir()

    def test_files_and_qr_are_added(self):
        with tempfile.TemporaryDirectory() as t:
            d = Path(t)
            self.make_case(d)
            rs.add_radiation(d, 0.9)
            for name in ("radiationProperties", "boundaryRadiationProperties"):
                self.assertTrue((d / "constant" / name).exists())
            self.assertTrue((d / "0" / "G").exists() and (d / "0" / "IDefault").exists())
            # three components + the shell now read qr, and the shell is still adiabatic (q = 0)
            T = (d / "0" / "T").read_text()
            self.assertEqual(T.count("qr              qr;"), 4)
            self.assertNotIn("zeroGradient", T)
            self.assertIn("emissivity      0.9;", (d / "constant" / "boundaryRadiationProperties").read_text())
            self.assertIn("Ii", (d / "system" / "fvSolution").read_text())
            self.assertIn("div(Ji,Ii_h)", (d / "system" / "fvSchemes").read_text())

    def test_wrong_template_fails(self):
        with tempfile.TemporaryDirectory() as t:
            d = Path(t)
            self.make_case(d)
            (d / "0" / "T").write_text("walls { type zeroGradient; }\n")
            with self.assertRaises(RuntimeError):
                rs.add_radiation(d, 0.9)

    def test_face_area_of_a_unit_square(self):
        with tempfile.TemporaryDirectory() as t:
            d = Path(t)
            (d / "points").write_text("4\n(\n(0 0 0)\n(1 0 0)\n(1 1 0)\n(0 1 0)\n)\n")
            (d / "faces").write_text("1\n(\n4(0 1 2 3)\n)\n")
            (d / "boundary").write_text("1\n(\n    top\n    {\n        type patch;\n        nFaces 1;\n        startFace 0;\n    }\n)\n")
            self.assertAlmostEqual(float(rs.face_areas(d, "top").sum()), 1.0)


class TestSlice3D(unittest.TestCase):
    def test_layers_add_up_and_are_symmetric(self):
        f = mcm.layer_fractions(12)
        self.assertEqual(len(f), 12)
        self.assertAlmostEqual(f[-1], 1.0)
        thick = [b - a for a, b in zip([0] + f[:-1], f)]
        self.assertAlmostEqual(thick[0], thick[-1])
        self.assertGreater(thick[5], thick[0])      # thicker in the middle than at the walls

    def test_one_layer_is_the_2d_case(self):
        self.assertEqual(mcm.layer_fractions(1), [1.0])

    def test_front_and_back_become_walls_only_in_3d(self):
        text = "front\n    {\n        type patch;\n    }\n"
        self.assertEqual(fix_boundary.TYPES["front"], "empty")
        self.assertEqual(fix_boundary.TYPES_3D["front"], "wall")
        self.assertEqual(fix_boundary.TYPES_3D["inlet"], fix_boundary.TYPES["inlet"])
        self.assertIn("wall", fix_boundary.fix("\n" + text + "\n".join(
            "\n    %s\n    {\n        type patch;\n    }" % n for n in
            ["back", "walls", "inlet", "outlet", "comp1", "comp2", "comp3"]), fix_boundary.TYPES_3D))

    def test_sides_as_walls_changes_every_field(self):
        with tempfile.TemporaryDirectory() as t:
            d = Path(t)
            (d / "0").mkdir()
            for f in (ROOT / "cabinet_case" / "0.orig").iterdir():
                text = f.read_text()
                for k, v in {"@UIN@": "1", "@Q1@": "100", "@Q2@": "60", "@Q3@": "30"}.items():
                    text = text.replace(k, v)
                (d / "0" / f.name).write_text(text)
            s3.sides_as_walls(d)
            for f in (d / "0").iterdir():
                text = f.read_text()
                self.assertIn("front|back", text)          # the side panels are named in a wall entry
                self.assertNotIn("empty", text)            # and nothing is 'empty' any more

    def test_unexpected_field_file_fails(self):
        with tempfile.TemporaryDirectory() as t:
            d = Path(t)
            (d / "0").mkdir()
            (d / "0" / "k").write_text("boundaryField { }\n")
            with self.assertRaises(RuntimeError):
                s3.sides_as_walls(d)


class TestCHT(unittest.TestCase):
    def template(self, name):
        return (ROOT / "cabinet_case" / "0.orig" / name).read_text()

    def test_power_is_flux_times_surface_of_the_block(self):
        # component 1: q = 100 W/m2 on a 0.7 m perimeter and 0.1 m depth is 7 W; components 2 and 3 get 0.6 and 0.3 of it
        self.assertAlmostEqual(cht.component_power(100, 1.0), 7.0)
        self.assertAlmostEqual(cht.component_power(100, 0.6), 4.2)
        self.assertAlmostEqual(cht.component_power(100, 0.3), 2.1)

    def test_air_temperature_has_three_interfaces_and_no_flux(self):
        T = cht.fluid_field("T", self.template("T"), 1.0)
        self.assertEqual(T.count("turbulentTemperatureRadCoupledMixed"), 3)
        for c in ("comp1", "comp2", "comp3"):
            self.assertIn("fluid_to_" + c, T)
        self.assertNotIn("externalWallHeatFluxTemperature", T.replace("walls", ""))
        self.assertNotIn("@", T)

    def test_interfaces_use_the_explicit_coupling(self):
        # useImplicit true lost about 14 % of the heat in my runs (see the README), so I check that it is not used
        for text in (cht.fluid_field("T", self.template("T"), 1.0), cht.SOLID_T, cht.FLUID_T_INTERFACE):
            self.assertNotIn("useImplicit     true", text)

    def test_other_fields_use_the_interface_patch_names(self):
        U = cht.fluid_field("U", self.template("U"), 0.5)
        self.assertIn("fluid_to_comp1", U)
        self.assertNotIn("|comp1", U)
        self.assertIn("0.5", U)      # the inlet velocity placeholder is filled

    def test_heat_to_air_of_a_simple_flow(self):
        # 2 faces on the inlet (flux -0.01 kg/s each, at 293.15 K) and 2 on the outlet (+0.01 kg/s at 303.15 K):
        # the heat is 1005 * 2 * 0.01 * 10 = 201 W
        with tempfile.TemporaryDirectory() as t:
            d = Path(t)
            (d / "fluid").mkdir()
            def field(path, inlet, outlet):
                (d / "fluid" / path).write_text(
                    "boundaryField\n{\n    inlet\n    {\n        value nonuniform List<scalar> 2(%s %s);\n    }\n"
                    "    outlet\n    {\n        value nonuniform List<scalar> 2(%s %s);\n    }\n}\n" % (*inlet, *outlet))
            field("phi", (-0.01, -0.01), (0.01, 0.01))
            field("T", (293.15, 293.15), (303.15, 303.15))
            self.assertAlmostEqual(cht.heat_to_air(d), 201.0, places=6)

    def test_wrong_temperature_file_fails(self):
        with self.assertRaises(RuntimeError):
            cht.fluid_field("T", "boundaryField { }", 1.0)

    def test_internal_max_reads_a_list_and_a_uniform_value(self):
        with tempfile.TemporaryDirectory() as t:
            f = Path(t) / "T"
            f.write_text("internalField   nonuniform List<scalar> 3(300 310.5 305);\n")
            self.assertEqual(cht.internal_max(f), 310.5)
            f.write_text("internalField   uniform 293.15;\n")
            self.assertEqual(cht.internal_max(f), 293.15)

    def test_solid_mesh_has_no_component_patches(self):
        with tempfile.TemporaryDirectory() as t:
            names, _ = mcm.build(str(Path(t) / "c.msh"), scale=3.0, solids=True)
        self.assertEqual(names, ["back", "front", "inlet", "outlet", "walls"])


if __name__ == "__main__":
    unittest.main()
