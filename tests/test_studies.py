"""Tests for the pure helpers of the Tier-2 study scripts (no OpenFOAM needed)."""
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import active_learning as al  # noqa: E402
import cabinet_mesh_study as ms  # noqa: E402
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


if __name__ == "__main__":
    unittest.main()
