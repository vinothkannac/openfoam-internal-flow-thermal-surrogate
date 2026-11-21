# Changelog

What I changed after the first complete version. All numbers come from `results/`.

## 2026-11-18 (Tier 2 studies)
- **Cabinet mesh study** (`scripts/cabinet_mesh_study.py`): 4 cases on 10k, 23k and 46k cells. T_max rise is 10 to 14 % higher on the fine mesh, apparent order 0.6 to 1.1, GCI 30 to 48 %: the cabinet temperatures are not mesh-converged. Pressure drop changes 4 to 7 %.
- **`linearUpwind` restart** (`scripts/linear_upwind_study.py`): +10 to +72 K in T_max on four converged cases while an `upwind` control moved by 0.1 K, and the second-order flow does not settle. The steady upwind results are biased low.
- **Transient sensitivity** (`scripts/transient_sensitivity.py`, `u0.2_q100`): time average changes by less than 0.15 K for half and double time step and double run length. One case only.
- **Active learning** (`scripts/active_learning.py`): 8 CFD runs, 4 accepted. Half of the GP's proposals were unsteady. Prospective T_max errors 0.2 to 5.3 K; the project surrogate was not retrained.
- Updated the README: new Part 2 sections for the four studies, two new rows in Results at a glance, a `studies` line in Quick start, and revised Assumptions, Decisions and Limitations rows (mesh, numerics, transient sensitivity, surrogate).
- Added `./run_all.sh studies`, unit tests for the study helpers (`tests/test_studies.py`), and a `root` argument to `run_sweep.prepare()` so the studies do not touch `cabinet_case/runs`.
- Roadmap items 6 to 9 ticked. New open items from the findings: time-average second-order (`linearUpwind`) runs for the sweep, and test wall treatment/prism layers against the mesh sensitivity.

## 2026-11-18
- Checked y⁺ on four cases that cover the sweep. It is 0.02 to 9.9 on the walls and components, below the log-law range. Details are in the README (Assumptions and Limitations); data in `results/yplus.csv`.
- Compared the MLP with a Gaussian process and a quadratic polynomial (`scripts/compare_surrogates.py`). The Gaussian process is about as good as the MLP, not better. Its 95 % interval was too narrow: it covered 81 % and 88 % of the left-out cases. The polynomial was clearly worse. I also added a flag for queries outside the sampled range.
- Added T_max and pressure-drop monitors to the cabinet case template, plus `monitor_drift()` to read them and a test. The 20 reported cases were run before this, so they don't have monitor data yet.
- Added a GitHub Actions workflow that runs the unit tests.
- Removed the duplicate `constant/momentumTransport`; ESI v2606 reads `turbulenceProperties`.
- Found a second source for the benchmark value 4.519. Both are still secondary; I haven't read the 1983 paper.
