# Changelog

What I changed after the first complete version. All numbers come from `results/`.

## 2026-11-24 (conjugate heat transfer)
- **Conjugate heat transfer** (`scripts/cht_study.py`): the three components meshed as solid blocks with a uniform heat source of the same total power as the sweep, coupled to the air with `chtMultiRegionSimpleFoam`; 3 cases x 2 assumed conductivities (200 and 5 W/(m K)), 6 runs, all converged. T_max is 12 to 74 K lower than in the sweep, but the mean surface temperature is only 0.4 to 2.6 K lower: the sweep's T_max is a hot-spot value of a prescribed flux. Pressure drop changes by 0.2 to 2.8 %. Assumptions are listed in the README.
- **Corrected mistake:** with the tutorial setting `useImplicit true` the coupling lost about 14 % of the heat (86 % of the power reached the air). I found it with an enthalpy-flux balance, switched to `useImplicit false` (balance within 0.4 % and 2.2 %, like the sweep), and discarded and re-ran my first results. I did not find why the implicit form loses heat.
- `make_cabinet_mesh.py` got a `--solids` option and `fix_boundary.py` a `--cht` option. Added the study to `./run_all.sh studies`, 8 tests in `tests/test_studies.py`, `results/cht_study.csv` (with the heat balance), and the run folders to `.gitignore`.
- Updated the README: new Part 2 section with all assumptions, a row in Results at a glance, and the Heat-sources and Simplified-physics rows.
- Roadmap item 10: all four parts are now done.

## 2026-11-24 (3D slice)
- **3D slice** (`scripts/slice3d_study.py`): same cross-section and in-plane mesh with 12 graded cells through the 0.1 m depth (271 632 cells) and no-slip adiabatic side panels instead of `empty`. Three cases, each on 6 cores, 55 to 64 minutes. Pressure drop is 16 to 20 % higher in all three. T_max is +8.9 K for `u1_q100` and −0.4 K for `u1_q600` (both converged); `u0.5_q400` does not converge in 3D (T_max spread 16.6 K over the last 500 iterations), so I do not use its number.
- `make_cabinet_mesh.py` got a `--layers` option (default 1, the 2D case, unchanged) and `fix_boundary.py` a `--3d` option that makes front and back walls.
- Added the study to `./run_all.sh studies`, 5 tests in `tests/test_studies.py`, `results/slice3d_study.csv`, and the run folders to `.gitignore`.
- Updated the README: new Part 2 section, a row in Results at a glance, and the 2D-section row in Limitations.
- Roadmap item 10: 3D part done; conjugate heat transfer is still open.

## 2026-11-24 (radiation)
- **Radiation study** (`scripts/radiation_study.py`): `fvDOM` surface radiation added to the same 2D cabinet, transparent air, black openings, shell and components with emissivity 0.9 or 0.3, on 3 cases (6 runs, all converged, run on 6 cores). T_max drops by 5 to 64 K (22 to 52 % of the rise), pressure drop changes by 0.4 to 13.5 %. The components radiate 10 to 31 % of their power; the radiative exchange sums to 0.1 to 0.3 % of the input, so it is consistent. 3.5 to 7.4 % of the power leaves through the open inlet and outlet, which depends on my black-opening assumption.
- Added the study to `./run_all.sh studies`, tests for `add_radiation` and `face_areas` in `tests/test_studies.py`, `results/radiation_study.csv`, and the run folder to `.gitignore`.
- Updated the README: new Part 2 section, a row in Results at a glance, and revised Radiation, Simplified-physics and Limitations rows.
- Roadmap item 10: radiation part done; 3D slice and conjugate heat transfer are still open.

## 2026-11-24 (inlet turbulence)
- **Inlet-turbulence sensitivity** (`scripts/inlet_turbulence_study.py`): 3 converged cases re-run with the inlet k multiplied by 0.1, 1, 10 and 100 (omega scaled with sqrt(k), so only the intensity changes). The k x1 control reproduces the sweep exactly. Up to k x10 (5 to 16 % intensity) T_max moves by less than 1 K and dp by at most 5.5 %; at k x100 (26 to 52 %) T_max moves by 1.4 to 7.2 K.
- Added the study to `./run_all.sh studies`, a test for `set_inlet` in `tests/test_studies.py`, and `results/inlet_turbulence_study.csv`.
- Updated the README: new Part 2 section, a row in Results at a glance, and revised Turbulence, Simplified-physics rows.
- Roadmap item 10: inlet-turbulence part done; 3D slice, conjugate heat transfer and radiation are still open.

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
