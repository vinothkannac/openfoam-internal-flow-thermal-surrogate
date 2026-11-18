# Changelog

What I changed after the first complete version. All numbers come from `results/`.

## 2026-11-18
- Checked y⁺ on four cases that cover the sweep. It is 0.02 to 9.9 on the walls and components, below the log-law range. Details are in the README (Assumptions and Limitations); data in `results/yplus.csv`.
- Compared the MLP with a Gaussian process and a quadratic polynomial (`scripts/compare_surrogates.py`). The Gaussian process is about as good as the MLP, not better. Its 95 % interval was too narrow: it covered 81 % and 88 % of the left-out cases. The polynomial was clearly worse. I also added a flag for queries outside the sampled range.
- Added T_max and pressure-drop monitors to the cabinet case template, plus `monitor_drift()` to read them and a test. The 20 reported cases were run before this, so they don't have monitor data yet.
- Added a GitHub Actions workflow that runs the unit tests.
- Removed the duplicate `constant/momentumTransport`; ESI v2606 reads `turbulenceProperties`.
- Found a second source for the benchmark value 4.519. Both are still secondary; I haven't read the 1983 paper.
