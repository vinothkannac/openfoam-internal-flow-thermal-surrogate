#!/usr/bin/env python3
"""Second-order (`linearUpwind`) re-run of the converged steady cases.

The steady sweep uses first-order `upwind` for U and h, and I changed it together with the
relaxation factors, so I could not say how much of any change came from which. Here I restart
cases that already have converged fields and run each of them two ways from the SAME start:

  control : the original `upwind` scheme, same relaxation, the same number of extra iterations
  linear  : `bounded Gauss linearUpwind grad(U)` and `... grad(h)`, same relaxation, same iterations

  scheme effect     = T_max(linear) - T_max(control)     (what changing the scheme did)
  continuation drift = T_max(control) - T_max(original)  (what only running on did: iteration noise)

Only the cases whose run directory (cabinet_case/runs/<case>) still holds the saved fields can be
restarted; the originals are copied, never changed. A case is also judged with the 500-iteration
spread of the monitors, because a restart can leave a transient that the two-write-time test misses.

    python3 scripts/linear_upwind_study.py --jobs 4
    python3 scripts/linear_upwind_study.py --collect-only
"""
import argparse
import csv
import re
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from parse_foam import case_metrics, monitor_drift, time_dirs  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "cabinet_case" / "runs"
RUNS = ROOT / "cabinet_case" / "runs_linupwind"
OUT = ROOT / "results" / "linear_upwind_study.csv"
T_INLET = 293.15

SCHEMES = {
    "control": {},
    "linear": {"div(phi,U)": "bounded Gauss linearUpwind grad(U)",
               "div(phi,h)": "bounded Gauss linearUpwind grad(h)"},
}


def set_schemes(text, new):
    """Replace the divergence scheme of each key; fail loudly if a key is missing."""
    for key, scheme in new.items():
        pat = r"(?m)^(\s*)" + re.escape(key) + r"\s+[^;]+;"
        if not re.search(pat, text):
            raise KeyError(f"{key} not found in fvSchemes")
        text = re.sub(pat, lambda m: f"{m.group(1)}{key:<15} {scheme};", text, count=1)
    return text


def prepare(case, arm, extra, relax_uh=None):
    d = RUNS / f"{case}__{arm}"
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)
    src = SRC / case
    shutil.copytree(src / "constant", d / "constant")
    shutil.copytree(src / "system", d / "system")
    last = time_dirs(src)[-1]
    shutil.copytree(last, d / last.name)          # latest saved fields = the restart state
    (d / "0").mkdir()
    for f in (ROOT / "cabinet_case" / "0.orig").iterdir():   # 0 is only needed to satisfy startFrom
        shutil.copy(f, d / "0" / f.name)
    cd = d / "system" / "controlDict"
    text = cd.read_text()
    text = re.sub(r"(?m)^startFrom\s+\S+;", "startFrom       latestTime;", text)
    text = re.sub(r"(?m)^endTime\s+\S+;", f"endTime         {float(last.name) + extra:g};", text)
    cd.write_text(text)
    fs = d / "system" / "fvSchemes"
    fs.write_text(set_schemes(fs.read_text(), SCHEMES[arm]))
    if relax_uh is not None:
        fv = d / "system" / "fvSolution"
        t = fv.read_text()
        t = re.sub(r'(?m)^(\s*)U\s+[0-9.]+;', rf"\g<1>U               {relax_uh};", t)
        t = re.sub(r'(?m)^(\s*)"\(h\|e\)"\s+[0-9.]+;', rf'\g<1>"(h|e)"         {relax_uh};', t)
        fv.write_text(t)
    return d


def run_one(task):
    case, arm, extra, relax_uh = task
    d = prepare(case, arm, extra, relax_uh)
    with open(d / "log.solver", "w") as fh:
        r = subprocess.run(["buoyantSimpleFoam"], cwd=d, stdout=fh, stderr=subprocess.STDOUT)
    print(f"{'done' if r.returncode == 0 else 'FAILED'} {case} {arm}", flush=True)


def collect(cases):
    rows = []
    for case in cases:
        base = case_metrics(SRC / case, log_name="log.solver_continue"
                            if (SRC / case / "log.solver_continue").exists() else "log.solver")
        for arm in SCHEMES:
            d = RUNS / f"{case}__{arm}"
            if not (d / "log.solver").exists():
                continue
            try:
                m = case_metrics(d)
            except Exception as exc:
                print(f"cannot read {d.name}: {exc}")
                continue
            drift = monitor_drift(d, window=500) or {}
            rows.append({"case": case, "arm": arm, "T_max_original_K": base["T_max_K"],
                         "T_max_K": m["T_max_K"], "dT_vs_original_K": m["T_max_K"] - base["T_max_K"],
                         "dp_original_Pa": base["dp_Pa"], "dp_Pa": m["dp_Pa"],
                         "dp_change_pct": 100 * (m["dp_Pa"] - base["dp_Pa"]) / base["dp_Pa"],
                         "converged": m["converged"], "end_time_s": m["end_time_s"],
                         "T_max_spread_last500_K": drift.get("T_max_spread_K"),
                         "dp_spread_last500_Pa": drift.get("dp_spread_Pa"),
                         "wall_time_s": m["wall_time_s"]})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", nargs="+", default=["u0.5_q400", "u1_q100", "u1_q600", "u0.2_q400"])
    ap.add_argument("--extra", type=int, default=4000, help="extra iterations after the saved state")
    ap.add_argument("--relax-uh", type=float, default=None,
                    help="override the U/h relaxation of both arms (default: keep each case's own)")
    ap.add_argument("--tag", default="", help="suffix for the run directory and CSV (for variant runs)")
    ap.add_argument("--arms", nargs="+", default=list(SCHEMES))
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--collect-only", action="store_true")
    args = ap.parse_args()
    global RUNS, OUT
    if args.tag:
        RUNS, OUT = RUNS.with_name(RUNS.name + args.tag), OUT.with_name(OUT.stem + args.tag + ".csv")

    if not args.collect_only:
        RUNS.mkdir(exist_ok=True)
        tasks = [(c, a, args.extra, args.relax_uh) for c in args.cases for a in args.arms]
        with ThreadPoolExecutor(max_workers=args.jobs) as pool:
            list(pool.map(run_one, tasks))
    rows = collect(args.cases)
    if not rows:
        sys.exit("no finished runs")
    with open(OUT, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(f"{r['case']:10s} {r['arm']:8s} T_max {r['T_max_K']:.2f} K ({r['dT_vs_original_K']:+.2f} vs original), "
              f"dp {r['dp_Pa']:.4f} Pa ({r['dp_change_pct']:+.1f} %), converged={r['converged']}, "
              f"spread500={r['T_max_spread_last500_K']}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
