#!/usr/bin/env python3
"""Active learning: let the Gaussian-process uncertainty choose the next CFD cases.

The surrogate is trained on the 16 converged sweep cases. Its weak region is the low-velocity
side, where four sweep cases are unsteady and absent from the training set. Instead of adding
cases on a regular grid, each round does this:

  1. fit one Gaussian process per target (T_max rise, dp) on the cases accepted so far;
  2. score every candidate (u_in, q_ref) on a fine grid inside the sampled box by the largest
     predictive standard deviation of the two targets, each divided by that target's spread
     in the training data, so the temperature and the pressure drop count equally;
  3. run the winner as a CFD case with the sweep settings (8000 iterations; if it has not met
     the convergence flag, the same lower-relaxation continuation of 2000 iterations);
  4. accept it into the training set only if it met the convergence flag. A case that does not
     is recorded as "unsteady" and its neighbourhood is excluded from later proposals, because
     the GP cannot learn from it and would otherwise propose the same spot again.

The score of a round is measured BEFORE the case is added: the surrogate's prediction (and the
GP's own uncertainty) against the CFD result for the case it had not seen. After all rounds,
`summarize` retrains on each stage and reports leave-one-out error, the error on the cases of
later rounds, and the remaining GP uncertainty over the low-velocity region.

    python3 scripts/active_learning.py run --rounds 4         # propose, run, record (resumable)
    python3 scripts/active_learning.py summarize              # metrics per round + figure
    python3 scripts/active_learning.py propose                # only print the next proposal

The sweep results are not modified: new cases go to results/active_learning_cases.csv.
"""
import argparse
import csv
import json
import shutil
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import LeaveOneOut

sys.path.insert(0, str(Path(__file__).parent))
import compare_surrogates as cs  # noqa: E402
import run_sweep  # noqa: E402
import train_surrogate as ts  # noqa: E402
from continue_unconverged import lower_relaxation  # noqa: E402
from parse_foam import case_metrics, run_time  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "cabinet_case" / "runs_active"
RESULTS = ROOT / "results"
CASES_CSV = RESULTS / "active_learning_cases.csv"
SWEEP_CSV = RESULTS / "sweep_results.csv"
GRID_U = np.round(np.arange(0.2, 1.0 + 1e-9, 0.025), 4)
GRID_Q = np.arange(100, 600 + 1e-9, 25.0)
LOW_U = 0.5            # "low-velocity region" of the uncertainty and subset-error metrics
BLOCK_RADIUS = 0.10    # exclusion radius around a failed case, in box-normalised coordinates
BOX = np.array([[0.2, 1.0], [100.0, 600.0]])
FIELDS = ["round", "case", "u_in", "q_ref", "T_max_K", "dp_Pa", "converged", "accepted",
          "end_time_s", "wall_time_s", "status"]


def norm(x):
    """Inputs in box-normalised coordinates, so distances in u and q are comparable."""
    x = np.atleast_2d(x)
    return (x - BOX[:, 0]) / (BOX[:, 1] - BOX[:, 0])


def fit_gps(x, y):
    warnings.simplefilter("ignore", ConvergenceWarning)
    return [cs.make_gp().fit(x, y[:, i]) for i in range(y.shape[1])]


def gp_predict(gps, x):
    out = [g.predict(np.atleast_2d(x), return_std=True) for g in gps]
    return (np.column_stack([o[0] for o in out]), np.column_stack([o[1] for o in out]))


def candidates():
    uu, qq = np.meshgrid(GRID_U, GRID_Q, indexing="ij")
    return np.column_stack([uu.ravel(), qq.ravel()])


def propose(x_train, y_train, blocked):
    """Grid point with the largest normalised predictive std that is not already sampled and is
    not within BLOCK_RADIUS of a blocked point. Returns (u, q, score, grid, score_grid)."""
    gps = fit_gps(x_train, y_train)
    grid = candidates()
    _, sd = gp_predict(gps, grid)
    score = (sd / y_train.std(axis=0)).max(axis=1)
    keep = np.ones(len(grid), bool)
    for p in list(x_train) + list(blocked):
        r = 1e-9 if any(np.allclose(p, t) for t in x_train) else BLOCK_RADIUS
        keep &= np.linalg.norm(norm(grid) - norm(p), axis=1) > r
    s = np.where(keep, score, -np.inf)
    i = int(np.argmax(s))
    return float(grid[i, 0]), float(grid[i, 1]), float(score[i]), grid, score


def load_training():
    df = pd.read_csv(SWEEP_CSV)
    df["dT_max_K"] = df["T_max_K"] - ts.T_INLET
    all_sweep = df
    df = ts.select_cases(df, True)
    blocked = all_sweep[~all_sweep["converged"].fillna(False).astype(bool)][["u_in", "q_ref"]].to_numpy()
    return df[ts.FEATURES].to_numpy(), df[ts.TARGETS].to_numpy(), blocked


def load_rounds():
    if not CASES_CSV.exists():
        return pd.DataFrame(columns=FIELDS)
    df = pd.read_csv(CASES_CSV)
    df["dT_max_K"] = df["T_max_K"] - ts.T_INLET
    return df


def training_state(rounds):
    x, y, blocked = load_training()
    blocked = list(blocked)
    for _, r in rounds.iterrows():
        if r["accepted"]:
            x = np.vstack([x, [r["u_in"], r["q_ref"]]])
            y = np.vstack([y, [r["dT_max_K"], r["dp_Pa"]]])
        else:
            blocked.append([r["u_in"], r["q_ref"]])
    return x, y, blocked


def run_cfd(u, q):
    """One steady case with the sweep settings, then the continuation if it is not converged."""
    d = run_sweep.prepare(u, q, root=RUNS)
    shutil.copytree(run_sweep.TEMPLATE / "constant" / "polyMesh", d / "constant" / "polyMesh",
                    dirs_exist_ok=True)
    run_sweep.execute(d)
    m = case_metrics(d)
    wall = m["wall_time_s"] or 0.0
    status = "converged after 8000 iterations" if m["converged"] else ""
    if not m["converged"]:
        lower_relaxation(d, 10000, 0.1, 0.2)
        with open(d / "log.solver_continue", "w") as fh:
            subprocess.run(["buoyantSimpleFoam"], cwd=d, stdout=fh, stderr=subprocess.STDOUT, check=True)
        m = case_metrics(d, log_name="log.solver_continue")
        wall += run_time(d / "log.solver_continue") or 0.0
        status = "converged after continuation to 10000" if m["converged"] else "not converged (unsteady)"
    m["wall_time_s"] = wall
    return m, status


def append_row(row):
    new = not CASES_CSV.exists()
    with open(CASES_CSV, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)


def cmd_run(args):
    RESULTS.mkdir(exist_ok=True)
    RUNS.mkdir(parents=True, exist_ok=True)
    while True:
        rounds = load_rounds()
        n_acc, n_run = int(rounds["accepted"].sum()) if len(rounds) else 0, len(rounds)
        if n_acc >= args.rounds or n_run >= args.max_runs:
            break
        x, y, blocked = training_state(rounds)
        u, q, score, _, _ = propose(x, y, blocked)
        name = run_sweep.case_name(u, q)
        print(f"round {n_run + 1}: proposing {name} (normalised GP std {score:.3f}), "
              f"{len(x)} training cases", flush=True)
        m, status = run_cfd(u, q)
        row = {"round": n_run + 1, "case": name, "u_in": u, "q_ref": q, "T_max_K": m["T_max_K"],
               "dp_Pa": m["dp_Pa"], "converged": m["converged"], "accepted": bool(m["converged"]),
               "end_time_s": m["end_time_s"], "wall_time_s": m["wall_time_s"], "status": status}
        append_row(row)
        print(f"  -> T_max {m['T_max_K']:.2f} K, dp {m['dp_Pa']:.4f} Pa, {status}", flush=True)
    print(f"{n_acc} accepted cases, {n_run} CFD runs")


def cmd_propose(args):
    x, y, blocked = training_state(load_rounds())
    u, q, score, _, _ = propose(x, y, blocked)
    print(f"next proposal: u_in = {u}, q_ref = {q}, normalised GP std = {score:.3f}")


def loo_metrics(x, y, low_mask):
    out = {}
    for name, (factory, with_std) in {"mlp": (lambda: ts.make_model(42), False),
                                      "gp": (cs.make_gp, True)}.items():
        pred = np.zeros_like(y)
        for tr, te in LeaveOneOut().split(x):
            mu, sd = y[tr].mean(axis=0), y[tr].std(axis=0)
            if name == "gp":
                pred[te] = [g.predict(x[te])[0] for g in fit_gps(x[tr], y[tr])]
            else:
                pred[te] = factory().fit(x[tr], (y[tr] - mu) / sd).predict(x[te]) * sd + mu
        for i, t in enumerate(["dT", "dp"]):
            out[f"{name}_loo_R2_{t}"] = r2_score(y[:, i], pred[:, i])
            out[f"{name}_loo_MAE_{t}"] = mean_absolute_error(y[:, i], pred[:, i])
            out[f"{name}_loo_MAE_{t}_lowU"] = mean_absolute_error(y[low_mask, i], pred[low_mask, i])
    return out


def cmd_summarize(args):
    rounds = load_rounds()
    if rounds.empty:
        sys.exit("no active-learning cases yet")
    x0, y0, blocked0 = load_training()
    grid = candidates()
    low_grid = grid[:, 0] <= LOW_U
    acc = rounds[rounds["accepted"].astype(bool)].reset_index(drop=True)
    x, y = x0.copy(), y0.copy()
    table = []

    def models(xx, yy):
        mu, sd = yy.mean(axis=0), yy.std(axis=0)
        mlp = ts.make_model(42).fit(xx, (yy - mu) / sd)
        return fit_gps(xx, yy), (lambda z: mlp.predict(z) * sd + mu)

    gps, mlp = models(x, y)
    _, sd = gp_predict(gps, grid)
    s = sd / y.std(axis=0)
    table.append({"stage": "start (16 sweep cases)", "n_train": len(x), **{
        "gp_std_low_mean": float(s[low_grid].max(axis=1).mean()),
        "gp_std_low_max": float(s[low_grid].max())},
        **loo_metrics(x, y, x[:, 0] <= LOW_U)})
    for k, r in rounds.iterrows():
        # prospective error: the models trained on everything accepted before this round
        xn = np.array([[r["u_in"], r["q_ref"]]])
        mu_g, sd_g = gp_predict(gps, xn)
        mu_m = mlp(xn)
        truth = np.array([r["dT_max_K"], r["dp_Pa"]])
        row = {"stage": f"round {int(r['round'])}: {r['case']}", "accepted": bool(r["accepted"]),
               "n_train": len(x), "truth_dT_K": truth[0], "truth_dp_Pa": truth[1],
               "gp_pred_dT_K": mu_g[0, 0], "gp_sd_dT_K": sd_g[0, 0], "mlp_pred_dT_K": mu_m[0, 0],
               "gp_pred_dp_Pa": mu_g[0, 1], "gp_sd_dp_Pa": sd_g[0, 1], "mlp_pred_dp_Pa": mu_m[0, 1],
               "gp_err_dT_K": mu_g[0, 0] - truth[0], "mlp_err_dT_K": mu_m[0, 0] - truth[0],
               "gp_err_dp_Pa": mu_g[0, 1] - truth[1], "mlp_err_dp_Pa": mu_m[0, 1] - truth[1],
               "gp_z_dT": (mu_g[0, 0] - truth[0]) / sd_g[0, 0],
               "gp_z_dp": (mu_g[0, 1] - truth[1]) / sd_g[0, 1]}
        table.append(row)
        if r["accepted"]:
            x, y = np.vstack([x, xn]), np.vstack([y, truth])
            gps, mlp = models(x, y)
            _, sd = gp_predict(gps, grid)
            s = sd / y.std(axis=0)
            table.append({"stage": f"after round {int(r['round'])} (+{len(x) - len(x0)})",
                          "n_train": len(x), "gp_std_low_mean": float(s[low_grid].max(axis=1).mean()),
                          "gp_std_low_max": float(s[low_grid].max()),
                          **loo_metrics(x, y, x[:, 0] <= LOW_U)})
    # error of each stage's models on the accepted cases that came LATER (never seen by that stage)
    xs, ys = x0.copy(), y0.copy()
    for j in range(len(acc) + 1):
        later = acc.iloc[j:]
        if len(later):
            gps_j, mlp_j = models(xs, ys)
            xl = later[["u_in", "q_ref"]].to_numpy()
            tl = later[["dT_max_K", "dp_Pa"]].to_numpy()
            pg, pm = gp_predict(gps_j, xl)[0], mlp_j(xl)
            label = "start (16 sweep cases)" if j == 0 else f"after round {int(acc.iloc[j - 1]['round'])} (+{j})"
            for t in table:
                if t["stage"] == label:
                    t.update({"later_n": len(later),
                              "gp_MAE_dT_later": float(np.abs(pg[:, 0] - tl[:, 0]).mean()),
                              "mlp_MAE_dT_later": float(np.abs(pm[:, 0] - tl[:, 0]).mean()),
                              "gp_MAE_dp_later": float(np.abs(pg[:, 1] - tl[:, 1]).mean()),
                              "mlp_MAE_dp_later": float(np.abs(pm[:, 1] - tl[:, 1]).mean())})
        if j < len(acc):
            xs = np.vstack([xs, acc.iloc[j][["u_in", "q_ref"]].to_numpy(float)])
            ys = np.vstack([ys, acc.iloc[j][["dT_max_K", "dp_Pa"]].to_numpy(float)])
    out = pd.DataFrame(table)
    out.to_csv(RESULTS / "active_learning_rounds.csv", index=False)
    (RESULTS / "active_learning_summary.json").write_text(json.dumps(table, indent=2, default=float))
    with pd.option_context("display.width", 250, "display.max_columns", 50):
        print(out.round(4).to_string(index=False))
    make_figure(rounds, x0, y0, blocked0, out)


def make_figure(rounds, x0, y0, blocked0, table):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    sys.path.insert(0, str(Path(__file__).parent))
    import make_figures as mf          # shared look, so this figure matches the others
    mf.style()
    gps = fit_gps(x0, y0)
    uu, qq = np.meshgrid(GRID_U, GRID_Q, indexing="ij")
    grid = np.column_stack([uu.ravel(), qq.ravel()])
    _, sd = gp_predict(gps, grid)
    s = (sd / y0.std(axis=0)).max(axis=1).reshape(uu.shape)
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.9), gridspec_kw={"width_ratios": [1.15, 1]})
    mf.headline(fig, "Active learning: letting the Gaussian process choose the next CFD cases",
                "Left: where the surrogate was least certain and what it asked for. "
                "Right: its error on each new case, predicted before the case was run.", header_in=0.95)
    fig.subplots_adjust(bottom=0.27, wspace=0.28, left=0.07, right=0.97)

    a0 = ax[0]
    a0.grid(False)
    c = a0.contourf(uu, qq, s, levels=12, cmap=mf.COOL)
    cb = fig.colorbar(c, ax=a0, pad=0.02, fraction=0.045)
    cb.set_label("relative GP uncertainty (std / data spread)", color=mf.INK_2)
    cb.outline.set_visible(False)
    a0.scatter(x0[:, 0], x0[:, 1], c=mf.INK, s=22, zorder=3)
    a0.scatter(blocked0[:, 0], blocked0[:, 1], facecolors="white", edgecolors=mf.INK, s=44, lw=1.2, zorder=3)
    for _, r in rounds.iterrows():
        ok = bool(r["accepted"])
        a0.scatter(r["u_in"], r["q_ref"], marker="*" if ok else "X", s=250 if ok else 100,
                   c=mf.ORANGE if ok else "#b2182b", edgecolors="w", linewidths=1.0, zorder=5)
        a0.annotate(str(int(r["round"])), (r["u_in"], r["q_ref"]), xytext=(7, 6), textcoords="offset points",
                    fontsize=9, fontweight="bold", color=mf.INK, zorder=6)
    a0.set_xlabel("inlet velocity $u_{in}$ (m/s)")
    a0.set_ylabel("heat flux $q_{ref}$ (W/m²)")
    a0.set_ylim(80, 640)
    a0.set_xlim(0.18, 1.02)
    a0.set_title("Where the GP was uncertain, and what it asked for")
    handles = [Line2D([], [], marker="o", ls="", color=mf.INK, ms=5, label="converged sweep case (training)"),
               Line2D([], [], marker="o", ls="", mfc="w", mec=mf.INK, ms=7, label="unsteady sweep case (excluded)"),
               Line2D([], [], marker="*", ls="", color=mf.ORANGE, mec="w", ms=14, label="new case, accepted"),
               Line2D([], [], marker="X", ls="", color="#b2182b", mec="w", ms=9, label="new case, not converged")]
    a0.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2, fontsize=8.5,
              columnspacing=1.6, handletextpad=0.5)

    a1 = ax[1]
    acc = rounds[rounds["accepted"].astype(bool)]
    rows = table[table["stage"].str.startswith("round")]
    rows = rows[rows["accepted"] == True]  # noqa: E712
    xs = np.arange(len(rows))
    w = 0.34
    a1.bar(xs - w / 2, rows["gp_err_dT_K"].abs(), w, color=mf.BLUE, label="GP error", zorder=3)
    a1.bar(xs + w / 2, rows["mlp_err_dT_K"].abs(), w, color=mf.AQUA, label="MLP error", zorder=3)
    a1.scatter(xs - w / 2, 1.96 * rows["gp_sd_dT_K"], marker="_", s=420, c=mf.INK, lw=2, zorder=4,
               label="GP 95 % half-width")
    for x, v in zip(xs - w / 2, rows["gp_err_dT_K"].abs()):
        a1.text(x, v + 0.12, f"{v:.1f}", ha="center", fontsize=8, color=mf.INK_2)
    for x, v in zip(xs + w / 2, rows["mlp_err_dT_K"].abs()):
        a1.text(x, v + 0.12, f"{v:.1f}", ha="center", fontsize=8, color=mf.INK_2)
    a1.set_xticks(xs)
    a1.set_xticklabels([f"round {int(r)}\n{c}" for c, r in zip(acc["case"], acc["round"])], fontsize=8.5)
    a1.set_ylabel(r"$|$prediction $-$ CFD$|$ of $T_{max}$ rise (K)")
    a1.set_ylim(0, max(rows["mlp_err_dT_K"].abs().max(), rows["gp_err_dT_K"].abs().max()) * 1.12)
    a1.grid(axis="x", visible=False)
    a1.set_title("Prospective error on each accepted case")
    a1.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=3, fontsize=8.5, columnspacing=1.4)
    out = ROOT / "docs" / "img" / "active_learning.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    print("wrote", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["run", "propose", "summarize"])
    ap.add_argument("--rounds", type=int, default=4, help="accepted (converged) new cases to collect")
    ap.add_argument("--max-runs", type=int, default=8, help="CFD runs allowed in total, converged or not")
    args = ap.parse_args()
    {"run": cmd_run, "propose": cmd_propose, "summarize": cmd_summarize}[args.stage](args)


if __name__ == "__main__":
    main()
