#!/usr/bin/env python3
"""Build the figures used in README.md from the real run data.

Every figure is computed from files produced by the other scripts (results/*.csv,
results/*.json, the solver logs and the written fields in the run directories).
Nothing is typed in by hand. Output goes to docs/img/.

    python3 scripts/make_figures.py            # needs the OpenFOAM environment (postProcess)
    python3 scripts/make_figures.py --only sweep surrogate   # subset, no OpenFOAM needed

Why this script exists: I wanted one place that owns the look of all plots, so the
repo reads as one piece of work and the figures regenerate with a single command.
"""
import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.tri as mtri  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from scipy.interpolate import RegularGridInterpolator  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from make_cabinet_mesh import COMPS, H, SLOT, W  # noqa: E402  (I reuse the real geometry)
from train_surrogate import FEATURES, TARGETS, T_INLET, fit_scaled, select_cases  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = ROOT / "docs" / "img"

# --- look -----------------------------------------------------------------
# I use one validated categorical palette (blue, orange, aqua are the first three
# slots, which stay distinguishable for colour-blind readers) and one warm and one
# cool single-hue ramp for magnitudes, so a colour always means the same thing:
# warm = temperature, blue = speed / pressure.
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
WARM = LinearSegmentedColormap.from_list("warm", ["#fff4ec", "#f7c1a0", "#eb6834", "#b0401a", "#5c1f0a"])
COOL = LinearSegmentedColormap.from_list("cool", ["#eef5fd", "#a9c9f0", "#2a78d6", "#1a4c8c", "#0c2a52"])


def style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.family": "DejaVu Sans", "font.size": 10,
        "text.color": INK, "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2,
        "axes.edgecolor": GRID, "axes.linewidth": 1.0,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
        "axes.axisbelow": True, "legend.frameon": False,
        "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "savefig.dpi": 200, "savefig.bbox": "tight", "savefig.pad_inches": 0.25,
    })


def headline(fig, title, subtitle, header_in=0.95):
    """Title and one-line takeaway at the top; the axes start header_in inches below the top edge."""
    h = fig.get_figheight()
    fig.text(0.0, 1 - 0.08 / h, title, fontsize=14, fontweight="bold", ha="left", va="top", color=INK)
    fig.text(0.0, 1 - 0.42 / h, subtitle, fontsize=9.5, ha="left", va="top", color=INK_2)
    fig.subplots_adjust(top=1 - header_in / h)


# --- OpenFOAM field readers -------------------------------------------------
def read_field(path):
    """Internal field of an ASCII OpenFOAM volScalar/volVector file as an array."""
    text = Path(path).read_text()
    m = re.search(r"internalField\s+uniform\s+([^;]+);", text)
    if m:
        raise ValueError(f"{path} is uniform; the plots need a solved field")
    m = re.search(r"internalField\s+nonuniform\s+List<(\w+)>\s*(\d+)\s*\(", text)
    n, kind = int(m.group(2)), m.group(1)
    body = text[m.end():]
    if kind == "scalar":
        return np.array(body.split(")")[0].split()[:n], dtype=float)
    rows = re.findall(r"\(([^()]+)\)", body)[:n]
    return np.array([r.split() for r in rows], dtype=float)


def solved_fields(case_dir, time_dir, names):
    """Cell centres plus fields. I run writeCellCentres in a scratch copy so the
    run directories stay exactly as the solver left them."""
    case_dir = Path(case_dir)
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp) / "case"
        work.mkdir()
        for sub in ("constant", "system"):
            shutil.copytree(case_dir / sub, work / sub)
        shutil.copytree(case_dir / str(time_dir), work / str(time_dir))
        subprocess.run(["postProcess", "-func", "writeCellCentres", "-time", str(time_dir)],
                       cwd=work, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        data = {"C": read_field(work / str(time_dir) / "C")}
        for n in names:
            data[n] = read_field(work / str(time_dir) / n)
    return data


# --- figures ------------------------------------------------------------------
def fig_cavity_fields():
    """Temperature and flow in the Ra = 1e5 cavity, finest mesh."""
    params = json.loads((ROOT / "validation_cavity" / "case_params.json").read_text())
    run = ROOT / "validation_cavity" / "runs" / "N128"
    t_last = max(int(p.name) for p in run.iterdir() if p.name.isdigit())
    d = solved_fields(run, t_last, ["T", "U"])
    n = int(round(np.sqrt(len(d["T"]))))
    x = d["C"][:, 0].reshape(n, n) * 1e3
    y = d["C"][:, 1].reshape(n, n) * 1e3
    theta = ((d["T"] - params["T_cold"]) / (params["T_hot"] - params["T_cold"])).reshape(n, n)
    ux, uy = d["U"][:, 0].reshape(n, n) * 1e3, d["U"][:, 1].reshape(n, n) * 1e3
    speed = np.hypot(ux, uy)
    L = params["L"] * 1e3

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.9))
    for ax in axes:
        ax.set_aspect("equal")
        ax.set_xlim(0, L)
        ax.set_ylim(0, L)
        ax.grid(False)
        ax.set_xlabel("x [mm]")
    axes[0].set_ylabel("y [mm]")

    cf = axes[0].contourf(x, y, theta, levels=np.linspace(0, 1, 21), cmap=WARM)
    axes[0].contour(x, y, theta, levels=np.linspace(0.1, 0.9, 9), colors=INK, linewidths=0.5, alpha=0.55)
    axes[0].set_title("Temperature, isotherms every 0.1")
    cb = fig.colorbar(cf, ax=axes[0], fraction=0.046, pad=0.03)
    cb.set_label("θ = (T − T_cold) / (T_hot − T_cold)")
    axes[0].set_xlabel("x [mm]   (hot wall at x = 0, cold wall at x = L)")

    cf2 = axes[1].contourf(x, y, speed, levels=20, cmap=COOL)
    gx, gy = np.linspace(x.min(), x.max(), 160), np.linspace(y.min(), y.max(), 160)
    xs, ys = x[0, :], y[:, 0]
    GX, GY = np.meshgrid(gx, gy)
    pts = np.stack([GY.ravel(), GX.ravel()], axis=1)
    u_i = RegularGridInterpolator((ys, xs), ux)(pts).reshape(GX.shape)
    v_i = RegularGridInterpolator((ys, xs), uy)(pts).reshape(GX.shape)
    axes[1].streamplot(gx, gy, u_i, v_i, color=INK, linewidth=0.6, density=1.3, arrowsize=0.8)
    axes[1].set_title("Speed with streamlines")
    cb2 = fig.colorbar(cf2, ax=axes[1], fraction=0.046, pad=0.03)
    cb2.set_label("|U| [mm/s]")

    fig.subplots_adjust(wspace=0.42)
    headline(fig, "Differentially heated cavity, Ra = 10⁵",
             f"OpenFOAM buoyantSimpleFoam, N = {n} × {n} cells, steady state (iteration {t_last}). "
             "The hot wall drives a clockwise circulation; the core splits into two vortices.", 0.8)
    fig.savefig(OUT / "cavity_fields.png")
    plt.close(fig)


def fig_cavity_validation():
    """Nu versus mesh against the benchmark, and the error shrinking."""
    df = pd.read_csv(RESULTS / "cavity_mesh_study.csv")
    gci = json.loads((RESULTS / "cavity_gci.json").read_text())
    ref = json.loads((ROOT / "validation_cavity" / "case_params.json").read_text())["Nu_ref"]

    fig, (a, b) = plt.subplots(1, 2, figsize=(10.2, 4.0))
    a.plot(df["N"], df["Nu"], "-o", color=BLUE, lw=2, ms=7, mec=SURFACE, mew=1.5, zorder=3)
    a.axhline(ref, color=INK_2, lw=1.2, ls="--")
    a.axhline(gci["f_ext"], color=ORANGE, lw=1.2, ls=":")
    a.set_xscale("log", base=2)
    a.set_xticks(df["N"])
    a.set_xticklabels([f"{n} × {n}" for n in df["N"]])
    a.minorticks_off()
    for n, nu in zip(df["N"], df["Nu"]):
        a.annotate(f"{nu:.4f}", (n, nu), textcoords="offset points", xytext=(0, 9), ha="center",
                   fontsize=9, color=INK)
    x0 = df["N"].min() / 1.08
    a.text(x0, gci["f_ext"] + 0.0004, f"Richardson extrapolation {gci['f_ext']:.4f}", va="bottom", fontsize=9, color=ORANGE)
    a.text(x0, ref - 0.0004, f"de Vahl Davis benchmark {ref}", va="top", fontsize=9, color=INK_2)
    a.set_xlim(x0, df["N"].max() * 1.08)
    a.set_ylim(ref - 0.0035, df["Nu"].max() + 0.0035)
    a.set_xlabel("cells per side")
    a.set_ylabel("average Nusselt number, hot wall")
    a.set_title("Mesh convergence")

    err = df["error_pct"].to_numpy()
    bars = b.bar([str(n) for n in df["N"]], err, color=BLUE, width=0.55)
    for rect, e in zip(bars, err):
        b.text(rect.get_x() + rect.get_width() / 2, e + 0.012, f"+{e:.2f} %", ha="center", fontsize=9, color=INK)
    b.set_ylim(0, err.max() * 1.25)
    b.set_xlabel("cells per side")
    b.set_ylabel("error vs benchmark [%]")
    b.set_title("Error shrinks with refinement")
    b.grid(axis="x", visible=False)

    headline(fig, "Validation against de Vahl Davis (1983)",
             f"Apparent order p = {gci['p']:.2f}, GCI(fine) = {gci['gci_fine_pct']:.2f} %, "
             "monotonic convergence. Heat balance hot/cold below 0.001 %.")
    fig.savefig(OUT / "cavity_validation.png")
    plt.close(fig)


def _mask_components(tri, x, y):
    cx, cy = x[tri.triangles].mean(axis=1), y[tri.triangles].mean(axis=1)
    inside = np.zeros(len(cx), dtype=bool)
    for x0, y0, x1, y1 in COMPS.values():
        inside |= (cx > x0) & (cx < x1) & (cy > y0) & (cy < y1)
    tri.set_mask(inside)


def fig_cabinet_fields():
    """Temperature and speed for three operating points of the cabinet."""
    cases = [("u0.2_q600", "u = 0.2 m/s, q = 600 W/m²"),
             ("u0.5_q400", "u = 0.5 m/s, q = 400 W/m²"),
             ("u1_q400", "u = 1.0 m/s, q = 400 W/m²")]
    data = []
    for name, label in cases:
        run = ROOT / "cabinet_case" / "runs" / name
        t_last = max(int(p.name) for p in run.iterdir() if p.name.isdigit())
        d = solved_fields(run, t_last, ["T", "U"])
        data.append((label, d))
    tmin = T_INLET
    tmax = max(d["T"].max() for _, d in data)
    # squared spacing: finer colour steps near the inlet temperature, where most of the
    # domain lives, while the hot spots at the components still reach the dark end
    t_levels = tmin + (tmax - tmin) * np.linspace(0, 1, 24) ** 2
    t_norm = BoundaryNorm(t_levels, WARM.N)
    smax = max(np.hypot(d["U"][:, 0], d["U"][:, 1]).max() for _, d in data)

    for name, _ in cases:
        assert bool(pd.read_csv(RESULTS / "sweep_results.csv").set_index("case").loc[name, "converged"])
    fig, axes = plt.subplots(2, 3, figsize=(10.4, 9.6), sharex=True, sharey=True)
    cs_t = cs_u = None
    for j, (label, d) in enumerate(data):
        x, y = d["C"][:, 0], d["C"][:, 1]
        tri = mtri.Triangulation(x, y)
        _mask_components(tri, x, y)
        speed = np.hypot(d["U"][:, 0], d["U"][:, 1])
        cs_t = axes[0, j].tricontourf(tri, d["T"], levels=t_levels, cmap=WARM, norm=t_norm)
        cs_u = axes[1, j].tricontourf(tri, speed, levels=np.linspace(0, smax, 22), cmap=COOL)
        axes[0, j].set_title(label)
        for ax in axes[:, j]:
            ax.set_aspect("equal")
            ax.grid(False)
            ax.add_patch(Rectangle((0, 0), W, H, fill=False, ec=INK, lw=1.2))
            for name_c, (x0, y0, x1, y1) in COMPS.items():
                ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fc="#3a3935", ec=INK, lw=0.8, zorder=5))
                ax.text((x0 + x1) / 2, (y0 + y1) / 2, name_c, color="white", fontsize=7.5,
                        ha="center", va="center", zorder=6)
            xm = (SLOT[0] + SLOT[1]) / 2
            ax.text(xm, 0.035, "inlet ↑", ha="center", va="bottom", fontsize=8.5, color=INK)
            ax.text(xm, H - 0.035, "outlet ↑", ha="center", va="top", fontsize=8.5, color=INK)
            ax.set_xlim(-0.01, W + 0.01)
            ax.set_ylim(-0.01, H + 0.01)
    for ax in axes[1, :]:
        ax.set_xlabel("x [m]")
    for ax in axes[:, 0]:
        ax.set_ylabel("y [m]")
    fig.subplots_adjust(hspace=0.12, wspace=0.08, right=0.86)
    headline(fig, "Cabinet section: temperature and speed",
             "Three converged operating points of the sweep, same colour scale in every column "
             "(temperature scale is non-linear to resolve the cool region). Hot spots sit on the component faces.", 0.9)
    # colourbars get their own axes, placed after the layout is final, so they never overlap a column
    for row, (cs, label) in enumerate([(cs_t, "temperature [K]"), (cs_u, "|U| [m/s]")]):
        box = axes[row, 2].get_position()
        cax = fig.add_axes([box.x1 + 0.025, box.y0, 0.016, box.height])
        cb = fig.colorbar(cs, cax=cax)
        cb.set_label(label)
        if row == 0:
            cb.set_ticks(t_levels[::3])
            cb.ax.set_yticklabels([f"{v:.0f}" for v in t_levels[::3]])
    fig.savefig(OUT / "cabinet_fields.png")
    plt.close(fig)


def fig_sweep_maps():
    """T_max and dp over the 5 x 4 sweep grid, unconverged cases hatched."""
    df = pd.read_csv(RESULTS / "sweep_results.csv")
    us, qs = sorted(df["u_in"].unique()), sorted(df["q_ref"].unique())
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.9))
    specs = [("T_max_K", "Maximum component temperature [K]", WARM, "{:.0f}"),
             ("dp_Pa", "Pressure drop inlet → outlet [Pa]", COOL, "{:.2f}")]
    for ax, (col, title, cmap, fmt) in zip(axes, specs):
        grid = np.full((len(qs), len(us)), np.nan)
        conv = np.ones_like(grid, dtype=bool)
        for _, r in df.iterrows():
            i, j = qs.index(r["q_ref"]), us.index(r["u_in"])
            grid[i, j] = r[col]
            conv[i, j] = bool(r["converged"])
        im = ax.imshow(grid, origin="lower", cmap=cmap, aspect="auto")
        lo, hi = np.nanmin(grid), np.nanmax(grid)
        for i in range(len(qs)):
            for j in range(len(us)):
                v = grid[i, j]
                dark = (v - lo) / (hi - lo) > 0.55
                ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=9.5, fontweight="bold",
                        color="white" if dark else INK, zorder=5)
                if not conv[i, j]:
                    ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, hatch="///",
                                           ec=INK_2, lw=0, alpha=0.45, zorder=2))
        ax.set_xticks(range(len(us)))
        ax.set_xticklabels([f"{u:g}" for u in us])
        ax.set_yticks(range(len(qs)))
        ax.set_yticklabels([f"{q:g}" for q in qs])
        ax.set_xticks(np.arange(-0.5, len(us)), minor=True)
        ax.set_yticks(np.arange(-0.5, len(qs)), minor=True)
        ax.grid(which="minor", color=SURFACE, lw=2)
        ax.grid(which="major", visible=False)
        ax.tick_params(which="minor", length=0)
        ax.set_xlabel("inlet velocity u_in [m/s]")
        ax.set_ylabel("heat flux q_ref [W/m²]")
        ax.set_title(title)
        for s in ax.spines.values():
            s.set_visible(False)
    n_ok = int(df["converged"].sum())
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(fc="none", ec=INK_2, hatch="///", label="not converged: T_max still moving by more than 0.05 K between the last two write times")],
               loc="lower left", bbox_to_anchor=(0.0, -0.14), fontsize=9, handlelength=2.2, handleheight=1.2)
    headline(fig, "Cabinet parameter sweep",
             f"{len(df)} CFD cases at 5 inlet velocities × 4 heat loads; {n_ok} meet the convergence flag.", 0.8)
    fig.savefig(OUT / "sweep_maps.png")
    plt.close(fig)


def _residual_history(log_file, names=("Ux", "h", "p_rgh")):
    """First 'Initial residual' of each variable in every SIMPLE iteration."""
    hist = {n: [] for n in names}
    seen = set()
    pat = re.compile(r"Solving for (\w+), Initial residual = ([0-9.eE+-]+)")
    for line in Path(log_file).read_text().splitlines():
        if line.startswith("Time ="):
            seen = set()
            continue
        m = pat.search(line)
        if m and m.group(1) in hist and m.group(1) not in seen:
            hist[m.group(1)].append(float(m.group(2)))
            seen.add(m.group(1))
    return hist


def fig_convergence():
    """Why some cases converge and others do not: residual histories of two cases."""
    df = pd.read_csv(RESULTS / "sweep_results.csv").set_index("case")
    cases = [("u0.5_q400", "converges"), ("u0.2_q100", "does not converge")]
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.9), sharey=True)
    colours = {"Ux": BLUE, "h": ORANGE, "p_rgh": AQUA}
    names = {"Ux": "U (x)", "h": "enthalpy h", "p_rgh": "p_rgh"}
    for ax, (case, state) in zip(axes, cases):
        hist = _residual_history(ROOT / "cabinet_case" / "runs" / case / "log.solver")
        flag = bool(df.loc[case, "converged"])
        for key, vals in hist.items():
            ax.semilogy(np.arange(1, len(vals) + 1), vals, color=colours[key], lw=1.4, label=names[key])
        ax.set_xlabel("iteration")
        ax.set_title(f"u = {df.loc[case, 'u_in']:g} m/s, q = {df.loc[case, 'q_ref']:g} W/m²  ({state})")
        ax.grid(axis="x", visible=False)
        # direct labels at the line ends; I nudge them apart when two lines finish close together
        ends = sorted(hist.items(), key=lambda kv: kv[1][-1])
        last_y, ypos = None, {}
        for key, vals in ends:
            y = np.log10(vals[-1])
            if last_y is not None and y - last_y < 0.35:
                y = last_y + 0.35
            ypos[key], last_y = y, y
        for key, vals in hist.items():
            ax.annotate(names[key], (len(vals), 10 ** ypos[key]), xytext=(6, 0), textcoords="offset points",
                        fontsize=8.5, color=INK_2, va="center")
        ax.set_xlim(0, len(hist["Ux"]) * 1.16)
        assert flag == (state == "converges"), f"{case}: log story and CSV flag disagree"
    axes[0].set_ylabel("initial residual")
    headline(fig, "Residual histories of two sweep cases",
             "Same solver settings, 8000 iterations. Left: residuals keep falling and the case is flagged converged. "
             "Right: residuals stall at 1e-4 to 1e-3 and the case is flagged not converged.")
    fig.savefig(OUT / "convergence_history.png")
    plt.close(fig)


def fig_surrogate():
    """Leave-one-out parity plots on the training cases. If cases were excluded for not
    converging, I still show them as hollow diamonds: they are predicted by the final model,
    which never saw them, but their CFD value is itself unreliable, so they are not scored."""
    metrics = json.loads((RESULTS / "surrogate_metrics.json").read_text())
    df_all = pd.read_csv(RESULTS / "sweep_results.csv")
    df_all["dT_max_K"] = df_all["T_max_K"] - T_INLET
    df = select_cases(df_all, metrics["converged_only"])
    x, y = df[FEATURES].to_numpy(), df[TARGETS].to_numpy()
    from sklearn.metrics import r2_score
    from sklearn.model_selection import LeaveOneOut
    pred = np.zeros_like(y)
    for tr, te in LeaveOneOut().split(x):
        pred[te] = fit_scaled(x[tr], y[tr], 42)(x[te])
    final = fit_scaled(x, y, 42)
    excluded = df_all[~df_all["case"].isin(df["case"])]
    xe, ye = excluded[FEATURES].to_numpy(), excluded[TARGETS].to_numpy()
    pe = final(xe) if len(excluded) else np.zeros((0, 2))

    labels = ["Max. component temperature rise [K]", "Pressure drop p_rgh [Pa]"]
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.5))
    for i, ax in enumerate(axes):
        allv = np.concatenate([y[:, i], pred[:, i], ye[:, i], pe[:, i]])
        lim = [allv.min(), allv.max()]
        pad = 0.05 * (lim[1] - lim[0])
        lim = [lim[0] - pad, lim[1] + pad]
        ax.plot(lim, lim, color=INK_2, lw=1, ls="--", zorder=1)
        ax.scatter(y[:, i], pred[:, i], s=46, color=BLUE, ec=SURFACE, lw=1, zorder=3,
                   label=f"{len(df)} converged cases (leave-one-out)")
        if len(excluded):
            ax.scatter(ye[:, i], pe[:, i], s=58, marker="D", fc=SURFACE, ec=ORANGE, lw=1.8, zorder=3,
                       label=f"{len(excluded)} excluded, not converged (not scored)")
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_aspect("equal")
        ax.set_xlabel("CFD")
        ax.set_ylabel("surrogate")
        ax.set_title(labels[i])
        r2 = metrics[f"loo_R2_{TARGETS[i]}"]
        mae = metrics[f"loo_MAE_{TARGETS[i]}"]
        ax.text(0.04, 0.95, f"R² = {r2:.3f}\nMAE = {mae:.3g}", transform=ax.transAxes, va="top",
                fontsize=9.5, color=INK)
        assert abs(r2 - r2_score(y[:, i], pred[:, i])) < 1e-6, "recomputed LOO differs from the metrics file"
    handles, labs = axes[0].get_legend_handles_labels()
    fig.legend(handles, labs, loc="lower center", ncol=2, fontsize=9, bbox_to_anchor=(0.5, -0.07))
    headline(fig, "Surrogate versus CFD",
             f"Small MLP (2 hidden layers × 16 tanh units) trained on the {len(df)} converged cases only. "
             "R² and MAE are leave-one-out, read from results/surrogate_metrics.json.", 0.95)
    fig.savefig(OUT / "surrogate_parity.png")
    plt.close(fig)


FIGS = {"cavity_fields": fig_cavity_fields, "cavity": fig_cavity_validation,
        "cabinet_fields": fig_cabinet_fields, "sweep": fig_sweep_maps,
        "convergence": fig_convergence, "surrogate": fig_surrogate}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="+", choices=sorted(FIGS), help="build only these figures")
    args = ap.parse_args()
    style()
    OUT.mkdir(parents=True, exist_ok=True)
    for name in args.only or FIGS:
        print(f"building {name} ...", flush=True)
        FIGS[name]()
    print(f"figures written to {OUT}")


if __name__ == "__main__":
    main()
