#!/usr/bin/env python3
"""Animation of the four transient cabinet cases (docs/img/transient_cabinet.gif).

Two stages:
  run     restart each transient case from t = 100 s and run 25 s more, writing a field
          every 0.25 s. The main transient runs only wrote every 5 s, which is far too
          coarse to see a plume move, and I did not want to re-run 100 s just for the picture.
  render  read T and U of every written time and draw temperature, speed and the
          T_max history of each case side by side.

    python3 scripts/make_movie.py run --jobs 2     # needs the OpenFOAM environment
    python3 scripts/make_movie.py render           # needs only Python (+ postProcess once for cell centres)
"""
import argparse
import re
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.tri as mtri  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from PIL import Image  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
import make_figures as mf  # noqa: E402  (I reuse the look, the field reader and the geometry)
from make_figures import COMPS, H, SLOT, W, WARM, COOL, INK, INK_2, SURFACE  # noqa: E402
from run_transient import load_series  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "cabinet_case" / "runs_transient"
MOVIE = ROOT / "cabinet_case" / "runs_movie"
OUT = ROOT / "docs" / "img" / "transient_cabinet.gif"
CASES = [("u0.2_q100", "0.2 m/s, 100 W/m²"), ("u0.2_q200", "0.2 m/s, 200 W/m²"),
         ("u0.35_q400", "0.35 m/s, 400 W/m²"), ("u0.35_q600", "0.35 m/s, 600 W/m²")]
T_START, T_END, DT_WRITE = 100.0, 125.0, 0.25
T_INLET = 293.15


def run_case(case):
    d = MOVIE / case
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)
    shutil.copytree(SRC / case / "constant", d / "constant")
    shutil.copytree(SRC / case / "system", d / "system")
    shutil.copytree(SRC / case / "100", d / "100")
    cd = d / "system" / "controlDict"
    t = cd.read_text()
    t = re.sub(r"(?m)^startTime\s+\S+;", f"startTime       {T_START:g};", t)
    t = re.sub(r"(?m)^endTime\s+\S+;", f"endTime         {T_END:g};", t)
    t = re.sub(r"(?m)^writeInterval\s+\S+;", f"writeInterval   {DT_WRITE:g};", t)
    cd.write_text(t)
    with open(d / "log.solver", "w") as fh:
        subprocess.run(["buoyantPimpleFoam"], cwd=d, stdout=fh, stderr=subprocess.STDOUT, check=True)
    return case


def times(d):
    ts = sorted(float(p.name) for p in d.iterdir() if p.is_dir() and re.fullmatch(r"[0-9.]+", p.name))
    return [t for t in ts if t >= T_START]


def tname(t):
    return f"{t:g}"


def render(fps):
    mf.style()
    centres = None
    frames = {}
    for case, _ in CASES:
        d = MOVIE / case
        ts = times(d)
        if centres is None:
            first = mf.solved_fields(d, tname(ts[0]), ["T"])
            centres = first["C"]
        frames[case] = (ts, [(mf.read_field(d / tname(t) / "T"), mf.read_field(d / tname(t) / "U")) for t in ts])
    n = min(len(v[0]) for v in frames.values())
    ts = frames[CASES[0][0]][0][:n]
    x, y = centres[:, 0], centres[:, 1]
    tri = mtri.Triangulation(x, y)
    mf._mask_components(tri, x, y)
    smax = max(np.hypot(u[:, 0], u[:, 1]).max() for v in frames.values() for _, u in v[1][:n])
    s_levels = np.linspace(0, smax, 22)

    fig = plt.figure(figsize=(11.6, 9.4))
    gs = fig.add_gridspec(3, 4, height_ratios=[1, 1, 0.42], hspace=0.2, wspace=0.3,
                          left=0.06, right=0.94, top=0.855, bottom=0.075)
    ax_t = [fig.add_subplot(gs[0, j]) for j in range(4)]
    ax_u = [fig.add_subplot(gs[1, j]) for j in range(4)]
    ax_h = [fig.add_subplot(gs[2, j]) for j in range(4)]
    fig.text(0.0, 0.985, "Unsteady cabinet flow: the four cases a steady solver could not converge", fontsize=14,
             fontweight="bold", ha="left", va="top", color=INK)
    fig.text(0.0, 0.952, "buoyantPimpleFoam, 25 s of flow time shown. Top: temperature (linear colour scale, fixed per case). "
             "Middle: speed (one scale for all). Bottom: hottest component face over time.",
             fontsize=9.5, ha="left", va="top", color=INK_2)
    clock = fig.text(0.985, 0.985, "", fontsize=13, fontweight="bold", ha="right", va="top", color=INK)

    levels, norms, mappables = {}, {}, []
    for j, (case, label) in enumerate(CASES):
        tmax = max(f[0].max() for f in frames[case][1][:n])
        tmin = min(f[0].min() for f in frames[case][1][:n])
        lv = np.linspace(tmin, tmax, 24)   # linear, so the colourbar means what it shows
        levels[case] = lv
        norms[case] = plt.Normalize(tmin, tmax)
        ax_t[j].set_title(label)
    th = {c: load_series(MOVIE / c) for c, _ in CASES}

    def draw(k):
        for j, (case, _) in enumerate(CASES):
            T, U = frames[case][1][k]
            for ax in (ax_t[j], ax_u[j]):
                ax.clear()
                ax.set_aspect("equal"); ax.grid(False)
                ax.set_xlim(-0.01, W + 0.01); ax.set_ylim(-0.01, H + 0.01)
                ax.set_xticks([]); ax.set_yticks([])
                ax.add_patch(Rectangle((0, 0), W, H, fill=False, ec=INK, lw=1.0))
                for (x0, y0, x1, y1) in COMPS.values():
                    ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fc="#3a3935", ec=INK, lw=0.6, zorder=5))
            ax_t[j].set_title(CASES[j][1], loc="left")
            ax_t[j].tricontourf(tri, T, levels=levels[case], cmap=WARM, norm=norms[case])
            ax_u[j].tricontourf(tri, np.hypot(U[:, 0], U[:, 1]), levels=s_levels, cmap=COOL, extend="max")
            ax_t[j].text(0.02, 0.985, f"T {T.min():.0f}–{T.max():.0f} K", transform=ax_t[j].transAxes,
                         fontsize=8, va="top", color=INK)
            t, tm, _ = th[case]
            m = (t >= T_START) & (t <= ts[k])
            ax = ax_h[j]
            ax.clear()
            ax.plot(t[(t >= T_START) & (t <= T_END)], tm[(t >= T_START) & (t <= T_END)], color="#d9d8d3", lw=1.0)
            ax.plot(t[m], tm[m], color=mf.ORANGE, lw=1.6)
            ax.set_xlim(T_START, T_END)
            lo, hi = tm[(t >= T_START)].min(), tm[(t >= T_START)].max()
            ax.set_ylim(lo - 0.05 * (hi - lo), hi + 0.05 * (hi - lo))
            ax.set_xlabel("flow time [s]" if True else "")
            ax.set_ylabel("T_max [K]" if j == 0 else "")
            ax.grid(True)
        clock.set_text(f"t = {ts[k]:.2f} s")

    # colourbars live on their own axes (created once, outside the panels that I clear every frame)
    draw(0)
    for j, (case, _) in enumerate(CASES):
        sm = plt.cm.ScalarMappable(norm=norms[case], cmap=WARM)
        box = ax_t[j].get_position()
        cax = fig.add_axes([box.x0, box.y0 - 0.022, box.width, 0.011])
        cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
        ticks = np.round(np.linspace(levels[case][0], levels[case][-1], 4))
        cb.set_ticks(ticks, labels=[f"{v:.0f} K" for v in ticks])
        cb.ax.tick_params(labelsize=7.5)
        cb.ax.minorticks_off()
        cb.ax.minorticks_off()
        cb.outline.set_visible(False)
    sm = plt.cm.ScalarMappable(norm=plt.Normalize(0, smax), cmap=COOL)
    box = ax_u[3].get_position()
    cax = fig.add_axes([box.x1 + 0.02, box.y0, 0.012, box.height])
    cb = fig.colorbar(sm, cax=cax); cb.set_label("|U| [m/s]"); cb.outline.set_visible(False)

    imgs = []
    for k in range(n):
        draw(k)
        fig.canvas.draw()
        a = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
        imgs.append(Image.fromarray(a).convert("P", palette=Image.ADAPTIVE, colors=128))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    imgs[0].save(OUT, save_all=True, append_images=imgs[1:], duration=int(1000 / fps), loop=0, optimize=True)
    # a still frame so the README has a fallback and I can inspect the layout
    Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[:, :, :3]).save(OUT.with_suffix(".png"))
    print(f"wrote {OUT} ({n} frames, {OUT.stat().st_size / 1e6:.1f} MB)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["run", "render"])
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--fps", type=int, default=12)
    args = ap.parse_args()
    if args.stage == "run":
        with ThreadPoolExecutor(max_workers=args.jobs) as ex:
            for c in ex.map(run_case, [c for c, _ in CASES]):
                print("finished", c, flush=True)
    else:
        render(args.fps)


if __name__ == "__main__":
    main()
