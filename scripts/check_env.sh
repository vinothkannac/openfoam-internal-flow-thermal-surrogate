#!/bin/sh
# Preflight check: OpenFOAM ESI v2606 environment, tools and Python packages.
# I run this first, because every later failure ("blockMesh: command not found",
# "No module named gmsh") is faster to diagnose here than in the middle of a sweep.
ok=1
say() { printf '%s\n' "$*"; }

if [ -z "$WM_PROJECT_VERSION" ]; then
    say "FAIL  no OpenFOAM environment. Run: source <OpenFOAM install>/etc/bashrc"
    say "      (e.g. /usr/lib/openfoam/openfoam2606/etc/bashrc, or the path of your WSL/Docker install)"
    ok=0
else
    say "OK    WM_PROJECT=$WM_PROJECT  WM_PROJECT_VERSION=$WM_PROJECT_VERSION"
    [ "$WM_PROJECT_VERSION" = "v2606" ] || say "WARN  this repo targets v2606, you have $WM_PROJECT_VERSION; expect small dictionary edits"
fi

for exe in blockMesh checkMesh buoyantSimpleFoam gmshToFoam postProcess; do
    if command -v "$exe" >/dev/null 2>&1; then say "OK    $exe"; else say "FAIL  $exe not on PATH"; ok=0; fi
done

if command -v python3 >/dev/null 2>&1; then
    python3 - <<'PY' || ok=0
import importlib, sys
bad = 0
for m in ("gmsh", "numpy", "pandas", "sklearn", "matplotlib"):
    try:
        importlib.import_module(m); print("OK    python:", m)
    except Exception as e:
        print("FAIL  python:", m, "-", e); bad = 1
sys.exit(bad)
PY
else
    say "FAIL  python3 not found"; ok=0
fi

[ "$ok" = 1 ] && say "\nEnvironment ready." || { say "\nFix the FAIL lines above (pip install -r requirements.txt for Python)."; exit 1; }
