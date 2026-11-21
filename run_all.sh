#!/bin/sh
# One entry point for the whole project (run from the repo root).
# I use a single script so a reader can reproduce every number and figure with the
# same few commands I used.
#   ./run_all.sh smoke      quick check that both cases run, cavity at the N in meshParams (about 5-15 min)
#   ./run_all.sh cavity     3-mesh validation study
#   ./run_all.sh cabinet    mesh + 20-case sweep
#   ./run_all.sh transient  transient re-run of the unconverged cases + animation (about 2 h on 2 cores)
#   ./run_all.sh ml         surrogate training + results summary
#   ./run_all.sh studies    Tier-2 studies: cabinet mesh study, linearUpwind restart, transient sensitivity, active learning (hours)
#   ./run_all.sh all        cavity, cabinet, ml
# JOBS=4 ./run_all.sh cabinet   runs 4 cabinet cases in parallel
cd "$(dirname "$0")" || exit 1
JOBS="${JOBS:-1}"   # cabinet cases are independent, so I run JOBS of them at once (I used 2)

stage_smoke() {
    echo "== smoke: cavity (mesh size N from validation_cavity/system/meshParams) =="
    python3 scripts/check_rayleigh.py validation_cavity || exit 1
    (cd validation_cavity && ./Allclean && ./Allrun) || exit 1
    echo "== smoke: cabinet, coarse mesh, 1 case, 300 iterations =="
    (cd cabinet_case && ./Allclean && SCALE=2 ./Allmesh) || exit 1
    python3 scripts/run_sweep.py --velocities 0.5 --loads 300 --end-time 300 || exit 1
    echo "Smoke test finished. Look at results/sweep_results.csv and the logs in cabinet_case/runs/*/log.solver"
}
# Each stage ends by building its figures into docs/img/ from the data it just produced.
stage_cavity()  { python3 scripts/check_rayleigh.py validation_cavity && python3 scripts/mesh_study.py \
                  && python3 scripts/make_figures.py --only cavity_fields cavity; }
stage_cabinet() { (cd cabinet_case && ./Allclean && ./Allmesh) && python3 scripts/run_sweep.py --jobs "$JOBS" \
                  && python3 scripts/make_figures.py --only cabinet_fields sweep convergence; }
stage_transient() { python3 scripts/run_transient.py --jobs "$JOBS" && python3 scripts/make_movie.py run --jobs "$JOBS" \
                  && python3 scripts/make_movie.py render && python3 scripts/recirculation.py; }
stage_ml()      { python3 scripts/train_surrogate.py --converged-only && python3 scripts/make_figures.py --only surrogate \
                  && python3 scripts/summarize_results.py; }

stage_studies() { python3 scripts/cabinet_mesh_study.py --jobs "$JOBS" && python3 scripts/linear_upwind_study.py --jobs "$JOBS" \
                  && python3 scripts/transient_sensitivity.py start && python3 scripts/transient_sensitivity.py run --jobs "$JOBS" \
                  && python3 scripts/transient_sensitivity.py collect && python3 scripts/active_learning.py run && python3 scripts/active_learning.py summarize; }

sh scripts/check_env.sh || exit 1
case "$1" in
    smoke)   stage_smoke ;;
    cavity)  stage_cavity ;;
    cabinet) stage_cabinet ;;
    transient) stage_transient ;;
    ml)      stage_ml ;;
    studies) stage_studies ;;
    all)     stage_cavity && stage_cabinet && stage_ml ;;
    *)       echo "usage: ./run_all.sh smoke|cavity|cabinet|transient|ml|studies|all"; exit 1 ;;
esac
