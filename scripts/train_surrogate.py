#!/usr/bin/env python3
"""Train a small MLP surrogate on the cabinet sweep and evaluate it.

Inputs : u_in [m/s], q_ref [W/m2]
Outputs: T_max rise above inlet temperature [K], pressure drop dp [Pa]

With only ~20 CFD cases a single hold-out split of 5 points is noisy, so I report
both a fixed hold-out split and leave-one-out cross-validation (every case is
predicted by a model that never saw it). The speed-up compares the mean CFD wall
time per case with the surrogate inference time. Figures: scripts/make_figures.py.
With --converged-only (what run_all.sh uses) I train and evaluate only on the cases
whose CFD run met the convergence flag. I do this because a surrogate can only be
as good as its training data, and the unconverged cases have values that are still
moving. Without the flag every case in the CSV is used.

    python3 scripts/train_surrogate.py [--csv results/sweep_results.csv]
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import mean_absolute_error, r2_score  # noqa: E402
from sklearn.model_selection import LeaveOneOut, train_test_split  # noqa: E402
from sklearn.neural_network import MLPRegressor  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

T_INLET = 293.15
FEATURES = ["u_in", "q_ref"]
TARGETS = ["dT_max_K", "dp_Pa"]


def select_cases(df, converged_only):
    """Rows used for training. The flag column can be empty (unknown), which I treat as
    not converged, so that a case is only used when I know it converged."""
    if not converged_only:
        return df
    return df[df["converged"].fillna(False).astype(bool)].reset_index(drop=True)


def make_model(seed=0):
    # I chose a small network (2 x 16 tanh units) with the L-BFGS solver because the
    # data set is tiny; a larger model would just memorise 20 points. tanh gives a
    # smooth response, and the small L2 penalty (alpha) keeps the weights modest.
    # Inputs are standardised so u_in (about 1) and q_ref (hundreds) are on one scale.
    return make_pipeline(
        StandardScaler(),
        MLPRegressor(hidden_layer_sizes=(16, 16), activation="tanh", solver="lbfgs",
                     alpha=1e-3, max_iter=5000, random_state=seed),
    )


def fit_scaled(x_tr, y_tr, seed=0):
    """Fit model on standardised targets, return predict function in physical units.
    I standardise the targets too, because the temperature rise (tens of K) and the
    pressure drop (about 1 Pa) differ by orders of magnitude and the loss would
    otherwise favour the larger one."""
    mu, sd = y_tr.mean(axis=0), y_tr.std(axis=0)
    model = make_model(seed).fit(x_tr, (y_tr - mu) / sd)
    return lambda x: model.predict(x) * sd + mu


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="results/sweep_results.csv")
    ap.add_argument("--out", default="results")
    ap.add_argument("--test-size", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--converged-only", action="store_true",
                    help="train and evaluate only on cases flagged converged")
    args = ap.parse_args()

    df_all = pd.read_csv(args.csv)
    df_all["dT_max_K"] = df_all["T_max_K"] - T_INLET
    df = select_cases(df_all, args.converged_only)
    x, y = df[FEATURES].to_numpy(), df[TARGETS].to_numpy()
    out = Path(args.out)
    out.mkdir(exist_ok=True)

    # 1) fixed hold-out split
    x_tr, x_te, y_tr, y_te = train_test_split(x, y, test_size=args.test_size, random_state=args.seed)
    predict = fit_scaled(x_tr, y_tr, args.seed)
    y_hat = predict(x_te)
    metrics = {"converged_only": bool(args.converged_only), "n_cases": len(df),
               "n_excluded": len(df_all) - len(df), "n_train": len(x_tr), "n_test": len(x_te)}
    for i, name in enumerate(TARGETS):
        metrics[f"holdout_R2_{name}"] = r2_score(y_te[:, i], y_hat[:, i])
        metrics[f"holdout_MAE_{name}"] = mean_absolute_error(y_te[:, i], y_hat[:, i])

    # 2) leave-one-out cross-validation
    loo_pred = np.zeros_like(y)
    for tr, te in LeaveOneOut().split(x):
        loo_pred[te] = fit_scaled(x[tr], y[tr], args.seed)(x[te])
    for i, name in enumerate(TARGETS):
        metrics[f"loo_R2_{name}"] = r2_score(y[:, i], loo_pred[:, i])
        metrics[f"loo_MAE_{name}"] = mean_absolute_error(y[:, i], loo_pred[:, i])

    # 3) speed-up against CFD
    predict_all = fit_scaled(x, y, args.seed)
    t0 = time.perf_counter()
    for _ in range(1000):
        predict_all(x[:1])
    t_inf = (time.perf_counter() - t0) / 1000
    # CFD cost per case is averaged over all 20 runs, because that is what a CFD run costs,
    # whether or not I later use its result for training.
    if "wall_time_s" in df_all and df_all["wall_time_s"].notna().any():
        t_cfd = float(df_all["wall_time_s"].mean())
        metrics["cfd_wall_time_s_mean"] = t_cfd
        metrics["surrogate_inference_s"] = t_inf
        metrics["speedup"] = t_cfd / t_inf

    (out / "surrogate_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
