#!/usr/bin/env python3
"""Compare three surrogates by leave-one-out on the converged cases: the MLP of
train_surrogate.py, a Gaussian process and a quadratic polynomial.

I add this because with 16 points a neural network is not obviously the best
choice. A Gaussian process (GP) also returns an uncertainty, which I use for two
things: a 95 % interval on every prediction, and a check of how often the true CFD
value of the left-out case lies inside that interval (coverage). I also flag any query
outside the sampled box of u_in and q_ref, where none of the models may be trusted.

    python3 scripts/compare_surrogates.py [--csv results/sweep_results.csv]

Writes results/surrogate_comparison.json. The existing results/surrogate_metrics.json
is not changed.
"""
import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import LeaveOneOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

import train_surrogate as ts  # noqa: E402  (scripts/ is on sys.path when run as a script)


def make_gp():
    # one length scale per input (anisotropic), because velocity and heat flux act on
    # the response on different scales; the white-noise term keeps the fit from
    # interpolating small solver noise exactly
    kernel = ConstantKernel(1.0) * RBF(length_scale=[1.0, 1.0], length_scale_bounds=(0.1, 100)) \
        + WhiteKernel(1e-3, noise_level_bounds=(1e-8, 1e-1))
    return make_pipeline(StandardScaler(),
                         GaussianProcessRegressor(kernel, normalize_y=True, n_restarts_optimizer=5,
                                                  random_state=0))


def make_poly():
    return make_pipeline(StandardScaler(), PolynomialFeatures(2), Ridge(alpha=1e-3))


def loo(x, y, factory, with_std=False):
    pred, std = np.zeros_like(y), np.zeros_like(y)
    for tr, te in LeaveOneOut().split(x):
        mu, sd = y[tr].mean(axis=0), y[tr].std(axis=0)
        m = factory().fit(x[tr], (y[tr] - mu) / sd)
        if with_std:
            p, s = m.predict(x[te], return_std=True)
            pred[te] = np.atleast_2d(p) * sd + mu
            std[te] = np.atleast_2d(s).T * sd if np.ndim(s) == 1 else s * sd
        else:
            pred[te] = m.predict(x[te]) * sd + mu
    return pred, std


def out_of_range(x_new, x_train):
    """True where a query lies outside the sampled box of the training inputs."""
    x_new = np.atleast_2d(x_new)
    return ((x_new < x_train.min(axis=0)) | (x_new > x_train.max(axis=0))).any(axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="results/sweep_results.csv")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    warnings.simplefilter("ignore", ConvergenceWarning)

    df = pd.read_csv(args.csv)
    df["dT_max_K"] = df["T_max_K"] - ts.T_INLET
    df = ts.select_cases(df, True)
    x, y = df[ts.FEATURES].to_numpy(), df[ts.TARGETS].to_numpy()

    res = {"n_cases": len(df), "models": {}}
    runs = {"mlp": (lambda: ts.make_model(42), False), "gaussian_process": (make_gp, True),
            "quadratic_polynomial": (make_poly, False)}
    for name, (factory, with_std) in runs.items():
        pred, std = loo(x, y, factory, with_std)
        r = {}
        for i, t in enumerate(ts.TARGETS):
            r[f"loo_R2_{t}"] = r2_score(y[:, i], pred[:, i])
            r[f"loo_MAE_{t}"] = mean_absolute_error(y[:, i], pred[:, i])
            if with_std:
                inside = np.abs(y[:, i] - pred[:, i]) <= 1.96 * std[:, i]
                r[f"loo_coverage95_{t}"] = float(inside.mean())
                r[f"loo_mean_halfwidth95_{t}"] = float((1.96 * std[:, i]).mean())
        res["models"][name] = r

    # worked example of the range flag, with a velocity of 0.1 m/s below the sampled 0.2 m/s
    res["range_check_example"] = {"query_u0.1_q300_out_of_range": bool(out_of_range([0.1, 300], x)[0]),
                                  "query_u0.5_q300_out_of_range": bool(out_of_range([0.5, 300], x)[0])}
    path = Path(args.out) / "surrogate_comparison.json"
    path.write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
