#!/usr/bin/env python3
"""Transient-aware surrogate (roadmap item 11).

The steady surrogate (train_surrogate.py) is trained on converged cases only and says nothing about the
unsteady corner (low velocity). This script covers it with three parts:

  1. regime  : logistic regression, P(unsteady | u_in, q_ref), from the steady-solver outcome of every case
               (converged = steady, not converged = unsteady);
  2. steady  : the MLP of train_surrogate.py on the converged cases (T_max rise, dp);
  3. unsteady: for the cases with a transient run, a ridge-linear model of the offset
               (time-mean T_max - steady-solver T_max) and of the standard deviation of T_max.

Prediction: expected T_max = T_steady + P(unsteady) * offset, plus the expected oscillation amplitude
P(unsteady) * std. Evaluation is leave-one-out over all cases, against the steady-only surrogate, whose
target for an unsteady case is the time mean from the transient run. Because the combined score mixes the three
parts, I also score each part against a simple baseline: the classifier against always answering with the more
common regime, and the two ridge models against the mean of the training cases. The transient runs use linearUpwind for
U and limitedLinear for h, the steady ones use upwind, so the offset contains the scheme effect as well as the
unsteadiness (see README). Treat it as "steady-solver value -> transient-run value".

    python3 scripts/transient_surrogate.py            # -> results/transient_surrogate.json, _loo.csv
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import train_surrogate as ts

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"
T_IN = ts.T_INLET


def load_cases():
    """One row per case: u_in, q_ref, steady-solver T_max/dp, unsteady flag, transient mean/std (if run)."""
    sw = pd.read_csv(R / "sweep_results.csv")
    sw = sw[["case", "u_in", "q_ref", "T_max_K", "dp_Pa", "converged"]]
    al = pd.read_csv(R / "active_learning_cases.csv")
    al = al[["case", "u_in", "q_ref", "T_max_K", "dp_Pa", "accepted"]].rename(columns={"accepted": "converged"})
    df = pd.concat([sw, al], ignore_index=True)
    df["converged"] = df["converged"].fillna(False).astype(bool)
    df["unsteady"] = ~df["converged"]
    tr = [pd.read_csv(R / f) for f in ("transient_results.csv", "transient_active_results.csv")
          if (R / f).exists()]
    tr = pd.concat(tr, ignore_index=True)[["case", "T_max_mean_K", "T_max_std_K", "dp_mean_Pa"]]
    df = df.merge(tr, on="case", how="left")
    df["offset_K"] = df["T_max_mean_K"] - df["T_max_K"]
    return df


def feats(df):
    return np.column_stack([np.log(df["u_in"]), np.log(df["q_ref"])])


class TransientSurrogate:
    def fit(self, df):
        self.lo, self.hi = df[["u_in", "q_ref"]].min().to_numpy(), df[["u_in", "q_ref"]].max().to_numpy()
        self.clf = make_pipeline(StandardScaler(), LogisticRegression(C=10.0)).fit(feats(df), df["unsteady"])
        st = df[df["converged"]]
        y = (st["T_max_K"] - T_IN).to_numpy()[:, None]
        self.steady = ts.fit_scaled(st[ts.FEATURES].to_numpy(), np.column_stack([y, st["dp_Pa"]]), 42)
        un = df[df["unsteady"] & df["offset_K"].notna()]
        self.off = make_pipeline(StandardScaler(), Ridge(alpha=1.0)).fit(feats(un), un["offset_K"])
        self.sd = make_pipeline(StandardScaler(), Ridge(alpha=1.0)).fit(feats(un), un["T_max_std_K"])
        return self

    def predict(self, u, q):
        x = np.array([[u, q]], float)
        f = np.log(x)
        p = float(self.clf.predict_proba(f)[0, 1])
        dT, dp = self.steady(x)[0]
        off, sd = float(self.off.predict(f)[0]), max(float(self.sd.predict(f)[0]), 0.0)
        return {"p_unsteady": p, "T_steady_K": T_IN + dT, "dp_steady_Pa": dp,
                "T_expected_K": T_IN + dT + p * off, "T_unsteady_mean_K": T_IN + dT + off,
                "T_osc_std_K": p * sd,
                "in_range": bool(np.all(x[0] >= self.lo) and np.all(x[0] <= self.hi))}


def part_scores(df):
    """Leave-one-out over the unsteady cases only, one part at a time, next to the plain training mean.
    The offset here is added to the steady CFD value of the left-out case, so the classifier and the
    MLP play no role in it."""
    rows = []
    for i in df.index[df["unsteady"]]:
        tr = df.drop(index=i)
        m = TransientSurrogate().fit(tr)
        x = feats(df.loc[[i]])
        un = tr[tr["unsteady"]]
        rows.append({"offset_ridge": m.off.predict(x)[0] - df.loc[i, "offset_K"],
                     "offset_mean": un["offset_K"].mean() - df.loc[i, "offset_K"],
                     "std_ridge": m.sd.predict(x)[0] - df.loc[i, "T_max_std_K"],
                     "std_mean": un["T_max_std_K"].mean() - df.loc[i, "T_max_std_K"],
                     # the full prediction if I already knew that the case is unsteady
                     "known_regime": m.predict(df.loc[i, "u_in"], df.loc[i, "q_ref"])["T_unsteady_mean_K"]
                                     - df.loc[i, "T_max_mean_K"]})
    return {k + "_MAE_K": float(v) for k, v in pd.DataFrame(rows).abs().mean().items()}


def main():
    df = load_cases()
    n_tr = int(df["offset_K"].notna().sum())
    truth = np.where(df["unsteady"], df["T_max_mean_K"], df["T_max_K"])   # what a user wants: the time mean if unsteady
    keep = ~np.isnan(truth)                                                # unsteady cases without a transient run are dropped
    df, truth = df[keep].reset_index(drop=True), truth[keep]
    rows = []
    for i in range(len(df)):
        tr = df.drop(index=i).reset_index(drop=True)
        if tr["unsteady"].sum() < 3 or tr["unsteady"].sum() == len(tr):
            continue
        m = TransientSurrogate().fit(tr)
        p = m.predict(df.loc[i, "u_in"], df.loc[i, "q_ref"])
        rows.append({"case": df.loc[i, "case"], "unsteady": bool(df.loc[i, "unsteady"]), "truth_K": truth[i],
                     "steady_only_K": p["T_steady_K"], "transient_aware_K": p["T_expected_K"],
                     "p_unsteady": p["p_unsteady"], "std_truth_K": df.loc[i, "T_max_std_K"],
                     "std_pred_K": p["T_osc_std_K"]})
    ev = pd.DataFrame(rows)
    ev["err_steady_only_K"] = ev["steady_only_K"] - ev["truth_K"]
    ev["err_aware_K"] = ev["transient_aware_K"] - ev["truth_K"]
    ev.to_csv(R / "transient_surrogate_loo.csv", index=False)
    un = ev[ev["unsteady"]]
    flagged = ev["p_unsteady"] > 0.5
    out = {
        "n_cases": int(len(df)), "n_unsteady_with_transient_run": n_tr,
        "regime_loo_accuracy": float((flagged == ev["unsteady"]).mean()),
        "regime_majority_baseline_accuracy": float(max(ev["unsteady"].mean(), 1 - ev["unsteady"].mean())),
        "regime_unsteady_found": int((flagged & ev["unsteady"]).sum()),
        "regime_false_alarms": int((flagged & ~ev["unsteady"]).sum()),
        "regime_base_rate_unsteady": float(df["unsteady"].mean()),
        "MAE_all_steady_only_K": float(ev["err_steady_only_K"].abs().mean()),
        "MAE_all_aware_K": float(ev["err_aware_K"].abs().mean()),
        "MAE_unsteady_steady_only_K": float(un["err_steady_only_K"].abs().mean()),
        "MAE_unsteady_aware_K": float(un["err_aware_K"].abs().mean()),
        "MAE_steady_steady_only_K": float(ev[~ev["unsteady"]]["err_steady_only_K"].abs().mean()),
        "MAE_steady_aware_K": float(ev[~ev["unsteady"]]["err_aware_K"].abs().mean()),
        "std_MAE_unsteady_K": float((un["std_pred_K"]).sub(un["std_truth_K"]).abs().mean()),
        "std_truth_mean_unsteady_K": float(un["std_truth_K"].mean()),
        "offset_mean_K": float(df["offset_K"].mean()), "offset_min_K": float(df["offset_K"].min()),
        "offset_max_K": float(df["offset_K"].max()),
        "parts_unsteady_loo": part_scores(df),
    }
    (R / "transient_surrogate.json").write_text(json.dumps(out, indent=2))
    print(ev.round(2).to_string(index=False))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
