"""
Regenerates every significance test that is NOT part of analyze.py's ladder table.

Writes four files into results/:
    significance_vs_best.csv   each ladder config against the best one (the
                               evidence behind the configuration chosen by
                               PROTOCOL.md section 8)
    significance_anchors.csv   H4 (dead columns) and H5 (op_condition index)
    significance_rq2.csv       Experiment 2: does reading cycles in order help?
    table3_metrics.csv         RMSE, MAE and PHM08 score per configuration
    table4_cost.csv            training and prediction seconds per model

Method is identical to analyze.py: paired bootstrap over test engines, 10,000
resamples, predictions averaged across seeds first, as specified in PROTOCOL.md
section 7. Run it after analyze.py.

    python analyze_extra.py
"""
import glob
import pandas as pd
import numpy as np
from analyze import paired_bootstrap

RESULTS = "results"
DS = ["FD001", "FD002", "FD003", "FD004"]
LADDER = ["C1_base", "C2_rolling", "C3_lag", "C4_trend", "C5_agg", "C6_full"]
BEST = "C6_full"          # best-scoring ladder config, averaged over subsets


def preds(exp, cfg, ds, model="lstm"):
    return sorted(glob.glob(f"{RESULTS}/predictions/{exp}_{cfg}_{ds}_{model}_s*.csv"))


def load(files):
    d = [pd.read_csv(f).sort_values("unit_id") for f in files]
    return d[0]["y_true"].values, np.mean([x["y_pred"].values for x in d], axis=0)


def verdict(lo, hi):
    if hi < 0:
        return "better"
    if lo > 0:
        return "worse"
    return "not measurable"


# --- 1. each ladder config vs the best one, per subset and averaged ----------
rows = []
for c in LADDER:
    per = []
    for ds in DS:
        m, (lo, hi), _ = paired_bootstrap(preds("exp1", BEST, ds), preds("exp1", c, ds))
        per.append(m)
        rows.append(dict(comparison=f"{c} vs {BEST}", dataset=ds, rmse_change=round(m, 3),
                         range_low=round(lo, 3), range_high=round(hi, 3),
                         verdict=verdict(lo, hi)))
    # averaged across the four subsets, which is what section 8 specifies
    rng = np.random.default_rng(0)
    diffs = []
    for ds in DS:
        yt, pb = load(preds("exp1", BEST, ds))
        _, pa = load(preds("exp1", c, ds))
        eb, ea = (pb - yt) ** 2, (pa - yt) ** 2
        idx = rng.integers(0, len(yt), size=(10000, len(yt)))
        diffs.append(np.sqrt(ea[idx].mean(1)) - np.sqrt(eb[idx].mean(1)))
    avg = np.mean(diffs, axis=0)
    lo, hi = np.percentile(avg, [2.5, 97.5])
    rows.append(dict(comparison=f"{c} vs {BEST}", dataset="AVERAGED", rmse_change=round(avg.mean(), 3),
                     range_low=round(lo, 3), range_high=round(hi, 3), verdict=verdict(lo, hi)))
pd.DataFrame(rows).to_csv(f"{RESULTS}/significance_vs_best.csv", index=False)
print("wrote significance_vs_best.csv")

# --- 2. the anchors: H4 and H5 ----------------------------------------------
rows = []
for a, b, h, q in [("C6_full", "C6_legacy", "H4", "do the dead columns change accuracy?"),
                   ("C6_legacy", "C6_expose", "H5", "does op_condition as a number change accuracy?")]:
    for ds in DS:
        m, (lo, hi), _ = paired_bootstrap(preds("exp1", a, ds), preds("exp1", b, ds))
        rows.append(dict(hypothesis=h, question=q, comparison=f"{b} vs {a}", dataset=ds,
                         rmse_change=round(m, 3), range_low=round(lo, 3),
                         range_high=round(hi, 3), verdict=verdict(lo, hi)))
pd.DataFrame(rows).to_csv(f"{RESULTS}/significance_anchors.csv", index=False)
print("wrote significance_anchors.csv")

# --- 3. Experiment 2 ---------------------------------------------------------
CFG = "C4_trend"
rows = []
for a, b, q in [("rf_flat", "lstm", "does reading the cycles IN ORDER help?"),
                ("rf", "rf_flat", "does seeing the 30-cycle window help at all?"),
                ("rf", "lstm", "LSTM vs Random Forest on the final cycle")]:
    for ds in DS:
        fa, fb = preds("exp2", CFG, ds, a), preds("exp2", CFG, ds, b)
        m, (lo, hi), _ = paired_bootstrap(fa, fb)
        rows.append(dict(question=q, comparison=f"{b} vs {a}", dataset=ds,
                         rmse_change=round(m, 3), range_low=round(lo, 3),
                         range_high=round(hi, 3), verdict=verdict(lo, hi),
                         seeds_a=len(fa), seeds_b=len(fb)))
pd.DataFrame(rows).to_csv(f"{RESULTS}/significance_rq2.csv", index=False)
print("wrote significance_rq2.csv")

# --- 4. metrics and cost tables ---------------------------------------------
d = pd.read_csv(f"{RESULTS}/runs.csv")
e1 = d[(d.experiment == "exp1") & (d.model == "lstm")]
t3 = e1.groupby(["config", "dataset"]).agg(
    rmse=("rmse", "mean"), mae=("mae", "mean"),
    phm08=("nasa_score", "mean"), n_features=("n_features", "mean")).round(2)
t3.to_csv(f"{RESULTS}/table3_metrics.csv")
print("wrote table3_metrics.csv")

t4 = d.groupby(["experiment", "model"]).agg(
    train_seconds=("train_seconds", "mean"),
    predict_seconds=("predict_seconds", "mean"), runs=("rmse", "size")).round(2)
t4.to_csv(f"{RESULTS}/table4_cost.csv")
print("wrote table4_cost.csv")
print("\nDone. All significance tests are now reproducible from committed files.")
