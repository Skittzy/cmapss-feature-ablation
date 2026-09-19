"""
Leave one out analysis for RQ1.

The ladder in analyze.py measures what a feature group adds *given the groups
before it*. This file measures what a group adds *given everything else*. The
two answer different questions and are allowed to disagree, which is the whole
reason PROTOCOL.md section 3 asks for both.

Reading the two together separates two cases the ladder cannot tell apart:

    ladder null + loo null       the group is genuinely redundant
    ladder null + loo measurable the group was masked, an earlier group had
                                 already supplied the same information

Writes into results/:
    significance_loo.csv    each L_no_<group> against L_full, per subset
                            and averaged
    significance_ffill.csv  L_full against the pre fix C6_full, which measures
                            what the 14 September forward fill repair did to
                            accuracy
    table5_loo.csv          RMSE, MAE, PHM08 and feature counts per loo config

Method is identical to analyze.py: paired bootstrap over test engines, 10,000
resamples, predictions averaged across seeds first, as specified in PROTOCOL.md
section 7. Run it after analyze.py.

    python analyze_loo.py
"""
import glob
import numpy as np
import pandas as pd
from analyze import paired_bootstrap

RESULTS = "results"
DS = ["FD001", "FD004"]            # loo was run on these two only
GROUPS = ["rolling", "lag", "trend", "agg", "cycle"]
REF = "L_full"


def preds(exp, cfg, ds, model="lstm"):
    return sorted(glob.glob(f"{RESULTS}/predictions/{exp}_{cfg}_{ds}_{model}_s*.csv"))


def load(files):
    d = [pd.read_csv(f).sort_values("unit_id") for f in files]
    return d[0]["y_true"].values, np.mean([x["y_pred"].values for x in d], axis=0)


def verdict_loo(lo, hi):
    # positive means removing the group made rmse worse, so the group was doing
    # something. negative means removing it helped, so the group was harmful.
    if lo > 0:
        return "group contributes"
    if hi < 0:
        return "group harmful"
    return "not measurable"


def averaged(pairs):
    # one bootstrap draw reused across subsets, same approach as analyze_extra
    rng = np.random.default_rng(0)
    diffs = []
    for files_a, files_b in pairs:
        yt, pa = load(files_a)
        _, pb = load(files_b)
        ea, eb = (pa - yt) ** 2, (pb - yt) ** 2
        idx = rng.integers(0, len(yt), size=(10000, len(yt)))
        diffs.append(np.sqrt(eb[idx].mean(1)) - np.sqrt(ea[idx].mean(1)))
    avg = np.mean(diffs, axis=0)
    lo, hi = np.percentile(avg, [2.5, 97.5])
    return float(avg.mean()), float(lo), float(hi)


# --- 1. each group removed, against the full set ----------------------------
rows = []
for g in GROUPS:
    cfg = f"L_no_{g}"
    pairs = []
    for ds in DS:
        a, b = preds("exp1_loo", REF, ds), preds("exp1_loo", cfg, ds)
        if not a or not b:
            raise SystemExit(f"missing prediction files for {cfg} on {ds}")
        pairs.append((a, b))
        m, (lo, hi), _ = paired_bootstrap(a, b)
        rows.append(dict(comparison=f"{cfg} vs {REF}", group=g, dataset=ds,
                         rmse_change=round(m, 3), range_low=round(lo, 3),
                         range_high=round(hi, 3), verdict=verdict_loo(lo, hi)))
    m, lo, hi = averaged(pairs)
    rows.append(dict(comparison=f"{cfg} vs {REF}", group=g, dataset="AVERAGED",
                     rmse_change=round(m, 3), range_low=round(lo, 3),
                     range_high=round(hi, 3), verdict=verdict_loo(lo, hi)))
loo = pd.DataFrame(rows)
loo.to_csv(f"{RESULTS}/significance_loo.csv", index=False)
print("wrote significance_loo.csv\n")
print("Leave one out: each group removed from the full set")
print("positive rmse_change = removing it made things worse = it was contributing\n")
print(loo.to_string(index=False))

# --- 2. what the forward fill repair did ------------------------------------
# L_full and C6_full are the same six groups with the same seeds. the only
# difference is that L_full was built after the ffill was grouped per engine,
# so this comparison isolates the repair.
rows = []
pairs = []
for ds in DS:
    a, b = preds("exp1", "C6_full", ds), preds("exp1_loo", REF, ds)
    pairs.append((a, b))
    m, (lo, hi), _ = paired_bootstrap(a, b)
    rows.append(dict(comparison="L_full vs C6_full (ffill repaired vs not)",
                     dataset=ds, rmse_change=round(m, 3), range_low=round(lo, 3),
                     range_high=round(hi, 3),
                     verdict="measurable" if (lo > 0 or hi < 0) else "not measurable"))
m, lo, hi = averaged(pairs)
rows.append(dict(comparison="L_full vs C6_full (ffill repaired vs not)",
                 dataset="AVERAGED", rmse_change=round(m, 3),
                 range_low=round(lo, 3), range_high=round(hi, 3),
                 verdict="measurable" if (lo > 0 or hi < 0) else "not measurable"))
ff = pd.DataFrame(rows)
ff.to_csv(f"{RESULTS}/significance_ffill.csv", index=False)
print("\n\nwrote significance_ffill.csv\n")
print("Effect of the forward fill repair, negative = the repaired build scored better\n")
print(ff.to_string(index=False))

# --- 3. the metrics table ---------------------------------------------------
d = pd.read_csv(f"{RESULTS}/runs.csv")
e = d[(d.experiment == "exp1_loo") & (d.model == "lstm")]
t5 = e.groupby(["config", "dataset"]).agg(
    rmse=("rmse", "mean"), rmse_sd=("rmse", "std"), mae=("mae", "mean"),
    phm08=("nasa_score", "mean"), n_features=("n_features", "mean")).round(2)
t5.to_csv(f"{RESULTS}/table5_loo.csv")
print("\n\nwrote table5_loo.csv\n")
print(t5.to_string())
