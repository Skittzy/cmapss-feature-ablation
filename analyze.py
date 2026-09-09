"""Turns results/runs.csv into the tables and figures for the paper.

    python analyze.py

Writes into results/:
    table1_ablation.csv       RMSE for each feature configuration
    table1b_anchors.csv       the three anchor runs side by side
    table2_architecture.csv   LSTM vs Random Forest vs Decision Tree
    significance.csv          does each rung of the ladder actually help?
    fig_ablation.png
    fig_architecture.png
"""

from pathlib import Path
import itertools
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS = Path("results")
PREDS = RESULTS / "predictions"
LADDER = ["C1_base", "C2_rolling", "C3_lag", "C4_trend", "C5_agg", "C6_full"]
ANCHORS = ["C6_full", "C6_legacy", "C6_expose"]


def load():
    df = pd.read_csv(RESULTS / "runs.csv")
    # if a run was repeated, keep the newest one
    return df.drop_duplicates(
        subset=["experiment", "config", "dataset", "model", "seed"], keep="last")


def summarise(df, index):
    """Average over seeds and measure how much the seeds disagreed."""
    g = df.groupby([index, "dataset"])["rmse"].agg(["mean", "std", "count"])
    return g.reset_index()


def as_table(g, index, order=None):
    g = g.copy()
    g["cell"] = g.apply(
        lambda r: f"{r['mean']:.2f} ± {r['std']:.2f}" if r["count"] > 1
        else f"{r['mean']:.2f}", axis=1)
    t = g.pivot(index=index, columns="dataset", values="cell")
    if order:
        t = t.reindex([c for c in order if c in t.index])
    return t


def paired_bootstrap(files_a, files_b, n=10000, seed=0):
    """Is B really better than A, or did we just get a lucky random start?

    Both configs are scored on the same randomly re-drawn set of engines,
    thousands of times. Pairing them this way cancels out "some engines are
    just harder to predict". Returns the average RMSE difference (B minus A),
    a 95% range for it, and how often B came out ahead.

    If the 95% range includes zero, the difference is not real -- it is inside
    the noise.
    """
    def stack(files):
        d = [pd.read_csv(f).sort_values("unit_id") for f in files]
        yt = d[0]["y_true"].values
        for x in d[1:]:
            assert np.allclose(x["y_true"].values, yt), \
                "true RUL differs between seed files -- check the pipeline"
        return yt, np.mean([x["y_pred"].values for x in d], axis=0)

    yt, pa = stack(files_a)
    _, pb = stack(files_b)
    ea, eb = (pa - yt) ** 2, (pb - yt) ** 2

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(yt), size=(n, len(yt)))
    diff = np.sqrt(eb[idx].mean(1)) - np.sqrt(ea[idx].mean(1))
    return (float(diff.mean()),
            (float(np.percentile(diff, 2.5)), float(np.percentile(diff, 97.5))),
            float((diff < 0).mean()))


def main():
    df = load()
    e1 = df[(df.experiment == "exp1") & (df.model == "lstm")]

    # ---- Table 1: the ladder ---------------------------------------------
    if len(e1):
        t1 = as_table(summarise(e1, "config"), "config", LADDER)
        t1.to_csv(RESULTS / "table1_ablation.csv")
        print("\nTable 1 - RMSE by feature configuration (lower is better)\n")
        print(t1.to_string())

        g = summarise(e1, "config")
        fig, ax = plt.subplots(figsize=(9, 5))
        for ds in sorted(g.dataset.unique()):
            s = (g[g.dataset == ds].set_index("config")
                 .reindex(LADDER).dropna(subset=["mean"]))
            ax.errorbar(range(len(s)), s["mean"], yerr=s["std"].fillna(0),
                        marker="o", capsize=4, label=ds)
        ax.set_xticks(range(len(LADDER)))
        ax.set_xticklabels([c.split("_", 1)[1] for c in LADDER], rotation=20)
        ax.set_xlabel("features added, one group at a time")
        ax.set_ylabel("Test RMSE (cycles)")
        ax.set_title("Feature ablation - average over seeds, bars show seed spread")
        ax.legend()
        fig.tight_layout()
        fig.savefig(RESULTS / "fig_ablation.png", dpi=150)
        plt.close(fig)

        # ---- Table 1b: the three anchors ---------------------------------
        anc = e1[e1.config.isin(ANCHORS)]
        if len(anc):
            t1b = as_table(summarise(anc, "config"), "config", ANCHORS)
            t1b.to_csv(RESULTS / "table1b_anchors.csv")
            print("\nTable 1b - clean pipeline vs GitHub master vs exposé run\n")
            print(t1b.to_string())

        # ---- does each rung actually help? -------------------------------
        rows = []
        for ds in sorted(e1.dataset.unique()):
            rungs = [c for c in LADDER if c in set(e1.config)]
            for a, b in itertools.pairwise(rungs):
                fa = sorted(PREDS.glob(f"exp1_{a}_{ds}_lstm_s*.csv"))
                fb = sorted(PREDS.glob(f"exp1_{b}_{ds}_lstm_s*.csv"))
                if not fa or not fb:
                    continue
                d, ci, win = paired_bootstrap(fa, fb)
                rows.append(dict(dataset=ds, added_group=b.split("_", 1)[1],
                                 rmse_change=round(d, 3),
                                 range_low=round(ci[0], 3),
                                 range_high=round(ci[1], 3),
                                 chance_it_helps=round(win, 3),
                                 real_improvement=bool(ci[1] < 0)))
        if rows:
            sig = pd.DataFrame(rows)
            sig.to_csv(RESULTS / "significance.csv", index=False)
            print("\nDoes each added feature group actually help?")
            print("(negative rmse_change = better; real only if range_high < 0)\n")
            print(sig.to_string(index=False))

    # ---- Table 2: model comparison ---------------------------------------
    e2 = df[df.experiment == "exp2"]
    if len(e2):
        t2 = as_table(summarise(e2, "model"), "model")
        t2.to_csv(RESULTS / "table2_architecture.csv")
        print("\nTable 2 - RMSE by model, all using the same features\n")
        print(t2.to_string())

        cost = (e2.groupby("model")[["train_seconds", "predict_seconds"]]
                .mean().round(2))
        print("\nAverage cost per run (seconds)\n")
        print(cost.to_string())

        g = summarise(e2, "model")
        models, datasets = sorted(g.model.unique()), sorted(g.dataset.unique())
        x, w = np.arange(len(datasets)), 0.8 / len(models)
        fig, ax = plt.subplots(figsize=(9, 5))
        for i, m in enumerate(models):
            s = g[g.model == m].set_index("dataset").reindex(datasets)
            ax.bar(x + i * w - 0.4 + w / 2, s["mean"], w,
                   yerr=s["std"].fillna(0), capsize=3, label=m)
        ax.set_xticks(x)
        ax.set_xticklabels(datasets)
        ax.set_ylabel("Test RMSE (cycles)")
        ax.set_title("Same features, different models")
        ax.legend()
        fig.tight_layout()
        fig.savefig(RESULTS / "fig_architecture.png", dpi=150)
        plt.close(fig)

    print(f"\nWritten to {RESULTS}/")


if __name__ == "__main__":
    main()
