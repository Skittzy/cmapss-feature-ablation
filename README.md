# Feature Engineering Ablation for Turbofan RUL Prediction (NASA C-MAPSS)

A controlled ablation study asking two questions. **RQ1:** which feature
engineering steps actually reduce prediction error, and which are redundant?
**RQ2:** does an LSTM's temporal processing add accuracy beyond what a good
temporal feature set already encodes — or can a Random Forest on the same features
match it? Six cumulative feature configurations × four C-MAPSS subsets ×
three random seeds, every run logged to `results/runs.csv`.

Solo research paper extending a group course project at FH Technikum Wien,
supervised by Prof. Patrick Link.

## Results

236 logged training runs. Experiment 1 is complete at five seeds per cell;
Experiment 2 is complete except for one single-seed cell, noted below.

### Experiment 1 — feature ablation (test RMSE, mean ± sd over 5 seeds)

| Config | Features | FD001 | FD002 | FD003 | FD004 |
|--------|---------:|------:|------:|------:|------:|
| C1_base    |  18 | 16.68 ± 1.08 | 26.32 ± 0.89 | 16.49 ± 0.99 | 28.73 ± 0.77 |
| C2_rolling |  83 | 18.00 ± 0.37 | 25.74 ± 0.77 | 20.32 ± 1.80 | 28.58 ± 0.48 |
| C3_lag     | 116 | 17.81 ± 0.59 | 26.34 ± 0.44 | 20.15 ± 2.22 | 28.58 ± 0.56 |
| C4_trend   | 132 | 17.12 ± 0.84 | 25.73 ± 0.80 | 16.16 ± 1.14 | 26.34 ± 0.68 |
| C5_agg     | 136 | 17.56 ± 0.78 | 25.58 ± 0.94 | 15.95 ± 1.36 | 26.37 ± 0.78 |
| C6_full    | 139 | 17.85 ± 1.12 | 24.15 ± 1.03 | 15.16 ± 0.59 | 24.69 ± 0.44 |

Of the 20 feature-engineering steps tested, **three produced a measurable
improvement, one produced a measurable harm, and the remaining sixteen were
inside the noise** (paired bootstrap at 95%). The three that helped are
degradation trend slopes on FD003 and FD004, and cycle transforms on FD004.
The one that hurt is rolling statistics on FD003.

### Why rolling statistics hurt, and which half is responsible

The rolling group bundles two different things. Splitting it (exploratory runs,
FD001 and FD003, five seeds, measured against `C1_base`):

| Half of the group | FD001 | FD003 |
|-------------------|-------|-------|
| rolling **means** only | −0.46 [−1.52, +0.50] not measurable | −0.37 [−1.14, +0.34] not measurable |
| rolling **standard deviations** only | **+1.98 [+0.10, +3.68] real harm** | **+4.36 [+1.65, +6.96] real harm** |
| both together | +1.11 [−0.44, +2.54] not measurable | **+3.41 [+0.97, +5.80] real harm** |

The standard deviations do all the damage; the means are harmless. They track
RUL about a fifth as strongly as the raw sensors (mean |r| of 0.10–0.14 against
0.34–0.54), and thirty of them are added to roughly eighteen informative
columns, burying the signal.

Note the FD001 row. The group as a whole is not measurably harmful there,
because the helpful means offset the harmful standard deviations, yet one half
of it clearly is. Bundling two features of opposite sign into a single ablation
step concealed a real effect.

### Experiment 2 — architecture, all arms on the same `C4_trend` features

| Model | Input | FD001 | FD002 | FD003 | FD004 |
|-------|-------|------:|------:|------:|------:|
| lstm    | 30 cycles, in order        | 17.12 ± 0.84 | 25.73 ± 0.80 | 16.16 ± 1.14 | 26.34 ± 0.68 |
| rf_flat | 30 cycles, order destroyed | 16.07 ± 0.02 | 25.04 ± 0.06 | 17.72 ± 0.12 | 27.96 |
| rf      | final cycle only           | 18.91 ± 0.09 | 26.75 ± 0.08 | 20.26 ± 0.08 | 28.07 ± 0.08 |
| dt      | final cycle only           | 20.92 ± 0.00 | 29.08 ± 0.00 | 22.24 ± 0.00 | 29.52 ± 0.00 |

`rf_flat` on FD004 rests on one seed rather than five; see `PROTOCOL.md`
section 10 for the deviation and its justification.

### The pattern both experiments point at

Sorting the subsets by how many fault modes they contain:

| Subset | Conditions × faults | Trend features help? | Reading in order helps? |
|--------|--------------------|----------------------|-------------------------|
| FD001  | 1 × 1 | no | no |
| FD002  | 6 × 1 | no | no |
| FD003  | 1 × **2** | **yes, −3.08** | **yes, −2.74** |
| FD004  | 6 × **2** | **yes, −2.00** | **yes, −3.10** |

Explicit rate-of-change features and implicit temporal modelling become
measurably useful on exactly the subsets with two failure modes, and on neither
single-fault subset. Operating-condition count does not predict it; fault-mode
count does.

### Anchors: what the original pipeline's quirks actually cost

| Config | Features | Dead | FD001 | FD002 | FD003 | FD004 |
|--------|---------:|-----:|------:|------:|------:|------:|
| C6_full   | 139 |  0 | 17.85 ± 1.12 | 24.15 ± 1.03 | 15.16 ± 0.59 | 24.69 ± 0.44 |
| C6_legacy | 179 | 34–44 | 17.12 ± 0.75 | 24.12 ± 0.54 | 14.77 ± 1.17 | 24.46 ± 0.42 |
| C6_expose | 180 | 34–44 | 17.33 ± 0.12 | 25.34 ± 0.66 | 14.96 ± 0.65 | 24.39 ± 0.96 |

The 34–44 constant columns in the original pipeline change accuracy on no
subset. Feeding the k-means operating-condition index in as a numeric feature
measurably hurts FD002 (+1.20, range +0.03 to +2.44), the one subset where that
index carries real information.

## Status

- **Done** — protocol pre-registered before any results; Experiment 1 (160 runs);
  Experiment 2 (76 runs); all significance tests
- **Closed** — the FD003 rolling-statistics result was investigated and
  explained. It is a property of the features, not a defect: the rolling
  standard deviations are responsible and the rolling means are harmless.
  Three candidate implementation defects were tested and ruled out. See
  `PROTOCOL.md` section 10
- **Known issue** — a forward-fill in `features_v2.py` leaked one row across
  each engine boundary, affecting 0.17% of cells. Fixed in code on 14 Sep.
  All reported results predate the fix and were not regenerated; the measured
  bound is far below the seed spread. Details in `PROTOCOL.md` section 10
- **Next** — leave-one-out robustness check; Setup and Results draft

## Reproduce

```bash
conda env create -f environment.yml && conda activate nasa-rul-project
python run_experiment.py --experiment exp1 --seeds 0 1 2 3 4
python run_experiment.py --experiment exp2 --best-config C4_trend --seeds 0 1 2 3 4
python analyze.py && python analyze_extra.py
```

Data is not committed. Place the C-MAPSS `.txt` files in `data/raw/`.

Every number above traces to a line in `results/runs.csv`, which logs one row
per training run with its seed, feature count, metrics and wall-clock cost.
Significance tests are regenerated by `analyze.py` and `analyze_extra.py` into
`results/significance*.csv`.

## Links

- Research exposé - [`docs/expose_v2.pdf`](docs/expose_v2.pdf)
- Pre-registered protocol - [`PROTOCOL.md`](PROTOCOL.md)
- Original group project — https://github.com/mezoabris/nasa-engine-rul-prediction