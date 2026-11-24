# Feature Engineering Ablation for Turbofan RUL Prediction (NASA C-MAPSS)

This controlled ablation study examines two questions about predicting a turbofan
engine's remaining useful life (RUL). **RQ1:** Which feature engineering steps
reduce prediction error, and which are redundant? **RQ2:** Does an LSTM gain
accuracy by processing cycles in sequence, or can a random forest match it using
the same well-chosen temporal features? The study compares six cumulative feature
configurations across four C-MAPSS subsets and five random seeds. A separate
leave-one-out experiment covers FD001 and FD004. Every run is logged in
`results/runs.csv`.

This single-author research paper extends a group course project at FH Technikum
Wien, supervised by Prof. Patrick Link.

## Results

The log contains 316 training runs. Experiment 1 is complete with five seeds in every cell.
Experiment 2 is complete except for the single-seed cell identified below.
The leave-one-out experiment is complete on FD001 and FD004 with five seeds.

### Cumulative feature ablation (Experiment 1)

For each subset, the table reports test RMSE in cycles (mean ± SD over 5 seeds).

| Config | Features | FD001 | FD002 | FD003 | FD004 |
|--------|---------:|------:|------:|------:|------:|
| C1_base    |  18 | 16.68 ± 1.08 | 26.32 ± 0.89 | 16.49 ± 0.99 | 28.73 ± 0.77 |
| C2_rolling |  83 | 18.00 ± 0.37 | 25.74 ± 0.77 | 20.32 ± 1.80 | 28.58 ± 0.48 |
| C3_lag     | 116 | 17.81 ± 0.59 | 26.34 ± 0.44 | 20.15 ± 2.22 | 28.58 ± 0.56 |
| C4_trend   | 132 | 17.12 ± 0.84 | 25.73 ± 0.80 | 16.16 ± 1.14 | 26.34 ± 0.68 |
| C5_agg     | 136 | 17.56 ± 0.78 | 25.58 ± 0.94 | 15.95 ± 1.36 | 26.37 ± 0.78 |
| C6_full    | 139 | 17.85 ± 1.12 | 24.15 ± 1.03 | 15.16 ± 0.59 | 24.69 ± 0.44 |

Of the 20 feature-engineering steps tested, three measurably reduced error,
one measurably increased it, and the remaining sixteen fell within the noise
(paired bootstrap at 95%). Degradation trend slopes reduced error on FD003 and
FD004, and cycle transforms reduced it on FD004. Rolling statistics increased
error on FD003. For the bootstrap, predictions are averaged across seeds for
each test engine. Both configurations are then scored on the same resampled
engines.

### Rolling means and standard deviations

Exploratory runs on FD001 and FD003 tested the two parts of the rolling group
separately. The table gives the change in RMSE relative to `C1_base` for each
subset, using five seeds.

| Half of the group | FD001 | FD003 |
|-------------------|-------|-------|
| rolling **means** only | −0.46 [−1.52, +0.50] not measurable | −0.37 [−1.14, +0.34] not measurable |
| rolling **standard deviations** only | **+1.98 [+0.10, +3.68] real harm** | **+4.36 [+1.65, +6.96] real harm** |
| both together | +1.11 [−0.44, +2.54] not measurable | **+3.41 [+0.97, +5.80] real harm** |

Rolling standard deviations account for all the harm, and rolling means are
harmless. The standard deviations correlate with RUL about a fifth as strongly
as the raw sensors (mean |r| of 0.10–0.14 against 0.34–0.54). Adding thirty
such columns to roughly eighteen informative columns dilutes the signal.

On FD001, the helpful means offset the harmful standard deviations, so the
combined group shows no measurable harm even though the standard deviations
alone do. Combining features with opposite effects in one ablation step
concealed a real effect.

### Leave-one-out ablation on FD001 and FD004

The cumulative ablation (the ladder) measures what each group adds to the
preceding groups. Leave-one-out measures what each group contributes when all
the others are present. The leave-one-out experiment used five seeds on FD001
and FD004. All six configurations were built after the forward-fill repair,
including the reference configuration, `L_full`, which rebuilds `C6_full` with
the current code.

A positive change in RMSE means that removing the group increased error, so
including it improved accuracy.

| Group removed | Features left (FD001) | FD001 | FD004 | The ladder said |
|---------------|----------------------:|-------|-------|-----------------|
| `rolling` |  69 | **−2.83 [−4.77, −0.98]** | **−1.34 [−2.34, −0.38]** | not measurable on either |
| `lag`     |  99 | −0.77 [−2.05, +0.49] | +0.04 [−0.64, +0.70] | not measurable on either |
| `trend`   | 114 | **+1.87 [+0.15, +3.62]** | **+1.88 [+0.69, +3.14]** | FD001 not measurable, FD004 helps |
| `agg`     | 125 | +0.01 [−0.92, +0.93] | +0.08 [−0.73, +0.91] | not measurable on either |
| `cycle`   | 126 | −0.53 [−2.18, +1.03] | **+1.37 [+0.11, +2.60]** | FD001 not measurable, FD004 helps |

**Lag and aggregate features are genuinely redundant**, not merely masked — null
by both methods on both subsets. That is 30 and 4 columns respectively doing
nothing at all.

The ladder concealed the contribution of trend features on FD001. Adding them
after the harmful rolling group produced no measurable difference in the ladder.
In the leave-one-out comparison, removing trend features increased RMSE by 1.87,
a measurable contribution.

Removing the rolling group measurably improved accuracy on both subsets.
`L_no_rolling` retains base, lag, trend, agg and cycle (69 features on FD001).
Its RMSE was **14.50 ± 0.72** on FD001 and **23.20 ± 0.34** on FD004, better
than any other configuration measured anywhere in this study, with
roughly half the columns of the full set.

Experiment 2 used `C4_trend`. The selection followed the rule in `PROTOCOL.md`
section 8, pre-registered before any results existed. The study retains that
selection after the leave-one-out experiment. Section 10 explains the decision.

A group's measured contribution depends on which other groups are present.
A single ablation path therefore cannot establish that a group contributes
nothing. The choice of path and the decision to bundle features are two
independent ways a cumulative ablation can conceal a real effect.

### Architecture comparison (Experiment 2)

All models use the same `C4_trend` features. The table reports test RMSE
in cycles, averaged over the available seeds, with the standard deviation
shown for cells with multiple seeds.

| Model | Input | FD001 | FD002 | FD003 | FD004 |
|-------|-------|------:|------:|------:|------:|
| lstm    | 30 cycles, in order        | 17.12 ± 0.84 | 25.73 ± 0.80 | 16.16 ± 1.14 | 26.34 ± 0.68 |
| rf_flat | 30 cycles, order destroyed | 16.07 ± 0.02 | 25.04 ± 0.06 | 17.72 ± 0.12 | 27.96 |
| rf      | final cycle only           | 18.91 ± 0.09 | 26.75 ± 0.08 | 20.26 ± 0.08 | 28.07 ± 0.08 |
| dt      | final cycle only           | 20.92 ± 0.00 | 29.08 ± 0.00 | 22.24 ± 0.00 | 29.52 ± 0.00 |

The `rf_flat` result on FD004 uses one seed instead of five. Section 10 of
`PROTOCOL.md` documents the deviation and its justification.

### Results by fault-mode count

The table groups the four subsets by the number of fault modes and shows the
number of operating conditions for each.

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

### Anchor configurations and the original feature pipeline

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

- **Completed.** The protocol was pre-registered before any results. The study
  completed 160 runs for Experiment 1, 76 runs for Experiment 2 and 60 runs for
  leave-one-out. All significance tests are complete.
- **FD003 rolling statistics.** The investigation is closed. It traced the loss
  in accuracy to the rolling standard deviations. The rolling means are harmless.
  Three candidate implementation defects were tested and ruled out. The result
  is a property of the features. See `PROTOCOL.md` section 10.
- **Forward-fill defect.** In `features_v2.py`, one row leaked across each engine
  boundary, affecting 0.17% of cells. The code was fixed on 14 Sep. All
  Experiment 1 and 2 results predate the repair and were not regenerated. The
  full configuration rebuilt after the repair (`L_full`) scored measurably
  better on FD001 by 0.77 RMSE and showed no difference on FD004. The effect
  is small but not negligible. Rerun variation confounds the comparison.
  Section 10 of `PROTOCOL.md` corrects the original justification, which rested
  on inference rather than measurement.
- **Planned writing.** The Setup and Results draft is next. Low-cost optional
  follow-ups include a same-seed determinism check, dropping only the rolling
  standard deviations from the full set, and a minimal `base + trend + cycle`
  configuration.

## Reproduce

Place the C-MAPSS `.txt` files in `data/raw/` before running the commands. The
data files are not committed to the repository.

```bash
conda env create -f environment.yml && conda activate nasa-rul-project
python run_experiment.py --experiment exp1 --seeds 0 1 2 3 4
python run_experiment.py --experiment exp2 --best-config C4_trend --seeds 0 1 2 3 4
python run_experiment.py --experiment exp1_loo --datasets FD001 FD004 --seeds 0 1 2 3 4
python analyze.py && python analyze_extra.py && python analyze_loo.py
```

Every number above traces to a line in `results/runs.csv`, which logs one row
per training run with its seed, feature count, metrics and wall-clock cost.
Significance tests are regenerated by `analyze.py`, `analyze_extra.py` and
`analyze_loo.py` into `results/significance*.csv`.

## Links

- Research exposé - [`docs/expose_v2.pdf`](docs/expose_v2.pdf)
- Pre-registered protocol - [`PROTOCOL.md`](PROTOCOL.md)
- Original group project — https://github.com/mezoabris/nasa-engine-rul-prediction