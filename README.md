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

Baseline, 21 Aug 2026 — full pipeline as currently implemented
(`C6_legacy`, 179 features, single seed):

| Subset | MAE | RMSE |
|--------|----:|-----:|
| FD001  | 12.35 | 17.14 |
| FD002  | 16.02 | 23.27 |
| FD003  | 10.63 | 14.89 |
| FD004  | 18.05 | 25.45 |

### Experiment 1 — feature ablation (RMSE, mean ± sd over 3 seeds)

| Config | Features added | FD001 | FD002 | FD003 | FD004 |
|--------|----------------|------:|------:|------:|------:|
| C1_base    | settings + raw sensors      | – | – | – | – |
| C2_rolling | + rolling mean/sd (5, 10)   | – | – | – | – |
| C3_lag     | + lags 1, 3                 | – | – | – | – |
| C4_trend   | + trend slopes (w=10)       | – | – | – | – |
| C5_agg     | + all-sensor aggregates     | – | – | – | – |
| C6_full    | + cycle transforms          | – | – | – | – |

## Status

- **Done** — environment restored, baseline reproduced, repository set up
- **Running** — protocol pre-registration, config-driven feature builder
- **Next** — seed-variance pilot, then the full grid

## Reproduce

```bash
conda env create -f environment.yml && conda activate nasa-rul-project
python run_experiment.py --experiment exp1 --seeds 0 1 2
python run_experiment.py --experiment exp2 --seeds 0 1 2
python analyze.py
```

Data is not committed. Place the C-MAPSS `.txt` files in `data/raw/`.

## Links

- Research exposé — `docs/expose_v2.pdf` *(corrected version being finalised)*
- Pre-registered protocol - [`PROTOCOL.md`](PROTOCOL.md)
- Original group project — https://github.com/mezoabris/nasa-engine-rul-prediction