# Experimental Protocol

**A feature ablation study on NASA C-MAPSS**

Matej Krsteski, FH Technikum Wien
Supervised by Ass.-Prof. Patrick Link

Initial protocol committed on 26 August 2026, before the recorded ablation runs.

---

## 1. Research questions

**RQ1. Which feature engineering steps contribute to predictive performance?**

The original pipeline derives roughly 180 features from 21 sensors using rolling statistics, lagged values, degradation trends, cross-sensor summaries and cycle transforms. RQ1 addresses their individual contributions, which have never been evaluated. Some groups may account for the entire result, while others may contribute nothing.

**RQ2. What does sequential processing add to the engineered features?**

An LSTM processes thirty cycles in order and carries information from earlier cycles in its hidden state. The additional computational cost is justified only if sequence order provides information beyond the history already encoded in features such as rolling means and lags. I compare the LSTM with a model that has no notion of time, using exactly the same features.

*Remaining useful life (RUL) is the number of cycles until an engine fails. A Long Short Term Memory (LSTM) network processes a sequence one step at a time, retaining information from earlier steps.*

---

## 2. Data and preprocessing

The study uses all four NASA C-MAPSS subsets, FD001 to FD004, from Kaggle (`behrad3d/nasa-cmaps`). It includes every engine and every row. Each training history ends at engine failure. The `RUL_FD00X.txt` files give one test label per engine, its true remaining life at the last recorded cycle.

All configurations follow the same five preprocessing steps in the order given below, with the anchor exception specified for the final step.

First, **RUL labelling with a cap**. The label is the engine's final cycle minus the current cycle, capped at 125 for FD001 and FD003 and at 150 for FD002 and FD004, following Saxena et al. (2008). Early in engine life, sensor readings are flat and indicate healthy operation. Distinguishing 280 from 300 remaining cycles on those readings would fit noise. All values above the cap receive the same "healthy" label.

Second, **operating condition grouping**. Fit k-means with k = 6 on the three settings columns in the training data and apply the fitted model unchanged to the test data. Each row receives a group number based on its similarity to other rows.

Third, **constant feature removal** on the raw columns, before anything is constructed.

Fourth, **per-condition scaling**. Fit a separate MinMax scaler to the training data for each operating condition. FD002 and FD004 each cover six flight conditions. A change in condition shifts the sensor readings far more than degradation does. Scaling within each condition prevents the model from interpreting "the aircraft climbed" as "the engine is dying".

Fifth, **constant feature removal after feature construction**. Drop any column that is constant across the training set, then apply the identical column list to the test set.

The original pipeline omitted this fifth step. Between 34 and 44 of its 179 columns were constant and carried no information. Constant sensors had been removed from the feature list but remained in the table and were included again. Other sensors varied globally but were constant within each operating condition. Per-condition scaling made those columns constant throughout. The original check preceded scaling and could not possibly detect them.

Step five is applied to every configuration except the two anchors described below, which keep their dead columns deliberately.

---

## 3. Feature groups and configurations

Six feature groups form the cumulative configurations. One further group is used only in an anchor configuration.

| Group | What it contains |
|---|---|
| `base` | Settings and sensors that survived constant removal |
| `rolling` | Rolling mean and standard deviation, windows 5 and 10 |
| `lag` | The value 1 cycle ago and 3 cycles ago |
| `trend` | Value now minus value 10 cycles ago |
| `agg` | Mean, sd, min, max and range across all sensors at once |
| `cycle` | Cycle number, its log and its square root |
| `opcond` | The k-means group number, fed in as an ordinary number (anchor `C6_expose` only) |

### Cumulative ablation (the ladder)

In the main experiment, each configuration adds exactly one feature group to its predecessor. Each comparison therefore measures one addition. The table gives the measured feature counts.

| Config | Groups | FD001 | FD002 | FD003 | FD004 |
|---|---|---:|---:|---:|---:|
| `C1_base` | base | 17 | 19 | 18 | 19 |
| `C2_rolling` | + rolling | 77 | 87 | 82 | 87 |
| `C3_lag` | + lag | 107 | 121 | 114 | 121 |
| `C4_trend` | + trend | 122 | 138 | 130 | 138 |
| `C5_agg` | + agg | 126 | 142 | 134 | 142 |
| `C6_full` | + cycle | 129 | 145 | 137 | 145 |

The final step isolates the cycle transforms. The original exposé included them in "the full pipeline", which would have made the final comparison measure a different change from the one its label described.

### The three anchors

| Config | What it is | Features |
|---|---|---:|
| `C6_full` | The clean pipeline this paper reports | 129 to 145 |
| `C6_legacy` | The group project code as it currently stands on GitHub | 179 |
| `C6_expose` | The code that produced the exposé's RMSE of 17.1, 23.7, 14.2 and 24.0 | 180 |

`C6_legacy` and `C6_expose` retain their constant columns. Their comparison with the clean pipeline measures how cleaning the feature set affects accuracy.

These configurations differ by exactly one column. `op_condition` supplies the k-means cluster label to the network as a numeric measurement. This encoding is questionable because the numerical label for cluster 5 does not imply five times the magnitude of cluster 1. The problem is more pronounced on FD001 and FD003, which each have a single physical operating condition. Their settings vary by about 0.002, essentially rounding noise, yet k-means still identified the six groups requested. The exposé model thus used a label for measurement noise as a feature.

### Leave-one-out ablation

A secondary experiment, run on FD001 and FD004 only: the full set minus each of `rolling`, `lag`, `trend`, `agg` and `cycle` in turn.

The ladder measures the contribution of each group conditional on those already added. Leave-one-out measures its contribution with all other groups present. These comparisons can disagree. If two groups supply the same information, the second will appear redundant in the ladder and both will appear redundant in leave-one-out. I report and discuss disagreements between the two analyses.

---

## 4. The models

All configurations use the same LSTM architecture. Only the number of input features varies.

```
Input(30 cycles, N features)
LSTM(128, return_sequences=True)  ->  Dropout(0.3)
LSTM(64)                          ->  Dropout(0.3)
Dense(32, relu)  ->  Dense(1)
```

Adam optimiser at learning rate 0.001, mean squared error loss, batch size 32, at most 60 epochs, early stopping after 10 epochs with no improvement, best weights restored.

*(Dropout switches off a random share of connections during training so the network cannot simply memorise. An epoch is one full pass over the training data. Early stopping halts training once it stops improving, rather than letting it run on and overfit.)*

Experiment 2 includes three models with no concept of time.

`rf` is a Random Forest of 500 trees, shown only the engine's final cycle. `dt` is a single decision tree with its depth chosen by cross-validation, also final cycle only. `rf_flat` is a Random Forest of 500 trees shown all thirty cycles laid out side by side in one long row.

The flattened-window forest provides the key architecture comparison. An advantage of the LSTM over `rf` alone has two possible explanations: sequential processing or access to thirty times as many inputs. `rf_flat` receives exactly the same values as the LSTM without any notion of their order, separating the two explanations. If `rf_flat` matches the LSTM, the features suffice and the architecture adds no benefit. If it falls short, processing the cycles in order improves prediction. If `rf` matches `rf_flat`, the final cycle already contains all the information and the other twenty-nine add nothing.

*(A Random Forest is a large number of decision trees that each vote on the answer. Cross-validation means training on part of the data and scoring on the rest, several times over, so a setting can be chosen without ever touching the test set.)*

---

## 5. Validation and test splits

**Validation holds out entire engines.** A seeded shuffle assigns twenty percent of the training engines to validation. Two windows from the same engine can share up to 29 of their 30 cycles. A window-level split would place near-copies of validation observations in training, so the early stopping decision would rely on data the model had effectively already seen.

Decision tree depth is selected by GroupKFold on `unit_id`. All observations from an engine stay on the same side of each cross-validation split.

The test set is reserved for final scoring and excluded from early stopping, hyperparameter selection and configuration selection.

All windows span 30 cycles. The step size for sliding training windows is recorded for each run, including any reduction made for speed. Each test engine contributes its final window, padded with leading zeros if its history is shorter than 30 cycles. `StandardScaler` is fitted on training windows only.

---

## 6. Evaluation metrics

**RMSE in cycles** is the primary metric. It gives larger errors more weight in the average. Almost all published C-MAPSS results report it, making it the only metric available for comparison with other studies.

**MAE in cycles** gives the mean absolute error. A few badly predicted engines affect MAE much less than RMSE, so the gap between the two metrics helps characterise the error distribution.

**The PHM08 score** is the field's own asymmetric metric. With `d` as predicted minus true, it sums `exp(-d/13) - 1` when the prediction is early and `exp(d/10) - 1` when it is late. Late predictions are punished harder, because saying an engine has 40 cycles left when it has 10 is a different kind of wrong from saying it has 10 when it has 40.

**Training seconds and prediction seconds** are recorded for every run. These costs form part of the RQ2 comparison of model complexity and performance.

**Number of features and number of dead features** are recorded per configuration to quantify the feature-set simplicity considered in RQ1.

Everything is reported per subset and averaged across the four.

---

## 7. Seeds and paired comparisons

Each cell runs three times, with seeds 0, 1 and 2. If the spread exceeds 1.0 RMSE in any cell, all cells extend to five seeds, and medians will be reported alongside means. The number of seeds is the one part of the plan that will not be reduced to save time.

All tables report the average across seeds together with the spread. A single value without its spread is insufficient evidence of performance.

The paired bootstrap compares configurations using test engines as the resampling unit. First average the predictions across seeds for each engine. Then sample test engines with replacement and score both configurations on the same sample. Repeat ten thousand times. A difference counts as real only if the central 95% of those ten thousand differences excludes zero.

Some engines are harder to predict than others. Pairing cancels out differences in engine difficulty that would otherwise obscure the comparison, because both configurations are scored on the same engines in each resample.

When the range includes zero, the difference is reported as unmeasurable. Neither "a slight improvement" nor "a trend towards" describes such a result. For example, a group that adds thirty columns and reduces RMSE by 0.2 with a range of plus or minus 0.8 is an outcome of direct interest to this study and does not qualify as a gain.

**Baseline reproducibility.** No reproducible baseline currently exists for this pipeline. The exposé used a code version that included `op_condition` as an input. The GitHub version omits it, and the saved models cannot load against it because they expect 180 inputs and the current version produces 179. The figures in the group's `report.md` match no saved artifact. Run-to-run variation has never been measured for this pipeline. Until it is, single-run comparisons cannot support conclusions.

---

## 8. Configuration selection

I recorded the following rule before seeing any results.

> The configuration carried forward into Experiment 2 is the one with the **fewest features that is not measurably worse than the best scoring configuration**, judged by the paired bootstrap at 95% and averaged across all four subsets.

When the difference falls within the noise, this rule selects the smaller feature set. The research question asks whether additional feature engineering earns its place, which is why the tie-break favours simplicity.

---

## 9. Hypotheses and falsification criteria

Each prediction has a corresponding criterion for falsification.

**H1. Rolling statistics account for most of the total improvement from `C1_base` to `C6_full`.** Wrong if the `C1` to `C2` step delivers less than half the total gain, or if its range includes zero.

**H2. At least one of `lag`, `trend`, `agg` and `cycle` is redundant given the groups that came before it.** Wrong if every rung's range excludes zero on every subset.

**H3. A Random Forest on the final cycle, given the chosen features, lands within 2 RMSE of the LSTM.** Wrong if the gap is larger than 2 RMSE on most subsets.

**H4. The 34 to 44 dead features in the original pipeline made no difference to accuracy.** Wrong if `C6_legacy` and `C6_full` differ measurably by paired bootstrap.

**H5. Feeding `op_condition` in as a number made no difference either.** Wrong if `C6_expose` and `C6_legacy` differ measurably.

I would welcome evidence against H4 or H5. A change in accuracy attributable to the dead columns or the noise-derived cluster index would be of greater interest than an unchanged result.

---

## 10. Changes and notes

These entries were added retrospectively. Each records the date, decision, rationale and whether I had seen results at the time.

**9 September 2026. Pilot complete.** Three seeds of `C6_expose` on FD001, which is the exposé's exact feature set. RMSE came out at 17.151, 17.345 and 17.339, giving a mean of 17.28 and a spread of 0.111. Mean MAE 12.78. Training took 156, 115 and 125 seconds, and the runs used 19, 14 and 15 epochs, all comfortably inside the 60-epoch cap. The configuration built 180 features of which 44 are constant, matching the figure quoted in Section 2.

I made two decisions before any ablation results were available. The observed spread was well below the 1.0 threshold in Section 7, so the seed count remains three. A full run takes roughly two minutes, making the grid affordable with every window retained. The window step therefore remains 1.

On reproducing the exposé baseline: the exposé reported 17.1 from a single unseeded run. The seeded mean is 17.28, and the exposé's figure sits at the optimistic end of the three-seed range (17.151 to 17.345) rather than at its centre. The pipeline is therefore reproduced, with the qualification that the number originally published was the better end of a distribution nobody had measured.

This estimate of spread comes from FD001 alone and three runs. Before starting the grid, I regard it as an indication of the order of magnitude of run-to-run variation. It provides no precise constant. Spreads are reported by subset, and FD002 and FD004 may prove noisier.

**10 September 2026. Experiment 1 complete, and the seed rule has triggered.** All 96 runs finished: six ladder configurations plus two anchors, four subsets, three seeds.

Seed spread exceeds the 1.0 RMSE threshold set in Section 7 in eight of the thirty-two cells: `C6_full` on FD001 (1.529) and FD002 (1.398), `C6_legacy` on FD003 (1.478) and FD001 (1.063), `C3_lag` on FD003 (1.376), `C5_agg` on FD003 (1.230), `C1_base` on FD001 (1.205), and `C2_rolling` on FD003 (1.169). Section 7 says seeds extend to five if the spread exceeds 1.0 anywhere, so the grid is being rerun with seeds 3 and 4 added, 64 further runs, and medians will be reported alongside means.

At the time of this decision, the additional runs had not started and no paired bootstrap had been computed. No significance result informed the decision.

The 9 September pilot measured a spread of 0.111 and led me to conclude that three seeds would suffice. The measurement for `C6_expose` / FD001 was sound, but this proved to be the least variable cell in the grid. Every other cell is between four and fourteen times noisier. Extrapolating from one cell to thirty-two was an error. A pilot on a single cell can substantially understate variation across the grid. I will report this limitation in the paper.

The FD003 result requires investigation before publication. The cumulative ablation departs systematically from a monotonic pattern. `C1_base` averages 16.51 RMSE, rising to 21.48 for `C2_rolling` and 21.60 for `C3_lag`, then falling to 16.61 for `C4_trend`. All three seeds agree at each step, establishing a feature-set effect beyond seed noise. Two explanations remain: a property of rolling statistics on a two-fault-mode subset, or a defect in the rolling and lag implementation. Reporting the result requires distinguishing these explanations.

**10 September 2026, later. Experiment 1 analysed, winning configuration selected.** All 160 runs complete, five seeds in every cell, and the paired bootstrap has been run.

Section 8 left two choices unspecified. The rule says "the fewest features that is not measurably worse than the best-scoring configuration, judged by the paired bootstrap at 95% and averaged across all four subsets." It does not say whether the candidate pool includes the two anchors, nor whether "not measurably worse" is tested per subset or on the average. I resolved both by following the wording of the rule. The anchors are excluded, because `C6_legacy` and `C6_expose` deliberately retain dead columns and carrying one into Experiment 2 would confound the architecture question with the dead-column question; the ladder is what Section 3 calls the main experiment. I interpreted "averaged across all four subsets" as a test on the average. I computed both comparisons. Averaged, the rule selects `C4_trend`. Tested per subset instead, it would have selected `C6_full`. The two differ by seven features out of roughly one hundred and forty, so the difference does not materially affect the study. I retained the choice specified by the written rule.

**The configuration carried into Experiment 2 is `C4_trend`, 132 features.** Against `C6_full` it is +0.704 RMSE averaged across the four subsets, with a 95% range of −0.073 to +1.514, which includes zero. `C5_agg` at 136 features is measurably worse (+0.823, range +0.037 to +1.641), so the aggregate features are not merely redundant but slightly harmful.

The RQ1 outcomes below use the falsification criteria specified in Section 9.

H1 is falsified. Rolling statistics were predicted to account for most of the improvement. They are not a measurable improvement on any subset, and on FD003 they are a measurable harm of +3.408 RMSE with a range of +0.972 to +5.800 lying entirely above zero.

H2 is supported. Lag and aggregate features are redundant on all four subsets, with every interval spanning zero.

H4 is supported. The 34 to 44 constant columns change accuracy on no subset; every interval includes zero. The largest effect is FD001 at −0.94 with a range of −2.09 to +0.15.

H5 is partly falsified. The operating-condition index fed in as a number is not measurable on FD001, FD003 or FD004, but on FD002 it is a measurable harm of +1.20 with a range of +0.03 to +2.44. FD002 has six genuine operating conditions, so this is the subset where the index carries real information and it still hurt.

One hypothesis for discussion remains untested. Degradation trend slopes measurably improve performance on exactly the two subsets with two fault modes, FD003 at −3.079 and FD004 at −2.004, and on neither single-fault subset. One plausible explanation is that rate of change distinguishes the failure modes where absolute levels do not. This mechanism would also explain the earlier FD003 rolling result: smoothing removes rate information and the trend step restores it almost exactly. This interpretation requires a direct test.

**12 September 2026. Experiment 2 complete, with one recorded deviation.**

**Deviation from Section 7.** Seventy-six of the eighty Experiment 2 runs completed. The four missing runs are seeds 1 to 4 of `rf_flat` on FD004, which is the most expensive cell in the study: a single run trains 500 trees over a flattened thirty-cycle window of 138 features, which is 4,140 input columns, and took three hours and twenty-eight minutes. Completing the cell would have cost a further fourteen hours of wall clock. The cell therefore rests on one seed instead of five.

The decision rests on measured seed variation. Section 7 set five seeds because run-to-run variation had never been quantified and might have been the same size as the effects. For `rf_flat` it has now been quantified on the three completed subsets, and the seed spread is 0.020 on FD001, 0.058 on FD002 and 0.118 on FD003. A Random Forest averages five hundred trees, so the choice of random seed has very little effect; the plain `rf` arm behaves the same way at 0.08 to 0.09, and the decision tree is deterministic to four decimal places. These spreads are between one and two orders of magnitude below the differences being tested. The rule's purpose is satisfied by the evidence. The four outstanding runs are deferred. Repeating the original command resumes the experiment and executes only those four runs. The paper reports this cell as single-seed.

**A defect in the results file, found and repaired.** `results/runs.csv` had become unparseable. The decision-tree branch of `run_experiment.py` adds a `best_depth` field that no other model reports, while the header is written once from the first row ever appended, which was an LSTM row without that field. Every one of the twenty `dt` rows therefore carried twenty-four values against a twenty-three column header, shifting their timestamps. The file has been rewritten with `best_depth` as a proper column, empty for every non-tree row, and the original preserved as `results/runs.csv.backup_2026-09-12`. No measurement changed; only the column layout. `run_experiment.py` has been patched so that every row is built with identical columns regardless of model. Cross-validated depth was 6 in all twenty tree runs.

**Results for RQ2.** All four arms use the `C4_trend` feature set. Paired bootstrap as specified in Section 7, with predictions averaged across seeds before resampling.

Reading the cycles in order, which is the LSTM against the order-blind `rf_flat` on identical numbers, is not measurable on FD001 (−0.44, range −2.21 to +1.17) or FD002 (−0.69, range −2.32 to +0.87), and is a real improvement on FD003 (−2.74, range −4.81 to −0.78) and FD004 (−3.10, range −5.35 to −1.03).

Seeing the thirty-cycle window at all, which is `rf_flat` against `rf` on the final cycle alone, is a real improvement on FD001 (−2.84), FD002 (−1.70) and FD003 (−2.54), and not measurable on FD004 (−0.12, range −2.39 to +2.32, the single-seed cell).

H3 is falsified. It predicted that a Random Forest on the final cycle would land within 2 RMSE of the LSTM. The gap is 3.28 on FD001, 2.39 on FD002, 5.29 on FD003 and 3.22 on FD004, exceeding 2 on all four subsets rather than on none.

**Per-seed and ensemble RMSE.** Table 2 averages the RMSE of individual seeded runs. The bootstrap in Section 7 averages predictions across seeds before scoring the ensemble. These quantities can favour different models. On FD001, mean per-seed RMSE favours `rf_flat` by 1.05, while ensemble RMSE favours the LSTM by 0.44. Averaging predictions benefits models with greater disagreement across seeds. The LSTM's seed spread is 0.68 to 1.14, compared with 0.02 to 0.12 for `rf_flat`, so the LSTM benefits much more. Both summaries are valid. The paper reports both and explains what each measures: what one run typically achieves and what an ensemble of five achieves.

**13–14 September 2026. The FD003 rolling result is explained, and my predicted mechanism was wrong.**

The anomaly flagged on 10 September has been investigated and closed. It is a property of the features, not a defect in the code.

**Feature-matrix checks before training.** I tested three candidate defects by inspecting the constructed feature matrix directly, without training a model. Missing values from the rolling warm-up: ruled out, the final matrix contains no NaNs on either subset. Near-constant columns diluting the input: ruled out, FD003 has four such columns, FD001 has four, FD002 four and FD004 two, so FD003 is not unusual. Train-test distribution mismatch: ruled out, and in the opposite direction, since FD003's rolling features drift less between train and test than any other subset (ratio 0.78 against FD002's 1.04).

**Exploratory rolling-feature comparisons.** The cumulative ablation combines rolling means and standard deviations, so it cannot identify which accounts for the loss in accuracy. Two exploratory configurations, `C2a_rollmean` and `C2b_rollstd`, each retain one half of the group. This unexpected result prompted the post-hoc diagnostics. They are defined separately in `DIAG_CONFIGS`, outside Experiment 1, preserving the pre-registered ladder in `CONFIGS`. Twenty runs covered FD001 and FD003 with five seeds each, using the same code as every earlier run.

Measured against `C1_base` by paired bootstrap: rolling means alone are not measurable on either subset (FD001 −0.46, range −1.52 to +0.50; FD003 −0.37, range −1.14 to +0.34). Rolling standard deviations alone are a real harm on both (FD001 +1.98, range +0.10 to +3.68; FD003 +4.36, range +1.65 to +6.96). Head to head on FD003 the means beat the standard deviations by 4.73, range −7.62 to −1.82.

**Interpretation.** The loss in accuracy is entirely attributable to the rolling standard deviations. The rolling means are harmless. The likely cause is dimensional dilution rather than anything about smoothing: the standard deviations track RUL about a fifth as strongly as the raw sensors (mean absolute correlation 0.10 to 0.14 against 0.34 to 0.54), and thirty of them are added to roughly eighteen informative columns. On FD003 they are flattest of all, with mean variance 0.0005 against 0.0016 to 0.0039 elsewhere, which is consistent with the effect being largest there.

**Falsification of my smoothing hypothesis.** On 10 September I wrote that the likely mechanism was rolling smoothing destroying rate-of-change information on the two subsets with two fault modes, and that the trend rung then restored it. Both parts of that explanation were wrong. The means are the smoothing operation and they are harmless, and the standard deviations harm FD001 as well, which has a single fault mode. The FD003 result is not a fault-mode phenomenon. It only appeared to be one because on FD001 the helpful means partly cancel the harmful standard deviations and pull the combined result back inside the noise.

**Effect of grouping the rolling features.** On FD001 the rolling group as a whole is not measurably harmful (+1.11, range −0.44 to +2.54) while its standard-deviation half alone clearly is. Bundling two features of opposite sign into one ablation rung concealed a real effect. This is an argument for ablating at a finer grain than is usual in this literature.

**A second defect in the results file, same class as the first.** Adding the two diagnostic groups extended `GROUP_ORDER` from seven entries to nine, so the twenty new rows were written with twenty-six fields against a twenty-four column header. Repaired the same way: `rolling_mean` and `rolling_std` are now proper columns, zero for all 236 earlier rows, with the original preserved as `results/runs.csv.backup_2026-09-14`. No measurement changed. Note that `group_sizes` classifies columns by name and both new kinds contain the substring `_rolling_`, so their counts continue to appear under `rolling`; the two new columns are therefore always zero and are retained only to keep the file rectangular.

**14 September 2026. A minor leak across engine boundaries, fixed in code but deliberately not re-run.**

While investigating the FD003 result a genuine defect was found in `features_v2.py`. Rolling and lag columns are undefined at the very start of each engine's history, and the code filled those gaps with `df.ffill()` applied to the whole table rather than to each engine separately. The consequence is that cycle 1 of one engine inherited the last recorded value of the engine before it. Every engine boundary in the training data leaked one row.

I measured the effect by rebuilding the `C2_rolling` features with both implementations and comparing individual cells. The implementations differ in 2,772 of 1,588,587 cells on FD001, which is 0.174%, and in 2,908 of 2,027,040 on FD003, which is 0.143%. The largest single change is 0.354 and the median change 0.056, on features scaled to roughly 0 to 1. The defect affects FD001 marginally more than FD003, so it cannot account for the FD003 result, and its magnitude is far below any difference this study reports.

`features_v2.py` now groups by engine before forward-filling, and the fix is verified: the first cycle of engine 2 no longer carries engine 1's final value.

**Every result in this repository was produced before that fix and has not been regenerated.** I chose to retain those results. Regenerating would mean re-running every configuration containing rolling or lag features across both experiments, roughly a day and a half of compute, to move numbers by an amount the measurement above bounds well below the seed spread. Git records the timing of the fix relative to the runs. Re-running remains an option if requested in review.

I ran the two diagnostic configurations of 13 September with the original code to keep them comparable with the 236 runs already in `runs.csv`. I applied the fix only afterwards. Otherwise the comparison would have combined a feature change with an implementation change.

**18–19 September 2026. Leave one out, and a defect in it caught before it produced a single result.**

Section 3 specified leave-one-out, but it had never been run. Before launching it, I printed `LOO_CONFIGS` and found that all five configurations were incorrect.

`LOO_CONFIGS` was built as a comprehension over `GROUP_ORDER`, taking every group except the one being left out. That was correct when it was written. The 13–14 September entry above extended `GROUP_ORDER` from seven entries to nine by adding `rolling_mean` and `rolling_std`. That edit was made for the FD003 investigation and had nothing to do with leave-one-out, but leave-one-out read from the same list.

Direct checks showed two distinct outcomes. `L_no_rolling` dropped `rolling` and kept `rolling_mean` and `rolling_std`, so the configuration meant to contain no rolling features contained all sixty of them. The other four contained `rolling` together with both of its halves, so the rolling columns were built twice; the duplicate column names then made `var[c]` return a Series rather than a scalar and the run failed during the dead-column check.

Four of the five configurations would therefore have raised an error within a minute. `L_no_rolling` would have completed training and produced plausible results suggesting that removal had no effect, despite retaining the rolling features. Given the 13–14 September findings, an undetected error in this configuration would have been more consequential there than in any other configuration.

No published result is affected. `results/runs.csv` contained no `L_` rows at the time, so leave-one-out had never run. The ladder and the anchors list their groups explicitly and never referred to `GROUP_ORDER`, so nothing in Experiment 1 or Experiment 2 touched the defect. The repair replaces the derivation with an explicit `LADDER_GROUPS` list, so a future edit to `GROUP_ORDER` cannot reach these configurations again. Committed as `8eb8b85` before any run started, which the git history records.

The 13–14 September entry recorded the change to `GROUP_ORDER` and its first consequence, two additional columns in the results file. The effect on leave-one-out went unnoticed for five days. This oversight illustrates the risk of using a shared list to define several unrelated parts of the study.

**A reference configuration added at the same time.** All 256 runs then in `runs.csv` were produced before the forward-fill repair of 14 September. Comparing new leave-one-out runs against the old `C6_full` would have measured two changes at once, which is the same mistake the 14 September entry deliberately avoided when it ran the diagnostics against the unfixed code. `L_full` was therefore added, holding the same six groups as `C6_full`, so that every leave-one-out comparison sits inside one code version. Its feature counts came out at 129 on FD001 and 145 on FD004, matching `C6_full` in Section 3 exactly.

**The runs.** Sixty runs — six configurations on FD001 and FD004, five seeds each — completed 19 September, no failures. Analysis is in a new `analyze_loo.py`, written because `analyze_extra.py` has no leave-one-out handling. It writes `results/significance_loo.csv`, `results/significance_ffill.csv` and `results/table5_loo.csv`, by the same method as everything else: paired bootstrap over test engines, 10,000 resamples, predictions averaged across seeds first.

**Results.** Each comparison removes one group from the full set and uses `L_full` as the reference. A positive difference means removal increased error, so including the group improved prediction.

| Group removed | FD001 | FD004 | Averaged | Reading |
|---|---|---|---|---|
| `rolling` | −2.825 [−4.771, −0.984] | −1.338 [−2.335, −0.384] | −2.081 [−3.150, −1.049] | **removing it helps, measurably, on both** |
| `lag` | −0.768 [−2.053, +0.488] | +0.035 [−0.642, +0.700] | −0.367 [−1.092, +0.342] | not measurable |
| `trend` | +1.865 [+0.149, +3.622] | +1.877 [+0.692, +3.140] | +1.871 [+0.818, +2.926] | **contributes on both** |
| `agg` | +0.006 [−0.921, +0.926] | +0.079 [−0.733, +0.913] | +0.041 [−0.575, +0.670] | not measurable |
| `cycle` | −0.532 [−2.180, +1.028] | +1.366 [+0.113, +2.602] | +0.419 [−0.622, +1.405] | contributes on FD004 only |

Mean test RMSE over five seeds, for reference: `L_full` 17.04 and 24.59; `L_no_rolling` 14.50 and 23.20; `L_no_lag` 16.51 and 24.56; `L_no_trend` 19.40 and 26.93; `L_no_agg` 16.99 and 24.53; `L_no_cycle` 16.66 and 26.07, on FD001 and FD004 respectively.

**Where the two methods agree.** Lag and aggregate features are not measurable by either method on either subset. They are genuinely redundant rather than merely masked, which is the strongest form the H2 result could take. Trend and cycle features on FD004 are measurable by both methods, in the same direction.

**Disagreements between the two methods.** The discrepancies are the more informative results to report.

Rolling statistics. The ladder calls them not measurable on FD001 (+1.111, range −0.445 to +2.540) and on FD004 (−0.493, range −3.441 to +2.216). Leave-one-out calls them a measurable harm on both. Dropping all sixty rolling columns from the full set improves FD001 by 2.83 RMSE and FD004 by 1.34. The ladder understated the harm because it adds rolling to a bare seventeen-column baseline, while leave-one-out removes it from a set that also contains trend and cycle. That is consistent with the dilution account established on 13–14 September: the more informative columns are present, the more thirty weakly-correlated standard deviations cost.

Trend on FD001. The ladder finds no measurable contribution (−0.504, range −1.872 to +0.811). Leave-one-out finds a real contribution of 1.87 RMSE, range +0.149 to +3.622. This is the masking case anticipated in Section 3.

**Interpretation across the two ablations.** The 13–14 September entry showed that grouping two features with opposite effects concealed a real effect. The present comparison shows that the measured contribution of a group also depends on the groups already included. A single ordering can therefore conceal effects. These independent findings reinforce the same conclusion. A single ablation path is insufficient to establish that a group contributes nothing. Together they are the study's most transferable finding.

**Experiment 2 retains the original selection despite the better leave-one-out configuration.** `L_no_rolling` scores 14.50 on FD001 and 23.20 on FD004, better than any other configuration measured on either subset, using 69 and 77 features against `C6_full`'s 129 and 145. Experiment 2 used `C4_trend`, selected by the rule fixed in Section 8 before any result existed.

Section 8 precludes re-selection after observing the results. I applied the pre-registered rule to the ladder as written. The secondary leave-one-out experiment followed later and covered two of the four subsets. The paper reports that the pre-registered rule did not select the best configuration found in the study. This outcome is itself a finding about pre-registration. The conclusions of Experiment 2 are unaffected because all four arms use identical features, making the architecture comparison internally valid for whichever features were selected.

**What the forward-fill repair actually did, and a correction to the 14 September entry.** `L_full` and `C6_full` hold the same six groups at the same five seeds. The only intended difference is that `L_full` was built after the forward fill was grouped per engine, so the comparison isolates the repair.

| Subset | `L_full` minus `C6_full` | 95% range | Verdict |
|---|---:|---|---|
| FD001 | −0.774 | [−1.548, −0.049] | repaired build measurably better |
| FD004 | −0.003 | [−0.739, +0.694] | not measurable |
| Averaged | −0.389 | [−0.913, +0.118] | not measurable |

The 14 September entry justified not regenerating on the grounds that the defect's magnitude was far below any difference this study reports. That justification overstated what the measurement established. It inferred the effect on accuracy from changes in feature values alone. On FD001 the repaired build is measurably better by 0.77 RMSE, which is comparable to differences this study does report: the margin by which `C4_trend` was selected over `C6_full` in Section 8 was 0.70.

On FD004 there is no measurable difference, and averaged over the two subsets there is none either. The comparison cannot cleanly separate the repair from run-to-run variation, because same-seed rerun reproducibility has never been measured on this pipeline and TensorFlow on Metal is not guaranteed to be bitwise deterministic. The repaired build is nowhere worse, and any effect of the repair is bounded well under 1.6 RMSE on FD001 and under 0.7 on FD004.

The decision not to regenerate stands for now: regeneration is still roughly a day and a half of compute, and neither the direction nor the averaged magnitude changes any conclusion in the paper. The decision now rests on a measurement subject to the confound above. A reviewer could reasonably request regeneration. A determinism check — rerunning one cell at an identical seed under identical code and comparing — would cost about four minutes and would remove the confound. The determinism check is the next proposed step and has not been run.

**Two stale generated files, found while regenerating.** `results/table3_metrics.csv` and `results/table4_cost.csv` had been committed before the twenty diagnostic runs of 13 September were added, and never regenerated, so they did not match `runs.csv`. Regenerating adds the four diagnostic rows to table 3 and moves the Experiment 1 LSTM mean training time in table 4 from 193.35 seconds over 160 runs to 186.60 over 180. No measurement changed and no significance file changed; all four are byte-identical after regeneration. Because table 3 now includes both the pre-registered ladder and exploratory diagnostics, explicit labels are required to distinguish them in the paper.
