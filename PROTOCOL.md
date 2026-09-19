# Experimental Protocol

**A feature ablation study on NASA C-MAPSS**

Matej Krsteski, FH Technikum Wien
Supervised by Ass.-Prof. Patrick Link

Committed 26 August 2026, before a single result existed.

---

## Why this file is here

This is a pre-registration. It is the plan, written down and committed to git before any experiment has been run, so that the order of events is a matter of record rather than a matter of trust. It says what I am going to run, what I am going to measure, and how I will decide that one thing is better than another.

If you are reading this in the repository, look at the date on the commit that added this file, then look at the date on the commit that first added `results/runs.csv`. This one comes first. That is the whole point of writing it.

The reason to bother is not paperwork. It is that I do not want to fool myself. Once you have a table of numbers in front of you it becomes very easy to notice, entirely honestly, that the comparison you meant to make all along happens to be the flattering one. Deciding the rules first closes that door.

Nothing below gets edited once results start arriving. Corrections and changes of plan go at the bottom, dated, with a note on whether I had already seen results when I made them.

---

## 1. The two questions

**RQ1. Which feature engineering steps actually matter?**

The pipeline this study inherits builds roughly 180 features out of 21 sensors: rolling statistics, lagged values, degradation trends, cross-sensor summaries and cycle transforms. Nobody ever checked which of those were doing the work. Some of them may be carrying the whole result. Some may be carrying nothing at all. RQ1 asks which is which.

**RQ2. Is the sequential model earning its keep?**

An LSTM reads thirty cycles in order and carries a memory forward as it goes. That is expensive, and it is only worth paying for if reading in order tells you something the features have not already said. Rolling means and lags are, after all, history written down as numbers. So: give a model with no sense of time the exact same features, and see what the ordering was actually buying.

*(RUL means Remaining Useful Life, the number of cycles an engine has left before it fails. LSTM stands for Long Short Term Memory, a neural network that walks through a sequence one step at a time and keeps a running memory of what it has seen.)*

---

## 2. The data, and what happens to it before any model sees it

NASA C-MAPSS, all four subsets FD001 to FD004, taken from Kaggle (`behrad3d/nasa-cmaps`). No engines are dropped and no rows are filtered. The training files run every engine to failure. Test labels come from the `RUL_FD00X.txt` files, one number per engine, giving its true remaining life at the last cycle recorded.

Every configuration in this study gets exactly the same preprocessing. Five steps, in this order.

First, **RUL labelling with a cap**. The label is the engine's final cycle minus the current cycle, capped at 125 for FD001 and FD003 and at 150 for FD002 and FD004, following Saxena et al. (2008). The cap exists because early in an engine's life the sensors are flat and healthy, and asking a model to distinguish 280 cycles remaining from 300 is asking it to learn noise. Above the cap everything is simply labelled "healthy".

Second, **operating condition grouping**. k-means with k = 6 on the three settings columns, fitted on training data and then applied unchanged to test data. (k-means sorts rows into k groups by similarity; each row comes out with a group number attached.)

Third, **constant feature removal** on the raw columns, before anything is constructed.

Fourth, **per-condition scaling**. Each operating condition group gets its own MinMax scaler, fitted on training data only. This matters more than it sounds: FD002 and FD004 switch between six flight conditions, and a condition switch moves the sensors far more than degradation does. Scaling within each condition stops the model from reading "the aircraft climbed" as "the engine is dying".

Fifth, **dead column removal after the features are built**. Any column whose value never changes anywhere in the training set is dropped, and the identical column list is then applied to the test set so the two never drift apart.

That fifth step is new. The original pipeline did not do it, and as a result somewhere between 34 and 44 of its 179 columns were constant, carrying no information at all. Two separate causes: constant sensors were removed from the feature list but never from the table, so they came straight back in; and some sensors that vary globally are flat *within* a single operating condition, so per-condition scaling flattened them everywhere. The constant check ran before the scaling, so it could not possibly have caught the second kind.

Step five is applied to every configuration except the two anchors described below, which keep their dead columns deliberately.

---

## 3. Feature groups and configurations

Six groups, plus one that exists only inside an anchor.

| Group | What it contains |
|---|---|
| `base` | Settings and sensors that survived constant removal |
| `rolling` | Rolling mean and standard deviation, windows 5 and 10 |
| `lag` | The value 1 cycle ago and 3 cycles ago |
| `trend` | Value now minus value 10 cycles ago |
| `agg` | Mean, sd, min, max and range across all sensors at once |
| `cycle` | Cycle number, its log and its square root |
| `opcond` | The k-means group number, fed in as an ordinary number (anchor `C6_expose` only) |

### The ladder

This is the main experiment. Each rung adds exactly one group to the one before it, so every step measures one thing. Feature counts below are measured, not estimated.

| Config | Groups | FD001 | FD002 | FD003 | FD004 |
|---|---|---:|---:|---:|---:|
| `C1_base` | base | 17 | 19 | 18 | 19 |
| `C2_rolling` | + rolling | 77 | 87 | 82 | 87 |
| `C3_lag` | + lag | 107 | 121 | 114 | 121 |
| `C4_trend` | + trend | 122 | 138 | 130 | 138 |
| `C5_agg` | + agg | 126 | 142 | 134 | 142 |
| `C6_full` | + cycle | 129 | 145 | 137 | 145 |

The cycle transforms get their own rung on purpose. In the original exposé they were folded into "the full pipeline", which would have quietly made the final step measure something other than what its label claimed.

### The three anchors

| Config | What it is | Features |
|---|---|---:|
| `C6_full` | The clean pipeline this paper reports | 129 to 145 |
| `C6_legacy` | The group project code as it currently stands on GitHub | 179 |
| `C6_expose` | The code that produced the exposé's RMSE of 17.1, 23.7, 14.2 and 24.0 | 180 |

`C6_legacy` and `C6_expose` keep their dead columns. They are here so the paper can say, with evidence rather than assertion, what cleaning the feature set actually did to the accuracy.

The two differ by exactly one column: `op_condition`, the k-means group number, handed to the network as though it were a measurement. That is already questionable, since cluster 5 is not five times cluster 1. On FD001 and FD003 it is worse than questionable. Those subsets have a single real operating condition and their settings vary by about 0.002, so k-means was asked to find six groups in what is essentially rounding noise, and obligingly found six. The exposé's model was given a label for measurement noise and told it was a feature.

### Leave one out

A secondary experiment, run on FD001 and FD004 only: the full set minus each of `rolling`, `lag`, `trend`, `agg` and `cycle` in turn.

The ladder and leave-one-out answer different questions and are allowed to disagree. The ladder measures what a group adds given the groups before it. Leave-one-out measures what a group adds given everything else. If two groups happen to do the same job, the ladder will make the second one look useless while leave-one-out makes both look useless. Where they disagree I report the disagreement and discuss it, rather than quietly picking whichever version reads better.

---

## 4. The models

The LSTM is identical for every configuration. The only thing that changes is how many numbers go in the front.

```
Input(30 cycles, N features)
LSTM(128, return_sequences=True)  ->  Dropout(0.3)
LSTM(64)                          ->  Dropout(0.3)
Dense(32, relu)  ->  Dense(1)
```

Adam optimiser at learning rate 0.001, mean squared error loss, batch size 32, at most 60 epochs, early stopping after 10 epochs with no improvement, best weights restored.

*(Dropout switches off a random share of connections during training so the network cannot simply memorise. An epoch is one full pass over the training data. Early stopping halts training once it stops improving, rather than letting it run on and overfit.)*

For Experiment 2, three models that have no concept of time:

`rf` is a Random Forest of 500 trees, shown only the engine's final cycle. `dt` is a single decision tree with its depth chosen by cross-validation, also final cycle only. `rf_flat` is a Random Forest of 500 trees shown all thirty cycles laid out side by side in one long row.

That third one is the important one. If the LSTM beats `rf`, there are two possible explanations and no way to tell them apart: either the LSTM understands sequence, or it simply saw thirty times as many numbers. `rf_flat` sees precisely the same numbers as the LSTM with no notion of their order, which separates the two. If `rf_flat` matches the LSTM, the architecture is not buying anything and the features were already sufficient. If it falls short, reading in order genuinely helps. And if `rf` matches `rf_flat`, the final cycle already contained everything and the other twenty nine were along for the ride.

*(A Random Forest is a large number of decision trees that each vote on the answer. Cross-validation means training on part of the data and scoring on the rest, several times over, so a setting can be chosen without ever touching the test set.)*

---

## 5. Splits, and keeping the test set clean

**The validation split is by engine, never by window.** Twenty percent of training engines are held back using a seeded shuffle. This is not a detail. Two windows from the same engine overlap by up to 29 of their 30 cycles, so splitting by window would scatter near-identical copies of the validation data through the training set, and the early stopping decision would be made against data the model had effectively already seen.

Decision tree depth is selected with GroupKFold on `unit_id`, so no engine ever appears on both sides of a cross-validation split.

The test set is not touched until final scoring. It is never used to decide when to stop, never used to pick a setting, and never used to choose a configuration.

Windows are 30 cycles throughout. Training uses sliding windows and the step size is recorded per run, so if I ever reduce it for speed the record says so. Test uses one final window per engine, padded with zeros at the front if the engine is shorter than 30 cycles. `StandardScaler` is fitted on training windows only.

---

## 6. What gets measured

**RMSE in cycles** is the headline number. It is the average error with large misses weighted heavily, and it is what almost every published C-MAPSS result reports, so it is the only figure that lets this study be compared to anything else.

**MAE in cycles** goes alongside it. Plain average error, much less swayed by a handful of badly predicted engines, and useful precisely because when RMSE and MAE disagree the gap tells you something about the shape of the errors.

**The PHM08 score** is the field's own asymmetric metric. With `d` as predicted minus true, it sums `exp(-d/13) - 1` when the prediction is early and `exp(d/10) - 1` when it is late. Late predictions are punished harder, because saying an engine has 40 cycles left when it has 10 is a different kind of wrong from saying it has 10 when it has 40.

**Training seconds and prediction seconds** are recorded for every run. RQ2 is a question about whether complexity is worth its cost, so cost is a result, not a footnote.

**Number of features and number of dead features** are recorded per configuration. RQ1 is partly a question about simplicity, and simplicity has to be measured to be argued about.

Everything is reported per subset and averaged across the four.

---

## 7. Seeds, and how I decide a difference is real

Every cell runs three times, with seeds 0, 1 and 2. If the spread between seeds exceeds 1.0 RMSE anywhere, every cell goes to five seeds and medians are reported alongside means. Seeds are the one thing in this plan that does not get cut for time.

All tables report the average across seeds with the spread next to it. A number without a spread beside it is not a result, it is an anecdote.

Differences between configurations are tested with a paired bootstrap over engines. Draw a random set of test engines with repeats allowed, score both configurations on that same set, and repeat ten thousand times. Predictions are averaged across seeds before this runs. A difference counts as real only when the middle 95% of those ten thousand differences excludes zero.

Pairing matters here. It is the difference between comparing two runners by racing them on the same course on the same afternoon, and comparing their times from two different meetings in two different countries. Some engines are simply harder to predict than others, and pairing cancels that out instead of letting it drown the signal.

Where a range includes zero, the write-up says the difference is not measurable. It does not say "a slight improvement" or "a trend towards". A feature group that costs thirty columns and buys 0.2 RMSE with a range of plus or minus 0.8 is exactly the finding this study exists to report, and dressing it up as a win would defeat the purpose of running it.

**Why none of this is optional.** There is currently no reproducible baseline for this pipeline at all. The numbers in the exposé came from a version of the code that fed `op_condition` in as an input. The version on GitHub today does not, and the saved model files will not even load against it, because they expect 180 inputs while the current code produces 179. The figures in the group's `report.md` match no saved artifact whatsoever. So run-to-run variation on this pipeline has never once been measured, and until it is, no single-run comparison here can support anything.

---

## 8. The rule for picking the winner

Fixed now, in writing, before I have seen a single number.

> The configuration carried forward into Experiment 2 is the one with the **fewest features that is not measurably worse than the best scoring configuration**, judged by the paired bootstrap at 95% and averaged across all four subsets.

This is a simplicity rule, and it deliberately prefers the smaller feature set whenever the difference falls inside the noise. That is not laziness dressed up as principle. The research question is whether the extra engineering earns its place, so the tie-break has to be the one that makes the extra engineering prove itself.

---

## 9. What I expect, and what would prove me wrong

Writing down the predictions is half of it. Writing down what would falsify them is the half that counts.

**H1. Rolling statistics account for most of the total improvement from `C1_base` to `C6_full`.** Wrong if the `C1` to `C2` step delivers less than half the total gain, or if its range includes zero.

**H2. At least one of `lag`, `trend`, `agg` and `cycle` is redundant given the groups that came before it.** Wrong if every rung's range excludes zero on every subset.

**H3. A Random Forest on the final cycle, given the chosen features, lands within 2 RMSE of the LSTM.** Wrong if the gap is larger than 2 RMSE on most subsets.

**H4. The 34 to 44 dead features in the original pipeline made no difference to accuracy.** Wrong if `C6_legacy` and `C6_full` differ measurably by paired bootstrap.

**H5. Feeding `op_condition` in as a number made no difference either.** Wrong if `C6_expose` and `C6_legacy` differ measurably.

I would be quite happy to be wrong about H4 and H5. If the dead columns or the noise-derived cluster index turn out to have changed the accuracy, that is a more interesting sentence in the paper than if they did not.

---

## 10. Changes and notes

Everything below is added after the fact. Each entry records the date, what changed, why, and whether I had already seen results at the time.

**9 September 2026. Pilot complete.** Three seeds of `C6_expose` on FD001, which is the exposé's exact feature set. RMSE came out at 17.151, 17.345 and 17.339, giving a mean of 17.28 and a spread of 0.111. Mean MAE 12.78. Training took 156, 115 and 125 seconds, and the runs used 19, 14 and 15 epochs, all comfortably inside the 60-epoch cap. The configuration built 180 features of which 44 are constant, matching the figure quoted in Section 2.

Two decisions follow from this, both taken before any ablation results existed. Seeds stay at three, because the observed spread is far below the 1.0 threshold set in Section 7. Window step stays at 1, because a full run costs roughly two minutes and the grid is affordable without thinning the windows.

On reproducing the exposé baseline: the exposé reported 17.1 from a single unseeded run. The seeded mean is 17.28, and the exposé's figure sits at the optimistic end of the three-seed range (17.151 to 17.345) rather than at its centre. The pipeline is therefore reproduced, with the qualification that the number originally published was the better end of a distribution nobody had measured.

One caveat recorded now, before the grid runs. This spread is measured on FD001 alone and from three runs, so it indicates the order of magnitude of run-to-run variation rather than a precise constant. Per-subset spreads are reported with the results, and FD002 and FD004 may prove noisier.

**10 September 2026. Experiment 1 complete, and the seed rule has triggered.** All 96 runs finished: six ladder configurations plus two anchors, four subsets, three seeds.

Seed spread exceeds the 1.0 RMSE threshold set in Section 7 in eight of the thirty-two cells: `C6_full` on FD001 (1.529) and FD002 (1.398), `C6_legacy` on FD003 (1.478) and FD001 (1.063), `C3_lag` on FD003 (1.376), `C5_agg` on FD003 (1.230), `C1_base` on FD001 (1.205), and `C2_rolling` on FD003 (1.169). Section 7 says seeds extend to five if the spread exceeds 1.0 anywhere, so the grid is being rerun with seeds 3 and 4 added, 64 further runs, and medians will be reported alongside means.

This entry is written before those runs start and before any paired bootstrap has been computed, so no significance result influenced the decision.

A correction to the pilot entry above. The 9 September pilot measured a spread of 0.111 and concluded that three seeds would be enough. That measurement was taken on `C6_expose` / FD001, which has turned out to be the quietest cell in the entire grid; every other cell is between four and fourteen times noisier. The pilot was not wrong about what it measured, it was wrong to generalise from one cell to thirty-two. The lesson is recorded here rather than edited away, and the limitation is worth a sentence in the paper: a single-cell variance pilot can badly understate the variance of a grid.

One observation flagged for investigation before any of it is written up. On FD003 the ladder is not monotonic and the departure is systematic rather than random. `C1_base` averages 16.51 RMSE, `C2_rolling` rises to 21.48 and `C3_lag` to 21.60, then `C4_trend` returns to 16.61. All three seeds agree at each rung, so this is a real effect of the feature set and not seed noise. It is either a genuine finding about rolling statistics on a two-fault-mode subset or a defect in how the rolling and lag features are built. Which of the two it is must be established before the result is reported either way.

**10 September 2026, later. Experiment 1 analysed, winning configuration selected.** All 160 runs complete, five seeds in every cell, and the paired bootstrap has been run.

An ambiguity in Section 8 had to be resolved, and it is recorded here rather than quietly decided. The rule says "the fewest features that is not measurably worse than the best-scoring configuration, judged by the paired bootstrap at 95% and averaged across all four subsets." It does not say whether the candidate pool includes the two anchors, nor whether "not measurably worse" is tested per subset or on the average. Both were settled by the plainest reading of what was written. The anchors are excluded, because `C6_legacy` and `C6_expose` deliberately retain dead columns and carrying one into Experiment 2 would confound the architecture question with the dead-column question; the ladder is what Section 3 calls the main experiment. And "averaged across all four subsets" is taken at its word, so the test is on the average rather than subset by subset. For transparency both readings were computed. Averaged, the rule selects `C4_trend`. Tested per subset instead, it would have selected `C6_full`. The two differ by seven features out of roughly one hundred and forty, so nothing of substance turns on it, but the written rule governs.

**The configuration carried into Experiment 2 is `C4_trend`, 132 features.** Against `C6_full` it is +0.704 RMSE averaged across the four subsets, with a 95% range of −0.073 to +1.514, which includes zero. `C5_agg` at 136 features is measurably worse (+0.823, range +0.037 to +1.641), so the aggregate features are not merely redundant but slightly harmful.

Hypothesis outcomes for RQ1, read against the falsification conditions written in Section 9.

H1 is falsified. Rolling statistics were predicted to account for most of the improvement. They are not a measurable improvement on any subset, and on FD003 they are a measurable harm of +3.408 RMSE with a range of +0.972 to +5.800 lying entirely above zero.

H2 is supported. Lag and aggregate features are redundant on all four subsets, with every interval spanning zero.

H4 is supported. The 34 to 44 constant columns change accuracy on no subset; every interval includes zero. The largest effect is FD001 at −0.94 with a range of −2.09 to +0.15.

H5 is partly falsified. The operating-condition index fed in as a number is not measurable on FD001, FD003 or FD004, but on FD002 it is a measurable harm of +1.20 with a range of +0.03 to +2.44. FD002 has six genuine operating conditions, so this is the subset where the index carries real information and it still hurt.

One observation for the discussion, offered as a hypothesis and not yet tested. Degradation trend slopes are a measurable improvement on exactly the two subsets that contain two fault modes, FD003 at −3.079 and FD004 at −2.004, and on neither single-fault subset. A plausible mechanism is that with two failure modes the rate of change discriminates between them where absolute levels do not. This would also explain the FD003 rolling anomaly recorded earlier, since smoothing removes rate information and the trend rung then restores it almost exactly. This should be tested directly rather than asserted.

**12 September 2026. Experiment 2 complete, with one recorded deviation.**

**Deviation from Section 7, stated plainly.** Seventy-six of the eighty Experiment 2 runs completed. The four missing runs are seeds 1 to 4 of `rf_flat` on FD004, which is the most expensive cell in the study: a single run trains 500 trees over a flattened thirty-cycle window of 138 features, which is 4,140 input columns, and took three hours and twenty-eight minutes. Completing the cell would have cost a further fourteen hours of wall clock. The cell therefore rests on one seed instead of five.

The justification, and it is a measurement rather than a preference. Section 7 set five seeds because run-to-run variation had never been quantified and might have been the same size as the effects. For `rf_flat` it has now been quantified on the three completed subsets, and the seed spread is 0.020 on FD001, 0.058 on FD002 and 0.118 on FD003. A Random Forest averages five hundred trees, so the random start is very nearly washed out; the plain `rf` arm behaves the same way at 0.08 to 0.09, and the decision tree is deterministic to four decimal places. These spreads are between one and two orders of magnitude below the differences being tested. The rule's purpose is satisfied by the evidence. The four runs are not abandoned, only deferred: the runner is resumable and re-issuing the identical command will execute exactly those four and nothing else. The paper reports this cell as single-seed.

**A defect in the results file, found and repaired.** `results/runs.csv` had become unparseable. The decision-tree branch of `run_experiment.py` adds a `best_depth` field that no other model reports, while the header is written once from the first row ever appended, which was an LSTM row without that field. Every one of the twenty `dt` rows therefore carried twenty-four values against a twenty-three column header, shifting their timestamps. The file has been rewritten with `best_depth` as a proper column, empty for every non-tree row, and the original preserved as `results/runs.csv.backup_2026-09-12`. No measurement changed; only the column layout. `run_experiment.py` has been patched so that every row is built with identical columns regardless of model. Cross-validated depth was 6 in all twenty tree runs.

**Results for RQ2.** All four arms use the `C4_trend` feature set. Paired bootstrap as specified in Section 7, with predictions averaged across seeds before resampling.

Reading the cycles in order, which is the LSTM against the order-blind `rf_flat` on identical numbers, is not measurable on FD001 (−0.44, range −2.21 to +1.17) or FD002 (−0.69, range −2.32 to +0.87), and is a real improvement on FD003 (−2.74, range −4.81 to −0.78) and FD004 (−3.10, range −5.35 to −1.03).

Seeing the thirty-cycle window at all, which is `rf_flat` against `rf` on the final cycle alone, is a real improvement on FD001 (−2.84), FD002 (−1.70) and FD003 (−2.54), and not measurable on FD004 (−0.12, range −2.39 to +2.32, the single-seed cell).

H3 is falsified. It predicted that a Random Forest on the final cycle would land within 2 RMSE of the LSTM. The gap is 3.28 on FD001, 2.39 on FD002, 5.29 on FD003 and 3.22 on FD004, exceeding 2 on all four subsets rather than on none.

**A methodological note that must appear in the paper.** Table 2 reports the mean of the per-seed RMSE values, whereas the bootstrap in Section 7 averages the predictions across seeds before scoring, which is an ensemble. The two are not the same quantity and they do not always agree in sign. On FD001 the mean-of-RMSE favours `rf_flat` by 1.05 while the ensemble comparison favours the LSTM by 0.44. The reason is that averaging predictions rewards a model whose seeds disagree, and the LSTM's seed spread is 0.68 to 1.14 while `rf_flat`'s is 0.02 to 0.12, so ensembling helps the LSTM far more. Both figures are honest and they answer different questions: what one run typically achieves, against what an ensemble of five achieves. Both are reported, and the distinction is explained rather than resolved by choosing whichever is more convenient.

**13–14 September 2026. The FD003 rolling result is explained, and my predicted mechanism was wrong.**

The anomaly flagged on 10 September has been investigated and closed. It is a property of the features, not a defect in the code.

**What was ruled out first, without training anything.** Three candidate defects were tested by inspecting the constructed feature matrix directly. Missing values from the rolling warm-up: ruled out, the final matrix contains no NaNs on either subset. Near-constant columns diluting the input: ruled out, FD003 has four such columns, FD001 has four, FD002 four and FD004 two, so FD003 is not unusual. Train-test distribution mismatch: ruled out, and in the opposite direction, since FD003's rolling features drift less between train and test than any other subset (ratio 0.78 against FD002's 1.04).

**The decisive test.** The rolling group bundles two different things, means and standard deviations, so the ladder cannot say which does the damage. Two exploratory configurations were added, `C2a_rollmean` and `C2b_rollstd`, each holding one half. These are post-hoc diagnostics prompted by an unexpected result. They are deliberately held in a separate `DIAG_CONFIGS` dictionary so that the pre-registered ladder in `CONFIGS` is unchanged, and they are not part of Experiment 1. Twenty runs on FD001 and FD003, five seeds each, using the same code as every earlier run.

Measured against `C1_base` by paired bootstrap: rolling means alone are not measurable on either subset (FD001 −0.46, range −1.52 to +0.50; FD003 −0.37, range −1.14 to +0.34). Rolling standard deviations alone are a real harm on both (FD001 +1.98, range +0.10 to +3.68; FD003 +4.36, range +1.65 to +6.96). Head to head on FD003 the means beat the standard deviations by 4.73, range −7.62 to −1.82.

**The conclusion.** The rolling standard deviations do all the damage and the rolling means are harmless. The likely cause is dimensional dilution rather than anything about smoothing: the standard deviations track RUL about a fifth as strongly as the raw sensors (mean absolute correlation 0.10 to 0.14 against 0.34 to 0.54), and thirty of them are added to roughly eighteen informative columns. On FD003 they are flattest of all, with mean variance 0.0005 against 0.0016 to 0.0039 elsewhere, which is consistent with the effect being largest there.

**A hypothesis of mine, recorded as falsified.** On 10 September I wrote that the likely mechanism was rolling smoothing destroying rate-of-change information on the two subsets with two fault modes, and that the trend rung then restored it. That is wrong on both counts. The means are the smoothing operation and they are harmless, and the standard deviations harm FD001 as well, which has a single fault mode. The FD003 result is not a fault-mode phenomenon. It only appeared to be one because on FD001 the helpful means partly cancel the harmful standard deviations and pull the combined result back inside the noise.

**A methodological point worth reporting.** On FD001 the rolling group as a whole is not measurably harmful (+1.11, range −0.44 to +2.54) while its standard-deviation half alone clearly is. Bundling two features of opposite sign into one ablation rung concealed a real effect. This is an argument for ablating at a finer grain than is usual in this literature.

**A second defect in the results file, same class as the first.** Adding the two diagnostic groups extended `GROUP_ORDER` from seven entries to nine, so the twenty new rows were written with twenty-six fields against a twenty-four column header. Repaired the same way: `rolling_mean` and `rolling_std` are now proper columns, zero for all 236 earlier rows, with the original preserved as `results/runs.csv.backup_2026-09-14`. No measurement changed. Note that `group_sizes` classifies columns by name and both new kinds contain the substring `_rolling_`, so their counts continue to appear under `rolling`; the two new columns are therefore always zero and are retained only to keep the file rectangular.

**14 September 2026. A minor leak across engine boundaries, fixed in code but deliberately not re-run.**

While investigating the FD003 result a genuine defect was found in `features_v2.py`. Rolling and lag columns are undefined at the very start of each engine's history, and the code filled those gaps with `df.ffill()` applied to the whole table rather than to each engine separately. The consequence is that cycle 1 of one engine inherited the last recorded value of the engine before it. Every engine boundary in the training data leaked one row.

The size of the effect was measured rather than estimated. Rebuilding the `C2_rolling` features both ways and comparing cell by cell: 2,772 of 1,588,587 cells change on FD001, which is 0.174%, and 2,908 of 2,027,040 on FD003, which is 0.143%. The largest single change is 0.354 and the median change 0.056, on features scaled to roughly 0 to 1. The defect affects FD001 marginally more than FD003, so it cannot account for the FD003 result, and its magnitude is far below any difference this study reports.

`features_v2.py` now groups by engine before forward-filling, and the fix is verified: the first cycle of engine 2 no longer carries engine 1's final value.

**Every result in this repository was produced before that fix and has not been regenerated.** That is a deliberate decision, recorded here rather than left implicit. Regenerating would mean re-running every configuration containing rolling or lag features across both experiments, roughly a day and a half of compute, to move numbers by an amount the measurement above bounds well below the seed spread. The timing of the fix relative to the runs is checkable in the git history, and re-running remains open as future work should a reviewer ask for it.

The ordering also mattered and was chosen deliberately. The two diagnostic configurations of 13 September were run against the unfixed code, so that they were comparable with the 236 runs already in `runs.csv`. The fix was applied only afterwards. Changing the feature pipeline in the middle of a comparison would have confounded the diagnosis with a second change.

**18–19 September 2026. Leave one out, and a defect in it caught before it produced a single result.**

Leave-one-out was specified in Section 3 and had never been run. Before launching it I printed the contents of `LOO_CONFIGS` as a check. All five configurations were wrong.

`LOO_CONFIGS` was built as a comprehension over `GROUP_ORDER`, taking every group except the one being left out. That was correct when it was written. The 13–14 September entry above extended `GROUP_ORDER` from seven entries to nine by adding `rolling_mean` and `rolling_std`. That edit was made for the FD003 investigation and had nothing to do with leave-one-out, but leave-one-out read from the same list.

The consequences, measured rather than inferred. `L_no_rolling` dropped `rolling` and kept `rolling_mean` and `rolling_std`, so the configuration meant to contain no rolling features contained all sixty of them. The other four contained `rolling` together with both of its halves, so the rolling columns were built twice; the duplicate column names then made `var[c]` return a Series rather than a scalar and the run died in the dead-column check.

Four of the five would therefore have failed loudly within a minute. `L_no_rolling` would not. It would have trained, written plausible numbers, and reported that removing the rolling group changes nothing while the rolling group was still present. That is the one configuration in this study that could least afford to be silently wrong, given what the 13–14 September entry established about rolling statistics.

No published result is affected. `results/runs.csv` contained no `L_` rows at the time, so leave-one-out had never run. The ladder and the anchors list their groups explicitly and never referred to `GROUP_ORDER`, so nothing in Experiment 1 or Experiment 2 touched the defect. The repair replaces the derivation with an explicit `LADDER_GROUPS` list, so a future edit to `GROUP_ORDER` cannot reach these configurations again. Committed as `8eb8b85` before any run started, which the git history records.

Worth noting what the 13–14 September entry did and did not catch. It recorded the `GROUP_ORDER` change and its first consequence, the results file gaining two columns. The second consequence went unnoticed for five days. A list that several unrelated things derive from turns out to be a poor place to keep a definition.

**A reference configuration added at the same time.** All 256 runs then in `runs.csv` were produced before the forward-fill repair of 14 September. Comparing new leave-one-out runs against the old `C6_full` would have measured two changes at once, which is the same mistake the 14 September entry deliberately avoided when it ran the diagnostics against the unfixed code. `L_full` was therefore added, holding the same six groups as `C6_full`, so that every leave-one-out comparison sits inside one code version. Its feature counts came out at 129 on FD001 and 145 on FD004, matching `C6_full` in Section 3 exactly.

**The runs.** Sixty runs — six configurations on FD001 and FD004, five seeds each — completed 19 September, no failures. Analysis is in a new `analyze_loo.py`, written because `analyze_extra.py` has no leave-one-out handling. It writes `results/significance_loo.csv`, `results/significance_ffill.csv` and `results/table5_loo.csv`, by the same method as everything else: paired bootstrap over test engines, 10,000 resamples, predictions averaged across seeds first.

**Results.** Each group removed from the full set and measured against `L_full`. A positive number means removing the group made the error worse, so the group was contributing.

| Group removed | FD001 | FD004 | Averaged | Reading |
|---|---|---|---|---|
| `rolling` | −2.825 [−4.771, −0.984] | −1.338 [−2.335, −0.384] | −2.081 [−3.150, −1.049] | **removing it helps, measurably, on both** |
| `lag` | −0.768 [−2.053, +0.488] | +0.035 [−0.642, +0.700] | −0.367 [−1.092, +0.342] | not measurable |
| `trend` | +1.865 [+0.149, +3.622] | +1.877 [+0.692, +3.140] | +1.871 [+0.818, +2.926] | **contributes on both** |
| `agg` | +0.006 [−0.921, +0.926] | +0.079 [−0.733, +0.913] | +0.041 [−0.575, +0.670] | not measurable |
| `cycle` | −0.532 [−2.180, +1.028] | +1.366 [+0.113, +2.602] | +0.419 [−0.622, +1.405] | contributes on FD004 only |

Mean test RMSE over five seeds, for reference: `L_full` 17.04 and 24.59; `L_no_rolling` 14.50 and 23.20; `L_no_lag` 16.51 and 24.56; `L_no_trend` 19.40 and 26.93; `L_no_agg` 16.99 and 24.53; `L_no_cycle` 16.66 and 26.07, on FD001 and FD004 respectively.

**Where the two methods agree.** Lag and aggregate features are not measurable by either method on either subset. They are genuinely redundant rather than merely masked, which is the strongest form the H2 result could take. Trend and cycle features on FD004 are measurable by both methods, in the same direction.

**Where they disagree, which is the half worth reporting.**

Rolling statistics. The ladder calls them not measurable on FD001 (+1.111, range −0.445 to +2.540) and on FD004 (−0.493, range −3.441 to +2.216). Leave-one-out calls them a measurable harm on both. Dropping all sixty rolling columns from the full set improves FD001 by 2.83 RMSE and FD004 by 1.34. The ladder understated the harm because it adds rolling to a bare seventeen-column baseline, while leave-one-out removes it from a set that also contains trend and cycle. That is consistent with the dilution account established on 13–14 September: the more informative columns are present, the more thirty weakly-correlated standard deviations cost.

Trend on FD001. The ladder calls it not measurable (−0.504, range −1.872 to +0.811). Leave-one-out calls it a real contribution of 1.87 RMSE, range +0.149 to +3.622. This is exactly the masked case Section 3 anticipated when it said the two experiments are allowed to disagree.

**A methodological point, and it is the second of its kind.** The 13–14 September entry recorded that bundling two features of opposite sign into one rung concealed a real effect. This entry records that a group's measured contribution depends on what is already present, so a single ordering conceals effects as well. Both point the same way: one ablation path through a feature set is not enough to support a claim that a group does not matter. The two findings are independent and mutually reinforcing, and together they are the most transferable thing in this study.

**The best configuration found is not the one carried into Experiment 2, and it is not being changed.** `L_no_rolling` scores 14.50 on FD001 and 23.20 on FD004, better than any other configuration measured on either subset, using 69 and 77 features against `C6_full`'s 129 and 145. Experiment 2 used `C4_trend`, selected by the rule fixed in Section 8 before any result existed.

Re-selecting now, with the results in hand, is precisely what Section 8 exists to prevent. The rule was pre-registered, it was applied to the ladder as written, and leave-one-out is a secondary experiment run afterwards on two of the four subsets. The paper reports plainly that the pre-registered rule did not select the best configuration the study found. That is a result about pre-registration, not an embarrassment to be tidied away. Experiment 2's conclusions are unaffected in any case: all four of its arms use identical features, so the comparison between architectures is internally valid whatever those features are.

**What the forward-fill repair actually did, and a correction to the 14 September entry.** `L_full` and `C6_full` hold the same six groups at the same five seeds. The only intended difference is that `L_full` was built after the forward fill was grouped per engine, so the comparison isolates the repair.

| Subset | `L_full` minus `C6_full` | 95% range | Verdict |
|---|---:|---|---|
| FD001 | −0.774 | [−1.548, −0.049] | repaired build measurably better |
| FD004 | −0.003 | [−0.739, +0.694] | not measurable |
| Averaged | −0.389 | [−0.913, +0.118] | not measurable |

The 14 September entry justified not regenerating on the grounds that the defect's magnitude was far below any difference this study reports. That was an inference from how much the feature cells changed, not a measurement of how much the accuracy changed, and it was too confident. On FD001 the repaired build is measurably better by 0.77 RMSE, which is comparable to differences this study does report: the margin by which `C4_trend` was selected over `C6_full` in Section 8 was 0.70.

Two qualifications, both real. On FD004 there is no measurable difference, and averaged over the two subsets there is none either. And this comparison cannot cleanly separate the repair from run-to-run variation, because same-seed rerun reproducibility has never been measured on this pipeline and TensorFlow on Metal is not guaranteed to be bitwise deterministic. What can be said is that the repaired build is nowhere worse, and that any effect of the repair is bounded well under 1.6 RMSE on FD001 and under 0.7 on FD004.

The decision not to regenerate stands for now: regeneration is still roughly a day and a half of compute, and neither the direction nor the averaged magnitude changes any conclusion in the paper. But the justification is now a measurement with a stated confound rather than an inference, and the honest position is that a reviewer would be within their rights to ask for the regeneration. A determinism check — rerunning one cell at an identical seed under identical code and comparing — would cost about four minutes and would remove the confound. It is recorded here as the obvious next step rather than done, so that the choice is visible rather than implied.

**Two stale generated files, found while regenerating.** `results/table3_metrics.csv` and `results/table4_cost.csv` had been committed before the twenty diagnostic runs of 13 September were added, and never regenerated, so they did not match `runs.csv`. Regenerating adds the four diagnostic rows to table 3 and moves the Experiment 1 LSTM mean training time in table 4 from 193.35 seconds over 160 runs to 186.60 over 180. No measurement changed and no significance file changed; all four are byte-identical after regeneration. Note that table 3 now mixes the pre-registered ladder with the exploratory diagnostics in a single table, which the paper must label explicitly rather than leave to the reader.
