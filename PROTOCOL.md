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
