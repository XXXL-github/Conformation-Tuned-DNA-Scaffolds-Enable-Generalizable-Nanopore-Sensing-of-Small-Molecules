# Analysis correspondence to the revised manuscript

The project title follows the October 2026 manuscript, supporting information and
cover letter. This code release implements event-feature analyses; event extraction
from raw traces, molecular docking and experimental datasets are outside its scope.

| Manuscript analysis | Code | Evaluation |
| --- | --- | --- |
| Fig. 5; Methods; Supplementary Table 4 | `run_eight_analyte_classification.py`, `contour_models.py` | 2,400 training / 800 validation; separate 800-event test |
| Fig. 3 scaffold discrimination | `run_scaffold_screening.py` | Four named classifiers; five-fold OOF validation |
| Fig. 3 scaffold scoring and bootstrap | `mechanism_ranking`, `bootstrap_ranking` | Reconstructed ranking; requires original-data verification |
| Fig. 6 / Supplementary Fig. 48 estrogen classification | `run_estrogen_stack_classification.py` | Archived five-fold OOF validation; not independent testing |
| Methods real-sample confidence rule | `run_real_sample_prediction.py` | External validated thresholds; unassigned events retained |

## Eight-analyte model specification

All 12 individual classifiers and Stack use Supplementary Table 4 settings,
including the individual Logistic-EN (`max_iter=5000`) and the Stack's Logistic-EN
(`max_iter=2000`). Both use elastic-net, SAGA, l1_ratio 0.2, C 2.0, balanced class
weights and random state 42. Stack uses the four specified base classifiers,
probability outputs and three-fold cross-validation. No threshold, hyperparameter
or model selection is fitted using independent-test data. Library versions are
recorded with local run outputs.

The event-level 75/25 development split follows the manuscript. Its event units
must not be described as held-out experiments. Independent-test checks reject
overlapping experiment IDs and exact reused feature vectors; these are consistency
checks on supplied provenance, not a substitute for genuinely separate acquisition.

## Recovered scaffold analysis and reconstruction

The recovered earlier analysis used paired LI/LII values for I/I0, fluctuation,
skewness and dwell time. Its event feature equations are retained:

- log10 dwell time for each level;
- LII minus LI for I/I0, ΔSD and skew;
- normalized amplitude change, 2 × (LII − LI) / (LII + LI);
- log10 of the LII/LI dwell-time ratio.

These yield 13 model inputs. The revised text's four model families are retained.
The archived SVM C=5, distance-weighted kNN k=9, and Random Forest with 400 trees
and balanced_subsample class weights are preserved. ExtraTrees with 400 trees and
balanced class weights is an explicit reconstruction because its historical
settings were not found. The manuscript does not currently state these scaffold
hyperparameters, feature equations or fold details.

The recovered ranking used additional classifiers and a random-weight sensitivity
analysis. Random weighting is not an event bootstrap. Consequently its outputs
cannot be relabelled as the current figure's bootstrap probabilities.

The reconstructed physical ranking has four equally ranked dimensions:

1. LI stability and LII stability separately: lower amplitude CV, ΔSD CV, skew SD,
   log10 dwell SD and log10 dwell IQR are preferred.
2. Transition consistency: higher signal-to-dispersion ratios and lower SD are
   preferred for the five level-transition features. The dwell-ratio measure uses
   absolute median/IQR; other ratios use absolute mean/SD. A machine epsilon lower
   bound handles zero dispersion in a bootstrap resample.
3. Compactness: lower log-determinant of a Ledoit-Wolf covariance matrix in the
   standardized 13-feature space is preferred. The descriptive scaler is fitted
   once to the scaffold dataset and fixed during bootstrap; it is separate from
   the training-fold scalers used for classifier validation.

Within each dimension, mean rank, geometric mean rank and Borda score are ranked,
then their ranks are averaged, following the recovered rank-fusion rule.
Dimension ranks are scaled to [0,1] (0.5 if all tied). The four dimension scores
are combined with the same rank-fusion rule. Bootstrap draws events with replacement
within each scaffold, preserves class sizes, recalculates all scores and divides
tied first-place credit equally. Default: 1,000 replicates, seed 42. Rank quantiles
and first-place frequencies are exported without smoothing or forced winners.

This release documents the computation and its input requirements. It does not
include experimental inputs, precomputed scores, rankings, accuracy values or
other numerical analysis results. The functions calculate these quantities only
when a user supplies external event tables. The reconstructed scaffold procedure
is described above; exact reproduction of a particular historical analysis is
not asserted.

## Outstanding specification details

The source manuscript does not state the estrogen outer-fold protocol or the
numerical class-confidence thresholds and selection rule. The estrogen script
preserves the recovered five-fold analysis, and real-sample assignment requires
externally supplied validation-derived thresholds. These details need author
confirmation for exact result reproduction.

Code Availability should point directly to this repository. Scripts, dependency
specifications, tests and method documentation are provided at the repository root.
