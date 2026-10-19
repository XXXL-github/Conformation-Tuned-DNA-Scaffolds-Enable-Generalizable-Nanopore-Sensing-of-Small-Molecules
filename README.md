# Conformation-Tuned DNA Scaffolds Enable Generalizable Nanopore Sensing of Small Molecules

Analysis code for CONTOUR-DNA scaffold screening and nanopore-event classification.
This repository contains source code, dependency specifications and documentation.
Experimental data, source-data tables, figures, trained models and generated results
are supplied separately and are not included here.

## Installation

Use Python 3.12 in an isolated environment, then install the pinned dependencies:

```bash
python -m pip install -r requirements.txt
```

## Eight-analyte training, validation and independent testing

The model definitions follow Supplementary Table 4 of the revised manuscript:
AdaBoost, Logistic-EN, kNN, QDA, SVM-RBF, GaussianNB, CatBoost, ExtraTrees,
Random Forest, LightGBM, HistGBDT, XGBoost and Stack. All classifiers use the same
development split. Preprocessing is fitted inside each training pipeline, including
within the stacking folds. Model settings are in `code/contour_models.py`.

| Partition | Events per analyte | Total events | Purpose |
| --- | ---: | ---: | --- |
| Training | 300 | 2,400 | Fit model parameters; internal 3-fold stacking |
| Validation | 100 | 800 | Compare classifiers; inspect learning curves |
| Independent test | 100 | 800 | Evaluate frozen classifiers using separate nanopore experiments |

Provide **two external feature tables**: a development table with 400 events per
analyte and an independent-test table with 100 events per analyte. A random split
of the development table cannot substitute for the independent experiment.
Each table must contain `Analyte`, `ExperimentID`, `EventID`, and these features:

```text
I/I0, ΔSD, skew, median I/I0, toff, kurt, IQR I/I0, CV
```

`toff` is in milliseconds. Classes are `Mal, IsoMal, Mal3, IsoMal3, 3-DG, 5-AMF,
5-EMF, 5-HMF`. EventID identifies an event within its experiment. Experiment IDs
must identify actual separately acquired experiments and must not overlap between
development and independent testing. The loader preserves row order, rejects
unknown labels/missing values, and never silently filters, imputes or augments events.

CSV or XLSX is accepted; XLSX uses sheet `Expanded_Data` unless `--sheet` is given.
The archived names `RMS (pA)`, `Skew`, `Median I/I0`, `Dwell Time (ms)`, `Kurtosis`
and `English abbreviation` are accepted as column aliases. DP2/IDP2/DP3/IDP3 are
mapped to Mal/IsoMal/Mal3/IsoMal3. These operations only rename existing fields;
they do not calculate a new ΔSD or change feature values.

```bash
python code/run_eight_analyte_classification.py train \
  --development /external/development.xlsx \
  --output-dir /external/results/development --learning-curve

python code/run_eight_analyte_classification.py test \
  --test /external/independent_test.xlsx \
  --models /external/results/development/development_models.joblib \
  --output-dir /external/results/independent_test
```

The train command never opens the test table. It saves training-only fitted models,
the exact training/validation assignments, validation metrics and predictions.
Validation macro-F1, followed by accuracy, determines the comparison's validation
selection; Stack remains the manuscript's prespecified final architecture. Both
names are recorded. The test command performs prediction only, retains that frozen
selection and reports all 13 classifiers in their predefined order. It does not
refit, tune or choose a model using test performance. Results include probabilities,
confusion matrices, ROC values and the Stack superclass matrix. Learning curves
use only training subsets and the fixed validation subset, without smoothing or
altering the calculated accuracies.

Use the same pinned environment for both stages. Record the original event order
and experiment provenance when preparing the external tables; random state alone
does not recover a historical partition if the input order or event membership changes.
The numerical results are calculated from supplied data; manuscript accuracies
are not embedded as expected outputs.

## DNA scaffold screening

```bash
python code/run_scaffold_screening.py \
  --input /external/scaffold_events.csv \
  --output-dir /external/results/scaffolds --bootstrap 1000
```

The input is a flat event table with `Scaffold`, `EventID` and paired LI/LII features:
`LI_I/I0, LI_ΔSD, LI_skew, LI_toff, LII_I/I0, LII_ΔSD, LII_skew, LII_toff`.
The recovered archive's `tag, event_id, LI_II0, LI_RMS, LI_Skew, LII_II0,
LII_RMS, LII_Skew` names are accepted. It uses the 12 reported loop variants,
five-fold out-of-fold validation of ExtraTrees, Random Forest, SVM-RBF and kNN,
fold-wise ExtraTrees feature importance, and a mean-probability consensus.

The earlier archive supplied the LI/LII feature equations and physical stability
and transition descriptors. The current figure's exact four-model/bootstrap
implementation was not recovered. The ranking is therefore explicitly reconstructed:
LI stability, LII stability, transition consistency and covariance compactness are
combined by rank aggregation, with events resampled within each scaffold.
See `docs/methods_crosswalk.md` for the formula and parameter provenance.
No precomputed scores, selected-scaffold results or bootstrap probabilities are
included. Bootstrap output describes event resampling, not an independent-experiment test.

## Steroidal estrogen classification

```bash
python code/run_estrogen_stack_classification.py \
  --input /external/estrogen_features.xlsx \
  --output-dir /external/results/estrogens
```

Use the same eight event features and the classes `NIL, EE2, MES`. This entry point
retains the recovered analysis's five-fold outer validation and three-fold internal
stacking. Its output is labelled **OOF validation**, not an independent test set.
The Stack settings follow the revised model specification. The manuscript does
not specify a separate estrogen experiment split; this script does not invent one.

## Real-sample assignment

```bash
python code/run_real_sample_prediction.py \
  --input /external/real_sample_features.xlsx \
  --models /external/results/development/development_models.joblib \
  --thresholds /external/validated_thresholds.csv \
  --output-dir /external/results/real_samples
```

The external threshold table must contain one `Analyte,threshold` entry per class,
established using validation data before real-sample or independent-test analysis.
The revised manuscript does not provide the numerical thresholds or their selection
criterion, so no thresholds are invented or bundled here. Events below the
predicted class's threshold remain `unassigned` and are excluded from target counts.

## Verification and code-only publication

```bash
python -m unittest discover -s tests -v
python tools/check_code_only_release.py
```

Tests generate synthetic inputs at runtime to check execution and data separation;
they do not validate experimental accuracy. All result directories must be new,
preventing accidental overwrite. Generated outputs stay local and are excluded
by `.gitignore`; the release checker enforces an explicit source-file allowlist.
