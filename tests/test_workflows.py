"""Executable checks; synthetic arrays are generated in memory, never bundled."""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.naive_bayes import GaussianNB

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
import contour_ml_common as common
from contour_models import build_eight_analyte_models


def synthetic_events(classes, n=24, seed=42, experiment="development"):
    rng = np.random.default_rng(seed)
    y = np.repeat(np.arange(len(classes)), n)
    X = rng.normal(size=(len(y), 8)) + y[:, None] * 0.3
    df = pd.DataFrame(X, columns=common.FEATURES)
    df[common.LABEL] = [classes[i] for i in y]
    df["ExperimentID"] = experiment
    df["EventID"] = [f"event-{i}" for i in range(len(y))]
    return df, X, y


class SpecificationTests(unittest.TestCase):
    def test_table4_model_names_and_elastic_net(self):
        models = build_eight_analyte_models()
        self.assertEqual(list(models), ["AdaBoost", "Logistic-EN", "kNN", "QDA", "SVM-RBF",
                         "GaussianNB", "CatBoost", "ExtraTrees", "Random Forest", "LightGBM",
                         "HistGBDT", "XGBoost", "Stack"])
        for estimator, iterations in ((models["Logistic-EN"][-1], 5000),
                                       (models["Stack"].final_estimator, 2000)):
            p = estimator.get_params()
            self.assertEqual((p["penalty"], p["solver"], p["l1_ratio"], p["C"], p["max_iter"],
                              p["class_weight"], p["random_state"]),
                             ("elasticnet", "saga", 0.2, 2.0, iterations, "balanced", 42))
        self.assertEqual(models["Stack"].cv, 3)
        self.assertEqual(models["Stack"].stack_method, "predict_proba")
        self.assertEqual([name for name, _ in models["Stack"].estimators], ["svm", "rf", "xgb", "gnb"])

    def test_manuscript_split_counts_and_reproducibility(self):
        y = np.repeat(np.arange(8), 400)
        train, validation = common.development_split(y)
        again = common.development_split(y)
        self.assertEqual((len(train), len(validation)), (2400, 800))
        self.assertFalse(set(train) & set(validation))
        np.testing.assert_array_equal(np.bincount(y[train]), np.repeat(300, 8))
        np.testing.assert_array_equal(np.bincount(y[validation]), np.repeat(100, 8))
        np.testing.assert_array_equal(train, again[0])

    def test_aliases_only_rename_and_bad_rows_are_rejected(self):
        df, X, _ = synthetic_events(common.EIGHT_ANALYTE_CLASSES)
        legacy = df.rename(columns={"skew": "Skew", "toff": "Dwell Time (ms)",
                                    "median I/I0": "Median I/I0", "kurt": "Kurtosis"})
        np.testing.assert_array_equal(common.normalise_columns(legacy)[list(common.FEATURES)], X)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "external.csv"
            legacy.to_csv(path, index=False)
            _, actual, _ = common.load_feature_table(path, common.EIGHT_ANALYTE_CLASSES, require_provenance=True)
            np.testing.assert_allclose(actual, X)
            legacy.loc[0, "Analyte"] = "unknown"
            legacy.to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "no silently discarded"):
                common.load_feature_table(path, common.EIGHT_ANALYTE_CLASSES)

    def test_duplicate_event_ids_rejected(self):
        df, _, _ = synthetic_events(common.EIGHT_ANALYTE_CLASSES)
        df.loc[1, "EventID"] = df.loc[0, "EventID"]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "external.csv"
            df.to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "Repeated ExperimentID/EventID"):
                common.load_feature_table(path, common.EIGHT_ANALYTE_CLASSES, require_provenance=True)

    def test_independence_checks_experiment_and_reused_events(self):
        df, X, _ = synthetic_events(common.EIGHT_ANALYTE_CLASSES)
        bundle = {"development_experiments": ["development"], "development_feature_hashes": list(common.feature_hashes(X))}
        with self.assertRaisesRegex(ValueError, "experiments must be disjoint"):
            common.check_test_independence(bundle, df, X)
        df["ExperimentID"] = "test"
        with self.assertRaisesRegex(ValueError, "feature vectors overlap"):
            common.check_test_independence(bundle, df, X)

    def test_unassigned_not_forced_to_target(self):
        result = common.apply_confidence_thresholds(np.array([[0.6, 0.4], [0.1, 0.9]]),
                                                     ("a", "b"), {"a": 0.8, "b": 0.8})
        self.assertEqual(result.tolist(), ["unassigned", "b"])
        with self.assertRaises(ValueError):
            common.apply_confidence_thresholds(np.array([[0.6, 0.4]]), ("a", "b"), {"a": float("nan"), "b": 0.5})

    def test_test_stage_never_fits_or_changes_selection(self):
        df, X, y = synthetic_events(common.EIGHT_ANALYTE_CLASSES)
        test_df, test_X, test_y = synthetic_events(common.EIGHT_ANALYTE_CLASSES, n=8, seed=5, experiment="test")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.object(common, "build_eight_analyte_models", return_value={"Stack": GaussianNB()}):
                bundle = common.train_development(df, X, y, root / "training")
            selection = bundle["selected_by_validation"]
            with patch.object(bundle["models"]["Stack"], "fit", side_effect=AssertionError("Test must never fit")):
                first = common.evaluate_independent_test(bundle, test_df, test_X, test_y, root / "test")
                common.evaluate_independent_test(bundle, test_df, test_X, (test_y + 1) % 8, root / "relabeled")
            self.assertEqual(bundle["selected_by_validation"], selection)
            self.assertEqual(first["partition"].tolist(), ["independent_test"])
            train, _ = common.development_split(y)
            np.testing.assert_allclose(bundle["models"]["Stack"].theta_,
                                       GaussianNB().fit(X[train], y[train]).theta_)


class RuntimeSmokeTests(unittest.TestCase):
    def test_estrogen_nested_oof_is_labelled_validation(self):
        _, X, y = synthetic_events(common.ESTROGEN_CLASSES, n=16)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "estrogens"
            scores = common.evaluate_stack_oof(X, y, output)
            self.assertEqual(scores["partition"].tolist(), ["5-fold OOF validation"])
            assignment = pd.read_csv(output / "validation_folds.csv")
            self.assertEqual(len(assignment), len(y))
            self.assertEqual(set(assignment["validation_fold"]), {1, 2, 3, 4, 5})

    def test_all_thirteen_prespecified_models_fit_and_predict(self):
        _, X, y = synthetic_events(common.EIGHT_ANALYTE_CLASSES, n=32)
        train, validation = common.development_split(y)
        for name, definition in build_eight_analyte_models().items():
            with self.subTest(model=name):
                fitted = clone(definition).fit(X[train], y[train])
                probabilities = fitted.predict_proba(X[validation])
                self.assertEqual(probabilities.shape, (len(validation), 8))
                self.assertTrue(np.isfinite(probabilities).all())
                np.testing.assert_allclose(probabilities.sum(axis=1), 1, atol=1e-5)
                self.assertEqual(common.predict_labels(fitted, X[validation]).shape, (len(validation),))

    def test_column_vector_predictions_have_correct_error_count(self):
        result = common.metrics(np.array([0, 1, 2]), np.array([[0], [1], [1]]),
                                np.array([[.8, .1, .1], [.1, .8, .1], [.1, .8, .1]]))
        self.assertEqual(result["n_errors"], 1)


if __name__ == "__main__":
    unittest.main()
