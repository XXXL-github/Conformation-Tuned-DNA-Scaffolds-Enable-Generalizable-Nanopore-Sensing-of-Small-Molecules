from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
from run_scaffold_screening import (
    SCAFFOLDS, RAW_FEATURES, MODEL_FEATURES, bootstrap_ranking, classify_scaffolds, construct_features, load_scaffolds, main, scaffold_models,
)


TEST_RANDOM_STATE = int(np.random.default_rng().integers(np.iinfo(np.int32).max))


class ScaffoldTests(unittest.TestCase):
    def test_cli_requires_explicit_run_settings(self):
        arguments = ["screening", "--input", "external.csv", "--output-dir", "local-output"]
        settings = (("--random-state", str(TEST_RANDOM_STATE)),
                    ("--bootstrap", str(len(SCAFFOLDS))))
        for setting in settings:
            with self.subTest(provided=setting[0]):
                with patch.object(sys, "argv", arguments + list(setting)), patch("sys.stderr"):
                    with self.assertRaises(SystemExit) as error:
                        main()
                self.assertEqual(error.exception.code, 2)

    @staticmethod
    def events():
        rng = np.random.default_rng()
        df = pd.DataFrame(rng.lognormal(0, .25, (len(SCAFFOLDS) * 16, 8)), columns=RAW_FEATURES)
        df["Scaffold"] = np.repeat(SCAFFOLDS, 16)
        return construct_features(df)

    def test_only_the_four_reported_model_families(self):
        self.assertEqual(list(scaffold_models(TEST_RANDOM_STATE)), ["ExtraTrees", "Random Forest", "SVM-RBF", "kNN"])

    def test_recovered_feature_equations(self):
        df = self.events()
        self.assertEqual(len(MODEL_FEATURES), 13)
        np.testing.assert_allclose(df["Delta_I/I0"], df["LII_I/I0"] - df["LI_I/I0"])
        np.testing.assert_allclose(df["log_toff_ratio"], np.log10(df["LII_toff"] / df["LI_toff"]))

    def test_external_flat_table_loads_without_changing_values(self):
        df = self.events()
        df["EventID"] = np.arange(len(df))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "external.csv"
            df.to_csv(path, index=False)
            loaded = load_scaffolds(path)
            np.testing.assert_allclose(loaded[list(MODEL_FEATURES)], df[list(MODEL_FEATURES)])

    def test_bootstrap_reproducible_and_all_candidates_receive_fair_credit(self):
        events = self.events()
        # A small synthetic test workload, unrelated to the study run settings.
        first, ranks = bootstrap_ranking(events, n_bootstrap=4, random_state=TEST_RANDOM_STATE)
        second, _ = bootstrap_ranking(events, n_bootstrap=4, random_state=TEST_RANDOM_STATE)
        pd.testing.assert_frame_equal(first, second)
        self.assertEqual(ranks.shape, (4, len(SCAFFOLDS)))
        self.assertAlmostEqual(first["bootstrap_rank1_probability"].sum(), 1)
        self.assertTrue(np.isfinite(first.to_numpy(dtype=float)).all())

    def test_four_model_oof_runtime_and_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            scores = classify_scaffolds(self.events(), Path(directory), TEST_RANDOM_STATE)
            self.assertEqual(set(scores["model"]), set(scaffold_models(TEST_RANDOM_STATE)))
            self.assertTrue((scores["n_events"] == len(SCAFFOLDS) * 16).all())
            self.assertTrue(scores["n_errors"].between(0, len(SCAFFOLDS) * 16).all())


if __name__ == "__main__":
    unittest.main()
