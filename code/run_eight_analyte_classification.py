from __future__ import annotations

import argparse
from pathlib import Path

import joblib

from contour_ml_common import (
    EIGHT_ANALYTE_CLASSES, check_counts, evaluate_independent_test,
    load_feature_table, train_development,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CONTOUR-DNA training/validation and separate independent testing.")
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train", help="Split development events 75/25; fit and validate 13 models.")
    train.add_argument("--development", type=Path, required=True, help="External 3200-event development table.")
    train.add_argument("--random-state", type=int, required=True)
    train.add_argument("--learning-curve", action="store_true", help="Export raw Stack train/validation learning curves.")
    test = commands.add_parser("test", help="Evaluate frozen models on 800 events from separate experiments.")
    test.add_argument("--test", type=Path, required=True, help="External independent-test table, not a random holdout.")
    test.add_argument("--models", type=Path, required=True, help="Local development_models.joblib from train.")
    for command in (train, test):
        command.add_argument("--sheet", default="Expanded_Data")
        command.add_argument("--output-dir", type=Path, required=True, help="New local directory; existing results are preserved.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    path = args.development if args.command == "train" else args.test
    df, X, y = load_feature_table(path, EIGHT_ANALYTE_CLASSES, args.sheet, require_provenance=True)
    if args.command == "train":
        check_counts(y, EIGHT_ANALYTE_CLASSES, 400, "Development set")
        bundle = train_development(df, X, y, args.output_dir, args.random_state, args.learning_curve)
        print(f"Training: {bundle['n_training']}; validation: {bundle['n_validation']}")
        print(f"Selected by validation: {bundle['selected_by_validation']}; manuscript final model: Stack")
    else:
        check_counts(y, EIGHT_ANALYTE_CLASSES, 100, "Independent test set")
        bundle = joblib.load(args.models)
        print(evaluate_independent_test(bundle, df, X, y, args.output_dir).to_string(index=False))


if __name__ == "__main__":
    main()
