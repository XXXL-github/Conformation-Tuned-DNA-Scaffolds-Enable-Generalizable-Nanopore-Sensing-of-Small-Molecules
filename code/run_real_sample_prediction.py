"""Apply the frozen Stack and externally supplied validation-derived thresholds."""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from contour_ml_common import (
    FEATURES, apply_confidence_thresholds, normalise_columns, read_table,
    runtime_versions, write_manifest,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Classify real-sample events; retain low-confidence events as unassigned.")
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True, help="External unlabeled event features.")
    parser.add_argument("--thresholds", type=Path, required=True,
                        help="External CSV with Analyte,threshold; established using validation, never independent test.")
    parser.add_argument("--sheet", default="Expanded_Data")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    bundle = joblib.load(args.models)
    if bundle.get("schema_version") != 2 or bundle["versions"] != runtime_versions():
        raise ValueError("Use a model bundle and the pinned environment from this release.")
    df = normalise_columns(read_table(args.input, args.sheet))
    X = df[list(FEATURES)].to_numpy(dtype=float)
    if not np.isfinite(X).all():
        raise ValueError("All input features must be finite.")
    threshold_df = pd.read_csv(args.thresholds)
    if threshold_df["Analyte"].duplicated().any():
        raise ValueError("Threshold labels must be unique.")
    thresholds = dict(zip(threshold_df["Analyte"], threshold_df["threshold"]))
    model = bundle["models"]["Stack"]
    proba = model.predict_proba(X)
    assigned = apply_confidence_thresholds(proba, bundle["classes"], thresholds)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    result = pd.DataFrame({"source_row": np.arange(len(X)), "assigned_class": assigned,
                           "max_probability": proba.max(axis=1)})
    for column in ("ExperimentID", "EventID"):
        if column in df:
            result[column] = df[column].to_numpy()
    result.to_csv(args.output_dir / "real_sample_assignments.csv", index=False)
    # Unassigned events remain in the event table but are not target-specific counts.
    pd.Series(assigned[assigned != "unassigned"]).value_counts().reindex(
        bundle["classes"], fill_value=0).rename("assigned_events").to_csv(
            args.output_dir / "target_event_counts.csv")
    write_manifest(args.output_dir, {"model": "Stack", "n_events": len(X),
                                    "n_unassigned": int(np.sum(assigned == "unassigned")),
                                    "thresholds": thresholds, "versions": runtime_versions()})


if __name__ == "__main__":
    main()
