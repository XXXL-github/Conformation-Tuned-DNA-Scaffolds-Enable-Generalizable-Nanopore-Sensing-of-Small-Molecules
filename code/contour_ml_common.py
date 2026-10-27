"""External feature input, separated development/testing, and local result export."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, log_loss, precision_score,
    recall_score, roc_curve, auc,
)
from sklearn.model_selection import StratifiedKFold, train_test_split

from contour_models import build_eight_analyte_models, build_stack_model


FEATURES = ("I/I0", "ΔSD", "skew", "median I/I0", "toff", "kurt", "IQR I/I0", "CV")
LABEL = "Analyte"
EIGHT_ANALYTE_CLASSES = ("Mal", "IsoMal", "Mal3", "IsoMal3", "3-DG", "5-AMF", "5-EMF", "5-HMF")
ESTROGEN_CLASSES = ("NIL", "EE2", "MES")
ALIASES = {
    LABEL: ("English abbreviation",), "skew": ("Skew",),
    "median I/I0": ("Median I/I0",), "toff": ("Dwell time (ms)", "Dwell Time (ms)"),
    "kurt": ("Kurt", "Kurtosis"),
}
LEGACY_LABELS = {"DP2": "Mal", "IDP2": "IsoMal", "DP3": "Mal3", "IDP3": "IsoMal3"}


def read_table(path: Path, sheet: str = "Expanded_Data") -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() == ".xlsx":
        return pd.read_excel(path, sheet_name=sheet)
    raise ValueError("Use an external .csv or .xlsx feature table.")


def normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for name, aliases in ALIASES.items():
        found = [column for column in (name, *aliases) if column in df.columns]
        if len(found) > 1:
            raise ValueError(f"Ambiguous feature names for {name}: {found}")
        if found:
            df = df.rename(columns={found[0]: name})
    return df


def load_feature_table(path: Path, classes: tuple, sheet: str = "Expanded_Data",
                       require_provenance: bool = False) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    df = normalise_columns(read_table(path, sheet))
    required = [LABEL, *FEATURES]
    if require_provenance:
        required += ["EventID", "ExperimentID"]
    missing = [column for column in required if column not in df]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    if df[LABEL].isna().any():
        raise ValueError("Analyte labels cannot be missing.")
    df[LABEL] = df[LABEL].astype(str).str.strip().replace(LEGACY_LABELS)
    if set(df[LABEL]) != set(classes):
        raise ValueError(f"Expected exactly these analytes, with no silently discarded rows: {classes}")
    X = df[list(FEATURES)].to_numpy(dtype=float)
    if not np.isfinite(X).all():
        raise ValueError("All eight features must be finite numeric values.")
    if require_provenance:
        for column in ("EventID", "ExperimentID"):
            if df[column].isna().any() or df[column].astype(str).str.strip().eq("").any():
                raise ValueError(f"{column} cannot be missing or blank.")
            df[column] = df[column].astype(str).str.strip()
        if df.duplicated(["ExperimentID", "EventID"]).any():
            raise ValueError("Repeated ExperimentID/EventID pairs; identify each physical event uniquely.")
    y = df[LABEL].map({label: index for index, label in enumerate(classes)}).to_numpy(dtype=int)
    return df.reset_index(drop=True), X, y


def check_counts(y: np.ndarray, classes: tuple, per_class: int, description: str) -> None:
    counts = np.bincount(y, minlength=len(classes))
    if not np.all(counts == per_class):
        raise ValueError(f"{description} requires {per_class} events per analyte; got {dict(zip(classes, counts.tolist()))}")


def development_split(y: np.ndarray, random_state: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """The same stratified 75/25 development split is reused for all models."""
    return train_test_split(np.arange(len(y)), test_size=0.25, stratify=y, random_state=random_state)


def feature_hashes(X: np.ndarray) -> set[str]:
    # Use numeric values only, so relabelling a reused event does not evade this check.
    return {hashlib.sha256(np.asarray(row, dtype="<f8").tobytes()).hexdigest() for row in X}


def check_test_independence(bundle: dict, df: pd.DataFrame, X: np.ndarray) -> None:
    if set(df["ExperimentID"]) & set(bundle["development_experiments"]):
        raise ValueError("Independent test experiments must be disjoint from development experiments.")
    if feature_hashes(X) & set(bundle["development_feature_hashes"]):
        raise ValueError("Exact feature vectors overlap development/test; verify event provenance.")


def predict_labels(model: object, X: np.ndarray) -> np.ndarray:
    # CatBoost returns (n, 1); sklearn estimators normally return (n,).
    return np.asarray(model.predict(X)).reshape(-1)


def metrics(y: np.ndarray, pred: np.ndarray, proba: np.ndarray) -> dict:
    pred = np.asarray(pred).reshape(-1)
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "macro_precision": float(precision_score(y, pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y, pred, average="macro", zero_division=0)),
        "log_loss": float(log_loss(y, proba, labels=np.arange(proba.shape[1]))),
        "n_events": len(y), "n_errors": int(np.sum(y != pred)),
    }


def write_manifest(output: Path, payload: dict) -> None:
    (output / "run_manifest.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def runtime_versions() -> dict:
    return {name: importlib.metadata.version(name) for name in (
        "numpy", "pandas", "scikit-learn", "xgboost", "lightgbm", "catboost", "joblib", "scipy", "threadpoolctl",
    )}


def export_predictions(output: Path, name: str, y: np.ndarray, pred: np.ndarray,
                       proba: np.ndarray, classes: tuple, row_ids: np.ndarray | None = None) -> None:
    table = pd.DataFrame({
        "source_row": np.arange(len(y)) if row_ids is None else row_ids,
        "true_class": [classes[int(i)] for i in y],
        "predicted_class": [classes[int(i)] for i in pred],
    })
    for index, label in enumerate(classes):
        table[f"prob_{label}"] = proba[:, index]
    table.to_csv(output / f"{name}_predictions.csv", index=False)
    cm = confusion_matrix(y, pred, labels=np.arange(len(classes)))
    pd.DataFrame(cm, index=classes, columns=classes).to_csv(output / f"{name}_confusion_counts.csv")
    pd.DataFrame(cm / cm.sum(axis=1, keepdims=True) * 100, index=classes, columns=classes).to_csv(
        output / f"{name}_confusion_percent.csv",
    )


def train_development(df: pd.DataFrame, X: np.ndarray, y: np.ndarray, output: Path,
                      random_state: int = 42, learning_curve: bool = False) -> dict:
    train, validation = development_split(y, random_state)
    if feature_hashes(X[train]) & feature_hashes(X[validation]):
        raise ValueError("Exact feature vectors overlap training/validation; verify event provenance.")
    models = build_eight_analyte_models(random_state)
    output.mkdir(parents=True, exist_ok=False)
    fitted_models, rows = {}, []
    for name, definition in models.items():
        print(f"Training and validating {name}", flush=True)
        fitted = clone(definition).fit(X[train], y[train])
        pred, proba = predict_labels(fitted, X[validation]), fitted.predict_proba(X[validation])
        rows.append({"model": name, "partition": "validation", **metrics(y[validation], pred, proba)})
        fitted_models[name] = fitted
        export_predictions(output, f"validation_{name.replace(' ', '_')}", y[validation], pred,
                           proba, EIGHT_ANALYTE_CLASSES, validation)
    scores = pd.DataFrame(rows)
    scores.to_csv(output / "validation_metrics.csv", index=False)
    best = scores.sort_values(["macro_f1", "accuracy"], ascending=False, kind="stable").iloc[0]["model"]
    split = df[["ExperimentID", "EventID", LABEL]].copy()
    split.insert(0, "source_row", np.arange(len(df)))
    split["partition"] = "training"
    split.loc[validation, "partition"] = "validation"
    split.to_csv(output / "development_split.csv", index=False)
    bundle = {
        "schema_version": 2, "analysis": "eight-analyte", "features": FEATURES,
        "classes": EIGHT_ANALYTE_CLASSES, "models": fitted_models,
        "selected_by_validation": best, "manuscript_final_model": "Stack",
        "development_experiments": sorted(set(df["ExperimentID"])),
        "development_feature_hashes": sorted(feature_hashes(X)),
        "n_training": len(train), "n_validation": len(validation),
        "random_state": random_state, "versions": runtime_versions(),
    }
    # Retain the models fitted only to the training subset. No refit on validation/test.
    joblib.dump(bundle, output / "development_models.joblib")
    write_manifest(output, {key: value for key, value in bundle.items() if key not in (
        "models", "development_feature_hashes",
    )})
    if learning_curve:
        export_learning_curve(X[train], y[train], X[validation], y[validation], output, random_state)
    return bundle


def export_learning_curve(X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray,
                          y_val: np.ndarray, output: Path, random_state: int) -> None:
    rows = []
    for fraction in (0.2, 0.4, 0.6, 0.8, 1.0):
        indices = np.arange(len(y_train))
        if fraction < 1:
            indices, _ = train_test_split(indices, train_size=fraction, stratify=y_train,
                                           random_state=random_state)
        model = build_stack_model(random_state).fit(X_train[indices], y_train[indices])
        rows.append({"n_training": len(indices),
                     "training_accuracy": float(accuracy_score(y_train[indices], model.predict(X_train[indices]))),
                     "validation_accuracy": float(accuracy_score(y_val, model.predict(X_val)))})
    pd.DataFrame(rows).to_csv(output / "learning_curve.csv", index=False)


def evaluate_independent_test(bundle: dict, df: pd.DataFrame, X: np.ndarray, y: np.ndarray,
                              output: Path) -> pd.DataFrame:
    if bundle.get("schema_version") != 2 or tuple(bundle["classes"]) != EIGHT_ANALYTE_CLASSES:
        raise ValueError("Use the eight-analyte bundle created by this version of the training command.")
    if bundle["versions"] != runtime_versions():
        raise ValueError("Training/testing library versions differ; use the same pinned environment.")
    check_test_independence(bundle, df, X)
    output.mkdir(parents=True, exist_ok=False)
    rows, roc_rows = [], []
    for name, fitted in bundle["models"].items():
        # Deliberately no fit, calibration, ranking, or model choice on the test set.
        pred, proba = predict_labels(fitted, X), fitted.predict_proba(X)
        rows.append({"model": name, "partition": "independent_test", **metrics(y, pred, proba)})
        export_predictions(output, f"test_{name.replace(' ', '_')}", y, pred, proba, EIGHT_ANALYTE_CLASSES)
        for index, label in enumerate(EIGHT_ANALYTE_CLASSES):
            fpr, tpr, threshold = roc_curve(y == index, proba[:, index])
            roc_rows.extend({"model": name, "class": label, "fpr": float(a), "tpr": float(b),
                             "threshold": float(c), "auc": float(auc(fpr, tpr))}
                            for a, b, c in zip(fpr, tpr, threshold))
        if name == "Stack":
            superclass = (y >= 4).astype(int)
            superclass_pred = (pred >= 4).astype(int)
            pd.DataFrame(confusion_matrix(superclass, superclass_pred, labels=[0, 1]),
                         index=["Glycans", "Aldehydes"], columns=["Glycans", "Aldehydes"]).to_csv(
                             output / "test_Stack_superclass_confusion_counts.csv")
    scores = pd.DataFrame(rows)
    scores.to_csv(output / "independent_test_metrics.csv", index=False)
    pd.DataFrame(roc_rows).to_csv(output / "independent_test_roc.csv", index=False)
    write_manifest(output, {
        "partition": "independent_test", "n_events": len(y), "n_training": bundle["n_training"],
        "n_validation": bundle["n_validation"], "models": list(bundle["models"]),
        "selected_by_validation": bundle["selected_by_validation"],
        "manuscript_final_model": bundle["manuscript_final_model"],
        "test_experiments": sorted(set(df["ExperimentID"])),
        "random_state": bundle["random_state"], "versions": runtime_versions(),
    })
    return scores


def apply_confidence_thresholds(proba: np.ndarray, classes: tuple, thresholds: dict) -> np.ndarray:
    if set(thresholds) != set(classes) or any(not 0 <= float(v) <= 1 for v in thresholds.values()):
        raise ValueError("Provide one validated probability threshold in [0, 1] for every class.")
    best = np.argmax(proba, axis=1)
    return np.array([classes[i] if proba[row, i] >= float(thresholds[classes[i]]) else "unassigned"
                     for row, i in enumerate(best)])


def evaluate_stack_oof(X: np.ndarray, y: np.ndarray, output: Path,
                        random_state: int = 42) -> pd.DataFrame:
    """Preserve the archived estrogen 5-fold validation; it is not independent testing."""
    if np.bincount(y, minlength=3).min() < 5:
        raise ValueError("Estrogen 5-fold validation requires at least five events per class.")
    if len(feature_hashes(X)) != len(X):
        raise ValueError("Repeated feature vectors could cross validation folds; verify provenance.")
    outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    pred, proba, assignment = np.empty_like(y), np.empty((len(y), 3)), np.empty(len(y), dtype=int)
    folds = []
    for fold, (train, validation) in enumerate(outer.split(X, y), start=1):
        model = build_stack_model(random_state).fit(X[train], y[train])
        pred[validation], proba[validation], assignment[validation] = (
            predict_labels(model, X[validation]), model.predict_proba(X[validation]), fold,
        )
        folds.append({"fold": fold, "n_training": len(train), "n_validation": len(validation)})
    output.mkdir(parents=True, exist_ok=False)
    scores = pd.DataFrame([{"model": "Stack", "partition": "5-fold OOF validation", **metrics(y, pred, proba)}])
    scores.to_csv(output / "stack_validation_metrics.csv", index=False)
    export_predictions(output, "stack_oof", y, pred, proba, ESTROGEN_CLASSES)
    pd.DataFrame({"source_row": np.arange(len(y)), "validation_fold": assignment}).to_csv(
        output / "validation_folds.csv", index=False)
    write_manifest(output, {"analysis": "estrogen", "evaluation": "5-fold OOF validation",
                            "independent_test": False, "classes": ESTROGEN_CLASSES, "features": FEATURES,
                            "folds": folds, "random_state": random_state, "versions": runtime_versions()})
    return scores
