"""Four-model scaffold screening and explicitly reconstructed bootstrap ranking.

The recovered archive predates the current figure. Its LI/LII feature equations
and stability/transition rank aggregation are retained. The four-model comparison
and event bootstrap below are an auditable reconstruction, not a claim that the
archived historical scores have been reproduced. See docs/methods_crosswalk.md.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.covariance import LedoitWolf
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from contour_ml_common import export_predictions, feature_hashes, metrics, predict_labels, read_table, write_manifest


SCAFFOLDS = ("TC11L311", "TC11L312", "TC11L321", "TC11L322", "TC11L131", "TC11L132",
             "TC11L231", "TC11L232", "TC11L113", "TC11L123", "TC11L213", "TC11L223")
RAW_FEATURES = tuple(f"{level}_{feature}" for level in ("LI", "LII")
                     for feature in ("I/I0", "ΔSD", "skew", "toff"))
MODEL_FEATURES = tuple(f"{level}_{feature}" for level in ("LI", "LII")
                       for feature in ("I/I0", "ΔSD", "skew", "log10_toff")) + (
                           "Delta_I/I0", "Normalized_Delta_I/I0", "Delta_ΔSD", "Delta_skew", "log_toff_ratio")


def load_scaffolds(path: Path, sheet: str = "Expanded_Data") -> pd.DataFrame:
    df = read_table(path, sheet)
    aliases = {"tag": "Scaffold", "event_id": "EventID"}
    for level in ("LI", "LII"):
        aliases.update({f"{level}_II0": f"{level}_I/I0", f"{level}_RMS": f"{level}_ΔSD",
                        f"{level}_Skew": f"{level}_skew"})
    for old, new in aliases.items():
        if old in df and new in df:
            raise ValueError(f"Ambiguous columns: {old} and {new}")
    df = df.rename(columns=aliases)
    missing = [column for column in ("Scaffold", "EventID", *RAW_FEATURES) if column not in df]
    if missing:
        raise ValueError(f"Missing scaffold columns: {missing}")
    if df[["Scaffold", "EventID"]].isna().any().any():
        raise ValueError("Scaffold and EventID must be present for every event.")
    df["Scaffold"] = df["Scaffold"].astype(str).str.strip()
    df["EventID"] = df["EventID"].astype(str).str.strip()
    if df["EventID"].eq("").any():
        raise ValueError("EventID cannot be blank.")
    if set(df["Scaffold"]) != set(SCAFFOLDS):
        raise ValueError("Provide all 12 reported scaffolds and no unrelated rows.")
    if df.duplicated(["Scaffold", "EventID"]).any():
        raise ValueError("Repeated scaffold/EventID pairs.")
    X = df[list(RAW_FEATURES)].to_numpy(dtype=float)
    if not np.isfinite(X).all() or len(feature_hashes(X)) != len(X):
        raise ValueError("Scaffold features must be finite and contain no repeated event vectors.")
    df[list(RAW_FEATURES)] = X
    for level in ("LI", "LII"):
        if (df[f"{level}_I/I0"] <= 0).any() or (df[f"{level}_ΔSD"] <= 0).any() or (df[f"{level}_toff"] <= 0).any():
            raise ValueError("Scaffold I/I0, ΔSD and dwell times must be positive for CV/log features.")
    if df.groupby("Scaffold").size().min() < 10:
        raise ValueError("At least ten events per scaffold are required for screening and bootstrap.")
    return construct_features(df)


def construct_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for level in ("LI", "LII"):
        df[f"{level}_log10_toff"] = np.log10(df[f"{level}_toff"])
    df["Delta_I/I0"] = df["LII_I/I0"] - df["LI_I/I0"]
    df["Normalized_Delta_I/I0"] = 2 * df["Delta_I/I0"] / (df["LII_I/I0"] + df["LI_I/I0"])
    for feature in ("ΔSD", "skew"):
        df[f"Delta_{feature}"] = df[f"LII_{feature}"] - df[f"LI_{feature}"]
    df["log_toff_ratio"] = np.log10(df["LII_toff"] / df["LI_toff"])
    return df


def scaffold_models(random_state: int = 42) -> dict:
    # SVM/RF/kNN settings recovered from the archive; ExtraTrees is the new
    # manuscript's additional model and its settings are explicit here.
    return {
        "ExtraTrees": ExtraTreesClassifier(n_estimators=400, class_weight="balanced",
                                             random_state=random_state, n_jobs=2),
        "Random Forest": RandomForestClassifier(n_estimators=400, class_weight="balanced_subsample",
                                                 random_state=random_state, n_jobs=2),
        "SVM-RBF": make_pipeline(StandardScaler(), SVC(kernel="rbf", C=5.0, gamma="scale",
                                                        probability=True, random_state=random_state)),
        "kNN": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=9, weights="distance")),
    }


def classify_scaffolds(df: pd.DataFrame, output: Path, random_state: int = 42) -> pd.DataFrame:
    X = df[list(MODEL_FEATURES)].to_numpy(dtype=float)
    y = df["Scaffold"].map({tag: index for index, tag in enumerate(SCAFFOLDS)}).to_numpy(dtype=int)
    folds = list(StratifiedKFold(5, shuffle=True, random_state=random_state).split(X, y))
    results, all_proba, importance, assignment = [], [], [], np.empty(len(y), dtype=int)
    for name, definition in scaffold_models(random_state).items():
        proba = np.empty((len(y), len(SCAFFOLDS)))
        pred = np.empty_like(y)
        for fold, (train, validation) in enumerate(folds, 1):
            fitted = clone(definition).fit(X[train], y[train])
            proba[validation] = fitted.predict_proba(X[validation])
            pred[validation] = predict_labels(fitted, X[validation])
            assignment[validation] = fold
            if name == "ExtraTrees":
                importance.append(fitted.feature_importances_)
        results.append({"model": name, "partition": "5-fold OOF validation", **metrics(y, pred, proba)})
        export_predictions(output, name.replace(" ", "_"), y, pred, proba, SCAFFOLDS)
        all_proba.append(proba)
    consensus = np.mean(all_proba, axis=0)
    export_predictions(output, "four_model_consensus", y, consensus.argmax(axis=1), consensus, SCAFFOLDS)
    pd.DataFrame({"feature": MODEL_FEATURES, "importance": np.mean(importance, axis=0)}).to_csv(
        output / "extratrees_feature_importance.csv", index=False)
    pd.DataFrame({"source_row": np.arange(len(y)), "validation_fold": assignment}).to_csv(
        output / "scaffold_validation_folds.csv", index=False)
    scores = pd.DataFrame(results)
    scores.to_csv(output / "scaffold_validation_metrics.csv", index=False)
    return scores


def rank_consensus(table: pd.DataFrame, directions: dict[str, bool]) -> pd.Series:
    """Recovered mean-rank/Borda/geometric-rank fusion; smaller is better."""
    ranks = pd.DataFrame({name: table[name].rank(ascending=not higher, method="average")
                          for name, higher in directions.items()})
    average, product = ranks.mean(axis=1), np.exp(np.log(ranks).mean(axis=1))
    borda = (len(ranks) + 1 - ranks).sum(axis=1)
    return (average.rank() + product.rank() + borda.rank(ascending=False)) / 3


def score_dimension(consensus: pd.Series) -> pd.Series:
    if consensus.max() == consensus.min():
        return pd.Series(0.5, index=consensus.index)
    return (consensus.max() - consensus) / (consensus.max() - consensus.min())


def mechanism_ranking(df: pd.DataFrame, scaler: StandardScaler) -> pd.DataFrame:
    """Reconstructed four-dimension score from archived physical descriptors.

    Dimensions: LI stability, LII stability, transition consistency, compactness.
    No classifier accuracy or manuscript result is used to force the ranking.
    """
    records = []
    steps = ("Delta_I/I0", "Normalized_Delta_I/I0", "Delta_ΔSD", "Delta_skew", "log_toff_ratio")
    for tag in SCAFFOLDS:
        group = df[df["Scaffold"] == tag]
        row = {"Scaffold": tag}
        for level in ("LI", "LII"):
            for feature in ("I/I0", "ΔSD"):
                values = group[f"{level}_{feature}"]
                row[f"{level}_{feature}_cv"] = values.std(ddof=1) / values.mean()
            row[f"{level}_skew_sd"] = group[f"{level}_skew"].std(ddof=1)
            logs = group[f"{level}_log10_toff"]
            row[f"{level}_log_toff_sd"] = logs.std(ddof=1)
            row[f"{level}_log_toff_iqr"] = logs.quantile(.75) - logs.quantile(.25)
        for feature in steps:
            values = group[feature]
            row[f"{feature}_sd"] = values.std(ddof=1)
            numerator = abs(values.median()) if feature == "log_toff_ratio" else abs(values.mean())
            denominator = values.quantile(.75) - values.quantile(.25) if feature == "log_toff_ratio" else values.std(ddof=1)
            row[f"{feature}_snr"] = numerator / max(denominator, np.finfo(float).eps)
        cov = LedoitWolf().fit(scaler.transform(group[list(MODEL_FEATURES)].to_numpy())).covariance_
        sign, logdet = np.linalg.slogdet(cov)
        if sign <= 0:
            raise ValueError("Degenerate scaffold covariance; inspect the input events.")
        row["covariance_logdet"] = logdet
        records.append(row)
    result = pd.DataFrame(records).set_index("Scaffold")
    dimensions = {}
    for level in ("LI", "LII"):
        names = [f"{level}_I/I0_cv", f"{level}_ΔSD_cv", f"{level}_skew_sd",
                 f"{level}_log_toff_sd", f"{level}_log_toff_iqr"]
        dimensions[f"{level}_stability"] = score_dimension(rank_consensus(result, {name: False for name in names}))
    directions = {f"{feature}_{stat}": stat == "snr" for feature in steps for stat in ("snr", "sd")}
    dimensions["transition_consistency"] = score_dimension(rank_consensus(result, directions))
    dimensions["compactness"] = score_dimension(result["covariance_logdet"].rank())
    dimension_table = pd.DataFrame(dimensions)
    dimension_table["consensus_rank"] = rank_consensus(dimension_table, {name: True for name in dimensions})
    return result.join(dimension_table)


def bootstrap_ranking(df: pd.DataFrame, n_bootstrap: int = 1000, random_state: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    if n_bootstrap < 1:
        raise ValueError("n_bootstrap must be positive.")
    rng = np.random.default_rng(random_state)
    scaler = StandardScaler().fit(df[list(MODEL_FEATURES)].to_numpy())
    point = mechanism_ranking(df, scaler)
    groups = [df[df["Scaffold"] == tag] for tag in SCAFFOLDS]
    ranks, top_one = [], np.zeros(len(SCAFFOLDS))
    for _ in range(n_bootstrap):
        sampled = pd.concat([group.iloc[rng.integers(0, len(group), size=len(group))] for group in groups], ignore_index=True)
        rank = mechanism_ranking(sampled, scaler)["consensus_rank"].to_numpy()
        ranks.append(rank)
        winners = np.isclose(rank, rank.min())
        # Divide tied first-place credit equally; never favor the first scaffold.
        top_one += winners / winners.sum()
    rank_matrix = np.asarray(ranks)
    summary = point.copy()
    summary["bootstrap_rank1_probability"] = top_one / n_bootstrap
    summary["bootstrap_rank_median"] = np.median(rank_matrix, axis=0)
    summary["bootstrap_rank_q025"] = np.quantile(rank_matrix, .025, axis=0)
    summary["bootstrap_rank_q975"] = np.quantile(rank_matrix, .975, axis=0)
    return summary, pd.DataFrame(rank_matrix, columns=SCAFFOLDS)


def main() -> None:
    parser = argparse.ArgumentParser(description="12-scaffold screening with the four reported classifiers and reconstructed bootstrap ranking.")
    parser.add_argument("--input", type=Path, required=True, help="External flat LI/LII event table.")
    parser.add_argument("--sheet", default="Expanded_Data")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--bootstrap", type=int, default=1000)
    args = parser.parse_args()
    df = load_scaffolds(args.input, args.sheet)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    print(classify_scaffolds(df, args.output_dir, args.random_state).to_string(index=False))
    summary, ranks = bootstrap_ranking(df, args.bootstrap, args.random_state)
    summary.to_csv(args.output_dir / "scaffold_mechanism_ranking.csv")
    ranks.to_csv(args.output_dir / "scaffold_bootstrap_ranks.csv", index=False)
    write_manifest(args.output_dir, {"analysis": "12-scaffold screening", "evaluation": "5-fold OOF validation",
                                    "independent_test": False, "ranking_status": "reconstructed, historical results not verified",
                                    "bootstrap_unit": "event within scaffold", "n_bootstrap": args.bootstrap,
                                    "models": list(scaffold_models()), "features": MODEL_FEATURES,
                                    "n_events": len(df), "random_state": args.random_state})


if __name__ == "__main__":
    main()
