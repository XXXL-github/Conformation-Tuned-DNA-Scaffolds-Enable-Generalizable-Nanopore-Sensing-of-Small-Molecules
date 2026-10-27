"""Classifiers and settings from Supplementary Table 4 (October 2026 draft)."""
from __future__ import annotations

from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import (
    AdaBoostClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier,
    RandomForestClassifier, StackingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier


def logistic_en(max_iter: int, random_state: int = 42) -> LogisticRegression:
    return LogisticRegression(
        penalty="elasticnet", solver="saga", l1_ratio=0.2, C=2.0,
        max_iter=max_iter, class_weight="balanced", random_state=random_state,
    )


def build_eight_analyte_models(random_state: int = 42) -> dict[str, object]:
    """Twelve individual classifiers and Stack, in Supplementary Table 4 order."""
    models = {
        "AdaBoost": AdaBoostClassifier(
            estimator=DecisionTreeClassifier(
                max_depth=2, min_samples_leaf=5, random_state=random_state,
            ), n_estimators=180, learning_rate=0.45, random_state=random_state,
        ),
        "Logistic-EN": make_pipeline(StandardScaler(), logistic_en(5000, random_state)),
        "kNN": make_pipeline(
            StandardScaler(), KNeighborsClassifier(n_neighbors=15, weights="distance"),
        ),
        "QDA": make_pipeline(StandardScaler(), QuadraticDiscriminantAnalysis(reg_param=0.05)),
        "SVM-RBF": make_pipeline(
            StandardScaler(), SVC(
                kernel="rbf", C=10, gamma="scale", probability=True,
                class_weight="balanced", random_state=random_state,
            ),
        ),
        "GaussianNB": GaussianNB(),
        "CatBoost": CatBoostClassifier(
            iterations=160, depth=4, learning_rate=0.055, loss_function="MultiClass",
            random_seed=random_state, thread_count=2, verbose=False, allow_writing_files=False,
        ),
        "ExtraTrees": ExtraTreesClassifier(
            n_estimators=350, min_samples_leaf=2, class_weight="balanced",
            random_state=random_state, n_jobs=2,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=350, min_samples_leaf=2, class_weight="balanced",
            random_state=random_state, n_jobs=2,
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=180, num_leaves=15, max_depth=4, learning_rate=0.055,
            subsample=0.9, colsample_bytree=0.9, random_state=random_state,
            n_jobs=2, verbose=-1,
        ),
        "HistGBDT": HistGradientBoostingClassifier(
            max_iter=180, learning_rate=0.055, max_leaf_nodes=15, random_state=random_state,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=180, max_depth=3, learning_rate=0.055, subsample=0.9,
            colsample_bytree=0.9, objective="multi:softprob", eval_metric="mlogloss",
            random_state=random_state, n_jobs=2,
        ),
    }
    models["Stack"] = build_stack_model(random_state, models)
    return models


def build_stack_model(random_state: int = 42, base_models: dict | None = None) -> StackingClassifier:
    if base_models is None:
        # The full factory supplies base_models, so this call does not recurse.
        return build_eight_analyte_models(random_state)["Stack"]
    return StackingClassifier(
        estimators=[
            ("svm", base_models["SVM-RBF"]), ("rf", base_models["Random Forest"]),
            ("xgb", base_models["XGBoost"]), ("gnb", base_models["GaussianNB"]),
        ],
        final_estimator=logistic_en(2000, random_state),
        stack_method="predict_proba", cv=3,
    )
