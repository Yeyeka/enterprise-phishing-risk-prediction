from pathlib import Path
import hashlib
import json
import os
import sys
import tempfile
import time

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "phishing_ml_matplotlib_cache"),
)

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

try:
    import xgboost
    from xgboost import XGBClassifier
except ImportError:
    print("Error: xgboost is not installed in the current Python environment.", file=sys.stderr)
    raise

from data_preprocessing import (
    RANDOM_SEED,
    build_tree_preprocessor,
    get_feature_sets,
    load_dataset,
    split_dataset,
    validate_dataset,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "enterprise_phishing_simulation_2026.csv"
METRICS_DIR = PROJECT_ROOT / "results" / "metrics"
FIGURES_DIR = PROJECT_ROOT / "results" / "figures"
MODELS_DIR = PROJECT_ROOT / "models"
MODEL_NAME = "XGBoost Tuned"
SELECTED_FEATURE_SET = "reduced"
EXCLUDED_FEATURE = "cybersecurity_awareness_score"
CLASS_LABELS = ["No", "Yes"]


PARAM_DISTRIBUTIONS = {
    "model__n_estimators": [150, 200, 300, 500, 700],
    "model__learning_rate": [0.02, 0.05, 0.08, 0.1, 0.15],
    "model__max_depth": [3, 4, 5, 6, 8],
    "model__min_child_weight": [1, 3, 5, 7],
    "model__gamma": [0.0, 0.1, 0.3, 0.5],
    "model__subsample": [0.7, 0.85, 1.0],
    "model__colsample_bytree": [0.7, 0.85, 1.0],
    "model__reg_alpha": [0.0, 0.01, 0.1, 0.5],
    "model__reg_lambda": [1.0, 2.0, 5.0, 10.0],
    "model__scale_pos_weight": [1.0, 1.3, 1.68],
}


PARAMETER_DESCRIPTIONS = {
    "n_estimators": "number of boosting trees",
    "learning_rate": "learning step size for each tree",
    "max_depth": "maximum depth of each tree",
    "min_child_weight": "minimum child weight required for further splitting",
    "gamma": "minimum loss reduction required for a split",
    "subsample": "training sample fraction used by each tree",
    "colsample_bytree": "feature fraction used by each tree",
    "reg_alpha": "L1 regularization strength",
    "reg_lambda": "L2 regularization strength",
    "scale_pos_weight": "positive-class sample weight",
}


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def ensure_directories():
    for path in [METRICS_DIR, FIGURES_DIR, MODELS_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def require_baseline_files():
    required_paths = [
        PROJECT_ROOT / "src" / "train_xgboost.py",
        METRICS_DIR / "xgboost_cv_summary.csv",
        METRICS_DIR / "xgboost_test_metrics.csv",
    ]
    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing required XGBoost baseline files. Stop tuning. Missing: "
            + "; ".join(missing)
        )


def split_features_by_type(df, features):
    numeric_features = [
        feature for feature in features if pd.api.types.is_numeric_dtype(df[feature])
    ]
    categorical_features = [
        feature for feature in features if feature not in numeric_features
    ]
    return numeric_features, categorical_features


def build_sparse_tree_preprocessor(numeric_features, categorical_features):
    preprocessor = build_tree_preprocessor(numeric_features, categorical_features)
    if hasattr(preprocessor, "sparse_threshold"):
        preprocessor.sparse_threshold = 1.0
    return preprocessor


def build_pipeline(numeric_features, categorical_features):
    return Pipeline(
        [
            (
                "preprocessor",
                build_sparse_tree_preprocessor(numeric_features, categorical_features),
            ),
            (
                "model",
                XGBClassifier(
                    objective="binary:logistic",
                    eval_metric="logloss",
                    tree_method="hist",
                    importance_type="gain",
                    random_state=RANDOM_SEED,
                    n_jobs=-1,
                    verbosity=0,
                ),
            ),
        ]
    )


def read_and_print_baseline_cv():
    baseline_cv = pd.read_csv(METRICS_DIR / "xgboost_cv_summary.csv")
    required_feature_sets = {"full", "reduced"}
    found_feature_sets = set(baseline_cv["feature_set"])
    if not required_feature_sets.issubset(found_feature_sets):
        raise ValueError("xgboost_cv_summary.csv must contain full and reduced rows.")
    print("XGBoost baseline training CV summary:")
    print(
        baseline_cv[
            [
                "feature_set",
                "raw_feature_count",
                "transformed_feature_count",
                "accuracy_mean",
                "precision_mean",
                "recall_mean",
                "f1_mean",
                "roc_auc_mean",
            ]
        ].to_string(index=False)
    )
    full = baseline_cv[baseline_cv["feature_set"] == "full"].iloc[0]
    reduced = baseline_cv[baseline_cv["feature_set"] == "reduced"].iloc[0]
    print(
        "Fixed feature set: reduced. Baseline CV basis: "
        f"reduced F1={reduced['f1_mean']:.6f} vs full F1={full['f1_mean']:.6f}; "
        f"reduced ROC-AUC={reduced['roc_auc_mean']:.6f} vs full ROC-AUC={full['roc_auc_mean']:.6f}; "
        "reduced uses fewer raw features."
    )
    return baseline_cv


def to_python(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: to_python(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_python(item) for item in value]
    return value


def prepare_cv_results(search):
    cv_results = pd.DataFrame(search.cv_results_)
    keep_columns = [
        "params",
        "param_model__n_estimators",
        "param_model__learning_rate",
        "param_model__max_depth",
        "param_model__min_child_weight",
        "param_model__gamma",
        "param_model__subsample",
        "param_model__colsample_bytree",
        "param_model__reg_alpha",
        "param_model__reg_lambda",
        "param_model__scale_pos_weight",
        "mean_test_accuracy",
        "std_test_accuracy",
        "mean_test_precision",
        "std_test_precision",
        "mean_test_recall",
        "std_test_recall",
        "mean_test_f1",
        "std_test_f1",
        "mean_test_roc_auc",
        "std_test_roc_auc",
        "mean_train_f1",
        "std_train_f1",
        "mean_fit_time",
        "std_fit_time",
        "mean_score_time",
        "rank_test_f1",
    ]
    missing = [column for column in keep_columns if column not in cv_results.columns]
    if missing:
        raise ValueError(f"Missing expected RandomizedSearchCV result columns: {missing}")
    cv_results = cv_results[keep_columns].copy()
    cv_results["params"] = cv_results["params"].map(
        lambda params: json.dumps(to_python(params), sort_keys=True)
    )
    return cv_results.sort_values("rank_test_f1").reset_index(drop=True)


def best_row_from_cv_results(cv_results):
    return cv_results.sort_values(["rank_test_f1", "mean_test_f1"], ascending=[True, False]).iloc[0]


def get_transformed_feature_count(estimator):
    return len(estimator.named_steps["preprocessor"].get_feature_names_out())


def evaluate_on_test(estimator, split):
    y_pred = estimator.predict(split["X_test"])
    y_probability = estimator.predict_proba(split["X_test"])[:, 1]
    cm = confusion_matrix(split["y_test"], y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    metrics = {
        "model": MODEL_NAME,
        "selected_feature_set": SELECTED_FEATURE_SET,
        "accuracy": accuracy_score(split["y_test"], y_pred),
        "precision": precision_score(split["y_test"], y_pred, pos_label=1, zero_division=0),
        "recall": recall_score(split["y_test"], y_pred, pos_label=1, zero_division=0),
        "f1": f1_score(split["y_test"], y_pred, pos_label=1, zero_division=0),
        "roc_auc": roc_auc_score(split["y_test"], y_probability),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }
    report = classification_report(
        split["y_test"],
        y_pred,
        labels=[0, 1],
        target_names=CLASS_LABELS,
        digits=4,
        zero_division=0,
    )
    predictions = pd.DataFrame(
        {
            "sample_index": split["test_index"],
            "y_true": split["y_test"].to_numpy(),
            "y_pred": y_pred,
            "y_probability": y_probability,
        }
    )
    return metrics, report, predictions, cm, y_probability


def make_feature_mapping(preprocessor, numeric_features, categorical_features):
    transformed_features = list(preprocessor.get_feature_names_out())
    mapping = {}

    for feature in numeric_features:
        transformed_name = f"numeric__{feature}"
        mapping[transformed_name] = feature

    if categorical_features:
        categorical_pipeline = preprocessor.named_transformers_["categorical"]
        encoder = categorical_pipeline.named_steps["onehot"]
        for original_feature, categories in zip(categorical_features, encoder.categories_):
            for category in categories:
                transformed_name = f"categorical__{original_feature}_{category}"
                mapping[transformed_name] = original_feature

    missing = sorted(set(transformed_features) - set(mapping))
    extra = sorted(set(mapping) - set(transformed_features))
    if missing or extra:
        raise ValueError(
            "Transformed-to-original feature mapping is incomplete. "
            f"Missing={missing}; extra={extra}"
        )
    return mapping


def make_feature_importance(estimator, numeric_features, categorical_features):
    preprocessor = estimator.named_steps["preprocessor"]
    model = estimator.named_steps["model"]
    transformed_features = list(preprocessor.get_feature_names_out())
    raw_importance = np.asarray(model.feature_importances_, dtype=float)
    if len(raw_importance) != len(transformed_features):
        raise ValueError("XGBoost feature importance length does not match transformed features.")
    total = raw_importance.sum()
    if total > 0:
        normalized = raw_importance / total
    else:
        normalized = raw_importance

    mapping = make_feature_mapping(preprocessor, numeric_features, categorical_features)
    transformed_df = pd.DataFrame(
        {
            "transformed_feature": transformed_features,
            "original_feature": [mapping[name] for name in transformed_features],
            "gain_importance": normalized,
        }
    )
    transformed_df["gain_importance_percent"] = transformed_df["gain_importance"] * 100
    transformed_df = transformed_df.sort_values("gain_importance", ascending=False).reset_index(drop=True)
    transformed_df.insert(0, "rank", range(1, len(transformed_df) + 1))

    original_df = (
        transformed_df.groupby("original_feature", as_index=False)["gain_importance"]
        .sum()
        .sort_values("gain_importance", ascending=False)
        .reset_index(drop=True)
    )
    original_df["gain_importance_percent"] = original_df["gain_importance"] * 100
    original_df.insert(0, "rank", range(1, len(original_df) + 1))

    if not np.isclose(transformed_df["gain_importance"].sum(), 1.0, atol=1e-8):
        raise ValueError("Transformed feature gain importance does not sum to 1.")
    if not np.isclose(original_df["gain_importance"].sum(), 1.0, atol=1e-8):
        raise ValueError("Original feature gain importance does not sum to 1.")
    return transformed_df, original_df


def make_baseline_comparison(tuned_metrics):
    baseline = pd.read_csv(METRICS_DIR / "xgboost_test_metrics.csv").iloc[0]
    rows = []
    for metric in ["accuracy", "precision", "recall", "f1", "roc_auc", "tn", "fp", "fn", "tp"]:
        baseline_value = float(baseline[metric])
        tuned_value = float(tuned_metrics[metric])
        absolute_change = tuned_value - baseline_value
        if metric in {"tn", "fp", "fn", "tp"}:
            relative_change = "not_applicable"
        elif baseline_value != 0:
            relative_change = absolute_change / baseline_value * 100
        else:
            relative_change = "not_applicable"
        rows.append(
            {
                "metric": metric,
                "baseline": baseline_value,
                "tuned": tuned_value,
                "absolute_change": absolute_change,
                "relative_change_percent": relative_change,
            }
        )
    return pd.DataFrame(rows)


def plot_tuning_top_results(cv_results, output_path):
    top = cv_results.head(10).sort_values("mean_test_f1", ascending=True).copy()
    labels = [f"Rank {rank}" for rank in top["rank_test_f1"]]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(labels, top["mean_test_f1"], xerr=top["std_test_f1"], color="#4C78A8")
    ax.set_title("XGBoost Top 10 Tuning Results by CV F1")
    ax.set_xlabel("Mean CV F1")
    ax.set_ylabel("Parameter Set")
    ax.set_xlim(max(0, top["mean_test_f1"].min() - 0.03), min(1, top["mean_test_f1"].max() + 0.03))
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrix(cm, output_path):
    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(cm, cmap="Blues")
    ax.set_title("Tuned XGBoost Test Confusion Matrix")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(CLASS_LABELS)
    ax.set_yticklabels(CLASS_LABELS)
    labels = [["TN", "FP"], ["FN", "TP"]]
    threshold = cm.max() / 2
    for row_index in range(2):
        for column_index in range(2):
            color = "white" if cm[row_index, column_index] > threshold else "black"
            ax.text(
                column_index,
                row_index,
                f"{labels[row_index][column_index]}\n{cm[row_index, column_index]:,}",
                ha="center",
                va="center",
                color=color,
                fontsize=12,
            )
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_roc_curve(y_test, y_probability, auc_value, output_path):
    fpr, tpr, _ = roc_curve(y_test, y_probability)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, label=f"Tuned XGBoost (AUC = {auc_value:.4f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random Classifier")
    ax.set_title("Tuned XGBoost Test ROC Curve")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_feature_importance(original_importance, output_path):
    top = original_importance.head(15).sort_values("gain_importance", ascending=True)
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(top["original_feature"], top["gain_importance"], color="#59A14F")
    ax.set_title("Tuned XGBoost Top Original Feature Gain Importance")
    ax.set_xlabel("Gain Importance")
    ax.set_ylabel("Original Feature")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_baseline_vs_tuned(comparison, output_path):
    metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    subset = comparison[comparison["metric"].isin(metrics)].copy()
    x = np.arange(len(metrics))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width / 2, subset["baseline"], width, label="Baseline", color="#9C755F")
    ax.bar(x + width / 2, subset["tuned"], width, label="Tuned", color="#4C78A8")
    ax.set_title("XGBoost Baseline vs Tuned Test Metrics")
    ax.set_xlabel("Metric")
    ax.set_ylabel("Score")
    ax.set_xticks(x)
    ax.set_xticklabels(["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"])
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def assert_figure_valid(path):
    if not path.exists() or path.stat().st_size == 0:
        raise ValueError(f"Figure was not created or is empty: {path}")
    image = mpimg.imread(path)
    if image.size == 0 or image.shape[0] < 10 or image.shape[1] < 10:
        raise ValueError(f"Figure appears invalid: {path}")
    if float(np.std(image)) == 0.0:
        raise ValueError(f"Figure appears blank: {path}")


def write_classification_report(
    output_path,
    train_rows,
    test_rows,
    best_params,
    best_row,
    test_metrics,
    report,
):
    lines = [
        "Tuned XGBoost Classification Report",
        "=" * 80,
        f"XGBoost version: {xgboost.__version__}",
        f"Train samples: {train_rows}",
        f"Test samples: {test_rows}",
        "Positive class: Yes / 1",
        f"Feature set: reduced, excluding {EXCLUDED_FEATURE}",
        "",
        "Parameter search range:",
    ]
    for parameter, values in PARAM_DISTRIBUTIONS.items():
        clean_name = parameter.replace("model__", "")
        lines.append(f"- {clean_name}: {values} ({PARAMETER_DESCRIPTIONS[clean_name]})")
    lines.extend(
        [
            "",
            "Search method:",
            "- RandomizedSearchCV with 30 random parameter combinations.",
            "- StratifiedKFold(n_splits=5, shuffle=True, random_state=42).",
            "- Total CV fits: 150.",
            "- Refit metric: mean CV F1 on the training set.",
            "- XGBClassifier n_jobs=-1; RandomizedSearchCV n_jobs=1.",
            "- No SMOTE, early stopping, or threshold adjustment.",
            "- Test set was evaluated only after best parameters were selected.",
            "",
            "Best parameters:",
            json.dumps(to_python(best_params), indent=2, sort_keys=True),
            "",
            "Best CV metrics:",
            f"- Accuracy: {best_row['mean_test_accuracy']:.6f} +/- {best_row['std_test_accuracy']:.6f}",
            f"- Precision: {best_row['mean_test_precision']:.6f} +/- {best_row['std_test_precision']:.6f}",
            f"- Recall: {best_row['mean_test_recall']:.6f} +/- {best_row['std_test_recall']:.6f}",
            f"- F1: {best_row['mean_test_f1']:.6f} +/- {best_row['std_test_f1']:.6f}",
            f"- ROC-AUC: {best_row['mean_test_roc_auc']:.6f} +/- {best_row['std_test_roc_auc']:.6f}",
            "",
            "Final test metrics:",
            f"- Accuracy: {test_metrics['accuracy']:.6f}",
            f"- Precision: {test_metrics['precision']:.6f}",
            f"- Recall: {test_metrics['recall']:.6f}",
            f"- F1: {test_metrics['f1']:.6f}",
            f"- ROC-AUC: {test_metrics['roc_auc']:.6f}",
            f"- TN={test_metrics['tn']}, FP={test_metrics['fp']}, FN={test_metrics['fn']}, TP={test_metrics['tp']}",
            "",
            "Classification report:",
            report,
            "",
            "Feature importance note:",
            "- Gain importance indicates predictive contribution, not causality.",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def validate_outputs(
    cv_results,
    test_metrics,
    transformed_importance,
    original_importance,
    predictions,
    figure_paths,
    csv_hash_before,
    csv_hash_after,
):
    if len(cv_results) != 30:
        raise AssertionError(f"Expected 30 parameter rows, got {len(cv_results)}.")
    numeric_cv = cv_results.select_dtypes(include="number")
    if cv_results.replace([np.inf, -np.inf], np.nan).isna().any().any():
        raise AssertionError("CV results contain NaN or infinite values.")
    metric_columns = [
        "mean_test_accuracy",
        "mean_test_precision",
        "mean_test_recall",
        "mean_test_f1",
        "mean_test_roc_auc",
    ]
    for column in metric_columns:
        if not cv_results[column].between(0, 1).all():
            raise AssertionError(f"{column} has values outside [0, 1].")
    if not np.isfinite(numeric_cv.to_numpy()).all():
        raise AssertionError("CV numeric results contain non-finite values.")
    if test_metrics["tn"] + test_metrics["fp"] + test_metrics["fn"] + test_metrics["tp"] != 20000:
        raise AssertionError("Confusion matrix total is not 20000.")
    if len(predictions) != 20000:
        raise AssertionError("Test predictions should contain 20000 rows.")
    if transformed_importance["transformed_feature"].duplicated().any():
        raise AssertionError("Duplicate transformed feature names found.")
    if not np.isclose(transformed_importance["gain_importance"].sum(), 1.0, atol=1e-8):
        raise AssertionError("Transformed feature importance sum is not close to 1.")
    if not np.isclose(original_importance["gain_importance"].sum(), 1.0, atol=1e-8):
        raise AssertionError("Original feature importance sum is not close to 1.")
    if csv_hash_before != csv_hash_after:
        raise AssertionError("Original CSV SHA256 changed.")
    for path in figure_paths:
        assert_figure_valid(path)


def main():
    start_time = time.perf_counter()
    ensure_directories()
    require_baseline_files()

    csv_hash_before = sha256_file(DATA_PATH)
    print(f"XGBoost version: {xgboost.__version__}")
    read_and_print_baseline_cv()

    df = load_dataset()
    validate_dataset(df)
    feature_sets = get_feature_sets(df)
    if SELECTED_FEATURE_SET not in feature_sets:
        raise ValueError("Reduced feature set is missing from get_feature_sets().")
    if EXCLUDED_FEATURE in feature_sets[SELECTED_FEATURE_SET]:
        raise ValueError("Reduced feature set still contains cybersecurity_awareness_score.")

    splits = split_dataset(df)
    split = splits[SELECTED_FEATURE_SET]
    features = feature_sets[SELECTED_FEATURE_SET]
    numeric_features, categorical_features = split_features_by_type(df, features)
    pipeline = build_pipeline(numeric_features, categorical_features)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc",
    }
    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=PARAM_DISTRIBUTIONS,
        n_iter=30,
        scoring=scoring,
        refit="f1",
        cv=cv,
        random_state=RANDOM_SEED,
        n_jobs=1,
        return_train_score=True,
        verbose=2,
        error_score="raise",
    )

    print(f"Selected feature set: {SELECTED_FEATURE_SET}")
    print(f"Raw feature count: {len(features)}")
    print("RandomizedSearchCV parameter combinations: 30")
    print("Total CV fits: 150")
    print("Starting XGBoost randomized search...")
    search.fit(split["X_train"], split["y_train"])
    search_time = time.perf_counter() - start_time
    print("Completed XGBoost randomized search.")

    cv_results = prepare_cv_results(search)
    best_row = best_row_from_cv_results(cv_results)
    best_params = to_python(search.best_params_)
    transformed_feature_count = get_transformed_feature_count(search.best_estimator_)

    test_metrics, report, predictions, cm, y_probability = evaluate_on_test(
        search.best_estimator_,
        split,
    )
    transformed_importance, original_importance = make_feature_importance(
        search.best_estimator_,
        numeric_features,
        categorical_features,
    )
    comparison = make_baseline_comparison(test_metrics)

    cv_results.to_csv(METRICS_DIR / "xgboost_tuning_cv_results.csv", index=False, encoding="utf-8")

    best_payload = {
        "model": MODEL_NAME,
        "xgboost_version": xgboost.__version__,
        "selected_feature_set": SELECTED_FEATURE_SET,
        "raw_feature_count": len(features),
        "transformed_feature_count": transformed_feature_count,
        "best_params": {
            key.replace("model__", ""): value for key, value in best_params.items()
        },
        "best_cv_accuracy": best_row["mean_test_accuracy"],
        "best_cv_precision": best_row["mean_test_precision"],
        "best_cv_recall": best_row["mean_test_recall"],
        "best_cv_f1": best_row["mean_test_f1"],
        "best_cv_roc_auc": best_row["mean_test_roc_auc"],
        "best_cv_f1_std": best_row["std_test_f1"],
        "total_search_time_seconds": search_time,
        "cv_method": "StratifiedKFold(n_splits=5, shuffle=True, random_state=42)",
        "selection_metric": "mean_test_f1",
        "random_state": RANDOM_SEED,
    }
    (METRICS_DIR / "xgboost_best_params.json").write_text(
        json.dumps(to_python(best_payload), indent=2, sort_keys=True),
        encoding="utf-8",
    )

    summary_row = {
        "model": MODEL_NAME,
        "selected_feature_set": SELECTED_FEATURE_SET,
        "best_n_estimators": best_payload["best_params"]["n_estimators"],
        "best_learning_rate": best_payload["best_params"]["learning_rate"],
        "best_max_depth": best_payload["best_params"]["max_depth"],
        "best_min_child_weight": best_payload["best_params"]["min_child_weight"],
        "best_gamma": best_payload["best_params"]["gamma"],
        "best_subsample": best_payload["best_params"]["subsample"],
        "best_colsample_bytree": best_payload["best_params"]["colsample_bytree"],
        "best_reg_alpha": best_payload["best_params"]["reg_alpha"],
        "best_reg_lambda": best_payload["best_params"]["reg_lambda"],
        "best_scale_pos_weight": best_payload["best_params"]["scale_pos_weight"],
        "cv_accuracy_mean": best_row["mean_test_accuracy"],
        "cv_accuracy_std": best_row["std_test_accuracy"],
        "cv_precision_mean": best_row["mean_test_precision"],
        "cv_precision_std": best_row["std_test_precision"],
        "cv_recall_mean": best_row["mean_test_recall"],
        "cv_recall_std": best_row["std_test_recall"],
        "cv_f1_mean": best_row["mean_test_f1"],
        "cv_f1_std": best_row["std_test_f1"],
        "cv_roc_auc_mean": best_row["mean_test_roc_auc"],
        "cv_roc_auc_std": best_row["std_test_roc_auc"],
        "mean_fit_time": best_row["mean_fit_time"],
        "total_search_time": search_time,
    }
    pd.DataFrame([summary_row]).to_csv(
        METRICS_DIR / "xgboost_tuning_summary.csv",
        index=False,
        encoding="utf-8",
    )
    pd.DataFrame([test_metrics]).to_csv(
        METRICS_DIR / "xgboost_tuned_test_metrics.csv",
        index=False,
        encoding="utf-8",
    )
    predictions.to_csv(
        METRICS_DIR / "xgboost_tuned_test_predictions.csv",
        index=False,
        encoding="utf-8",
    )
    transformed_importance.to_csv(
        METRICS_DIR / "xgboost_tuned_feature_importance.csv",
        index=False,
        encoding="utf-8",
    )
    original_importance.to_csv(
        METRICS_DIR / "xgboost_tuned_original_feature_importance.csv",
        index=False,
        encoding="utf-8",
    )
    comparison.to_csv(
        METRICS_DIR / "xgboost_baseline_vs_tuned.csv",
        index=False,
        encoding="utf-8",
    )
    write_classification_report(
        METRICS_DIR / "xgboost_tuned_classification_report.txt",
        len(split["y_train"]),
        len(split["y_test"]),
        best_payload["best_params"],
        best_row,
        test_metrics,
        report,
    )

    figure_paths = [
        FIGURES_DIR / "xgboost_tuning_top_results.png",
        FIGURES_DIR / "xgboost_tuned_confusion_matrix.png",
        FIGURES_DIR / "xgboost_tuned_roc_curve.png",
        FIGURES_DIR / "xgboost_tuned_feature_importance.png",
        FIGURES_DIR / "xgboost_baseline_vs_tuned.png",
    ]
    plot_tuning_top_results(cv_results, figure_paths[0])
    plot_confusion_matrix(cm, figure_paths[1])
    plot_roc_curve(split["y_test"], y_probability, test_metrics["roc_auc"], figure_paths[2])
    plot_feature_importance(original_importance, figure_paths[3])
    plot_baseline_vs_tuned(comparison, figure_paths[4])

    joblib.dump(search.best_estimator_, MODELS_DIR / "xgboost_tuned.joblib")

    csv_hash_after = sha256_file(DATA_PATH)
    validate_outputs(
        cv_results,
        test_metrics,
        transformed_importance,
        original_importance,
        predictions,
        figure_paths,
        csv_hash_before,
        csv_hash_after,
    )

    train_f1_gap = float(best_row["mean_train_f1"] - best_row["mean_test_f1"])
    print("Best parameters:")
    print(json.dumps(best_payload["best_params"], indent=2, sort_keys=True))
    print("Best CV metrics:")
    print(
        f"accuracy={best_row['mean_test_accuracy']:.6f}, "
        f"precision={best_row['mean_test_precision']:.6f}, "
        f"recall={best_row['mean_test_recall']:.6f}, "
        f"f1={best_row['mean_test_f1']:.6f}, "
        f"roc_auc={best_row['mean_test_roc_auc']:.6f}"
    )
    print("Final test metrics:")
    print(
        f"accuracy={test_metrics['accuracy']:.6f}, "
        f"precision={test_metrics['precision']:.6f}, "
        f"recall={test_metrics['recall']:.6f}, "
        f"f1={test_metrics['f1']:.6f}, "
        f"roc_auc={test_metrics['roc_auc']:.6f}"
    )
    print(
        f"TN={test_metrics['tn']}, FP={test_metrics['fp']}, "
        f"FN={test_metrics['fn']}, TP={test_metrics['tp']}"
    )
    print("Baseline vs tuned changes:")
    print(comparison.to_string(index=False))
    print(f"Mean train F1 minus mean CV F1 for best params: {train_f1_gap:.6f}")
    print(f"Total search time: {search_time:.2f} seconds")
    print(f"Original CSV SHA256 unchanged: {csv_hash_before == csv_hash_after}")

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
