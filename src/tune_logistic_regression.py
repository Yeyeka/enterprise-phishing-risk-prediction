from pathlib import Path
import json
import os
import sys
import tempfile
import time
import warnings

import matplotlib

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "phishing_ml_matplotlib_cache"))
matplotlib.use("Agg")

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
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
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

from data_preprocessing import (
    RANDOM_SEED,
    build_linear_preprocessor,
    feature_types_for_feature_set,
    get_feature_sets,
    identify_feature_types,
    load_dataset,
    split_dataset,
    validate_dataset,
)


MODEL_NAME = "Logistic Regression Tuned"
BASELINE_MODEL_NAME = "Logistic Regression"
SELECTED_FEATURE_SET = "reduced"
METRICS_DIR = Path("results") / "metrics"
FIGURES_DIR = Path("results") / "figures"
MODELS_DIR = Path("models")
DPI = 150


def get_project_root():
    return Path(__file__).resolve().parents[1]


def ensure_output_dirs(project_root):
    metrics_dir = project_root / METRICS_DIR
    figures_dir = project_root / FIGURES_DIR
    models_dir = project_root / MODELS_DIR
    metrics_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    return metrics_dir, figures_dir, models_dir


def make_pipeline(numeric_features, categorical_features):
    return Pipeline(
        steps=[
            ("preprocessor", build_linear_preprocessor(numeric_features, categorical_features)),
            (
                "model",
                LogisticRegression(
                    max_iter=2000,
                    solver="lbfgs",
                    penalty="l2",
                    random_state=RANDOM_SEED,
                ),
            ),
        ]
    )


def make_grid_search(pipeline):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc",
    }
    param_grid = {
        "model__C": [0.001, 0.01, 0.1, 1, 10, 100],
        "model__class_weight": [None, "balanced"],
    }
    return GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring=scoring,
        refit="f1",
        cv=cv,
        n_jobs=1,
        return_train_score=False,
        error_score="raise",
    )


def clean_cv_results(grid_search):
    results = pd.DataFrame(grid_search.cv_results_)
    keep_columns = [
        "param_model__C",
        "param_model__class_weight",
        "mean_test_accuracy",
        "std_test_accuracy",
        "rank_test_accuracy",
        "mean_test_precision",
        "std_test_precision",
        "rank_test_precision",
        "mean_test_recall",
        "std_test_recall",
        "rank_test_recall",
        "mean_test_f1",
        "std_test_f1",
        "rank_test_f1",
        "mean_test_roc_auc",
        "std_test_roc_auc",
        "rank_test_roc_auc",
        "mean_fit_time",
        "std_fit_time",
        "mean_score_time",
        "std_score_time",
    ]
    cleaned = results[keep_columns].copy()
    cleaned.insert(0, "model", MODEL_NAME)
    cleaned.insert(1, "feature_set", SELECTED_FEATURE_SET)
    cleaned.insert(2, "selected_feature_set", SELECTED_FEATURE_SET)
    cleaned = cleaned.rename(
        columns={
            "param_model__C": "C",
            "param_model__class_weight": "class_weight",
        }
    )
    cleaned["class_weight"] = cleaned["class_weight"].astype(object).where(
        cleaned["class_weight"].notna(), "none"
    )
    return cleaned.sort_values("rank_test_f1").reset_index(drop=True)


def make_tuning_summary(
    grid_search,
    cv_results,
    raw_feature_count,
    transformed_feature_count,
    total_search_time_seconds,
    total_runtime_seconds,
):
    best_index = grid_search.best_index_
    raw_results = pd.DataFrame(grid_search.cv_results_)
    best_row = raw_results.loc[best_index]
    best_params = grid_search.best_params_
    return pd.DataFrame(
        [
            {
                "model": MODEL_NAME,
                "feature_set": SELECTED_FEATURE_SET,
                "selected_feature_set": SELECTED_FEATURE_SET,
                "raw_feature_count": raw_feature_count,
                "transformed_feature_count": transformed_feature_count,
                "best_C": best_params["model__C"],
                "best_class_weight": "none"
                if best_params["model__class_weight"] is None
                else best_params["model__class_weight"],
                "accuracy_mean": best_row["mean_test_accuracy"],
                "accuracy_std": best_row["std_test_accuracy"],
                "precision_mean": best_row["mean_test_precision"],
                "precision_std": best_row["std_test_precision"],
                "recall_mean": best_row["mean_test_recall"],
                "recall_std": best_row["std_test_recall"],
                "f1_mean": best_row["mean_test_f1"],
                "f1_std": best_row["std_test_f1"],
                "roc_auc_mean": best_row["mean_test_roc_auc"],
                "roc_auc_std": best_row["std_test_roc_auc"],
                "fit_time_mean": best_row["mean_fit_time"],
                "score_time_mean": best_row["mean_score_time"],
                "total_search_time_seconds": float(total_search_time_seconds),
                "total_runtime_seconds": float(total_runtime_seconds),
                "grid_search_refit_metric": grid_search.refit,
                "parameter_combination_count": len(cv_results),
            }
        ]
    )


def save_best_params(path, grid_search, total_search_time_seconds, total_runtime_seconds):
    best_params = {
        "model": MODEL_NAME,
        "feature_set": SELECTED_FEATURE_SET,
        "selected_feature_set": SELECTED_FEATURE_SET,
        "selection_data": "80% training set only",
        "cv": {
            "type": "StratifiedKFold",
            "n_splits": 5,
            "shuffle": True,
            "random_state": RANDOM_SEED,
        },
        "refit": "f1",
        "best_params": {
            "C": grid_search.best_params_["model__C"],
            "class_weight": grid_search.best_params_["model__class_weight"],
            "solver": "lbfgs",
            "penalty": "l2",
            "max_iter": 2000,
            "random_state": RANDOM_SEED,
        },
        "best_cv_f1": grid_search.best_score_,
        "total_search_time_seconds": float(total_search_time_seconds),
        "total_runtime_seconds": float(total_runtime_seconds),
    }
    path.write_text(json.dumps(best_params, indent=2), encoding="utf-8")


def evaluate_on_test(best_estimator, split):
    y_test = split["y_test"]
    y_pred = best_estimator.predict(split["X_test"])
    y_proba = best_estimator.predict_proba(split["X_test"])[:, 1]
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
    metrics = {
        "model": MODEL_NAME,
        "selected_feature_set": SELECTED_FEATURE_SET,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }
    report = classification_report(
        y_test,
        y_pred,
        labels=[0, 1],
        target_names=["No", "Yes"],
        digits=4,
    )
    predictions = pd.DataFrame(
        {
            "index": split["X_test"].index,
            "y_true": y_test.to_numpy(),
            "y_pred": y_pred,
            "y_proba_yes": y_proba,
        }
    )
    return metrics, report, predictions, y_pred, y_proba


def save_classification_report(path, report, metrics, summary, baseline_metrics):
    lines = []
    lines.append("Tuned Logistic Regression Classification Report")
    lines.append("=" * 80)
    lines.append("Model: LogisticRegression(max_iter=2000, solver='lbfgs', penalty='l2', random_state=42)")
    lines.append("Parameter grid: C=[0.001, 0.01, 0.1, 1, 10, 100], class_weight=[None, 'balanced']")
    lines.append(f"Feature set: {SELECTED_FEATURE_SET}, selected by the prior baseline cross-validation stage.")
    lines.append("Parameter selection used only the 80% training set with 5-fold StratifiedKFold.")
    lines.append("GridSearchCV refit metric: F1")
    lines.append("The test set was evaluated once after best_params_ was selected.")
    lines.append("No SMOTE, threshold adjustment, or test-set-driven parameter change was used.")
    lines.append("Positive class: Yes / 1")
    lines.append("")
    lines.append("Best cross-validation metrics:")
    best = summary.iloc[0]
    for metric in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
        lines.append(f"- {metric}: {best[f'{metric}_mean']:.6f} +/- {best[f'{metric}_std']:.6f}")
    lines.append("")
    lines.append("Test classification report:")
    lines.append(report)
    lines.append("")
    lines.append("Test metrics:")
    for metric in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
        delta = metrics[metric] - baseline_metrics[metric]
        lines.append(f"- {metric}: {metrics[metric]:.6f} (baseline delta {delta:+.6f})")
    lines.append(f"- TN={metrics['tn']}, FP={metrics['fp']}, FN={metrics['fn']}, TP={metrics['tp']}")
    path.write_text("\n".join(lines), encoding="utf-8")


def plot_parameter_comparison(path, cv_results):
    metrics = ["mean_test_accuracy", "mean_test_precision", "mean_test_recall", "mean_test_f1", "mean_test_roc_auc"]
    labels = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
    c_values = sorted(cv_results["C"].unique())

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for ax, class_weight_label in zip(axes, ["none", "balanced"]):
        subset = cv_results[cv_results["class_weight"] == class_weight_label].sort_values("C")
        for metric, label in zip(metrics, labels):
            ax.plot(subset["C"], subset[metric], marker="o", label=label)
        ax.set_xscale("log")
        ax.set_xticks(c_values)
        ax.set_xticklabels([str(value) for value in c_values], rotation=30)
        ax.set_ylim(0, 1)
        ax.set_title(f"class_weight={class_weight_label}")
        ax.set_xlabel("C")
        ax.set_ylabel("Mean CV Score")
        ax.grid(alpha=0.3)
    axes[1].legend(loc="lower right")
    fig.suptitle("Logistic Regression Tuning Parameter Comparison")
    fig.tight_layout()
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrix(path, metrics):
    matrix = np.array([[metrics["tn"], metrics["fp"]], [metrics["fn"], metrics["tp"]]])
    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_title("Tuned Logistic Regression Confusion Matrix")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["No", "Yes"])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["No", "Yes"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{matrix[i, j]:,}", ha="center", va="center", color="black")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def plot_roc_curve(path, y_test, y_proba, roc_auc):
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, label=f"Tuned Logistic Regression (AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random Baseline")
    ax.set_title("Tuned Logistic Regression ROC Curve")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def validate_outputs(cv_results, summary, metrics, predictions):
    metric_columns = [
        "mean_test_accuracy",
        "mean_test_precision",
        "mean_test_recall",
        "mean_test_f1",
        "mean_test_roc_auc",
    ]
    if len(cv_results) != 12:
        raise ValueError(f"Expected 12 parameter combinations, found {len(cv_results)}.")
    for column in metric_columns:
        if not cv_results[column].between(0, 1).all():
            raise ValueError(f"Grid search metric out of range: {column}")
    if cv_results.isna().any().any() or summary.isna().any().any() or predictions.isna().any().any():
        raise ValueError("NaN found in tuning outputs.")
    numeric_arrays = [
        cv_results.select_dtypes(include="number").to_numpy(),
        summary.select_dtypes(include="number").to_numpy(),
        predictions.select_dtypes(include="number").to_numpy(),
    ]
    if any((~np.isfinite(array)).any() for array in numeric_arrays):
        raise ValueError("Infinite value found in tuning outputs.")
    if metrics["tn"] + metrics["fp"] + metrics["fn"] + metrics["tp"] != 20000:
        raise ValueError("Confusion matrix values do not sum to 20000.")
    for metric in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
        if not (0 <= metrics[metric] <= 1):
            raise ValueError(f"Test metric out of range: {metric}")


def main():
    start_time = time.perf_counter()
    project_root = get_project_root()
    metrics_dir, figures_dir, models_dir = ensure_output_dirs(project_root)

    baseline_path = metrics_dir / "logistic_test_metrics.csv"
    if not baseline_path.exists():
        raise FileNotFoundError("Baseline logistic test metrics are required for comparison.")
    baseline_metrics = pd.read_csv(baseline_path).iloc[0].to_dict()
    baseline_selected = baseline_metrics["selected_feature_set"]
    if baseline_selected != SELECTED_FEATURE_SET:
        raise ValueError(
            f"Expected baseline selected feature set {SELECTED_FEATURE_SET}, got {baseline_selected}."
        )

    df = load_dataset()
    validate_dataset(df)
    feature_sets = get_feature_sets(df)
    splits = split_dataset(df)
    numeric_features, categorical_features = identify_feature_types(df)
    selected_features = feature_sets[SELECTED_FEATURE_SET]
    selected_split = splits[SELECTED_FEATURE_SET]
    feature_numeric, feature_categorical = feature_types_for_feature_set(
        selected_features, numeric_features, categorical_features
    )

    pipeline = make_pipeline(feature_numeric, feature_categorical)
    grid_search = make_grid_search(pipeline)

    print("Starting Logistic Regression GridSearchCV.")
    print(f"Feature set: {SELECTED_FEATURE_SET}")
    print("Parameter combinations: 12")
    convergence_warnings = []
    search_start_time = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always", ConvergenceWarning)
        grid_search.fit(selected_split["X_train"], selected_split["y_train"])
        convergence_warnings = [
            warning for warning in caught_warnings if issubclass(warning.category, ConvergenceWarning)
        ]
    total_search_time_seconds = float(time.perf_counter() - search_start_time)
    print("Completed Logistic Regression GridSearchCV.")

    cv_results = clean_cv_results(grid_search)
    transformed_feature_count = len(
        grid_search.best_estimator_.named_steps["preprocessor"].get_feature_names_out()
    )
    report_summary = make_tuning_summary(
        grid_search,
        cv_results,
        raw_feature_count=len(selected_features),
        transformed_feature_count=transformed_feature_count,
        total_search_time_seconds=0.0,
        total_runtime_seconds=0.0,
    )
    metrics, report, predictions, y_pred, y_proba = evaluate_on_test(
        grid_search.best_estimator_, selected_split
    )
    validate_outputs(cv_results, report_summary, metrics, predictions)

    cv_results.to_csv(metrics_dir / "logistic_tuning_cv_results.csv", index=False, encoding="utf-8")
    pd.DataFrame([metrics]).to_csv(
        metrics_dir / "logistic_tuned_test_metrics.csv", index=False, encoding="utf-8"
    )
    save_classification_report(
        metrics_dir / "logistic_tuned_classification_report.txt",
        report,
        metrics,
        report_summary,
        baseline_metrics,
    )
    predictions.to_csv(
        metrics_dir / "logistic_tuned_test_predictions.csv", index=False, encoding="utf-8"
    )

    plot_parameter_comparison(
        figures_dir / "logistic_tuning_parameter_comparison.png", cv_results
    )
    plot_confusion_matrix(figures_dir / "logistic_tuned_confusion_matrix.png", metrics)
    plot_roc_curve(
        figures_dir / "logistic_tuned_roc_curve.png",
        selected_split["y_test"],
        y_proba,
        metrics["roc_auc"],
    )
    joblib.dump(
        grid_search.best_estimator_,
        models_dir / "logistic_regression_tuned.joblib",
    )

    # total_runtime_seconds intentionally excludes only the final tiny metadata writes
    # for logistic_tuning_summary.csv and logistic_best_params.json.
    total_runtime_seconds = float(time.perf_counter() - start_time)
    summary = make_tuning_summary(
        grid_search,
        cv_results,
        raw_feature_count=len(selected_features),
        transformed_feature_count=transformed_feature_count,
        total_search_time_seconds=total_search_time_seconds,
        total_runtime_seconds=total_runtime_seconds,
    )
    summary.to_csv(metrics_dir / "logistic_tuning_summary.csv", index=False, encoding="utf-8")
    save_best_params(
        metrics_dir / "logistic_best_params.json",
        grid_search,
        total_search_time_seconds,
        total_runtime_seconds,
    )

    best_params = grid_search.best_params_
    print(f"Best params: C={best_params['model__C']}, class_weight={best_params['model__class_weight']}")
    print("Best CV metrics:")
    for metric in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
        print(
            f"- {metric}: {summary.loc[0, f'{metric}_mean']:.4f} "
            f"+/- {summary.loc[0, f'{metric}_std']:.4f}"
        )
    print("Tuned test metrics:")
    for metric in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
        delta = metrics[metric] - baseline_metrics[metric]
        print(f"- {metric}: {metrics[metric]:.4f} (baseline delta {delta:+.4f})")
    print(
        f"Confusion matrix: TN={metrics['tn']}, FP={metrics['fp']}, "
        f"FN={metrics['fn']}, TP={metrics['tp']}"
    )
    print(f"Convergence warnings: {len(convergence_warnings)}")
    print(f"Grid search time seconds: {total_search_time_seconds:.6f}")
    print(f"Total runtime seconds: {total_runtime_seconds:.6f}")
    _ = y_pred
    return 0


if __name__ == "__main__":
    sys.exit(main())
