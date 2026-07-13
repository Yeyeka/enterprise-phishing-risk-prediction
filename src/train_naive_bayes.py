from pathlib import Path
import os
import sys
import tempfile
import time

import matplotlib

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "phishing_ml_matplotlib_cache"))
matplotlib.use("Agg")

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
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline

from data_preprocessing import (
    RANDOM_SEED,
    build_naive_bayes_preprocessor,
    feature_types_for_feature_set,
    get_feature_sets,
    identify_feature_types,
    load_dataset,
    split_dataset,
    validate_dataset,
)


MODEL_NAME = "Gaussian Naive Bayes"
METRICS_DIR = Path("results") / "metrics"
FIGURES_DIR = Path("results") / "figures"
DPI = 150
TIE_THRESHOLD = 0.001
DENSE_MEMORY_LIMIT_MB = 1024


def get_project_root():
    return Path(__file__).resolve().parents[1]


def ensure_output_dirs(project_root):
    metrics_dir = project_root / METRICS_DIR
    figures_dir = project_root / FIGURES_DIR
    metrics_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    return metrics_dir, figures_dir


def make_pipeline(numeric_features, categorical_features):
    return Pipeline(
        steps=[
            ("preprocessor", build_naive_bayes_preprocessor(numeric_features, categorical_features)),
            ("model", GaussianNB()),
        ]
    )


def get_transformed_feature_count_and_memory(x_train, x_test, numeric_features, categorical_features):
    preprocessor = build_naive_bayes_preprocessor(numeric_features, categorical_features)
    preprocessor.fit(x_train)
    feature_count = len(preprocessor.get_feature_names_out())
    dense_memory_mb = round((len(x_train) + len(x_test)) * feature_count * 8 / (1024**2), 2)
    if dense_memory_mb > DENSE_MEMORY_LIMIT_MB:
        raise MemoryError(
            f"Estimated dense preprocessing output is too large: "
            f"{dense_memory_mb} MB > {DENSE_MEMORY_LIMIT_MB} MB"
        )
    return feature_count, dense_memory_mb


def run_cross_validation(feature_set_name, features, split, numeric_features, categorical_features):
    feature_numeric, feature_categorical = feature_types_for_feature_set(
        features, numeric_features, categorical_features
    )
    transformed_feature_count, dense_memory_mb = get_transformed_feature_count_and_memory(
        split["X_train"],
        split["X_test"],
        feature_numeric,
        feature_categorical,
    )
    pipeline = make_pipeline(feature_numeric, feature_categorical)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc",
    }

    print(f"Starting 5-fold CV for feature set: {feature_set_name}")
    cv_results = cross_validate(
        pipeline,
        split["X_train"],
        split["y_train"],
        cv=cv,
        scoring=scoring,
        return_train_score=False,
        n_jobs=1,
        error_score="raise",
    )

    fold_rows = []
    for fold_index in range(5):
        fold_rows.append(
            {
                "model": MODEL_NAME,
                "feature_set": feature_set_name,
                "fold": fold_index + 1,
                "accuracy": cv_results["test_accuracy"][fold_index],
                "precision": cv_results["test_precision"][fold_index],
                "recall": cv_results["test_recall"][fold_index],
                "f1": cv_results["test_f1"][fold_index],
                "roc_auc": cv_results["test_roc_auc"][fold_index],
                "fit_time": cv_results["fit_time"][fold_index],
                "score_time": cv_results["score_time"][fold_index],
            }
        )

    summary_row = {
        "model": MODEL_NAME,
        "feature_set": feature_set_name,
        "raw_feature_count": len(features),
        "transformed_feature_count": transformed_feature_count,
        "estimated_dense_memory_mb": dense_memory_mb,
    }
    for metric in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
        scores = cv_results[f"test_{metric}"]
        summary_row[f"{metric}_mean"] = scores.mean()
        summary_row[f"{metric}_std"] = scores.std(ddof=1)
    summary_row["fit_time_mean"] = cv_results["fit_time"].mean()
    summary_row["fit_time_std"] = cv_results["fit_time"].std(ddof=1)
    summary_row["score_time_mean"] = cv_results["score_time"].mean()
    summary_row["score_time_std"] = cv_results["score_time"].std(ddof=1)

    print(f"Completed 5-fold CV for feature set: {feature_set_name}")
    print(
        f"{feature_set_name}: "
        f"accuracy={summary_row['accuracy_mean']:.4f} +/- {summary_row['accuracy_std']:.4f}, "
        f"precision={summary_row['precision_mean']:.4f} +/- {summary_row['precision_std']:.4f}, "
        f"recall={summary_row['recall_mean']:.4f} +/- {summary_row['recall_std']:.4f}, "
        f"f1={summary_row['f1_mean']:.4f} +/- {summary_row['f1_std']:.4f}, "
        f"roc_auc={summary_row['roc_auc_mean']:.4f} +/- {summary_row['roc_auc_std']:.4f}"
    )
    return fold_rows, summary_row


def choose_feature_set(cv_summary):
    full_row = cv_summary[cv_summary["feature_set"] == "full"].iloc[0]
    reduced_row = cv_summary[cv_summary["feature_set"] == "reduced"].iloc[0]

    f1_difference = full_row["f1_mean"] - reduced_row["f1_mean"]
    if abs(f1_difference) > TIE_THRESHOLD:
        selected = "full" if f1_difference > 0 else "reduced"
        reason = (
            f"Selected by higher mean F1 on training 5-fold CV. "
            f"F1 difference={abs(f1_difference):.6f}."
        )
        return selected, reason

    roc_auc_difference = full_row["roc_auc_mean"] - reduced_row["roc_auc_mean"]
    if abs(roc_auc_difference) > TIE_THRESHOLD:
        selected = "full" if roc_auc_difference > 0 else "reduced"
        reason = (
            "Mean F1 values were close, so selected by higher mean ROC-AUC "
            "on training 5-fold CV."
        )
        return selected, reason

    recall_difference = full_row["recall_mean"] - reduced_row["recall_mean"]
    if abs(recall_difference) > TIE_THRESHOLD:
        selected = "full" if recall_difference > 0 else "reduced"
        reason = (
            "Mean F1 and ROC-AUC values were close, so selected by higher "
            "Yes-class Recall on training 5-fold CV."
        )
        return selected, reason

    return (
        "reduced",
        "Mean F1, ROC-AUC, and Recall were almost identical on training CV, so reduced was selected.",
    )


def evaluate_final_pipeline(selected_feature_set, features, split, numeric_features, categorical_features):
    feature_numeric, feature_categorical = feature_types_for_feature_set(
        features, numeric_features, categorical_features
    )
    pipeline = make_pipeline(feature_numeric, feature_categorical)
    pipeline.fit(split["X_train"], split["y_train"])

    y_pred = pipeline.predict(split["X_test"])
    y_proba = pipeline.predict_proba(split["X_test"])[:, 1]
    y_test = split["y_test"]

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
    metrics = {
        "model": MODEL_NAME,
        "selected_feature_set": selected_feature_set,
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
    return pipeline, y_pred, y_proba, metrics, report


def save_classification_report(path, report, selected_feature_set, selection_reason, split, metrics, cv_summary):
    lines = []
    lines.append("Gaussian Naive Bayes Baseline Classification Report")
    lines.append("=" * 80)
    lines.append("Model parameters: GaussianNB(var_smoothing=1e-9 default).")
    lines.append(f"Train size: {len(split['y_train'])}")
    lines.append(f"Test size: {len(split['y_test'])}")
    lines.append("Cross-validation: StratifiedKFold(n_splits=5, shuffle=True, random_state=42).")
    lines.append("Feature set CV summary:")
    for _, row in cv_summary.iterrows():
        lines.append(
            f"- {row['feature_set']}: "
            f"Accuracy={row['accuracy_mean']:.4f} +/- {row['accuracy_std']:.4f}, "
            f"Precision={row['precision_mean']:.4f} +/- {row['precision_std']:.4f}, "
            f"Recall={row['recall_mean']:.4f} +/- {row['recall_std']:.4f}, "
            f"F1={row['f1_mean']:.4f} +/- {row['f1_std']:.4f}, "
            f"ROC-AUC={row['roc_auc_mean']:.4f} +/- {row['roc_auc_std']:.4f}."
        )
    lines.append(f"Selected feature set: {selected_feature_set}")
    lines.append(f"Selection basis: {selection_reason}")
    lines.append("Positive class: Yes / 1")
    lines.append("The test set was evaluated once after selecting the feature set using training CV only.")
    lines.append("This baseline model has not been tuned.")
    lines.append("")
    lines.append("Test classification report:")
    lines.append(report)
    lines.append("")
    lines.append("Test metrics:")
    lines.append(
        f"Accuracy={metrics['accuracy']:.6f}, Precision={metrics['precision']:.6f}, "
        f"Recall={metrics['recall']:.6f}, F1={metrics['f1']:.6f}, ROC-AUC={metrics['roc_auc']:.6f}"
    )
    lines.append(f"TN={metrics['tn']}, FP={metrics['fp']}, FN={metrics['fn']}, TP={metrics['tp']}")
    path.write_text("\n".join(lines), encoding="utf-8")


def plot_confusion_matrix(path, metrics):
    matrix = np.array([[metrics["tn"], metrics["fp"]], [metrics["fn"], metrics["tp"]]])
    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_title("Gaussian Naive Bayes Confusion Matrix")
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
    ax.plot(fpr, tpr, label=f"GaussianNB (AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random Baseline")
    ax.set_title("Gaussian Naive Bayes ROC Curve")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def plot_cv_comparison(path, cv_summary):
    metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    x = np.arange(len(metrics))
    width = 0.35
    full_values = [
        cv_summary.loc[cv_summary["feature_set"] == "full", f"{metric}_mean"].iloc[0]
        for metric in metrics
    ]
    reduced_values = [
        cv_summary.loc[cv_summary["feature_set"] == "reduced", f"{metric}_mean"].iloc[0]
        for metric in metrics
    ]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - width / 2, full_values, width, label="full")
    ax.bar(x + width / 2, reduced_values, width, label="reduced")
    ax.set_title("Gaussian Naive Bayes CV Metric Comparison")
    ax.set_xlabel("Metric")
    ax.set_ylabel("Mean CV Score")
    ax.set_xticks(x)
    ax.set_xticklabels(["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"])
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def validate_outputs(cv_fold_results, cv_summary, test_metrics):
    if len(cv_fold_results) != 10:
        raise ValueError("Expected 10 CV fold rows: 5 folds for each of 2 feature sets.")
    fold_counts = cv_fold_results.groupby("feature_set")["fold"].count().to_dict()
    if fold_counts.get("full") != 5 or fold_counts.get("reduced") != 5:
        raise ValueError(f"Unexpected CV fold counts: {fold_counts}")

    metric_columns = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    for column in metric_columns:
        if not cv_fold_results[column].between(0, 1).all():
            raise ValueError(f"CV metric out of range: {column}")
        if not (0 <= test_metrics[column] <= 1):
            raise ValueError(f"Test metric out of range: {column}")

    if cv_fold_results.isna().any().any() or cv_summary.isna().any().any():
        raise ValueError("NaN found in CV results.")
    if not np.isfinite(cv_fold_results.select_dtypes(include="number").to_numpy()).all():
        raise ValueError("Infinite value found in CV fold results.")
    if not np.isfinite(cv_summary.select_dtypes(include="number").to_numpy()).all():
        raise ValueError("Infinite value found in CV summary.")

    if test_metrics["tn"] + test_metrics["fp"] + test_metrics["fn"] + test_metrics["tp"] != 20000:
        raise ValueError("Confusion matrix values do not sum to 20000.")


def main():
    start_time = time.perf_counter()
    project_root = get_project_root()
    metrics_dir, figures_dir = ensure_output_dirs(project_root)

    df = load_dataset()
    validate_dataset(df)
    feature_sets = get_feature_sets(df)
    splits = split_dataset(df)
    numeric_features, categorical_features = identify_feature_types(df)

    cv_fold_rows = []
    cv_summary_rows = []

    for feature_set_name in ["full", "reduced"]:
        fold_rows, summary_row = run_cross_validation(
            feature_set_name,
            feature_sets[feature_set_name],
            splits[feature_set_name],
            numeric_features,
            categorical_features,
        )
        cv_fold_rows.extend(fold_rows)
        cv_summary_rows.append(summary_row)

    cv_fold_results = pd.DataFrame(cv_fold_rows)
    cv_summary = pd.DataFrame(cv_summary_rows)
    selected_feature_set, selection_reason = choose_feature_set(cv_summary)
    print(f"Selected feature set: {selected_feature_set}")
    print(f"Selection reason: {selection_reason}")

    _, y_pred, y_proba, test_metrics, report = evaluate_final_pipeline(
        selected_feature_set,
        feature_sets[selected_feature_set],
        splits[selected_feature_set],
        numeric_features,
        categorical_features,
    )

    validate_outputs(cv_fold_results, cv_summary, test_metrics)

    cv_fold_results.to_csv(metrics_dir / "naive_bayes_cv_fold_results.csv", index=False, encoding="utf-8")
    cv_summary.to_csv(metrics_dir / "naive_bayes_cv_summary.csv", index=False, encoding="utf-8")
    pd.DataFrame([test_metrics]).to_csv(
        metrics_dir / "naive_bayes_test_metrics.csv", index=False, encoding="utf-8"
    )
    save_classification_report(
        metrics_dir / "naive_bayes_classification_report.txt",
        report,
        selected_feature_set,
        selection_reason,
        splits[selected_feature_set],
        test_metrics,
        cv_summary,
    )

    y_test = splits[selected_feature_set]["y_test"]
    plot_confusion_matrix(figures_dir / "naive_bayes_confusion_matrix.png", test_metrics)
    plot_roc_curve(
        figures_dir / "naive_bayes_roc_curve.png",
        y_test,
        y_proba,
        test_metrics["roc_auc"],
    )
    plot_cv_comparison(figures_dir / "naive_bayes_cv_comparison.png", cv_summary)

    elapsed_time = time.perf_counter() - start_time
    print("Final test metrics:")
    print(
        f"accuracy={test_metrics['accuracy']:.4f}, "
        f"precision={test_metrics['precision']:.4f}, "
        f"recall={test_metrics['recall']:.4f}, "
        f"f1={test_metrics['f1']:.4f}, "
        f"roc_auc={test_metrics['roc_auc']:.4f}"
    )
    print(
        f"Confusion matrix: TN={test_metrics['tn']}, FP={test_metrics['fp']}, "
        f"FN={test_metrics['fn']}, TP={test_metrics['tp']}"
    )
    print(f"Completed 10 cross-validation fits.")
    print(f"Total runtime seconds: {elapsed_time:.2f}")
    _ = y_pred
    return 0


if __name__ == "__main__":
    sys.exit(main())
