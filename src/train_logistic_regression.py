from pathlib import Path
import sys
import time
import warnings

import matplotlib
import os
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "phishing_ml_matplotlib_cache"))
matplotlib.use("Agg")

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
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline

from data_preprocessing import (
    RANDOM_SEED,
    TARGET_MAPPING,
    build_linear_preprocessor,
    feature_types_for_feature_set,
    get_feature_sets,
    identify_feature_types,
    load_dataset,
    split_dataset,
    validate_dataset,
)


MODEL_NAME = "Logistic Regression"
METRICS_DIR = Path("results") / "metrics"
FIGURES_DIR = Path("results") / "figures"
DPI = 150
F1_TIE_THRESHOLD = 0.001
ROC_AUC_TIE_THRESHOLD = 0.0005
RECALL_TIE_THRESHOLD = 0.001


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
            ("preprocessor", build_linear_preprocessor(numeric_features, categorical_features)),
            (
                "model",
                LogisticRegression(
                    max_iter=2000,
                    solver="lbfgs",
                    random_state=RANDOM_SEED,
                ),
            ),
        ]
    )


def get_transformed_feature_names(x_train, numeric_features, categorical_features):
    preprocessor = build_linear_preprocessor(numeric_features, categorical_features)
    preprocessor.fit(x_train)
    return preprocessor.get_feature_names_out()


def run_cross_validation(feature_set_name, features, split, numeric_features, categorical_features):
    feature_numeric, feature_categorical = feature_types_for_feature_set(
        features, numeric_features, categorical_features
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
    convergence_warnings = []
    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always", ConvergenceWarning)
        cv_results = cross_validate(
            pipeline,
            split["X_train"],
            split["y_train"],
            cv=cv,
            scoring=scoring,
            return_train_score=False,
        )
        convergence_warnings = [
            warning for warning in caught_warnings if issubclass(warning.category, ConvergenceWarning)
        ]

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

    transformed_feature_count = len(
        get_transformed_feature_names(split["X_train"], feature_numeric, feature_categorical)
    )
    summary_row = {
        "model": MODEL_NAME,
        "feature_set": feature_set_name,
        "raw_feature_count": len(features),
        "transformed_feature_count": transformed_feature_count,
    }
    for metric in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
        scores = cv_results[f"test_{metric}"]
        summary_row[f"{metric}_mean"] = scores.mean()
        summary_row[f"{metric}_std"] = scores.std(ddof=1)
    summary_row["fit_time_mean"] = cv_results["fit_time"].mean()
    summary_row["score_time_mean"] = cv_results["score_time"].mean()
    summary_row["convergence_warning_count"] = len(convergence_warnings)

    print(f"Completed 5-fold CV for feature set: {feature_set_name}")
    print(
        f"{feature_set_name}: "
        f"accuracy={summary_row['accuracy_mean']:.4f} +/- {summary_row['accuracy_std']:.4f}, "
        f"precision={summary_row['precision_mean']:.4f} +/- {summary_row['precision_std']:.4f}, "
        f"recall={summary_row['recall_mean']:.4f} +/- {summary_row['recall_std']:.4f}, "
        f"f1={summary_row['f1_mean']:.4f} +/- {summary_row['f1_std']:.4f}, "
        f"roc_auc={summary_row['roc_auc_mean']:.4f} +/- {summary_row['roc_auc_std']:.4f}"
    )

    return fold_rows, summary_row, convergence_warnings


def choose_feature_set(cv_summary):
    full_row = cv_summary[cv_summary["feature_set"] == "full"].iloc[0]
    reduced_row = cv_summary[cv_summary["feature_set"] == "reduced"].iloc[0]
    f1_difference = abs(full_row["f1_mean"] - reduced_row["f1_mean"])

    if f1_difference > F1_TIE_THRESHOLD:
        selected = "full" if full_row["f1_mean"] > reduced_row["f1_mean"] else "reduced"
        reason = (
            f"Selected by higher mean F1 on training 5-fold CV. "
            f"F1 difference={f1_difference:.6f}, tolerance={F1_TIE_THRESHOLD:.6f}."
        )
        return selected, reason

    roc_auc_difference = abs(full_row["roc_auc_mean"] - reduced_row["roc_auc_mean"])
    if roc_auc_difference > ROC_AUC_TIE_THRESHOLD:
        selected = "full" if full_row["roc_auc_mean"] > reduced_row["roc_auc_mean"] else "reduced"
        reason = (
            "Mean F1 values were practically tied, so selected by higher ROC-AUC "
            f"on training 5-fold CV. ROC-AUC difference={roc_auc_difference:.6f}, "
            f"tolerance={ROC_AUC_TIE_THRESHOLD:.6f}."
        )
        return selected, reason

    recall_difference = abs(full_row["recall_mean"] - reduced_row["recall_mean"])
    if recall_difference > RECALL_TIE_THRESHOLD:
        selected = "full" if full_row["recall_mean"] > reduced_row["recall_mean"] else "reduced"
        reason = (
            "Mean F1 and ROC-AUC values were practically tied, so selected by higher "
            f"Yes-class Recall on training 5-fold CV. Recall difference={recall_difference:.6f}, "
            f"tolerance={RECALL_TIE_THRESHOLD:.6f}."
        )
        return selected, reason

    selected = (
        "full"
        if full_row["raw_feature_count"] < reduced_row["raw_feature_count"]
        else "reduced"
    )
    reason = (
        "Mean F1, ROC-AUC, and Yes-class Recall were all within practical tie "
        "tolerances on training 5-fold CV; selected the feature set with fewer raw "
        f"features ({selected})."
    )

    return selected, reason


def evaluate_final_pipeline(selected_feature_set, features, split, numeric_features, categorical_features):
    feature_numeric, feature_categorical = feature_types_for_feature_set(
        features, numeric_features, categorical_features
    )
    pipeline = make_pipeline(feature_numeric, feature_categorical)

    convergence_warnings = []
    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always", ConvergenceWarning)
        pipeline.fit(split["X_train"], split["y_train"])
        convergence_warnings = [
            warning for warning in caught_warnings if issubclass(warning.category, ConvergenceWarning)
        ]

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
        "convergence_warning_count": len(convergence_warnings),
    }

    feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()
    coefficients = pipeline.named_steps["model"].coef_[0]
    coefficient_table = pd.DataFrame(
        {
            "feature": feature_names,
            "coefficient": coefficients,
        }
    )
    coefficient_table["abs_coefficient"] = coefficient_table["coefficient"].abs()
    coefficient_table["direction"] = np.where(
        coefficient_table["coefficient"] > 0,
        "More likely Yes",
        np.where(coefficient_table["coefficient"] < 0, "More likely No", "Neutral"),
    )
    coefficient_table = coefficient_table.sort_values(
        "abs_coefficient", ascending=False
    ).reset_index(drop=True)

    report = classification_report(
        y_test,
        y_pred,
        labels=[0, 1],
        target_names=["No", "Yes"],
        digits=4,
    )

    return pipeline, y_pred, y_proba, metrics, coefficient_table, report, convergence_warnings


def save_classification_report(path, report, selected_feature_set, selection_reason, split, metrics):
    text = []
    text.append("Logistic Regression Baseline Classification Report")
    text.append("=" * 80)
    text.append(f"Train size: {len(split['y_train'])}")
    text.append(f"Test size: {len(split['y_test'])}")
    text.append(f"Selected feature set: {selected_feature_set}")
    text.append(f"Selection basis: {selection_reason}")
    text.append("The test set was evaluated once after selecting the feature set using training CV only.")
    text.append("Positive class: Yes / 1")
    text.append("")
    text.append(report)
    text.append("")
    text.append("Confusion matrix values:")
    text.append(f"TN={metrics['tn']}, FP={metrics['fp']}, FN={metrics['fn']}, TP={metrics['tp']}")
    path.write_text("\n".join(text), encoding="utf-8")


def plot_confusion_matrix(path, metrics):
    matrix = np.array([[metrics["tn"], metrics["fp"]], [metrics["fn"], metrics["tp"]]])
    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_title("Logistic Regression Confusion Matrix")
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
    ax.plot(fpr, tpr, label=f"Logistic Regression (AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random Baseline")
    ax.set_title("Logistic Regression ROC Curve")
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
    ax.set_title("Logistic Regression CV Metric Comparison")
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


def validate_outputs(cv_fold_results, cv_summary, test_metrics, coefficient_table):
    if len(cv_fold_results) != 10:
        raise ValueError("Expected 10 CV fold rows: 5 folds for each of 2 feature sets.")
    fold_counts = cv_fold_results.groupby("feature_set")["fold"].count().to_dict()
    if fold_counts.get("full") != 5 or fold_counts.get("reduced") != 5:
        raise ValueError(f"Unexpected CV fold counts: {fold_counts}")

    metric_columns = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    for column in metric_columns:
        if not cv_fold_results[column].between(0, 1).all():
            raise ValueError(f"CV metric out of range: {column}")
    for column in [f"{metric}_mean" for metric in metric_columns] + [
        f"{metric}_std" for metric in metric_columns
    ]:
        if cv_summary[column].isna().any():
            raise ValueError(f"NaN found in CV summary: {column}")

    for column in metric_columns:
        if not (0 <= test_metrics[column] <= 1):
            raise ValueError(f"Test metric out of range: {column}")
    if test_metrics["tn"] + test_metrics["fp"] + test_metrics["fn"] + test_metrics["tp"] != 20000:
        raise ValueError("Confusion matrix values do not sum to 20000.")
    if len(coefficient_table) != int(
        cv_summary.loc[
            cv_summary["feature_set"] == test_metrics["selected_feature_set"],
            "transformed_feature_count",
        ].iloc[0]
    ):
        raise ValueError("Coefficient count does not match transformed feature count.")


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
    all_cv_convergence_warnings = []

    for feature_set_name in ["full", "reduced"]:
        fold_rows, summary_row, convergence_warnings = run_cross_validation(
            feature_set_name,
            feature_sets[feature_set_name],
            splits[feature_set_name],
            numeric_features,
            categorical_features,
        )
        cv_fold_rows.extend(fold_rows)
        cv_summary_rows.append(summary_row)
        all_cv_convergence_warnings.extend(convergence_warnings)

    cv_fold_results = pd.DataFrame(cv_fold_rows)
    cv_summary = pd.DataFrame(cv_summary_rows)
    selected_feature_set, selection_reason = choose_feature_set(cv_summary)
    print(f"Selected feature set: {selected_feature_set}")
    print(f"Selection reason: {selection_reason}")

    (
        final_pipeline,
        y_pred,
        y_proba,
        test_metrics,
        coefficient_table,
        report,
        final_convergence_warnings,
    ) = evaluate_final_pipeline(
        selected_feature_set,
        feature_sets[selected_feature_set],
        splits[selected_feature_set],
        numeric_features,
        categorical_features,
    )

    validate_outputs(cv_fold_results, cv_summary, test_metrics, coefficient_table)

    cv_fold_results.to_csv(metrics_dir / "logistic_cv_fold_results.csv", index=False, encoding="utf-8")
    cv_summary.to_csv(metrics_dir / "logistic_cv_summary.csv", index=False, encoding="utf-8")
    pd.DataFrame([test_metrics]).to_csv(
        metrics_dir / "logistic_test_metrics.csv", index=False, encoding="utf-8"
    )
    save_classification_report(
        metrics_dir / "logistic_classification_report.txt",
        report,
        selected_feature_set,
        selection_reason,
        splits[selected_feature_set],
        test_metrics,
    )
    coefficient_table.to_csv(
        metrics_dir / "logistic_coefficients.csv", index=False, encoding="utf-8"
    )

    y_test = splits[selected_feature_set]["y_test"]
    plot_confusion_matrix(figures_dir / "logistic_confusion_matrix.png", test_metrics)
    plot_roc_curve(
        figures_dir / "logistic_roc_curve.png",
        y_test,
        y_proba,
        test_metrics["roc_auc"],
    )
    plot_cv_comparison(figures_dir / "logistic_cv_comparison.png", cv_summary)

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
    print(f"CV convergence warnings: {len(all_cv_convergence_warnings)}")
    print(f"Final fit convergence warnings: {len(final_convergence_warnings)}")
    print(f"Total runtime seconds: {elapsed_time:.2f}")

    # Keep a reference so static checkers do not treat the fitted pipeline as accidental.
    _ = final_pipeline
    return 0


if __name__ == "__main__":
    sys.exit(main())
