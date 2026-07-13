from pathlib import Path
import json
import os
import sys
import tempfile
import time

import matplotlib

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "phishing_ml_matplotlib_cache"))
matplotlib.use("Agg")

import joblib
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
from sklearn.model_selection import GridSearchCV, StratifiedKFold
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


MODEL_NAME = "Gaussian Naive Bayes Tuned"
BASELINE_MODEL_NAME = "Gaussian Naive Bayes"
METRICS_DIR = Path("results") / "metrics"
FIGURES_DIR = Path("results") / "figures"
MODELS_DIR = Path("models")
DPI = 150
TIE_THRESHOLD = 0.001
VAR_SMOOTHING_VALUES = [
    1e-12,
    3e-12,
    1e-11,
    3e-11,
    1e-10,
    3e-10,
    1e-9,
    3e-9,
    1e-8,
    3e-8,
    1e-7,
    3e-7,
    1e-6,
]
RATIO_METRICS = ["accuracy", "precision", "recall", "f1", "roc_auc"]
COUNT_METRICS = ["tn", "fp", "fn", "tp"]


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


def require_baseline_files(project_root):
    required_paths = [
        project_root / "src" / "train_naive_bayes.py",
        project_root / "results" / "metrics" / "naive_bayes_cv_summary.csv",
        project_root / "results" / "metrics" / "naive_bayes_test_metrics.csv",
    ]
    missing_paths = [path for path in required_paths if not path.exists()]
    if missing_paths:
        missing_text = "\n".join(str(path) for path in missing_paths)
        raise FileNotFoundError(f"Required baseline files are missing:\n{missing_text}")


def choose_feature_set_from_baseline(cv_summary):
    full_row = cv_summary[cv_summary["feature_set"] == "full"].iloc[0]
    reduced_row = cv_summary[cv_summary["feature_set"] == "reduced"].iloc[0]

    f1_difference = full_row["f1_mean"] - reduced_row["f1_mean"]
    if abs(f1_difference) > TIE_THRESHOLD:
        selected = "full" if f1_difference > 0 else "reduced"
        reason = (
            f"Selected by higher baseline training-CV mean F1. "
            f"F1 difference={abs(f1_difference):.6f}."
        )
        return selected, reason

    roc_auc_difference = full_row["roc_auc_mean"] - reduced_row["roc_auc_mean"]
    if abs(roc_auc_difference) > TIE_THRESHOLD:
        selected = "full" if roc_auc_difference > 0 else "reduced"
        reason = (
            "Baseline training-CV mean F1 values were very close; selected by higher "
            "baseline training-CV mean ROC-AUC."
        )
        return selected, reason

    recall_difference = full_row["recall_mean"] - reduced_row["recall_mean"]
    if abs(recall_difference) > TIE_THRESHOLD:
        selected = "full" if recall_difference > 0 else "reduced"
        reason = (
            "Baseline training-CV mean F1 and ROC-AUC values were very close; selected by "
            "higher baseline training-CV mean Recall."
        )
        return selected, reason

    return (
        "reduced",
        "Baseline training-CV F1, ROC-AUC, and Recall were nearly identical; selected the feature set with fewer raw features.",
    )


def make_pipeline(numeric_features, categorical_features):
    return Pipeline(
        steps=[
            ("preprocessor", build_naive_bayes_preprocessor(numeric_features, categorical_features)),
            ("model", GaussianNB()),
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
    param_grid = {"model__var_smoothing": VAR_SMOOTHING_VALUES}
    return GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring=scoring,
        refit="f1",
        cv=cv,
        n_jobs=1,
        return_train_score=True,
        verbose=2,
        error_score="raise",
    )


def clean_cv_results(grid_search):
    results = pd.DataFrame(grid_search.cv_results_)
    keep_columns = [
        "param_model__var_smoothing",
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
    cleaned = results[keep_columns].copy()
    cleaned = cleaned.sort_values("rank_test_f1").reset_index(drop=True)
    cleaned["param_model__var_smoothing"] = cleaned["param_model__var_smoothing"].astype(float)
    return cleaned


def make_best_params_dict(
    grid_search,
    cv_results,
    selected_feature_set,
    raw_feature_count,
    transformed_feature_count,
    total_search_time,
):
    best = cv_results.iloc[0]
    best_var_smoothing = float(grid_search.best_params_["model__var_smoothing"])
    return {
        "model": MODEL_NAME,
        "selected_feature_set": selected_feature_set,
        "raw_feature_count": int(raw_feature_count),
        "transformed_feature_count": int(transformed_feature_count),
        "best_params": {"var_smoothing": best_var_smoothing},
        "best_var_smoothing": best_var_smoothing,
        "best_cv_accuracy": float(best["mean_test_accuracy"]),
        "best_cv_precision": float(best["mean_test_precision"]),
        "best_cv_recall": float(best["mean_test_recall"]),
        "best_cv_f1": float(best["mean_test_f1"]),
        "best_cv_roc_auc": float(best["mean_test_roc_auc"]),
        "best_cv_f1_std": float(best["std_test_f1"]),
        "total_search_time_seconds": float(total_search_time),
        "cv_method": "StratifiedKFold(n_splits=5, shuffle=True, random_state=42)",
        "selection_metric": "mean_test_f1",
        "random_state": int(RANDOM_SEED),
    }


def make_tuning_summary_from_row(best_params, best_row):
    return pd.DataFrame(
        [
            {
                "model": MODEL_NAME,
                "selected_feature_set": best_params["selected_feature_set"],
                "best_var_smoothing": best_params["best_var_smoothing"],
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
                "total_search_time": best_params["total_search_time_seconds"],
            }
        ]
    )


def evaluate_on_test(best_estimator, split, selected_feature_set, best_var_smoothing):
    y_test = split["y_test"]
    y_pred = best_estimator.predict(split["X_test"])
    y_probability = best_estimator.predict_proba(split["X_test"])[:, 1]
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
    metrics = {
        "model": MODEL_NAME,
        "selected_feature_set": selected_feature_set,
        "best_var_smoothing": float(best_var_smoothing),
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_probability),
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
            "sample_index": split["X_test"].index,
            "y_true": y_test.to_numpy(),
            "y_pred": y_pred,
            "y_probability": y_probability,
        }
    )
    return metrics, report, predictions, y_probability


def make_baseline_comparison(baseline_metrics, tuned_metrics):
    rows = []
    for metric in RATIO_METRICS + COUNT_METRICS:
        baseline = float(baseline_metrics[metric])
        tuned = float(tuned_metrics[metric])
        absolute_change = tuned - baseline
        if metric in RATIO_METRICS and baseline != 0:
            relative_change_percent = absolute_change / baseline * 100
        else:
            relative_change_percent = "not_applicable"
        rows.append(
            {
                "metric": metric,
                "baseline": baseline,
                "tuned": tuned,
                "absolute_change": absolute_change,
                "relative_change_percent": relative_change_percent,
            }
        )
    return pd.DataFrame(rows)


def save_classification_report(
    path,
    report,
    selected_feature_set,
    selection_reason,
    split,
    cv_results,
    best_params,
    test_metrics,
    comparison,
):
    best = cv_results.iloc[0]
    lines = []
    lines.append("Tuned Gaussian Naive Bayes Classification Report")
    lines.append("=" * 80)
    lines.append("Model: GaussianNB")
    lines.append(f"Train size: {len(split['y_train'])}")
    lines.append(f"Test size: {len(split['y_test'])}")
    lines.append("Positive class: Yes / 1")
    lines.append(f"Selected feature set: {selected_feature_set}")
    lines.append(f"Feature-set selection basis: {selection_reason}")
    lines.append(f"Parameter search values: {VAR_SMOOTHING_VALUES}")
    lines.append("Cross-validation: StratifiedKFold(n_splits=5, shuffle=True, random_state=42)")
    lines.append(f"Best var_smoothing: {best_params['best_var_smoothing']:.0e}")
    lines.append("Best cross-validation metrics:")
    lines.append(
        f"Accuracy={best['mean_test_accuracy']:.6f}, Precision={best['mean_test_precision']:.6f}, "
        f"Recall={best['mean_test_recall']:.6f}, F1={best['mean_test_f1']:.6f}, "
        f"ROC-AUC={best['mean_test_roc_auc']:.6f}"
    )
    lines.append("Test metrics:")
    lines.append(
        f"Accuracy={test_metrics['accuracy']:.6f}, Precision={test_metrics['precision']:.6f}, "
        f"Recall={test_metrics['recall']:.6f}, F1={test_metrics['f1']:.6f}, "
        f"ROC-AUC={test_metrics['roc_auc']:.6f}"
    )
    lines.append(f"TN={test_metrics['tn']}, FP={test_metrics['fp']}, FN={test_metrics['fn']}, TP={test_metrics['tp']}")
    lines.append("")
    lines.append("Full classification_report:")
    lines.append(report)
    lines.append("")
    lines.append("Baseline vs tuned changes:")
    for _, row in comparison.iterrows():
        lines.append(
            f"- {row['metric']}: baseline={row['baseline']}, tuned={row['tuned']}, "
            f"absolute_change={row['absolute_change']}"
        )
    lines.append("")
    lines.append("The test set was evaluated only after selecting var_smoothing from training CV.")
    lines.append("No SMOTE, priors tuning, threshold adjustment, or test-set-driven search-space change was used.")
    path.write_text("\n".join(lines), encoding="utf-8")


def plot_tuning_curve(path, cv_results, best_var_smoothing):
    fig, ax = plt.subplots(figsize=(8, 5))
    x = cv_results["param_model__var_smoothing"].astype(float)
    for metric, label in [
        ("f1", "F1"),
        ("recall", "Recall"),
        ("roc_auc", "ROC-AUC"),
    ]:
        ax.errorbar(
            x,
            cv_results[f"mean_test_{metric}"],
            yerr=cv_results[f"std_test_{metric}"],
            marker="o",
            capsize=3,
            label=label,
        )
    ax.axvline(best_var_smoothing, color="black", linestyle="--", label="Best var_smoothing")
    ax.set_xscale("log")
    ax.set_title("GaussianNB Tuning Curve")
    ax.set_xlabel("var_smoothing")
    ax.set_ylabel("Mean CV Score")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def plot_top_results(path, cv_results):
    top = cv_results.sort_values("rank_test_f1").head(10).copy()
    labels = [f"{value:.0e}" for value in top["param_model__var_smoothing"].astype(float)]
    positions = np.arange(len(top))
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(positions, top["mean_test_f1"], color="#4C78A8")
    ax.set_title("Top GaussianNB Tuning Results by CV F1")
    ax.set_xlabel("var_smoothing")
    ax.set_ylabel("Mean CV F1")
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylim(max(0, top["mean_test_f1"].min() - 0.02), min(1, top["mean_test_f1"].max() + 0.02))
    for bar, value in zip(bars, top["mean_test_f1"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.4f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    fig.tight_layout()
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrix(path, metrics):
    matrix = np.array([[metrics["tn"], metrics["fp"]], [metrics["fn"], metrics["tp"]]])
    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_title("Tuned GaussianNB Confusion Matrix")
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


def plot_roc_curve(path, y_test, y_probability, roc_auc):
    fpr, tpr, _ = roc_curve(y_test, y_probability)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, label=f"Tuned GaussianNB (AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random Baseline")
    ax.set_title("Tuned GaussianNB ROC Curve")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def plot_baseline_vs_tuned(path, comparison):
    subset = comparison[comparison["metric"].isin(RATIO_METRICS)].copy()
    positions = np.arange(len(subset))
    width = 0.35
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(positions - width / 2, subset["baseline"], width, label="Baseline")
    ax.bar(positions + width / 2, subset["tuned"], width, label="Tuned")
    ax.set_title("GaussianNB Baseline vs Tuned Test Metrics")
    ax.set_xlabel("Metric")
    ax.set_ylabel("Score")
    ax.set_xticks(positions)
    ax.set_xticklabels(["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"])
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def validate_outputs(cv_results, summary, best_params, test_metrics, predictions, comparison):
    if len(cv_results) != len(VAR_SMOOTHING_VALUES):
        raise ValueError(f"Expected 13 parameter rows, got {len(cv_results)}.")
    metric_columns = [
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
    ]
    for column in metric_columns:
        if not cv_results[column].between(0, 1).all():
            raise ValueError(f"CV metric out of range: {column}")
    checked_frames = [cv_results, summary, pd.DataFrame([test_metrics]), predictions, comparison]
    for frame in checked_frames:
        if frame.isna().any().any():
            raise ValueError("NaN found in output tables.")
    for frame in checked_frames:
        numeric_values = frame.select_dtypes(include="number").to_numpy()
        if numeric_values.size and not np.isfinite(numeric_values).all():
            raise ValueError("Infinite value found in output tables.")
    if test_metrics["tn"] + test_metrics["fp"] + test_metrics["fn"] + test_metrics["tp"] != 20000:
        raise ValueError("Confusion matrix values do not sum to 20000.")
    if len(predictions) != 20000:
        raise ValueError("Prediction row count is not 20000.")
    if best_params["best_var_smoothing"] not in VAR_SMOOTHING_VALUES:
        raise ValueError("Best var_smoothing is not in the search space.")


def main():
    start_time = time.perf_counter()
    project_root = get_project_root()
    metrics_dir, figures_dir, models_dir = ensure_output_dirs(project_root)
    require_baseline_files(project_root)

    baseline_cv_summary = pd.read_csv(metrics_dir / "naive_bayes_cv_summary.csv")
    baseline_test_metrics = pd.read_csv(metrics_dir / "naive_bayes_test_metrics.csv").iloc[0]
    selected_feature_set, selection_reason = choose_feature_set_from_baseline(baseline_cv_summary)

    print("Baseline feature-set CV metrics:")
    print(baseline_cv_summary.to_string(index=False))
    print(f"Selected feature set for tuning: {selected_feature_set}")
    print(f"Selection reason: {selection_reason}")
    print(f"Parameter combinations: {len(VAR_SMOOTHING_VALUES)}")
    print(f"Total CV fits: {len(VAR_SMOOTHING_VALUES) * 5}")

    df = load_dataset()
    validate_dataset(df)
    feature_sets = get_feature_sets(df)
    splits = split_dataset(df)
    numeric_features, categorical_features = identify_feature_types(df)
    selected_features = feature_sets[selected_feature_set]
    selected_split = splits[selected_feature_set]
    feature_numeric, feature_categorical = feature_types_for_feature_set(
        selected_features, numeric_features, categorical_features
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", build_naive_bayes_preprocessor(feature_numeric, feature_categorical)),
            ("model", GaussianNB()),
        ]
    )
    grid_search = make_grid_search(pipeline)
    grid_search.fit(selected_split["X_train"], selected_split["y_train"])
    search_time = time.perf_counter() - start_time

    cv_results = clean_cv_results(grid_search)
    transformed_feature_count = len(
        grid_search.best_estimator_.named_steps["preprocessor"].get_feature_names_out()
    )
    best_params = make_best_params_dict(
        grid_search,
        cv_results,
        selected_feature_set,
        raw_feature_count=len(selected_features),
        transformed_feature_count=transformed_feature_count,
        total_search_time=search_time,
    )
    best_row = cv_results.iloc[0]
    summary = make_tuning_summary_from_row(best_params, best_row)
    test_metrics, report, predictions, y_probability = evaluate_on_test(
        grid_search.best_estimator_,
        selected_split,
        selected_feature_set,
        best_params["best_var_smoothing"],
    )
    comparison = make_baseline_comparison(baseline_test_metrics, test_metrics)

    validate_outputs(cv_results, summary, best_params, test_metrics, predictions, comparison)

    cv_results.to_csv(metrics_dir / "naive_bayes_tuning_cv_results.csv", index=False, encoding="utf-8")
    (metrics_dir / "naive_bayes_best_params.json").write_text(
        json.dumps(best_params, indent=2), encoding="utf-8"
    )
    summary.to_csv(metrics_dir / "naive_bayes_tuning_summary.csv", index=False, encoding="utf-8")
    pd.DataFrame([test_metrics]).to_csv(
        metrics_dir / "naive_bayes_tuned_test_metrics.csv", index=False, encoding="utf-8"
    )
    save_classification_report(
        metrics_dir / "naive_bayes_tuned_classification_report.txt",
        report,
        selected_feature_set,
        selection_reason,
        selected_split,
        cv_results,
        best_params,
        test_metrics,
        comparison,
    )
    predictions.to_csv(
        metrics_dir / "naive_bayes_tuned_test_predictions.csv", index=False, encoding="utf-8"
    )
    comparison.to_csv(
        metrics_dir / "naive_bayes_baseline_vs_tuned.csv", index=False, encoding="utf-8"
    )

    plot_tuning_curve(
        figures_dir / "naive_bayes_tuning_curve.png",
        cv_results.sort_values("param_model__var_smoothing"),
        best_params["best_var_smoothing"],
    )
    plot_top_results(figures_dir / "naive_bayes_tuning_top_results.png", cv_results)
    plot_confusion_matrix(figures_dir / "naive_bayes_tuned_confusion_matrix.png", test_metrics)
    plot_roc_curve(
        figures_dir / "naive_bayes_tuned_roc_curve.png",
        selected_split["y_test"],
        y_probability,
        test_metrics["roc_auc"],
    )
    plot_baseline_vs_tuned(figures_dir / "naive_bayes_baseline_vs_tuned.png", comparison)

    joblib.dump(grid_search.best_estimator_, models_dir / "naive_bayes_tuned.joblib")

    elapsed_time = time.perf_counter() - start_time
    print(f"Best var_smoothing: {best_params['best_var_smoothing']:.0e}")
    print("Best CV metrics:")
    print(
        f"Accuracy={best_params['best_cv_accuracy']:.4f}, "
        f"Precision={best_params['best_cv_precision']:.4f}, "
        f"Recall={best_params['best_cv_recall']:.4f}, "
        f"F1={best_params['best_cv_f1']:.4f}, "
        f"ROC-AUC={best_params['best_cv_roc_auc']:.4f}"
    )
    print("Tuned test metrics:")
    print(
        f"Accuracy={test_metrics['accuracy']:.4f}, Precision={test_metrics['precision']:.4f}, "
        f"Recall={test_metrics['recall']:.4f}, F1={test_metrics['f1']:.4f}, "
        f"ROC-AUC={test_metrics['roc_auc']:.4f}"
    )
    print(
        f"Confusion matrix: TN={test_metrics['tn']}, FP={test_metrics['fp']}, "
        f"FN={test_metrics['fn']}, TP={test_metrics['tp']}"
    )
    print("Baseline vs tuned changes:")
    print(comparison.to_string(index=False))
    print(f"Total runtime seconds: {elapsed_time:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
