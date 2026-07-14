from pathlib import Path
import hashlib
import json
import os
import sys
import tempfile

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "phishing_ml_matplotlib_cache"),
)

import matplotlib

matplotlib.use("Agg")

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
METRICS_DIR = PROJECT_ROOT / "results" / "metrics"
FIGURES_DIR = PROJECT_ROOT / "results" / "figures"
DATA_PATH = PROJECT_ROOT / "data" / "enterprise_phishing_simulation_2026.csv"
CLASS_LABELS = ["No", "Yes"]
METRIC_COLUMNS = ["accuracy", "precision", "recall", "f1", "roc_auc"]


MODELS = [
    {
        "key": "logistic",
        "name": "Logistic Regression",
        "method": "GridSearchCV over C and class_weight, refit by training CV F1.",
    },
    {
        "key": "naive_bayes",
        "name": "Gaussian Naive Bayes",
        "method": "GridSearchCV over var_smoothing, refit by training CV F1.",
    },
    {
        "key": "random_forest",
        "name": "Random Forest",
        "method": "RandomizedSearchCV with 30 parameter combinations, refit by training CV F1.",
    },
    {
        "key": "xgboost",
        "name": "XGBoost",
        "method": "RandomizedSearchCV with 30 parameter combinations, refit by training CV F1.",
    },
]


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def required_files():
    paths = []
    for model in MODELS:
        key = model["key"]
        paths.extend(
            [
                METRICS_DIR / f"{key}_test_metrics.csv",
                METRICS_DIR / f"{key}_tuned_test_metrics.csv",
                METRICS_DIR / f"{key}_tuning_summary.csv",
                METRICS_DIR / f"{key}_best_params.json",
                METRICS_DIR / f"{key}_tuned_test_predictions.csv",
            ]
        )
    return paths


def ensure_required_files_exist():
    missing = [str(path.relative_to(PROJECT_ROOT)) for path in required_files() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing required result files. Stop comparison without retraining: "
            + "; ".join(missing)
        )


def read_single_row_csv(path):
    df = pd.read_csv(path)
    if len(df) != 1:
        raise ValueError(f"Expected exactly one row in {path}, got {len(df)}.")
    return df.iloc[0]


def read_json(path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def safe_get(row, candidates, default="not_recorded"):
    for column in candidates:
        if column in row.index:
            value = row[column]
            if pd.isna(value):
                return default
            return value
    return default


def safe_json_get(data, candidates, default="not_recorded"):
    for key in candidates:
        if key in data and data[key] is not None:
            return data[key]
    return default


def add_derived_metrics(record):
    tn = float(record["tn"])
    fp = float(record["fp"])
    fn = float(record["fn"])
    tp = float(record["tp"])
    recall_value = float(record["recall"]) if "recall" in record else float(record["test_recall"])
    record["false_positive_rate"] = fp / (fp + tn) if (fp + tn) else np.nan
    record["false_negative_rate"] = fn / (fn + tp) if (fn + tp) else np.nan
    record["specificity"] = tn / (tn + fp) if (tn + fp) else np.nan
    record["balanced_accuracy"] = (recall_value + record["specificity"]) / 2
    return record


def standardize_high(values):
    values = pd.Series(values, dtype=float)
    min_value = values.min()
    max_value = values.max()
    if np.isclose(max_value, min_value):
        return pd.Series(np.ones(len(values)), index=values.index)
    return (values - min_value) / (max_value - min_value)


def standardize_low(values):
    values = pd.Series(values, dtype=float)
    min_value = values.min()
    max_value = values.max()
    if np.isclose(max_value, min_value):
        return pd.Series(np.ones(len(values)), index=values.index)
    return (max_value - values) / (max_value - min_value)


def validate_predictions(model_name, predictions, reported_metrics):
    expected_columns = {"sample_index", "y_true", "y_pred", "y_probability"}
    missing_columns = expected_columns - set(predictions.columns)
    if missing_columns:
        raise ValueError(f"{model_name} prediction file missing columns: {sorted(missing_columns)}")
    if len(predictions) != 20000:
        raise ValueError(f"{model_name} prediction row count is {len(predictions)}, expected 20000.")
    if predictions["sample_index"].duplicated().any():
        raise ValueError(f"{model_name} prediction file contains duplicated sample_index values.")
    if predictions.replace([np.inf, -np.inf], np.nan).isna().any().any():
        raise ValueError(f"{model_name} prediction file contains NaN or infinite values.")
    if not set(predictions["y_pred"].unique()).issubset({0, 1}):
        raise ValueError(f"{model_name} y_pred contains values outside 0 and 1.")
    if not set(predictions["y_true"].unique()).issubset({0, 1}):
        raise ValueError(f"{model_name} y_true contains values outside 0 and 1.")
    if not predictions["y_probability"].between(0, 1).all():
        raise ValueError(f"{model_name} y_probability contains values outside [0, 1].")

    y_true = predictions["y_true"].to_numpy()
    y_pred = predictions["y_pred"].to_numpy()
    y_probability = predictions["y_probability"].to_numpy()
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    recalculated = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        "recall": recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        "f1": f1_score(y_true, y_pred, pos_label=1, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_probability),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }
    if int(tn + fp + fn + tp) != 20000:
        raise ValueError(f"{model_name} confusion matrix total is not 20000.")

    for metric in METRIC_COLUMNS:
        if not np.isclose(float(reported_metrics[metric]), recalculated[metric], atol=1e-6):
            raise ValueError(
                f"{model_name} {metric} mismatch: reported={reported_metrics[metric]}, "
                f"recalculated={recalculated[metric]}"
            )
    for metric in ["tn", "fp", "fn", "tp"]:
        if int(reported_metrics[metric]) != recalculated[metric]:
            raise ValueError(
                f"{model_name} {metric} mismatch: reported={reported_metrics[metric]}, "
                f"recalculated={recalculated[metric]}"
            )
    return recalculated


def normalize_prediction_columns(model_name, predictions):
    rename_map = {}
    if "sample_index" not in predictions.columns and "index" in predictions.columns:
        rename_map["index"] = "sample_index"
    if "y_probability" not in predictions.columns and "y_proba_yes" in predictions.columns:
        rename_map["y_proba_yes"] = "y_probability"
    if rename_map:
        print(f"{model_name} prediction columns mapped in memory: {rename_map}")
        predictions = predictions.rename(columns=rename_map)
    return predictions


def validate_prediction_consistency(prediction_map):
    reference_name = MODELS[0]["name"]
    reference = prediction_map[reference_name]
    for model in MODELS[1:]:
        name = model["name"]
        current = prediction_map[name]
        if not reference["sample_index"].equals(current["sample_index"]):
            raise ValueError(f"{name} sample_index does not match {reference_name}.")
        if not reference["y_true"].equals(current["y_true"]):
            raise ValueError(f"{name} y_true does not match {reference_name}.")


def collect_results():
    all_rows = []
    tuned_rows = []
    change_rows = []
    predictions = {}
    best_params = {}

    for model in MODELS:
        key = model["key"]
        name = model["name"]
        baseline = read_single_row_csv(METRICS_DIR / f"{key}_test_metrics.csv")
        tuned = read_single_row_csv(METRICS_DIR / f"{key}_tuned_test_metrics.csv")
        summary = read_single_row_csv(METRICS_DIR / f"{key}_tuning_summary.csv")
        best_json = read_json(METRICS_DIR / f"{key}_best_params.json")
        best_params[name] = best_json.get("best_params", {})

        for version, source in [("baseline", baseline), ("tuned", tuned)]:
            row = {
                "model": name,
                "version": version,
                "selected_feature_set": safe_get(
                    source,
                    ["selected_feature_set", "feature_set"],
                ),
            }
            for metric in METRIC_COLUMNS + ["tn", "fp", "fn", "tp"]:
                row[metric] = source[metric]
            all_rows.append(add_derived_metrics(row))

        cv_row = {
            "model": name,
            "selected_feature_set": safe_get(
                summary,
                ["selected_feature_set", "feature_set"],
            ),
            "cv_accuracy_mean": safe_get(summary, ["cv_accuracy_mean", "accuracy_mean"]),
            "cv_accuracy_std": safe_get(summary, ["cv_accuracy_std", "accuracy_std"]),
            "cv_precision_mean": safe_get(summary, ["cv_precision_mean", "precision_mean"]),
            "cv_precision_std": safe_get(summary, ["cv_precision_std", "precision_std"]),
            "cv_recall_mean": safe_get(summary, ["cv_recall_mean", "recall_mean"]),
            "cv_recall_std": safe_get(summary, ["cv_recall_std", "recall_std"]),
            "cv_f1_mean": safe_get(summary, ["cv_f1_mean", "f1_mean"]),
            "cv_f1_std": safe_get(summary, ["cv_f1_std", "f1_std"]),
            "cv_roc_auc_mean": safe_get(summary, ["cv_roc_auc_mean", "roc_auc_mean"]),
            "cv_roc_auc_std": safe_get(summary, ["cv_roc_auc_std", "roc_auc_std"]),
            "test_accuracy": tuned["accuracy"],
            "test_precision": tuned["precision"],
            "test_recall": tuned["recall"],
            "test_f1": tuned["f1"],
            "test_roc_auc": tuned["roc_auc"],
            "tn": tuned["tn"],
            "fp": tuned["fp"],
            "fn": tuned["fn"],
            "tp": tuned["tp"],
            "total_search_time_seconds": safe_get(
                summary,
                ["total_search_time_seconds", "total_search_time"],
                safe_json_get(best_json, ["total_search_time_seconds"], "not_recorded"),
            ),
        }
        tuned_rows.append(add_derived_metrics(cv_row))

        changes = {"model": name}
        for metric in METRIC_COLUMNS + ["tn", "fp", "fn", "tp"]:
            changes[f"{metric}_change"] = float(tuned[metric]) - float(baseline[metric])
        change_rows.append(changes)

        prediction_df = pd.read_csv(METRICS_DIR / f"{key}_tuned_test_predictions.csv")
        prediction_df = normalize_prediction_columns(name, prediction_df)
        validate_predictions(name, prediction_df, tuned)
        predictions[name] = prediction_df

    validate_prediction_consistency(predictions)
    return (
        pd.DataFrame(all_rows),
        pd.DataFrame(tuned_rows),
        pd.DataFrame(change_rows),
        predictions,
        best_params,
    )


def make_ranking(tuned_df):
    ranking = tuned_df[
        [
            "model",
            "cv_f1_mean",
            "cv_roc_auc_mean",
            "cv_recall_mean",
            "test_f1",
            "test_roc_auc",
            "test_recall",
            "test_precision",
            "test_accuracy",
            "fn",
        ]
    ].copy()
    ranking["cv_f1_rank"] = ranking["cv_f1_mean"].rank(ascending=False, method="average")
    ranking["cv_roc_auc_rank"] = ranking["cv_roc_auc_mean"].rank(ascending=False, method="average")
    ranking["cv_recall_rank"] = ranking["cv_recall_mean"].rank(ascending=False, method="average")
    ranking["test_f1_rank"] = ranking["test_f1"].rank(ascending=False, method="average")
    ranking["test_roc_auc_rank"] = ranking["test_roc_auc"].rank(ascending=False, method="average")
    ranking["test_recall_rank"] = ranking["test_recall"].rank(ascending=False, method="average")
    ranking["test_precision_rank"] = ranking["test_precision"].rank(ascending=False, method="average")
    ranking["test_accuracy_rank"] = ranking["test_accuracy"].rank(ascending=False, method="average")
    ranking["fn_rank"] = ranking["fn"].rank(ascending=True, method="average")

    ranking["norm_cv_f1"] = standardize_high(ranking["cv_f1_mean"])
    ranking["norm_test_f1"] = standardize_high(ranking["test_f1"])
    ranking["norm_test_recall"] = standardize_high(ranking["test_recall"])
    ranking["norm_test_roc_auc"] = standardize_high(ranking["test_roc_auc"])
    ranking["norm_low_fn_advantage"] = standardize_low(ranking["fn"])
    ranking["composite_reference_score"] = (
        0.35 * ranking["norm_cv_f1"]
        + 0.25 * ranking["norm_test_f1"]
        + 0.20 * ranking["norm_test_recall"]
        + 0.10 * ranking["norm_test_roc_auc"]
        + 0.10 * ranking["norm_low_fn_advantage"]
    )
    ranking["composite_rank"] = ranking["composite_reference_score"].rank(
        ascending=False, method="average"
    )
    return ranking.sort_values("composite_rank").reset_index(drop=True)


def annotate_bars(ax, bars, rotation=0):
    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            f"{height:.3f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
            rotation=rotation,
        )


def plot_tuned_metrics(tuned_df):
    metrics = ["test_accuracy", "test_precision", "test_recall", "test_f1", "test_roc_auc"]
    labels = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
    x = np.arange(len(metrics))
    width = 0.18
    fig, ax = plt.subplots(figsize=(12, 6))
    for offset_index, (_, row) in enumerate(tuned_df.iterrows()):
        bars = ax.bar(
            x + (offset_index - 1.5) * width,
            [row[metric] for metric in metrics],
            width,
            label=row["model"],
        )
        annotate_bars(ax, bars, rotation=90)
    ax.set_title("Tuned Model Test Metrics Comparison")
    ax.set_xlabel("Metric")
    ax.set_ylabel("Score")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.08)
    ax.legend(loc="lower right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "tuned_model_metrics_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_roc_comparison(predictions, tuned_df):
    fig, ax = plt.subplots(figsize=(7, 6))
    for _, row in tuned_df.iterrows():
        name = row["model"]
        pred = predictions[name]
        fpr, tpr, _ = roc_curve(pred["y_true"], pred["y_probability"])
        ax.plot(fpr, tpr, label=f"{name} (AUC = {row['test_roc_auc']:.4f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random Classifier")
    ax.set_title("Tuned Model ROC Curves")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "tuned_model_roc_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_baseline_vs_tuned(all_df):
    metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    labels = ["Acc", "Prec", "Recall", "F1", "AUC"]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharey=True)
    axes = axes.ravel()
    for ax, model_name in zip(axes, [model["name"] for model in MODELS]):
        subset = all_df[all_df["model"] == model_name].set_index("version")
        x = np.arange(len(metrics))
        width = 0.35
        ax.bar(x - width / 2, [subset.loc["baseline", metric] for metric in metrics], width, label="Baseline")
        ax.bar(x + width / 2, [subset.loc["tuned", metric] for metric in metrics], width, label="Tuned")
        ax.set_title(model_name)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylim(0, 1)
        ax.grid(axis="y", alpha=0.3)
    axes[0].legend(loc="lower right")
    fig.suptitle("Baseline vs Tuned Metrics by Model")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "baseline_vs_tuned_all_models.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_counts(tuned_df):
    x = np.arange(len(tuned_df))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 6))
    bars_fp = ax.bar(x - width / 2, tuned_df["fp"], width, label="FP")
    bars_fn = ax.bar(x + width / 2, tuned_df["fn"], width, label="FN")
    for bars in [bars_fp, bars_fn]:
        for bar in bars:
            ax.annotate(
                f"{int(bar.get_height())}",
                xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    ax.set_title("Tuned Model False Positive and False Negative Counts")
    ax.set_xlabel("Model")
    ax.set_ylabel("Count")
    ax.set_xticks(x)
    ax.set_xticklabels(tuned_df["model"], rotation=15, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "confusion_counts_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_cv_vs_test_f1(tuned_df):
    x = np.arange(len(tuned_df))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 6))
    bars_cv = ax.bar(
        x - width / 2,
        tuned_df["cv_f1_mean"],
        width,
        yerr=tuned_df["cv_f1_std"],
        capsize=4,
        label="CV F1",
    )
    bars_test = ax.bar(x + width / 2, tuned_df["test_f1"], width, label="Test F1")
    annotate_bars(ax, bars_cv, rotation=90)
    annotate_bars(ax, bars_test, rotation=90)
    ax.set_title("CV F1 vs Test F1")
    ax.set_xlabel("Model")
    ax.set_ylabel("F1 Score")
    ax.set_xticks(x)
    ax.set_xticklabels(tuned_df["model"], rotation=15, ha="right")
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "cv_vs_test_f1.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_ranking_reference(ranking):
    components = [
        "norm_cv_f1",
        "norm_test_f1",
        "norm_test_recall",
        "norm_test_roc_auc",
        "norm_low_fn_advantage",
    ]
    labels = ["CV F1", "Test F1", "Recall", "ROC-AUC", "Low FN"]
    x = np.arange(len(components))
    width = 0.18
    fig, ax = plt.subplots(figsize=(12, 6))
    for offset_index, (_, row) in enumerate(ranking.sort_values("model").iterrows()):
        ax.bar(
            x + (offset_index - 1.5) * width,
            [row[component] for component in components],
            width,
            label=row["model"],
        )
    ax.set_title("Multi-Metric Reference Comparison (Not Statistical Significance)")
    ax.set_xlabel("Reference Component")
    ax.set_ylabel("Min-Max Normalized Score")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "model_ranking_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def assert_figure_valid(path):
    if not path.exists() or path.stat().st_size == 0:
        raise ValueError(f"Figure was not created or is empty: {path}")
    image = mpimg.imread(path)
    if image.size == 0 or image.shape[0] < 10 or image.shape[1] < 10:
        raise ValueError(f"Figure appears invalid: {path}")
    if float(np.std(image)) == 0.0:
        raise ValueError(f"Figure appears blank: {path}")


def format_best_params(best_params):
    lines = []
    for name, params in best_params.items():
        if not params:
            lines.append(f"- {name}: not recorded")
        else:
            lines.append(f"- {name}: {json.dumps(params, ensure_ascii=False, sort_keys=True)}")
    return "\n".join(lines)


def top_model(df, metric, higher_is_better=True):
    idx = df[metric].idxmax() if higher_is_better else df[metric].idxmin()
    return df.loc[idx, "model"], df.loc[idx, metric]


def write_findings(all_df, tuned_df, changes, ranking, best_params):
    accuracy_top = top_model(tuned_df, "test_accuracy")
    precision_top = top_model(tuned_df, "test_precision")
    recall_top = top_model(tuned_df, "test_recall")
    f1_top = top_model(tuned_df, "test_f1")
    roc_top = top_model(tuned_df, "test_roc_auc")
    fn_top = top_model(tuned_df, "fn", higher_is_better=False)
    stable_top = top_model(tuned_df, "cv_f1_std", higher_is_better=False)
    composite_top = ranking.sort_values("composite_rank").iloc[0]
    high_recall_top = tuned_df.sort_values(["test_recall", "fn"], ascending=[False, True]).iloc[0]
    low_fp_top = tuned_df.sort_values(["test_precision", "fp"], ascending=[False, True]).iloc[0]

    lines = [
        "Model Comparison Findings",
        "=" * 80,
        "Data sources:",
        "- Baseline test metrics, tuned test metrics, tuning summaries, best-parameter JSON files, and tuned test prediction files for four models.",
        "- No model was retrained and no hyperparameter search was rerun in this comparison stage.",
        "- The dataset is synthetic simulated data; results support method feasibility checks and should not be treated as direct evidence for real enterprise deployment effects.",
        "",
        "Tuning methods:",
    ]
    for model in MODELS:
        lines.append(f"- {model['name']}: {model['method']}")
    lines.extend(["", "Best parameter summary:", format_best_params(best_params), ""])

    lines.append("Tuned model CV metrics:")
    for _, row in tuned_df.iterrows():
        lines.append(
            f"- {row['model']}: Accuracy={row['cv_accuracy_mean']:.6f} +/- {row['cv_accuracy_std']:.6f}, "
            f"Precision={row['cv_precision_mean']:.6f} +/- {row['cv_precision_std']:.6f}, "
            f"Recall={row['cv_recall_mean']:.6f} +/- {row['cv_recall_std']:.6f}, "
            f"F1={row['cv_f1_mean']:.6f} +/- {row['cv_f1_std']:.6f}, "
            f"ROC-AUC={row['cv_roc_auc_mean']:.6f} +/- {row['cv_roc_auc_std']:.6f}."
        )
    lines.extend(["", "Tuned model test metrics:"])
    for _, row in tuned_df.iterrows():
        lines.append(
            f"- {row['model']}: Accuracy={row['test_accuracy']:.6f}, "
            f"Precision={row['test_precision']:.6f}, Recall={row['test_recall']:.6f}, "
            f"F1={row['test_f1']:.6f}, ROC-AUC={row['test_roc_auc']:.6f}, "
            f"TN={int(row['tn'])}, FP={int(row['fp'])}, FN={int(row['fn'])}, TP={int(row['tp'])}."
        )
    lines.extend(["", "Baseline to tuned changes:"])
    for _, row in changes.iterrows():
        lines.append(
            f"- {row['model']}: Accuracy {row['accuracy_change']:+.6f}, "
            f"Precision {row['precision_change']:+.6f}, Recall {row['recall_change']:+.6f}, "
            f"F1 {row['f1_change']:+.6f}, ROC-AUC {row['roc_auc_change']:+.6f}, "
            f"FN {int(row['fn_change']):+d}, FP {int(row['fp_change']):+d}."
        )
    lines.extend(
        [
            "",
            f"Highest Accuracy: {accuracy_top[0]} ({accuracy_top[1]:.6f})",
            f"Highest Precision: {precision_top[0]} ({precision_top[1]:.6f})",
            f"Highest Recall: {recall_top[0]} ({recall_top[1]:.6f})",
            f"Highest F1: {f1_top[0]} ({f1_top[1]:.6f})",
            f"Highest ROC-AUC: {roc_top[0]} ({roc_top[1]:.6f})",
            f"Fewest FN: {fn_top[0]} ({int(fn_top[1])})",
            f"Most stable CV F1: {stable_top[0]} (std={stable_top[1]:.6f})",
            "",
            "Model selection notes:",
            f"- Comprehensive reference recommendation: {composite_top['model']}. It has the highest auxiliary composite score, but the score is only a course-project summary aid.",
            f"- Main CV-F1 view: {top_model(tuned_df, 'cv_f1_mean')[0]} is numerically highest on training CV F1.",
            f"- High Recall / low missed-failure scenario: {high_recall_top['model']} is preferred because it has the highest test Recall and the fewest FN.",
            f"- Low false-alarm / high Precision scenario: {low_fp_top['model']} is preferred because it has the highest test Precision among tuned models.",
            "- Interpretability and deployment efficiency: Logistic Regression is preferred because coefficients are easier to explain and inference is lightweight; Gaussian Naive Bayes is also simple but has weaker overall metrics here.",
            "- The tuned class-weight settings in Logistic Regression, Random Forest, and XGBoost increased Recall and reduced FN, while also increasing FP and lowering Precision compared with their baselines.",
            "- Several tuned models are close in F1 and ROC-AUC, so the differences should be described as numerical differences rather than statistical superiority.",
            "- The test set is not used for further tuning after this comparison.",
            "",
            "Composite reference score explanation:",
            "- The composite score is only an auxiliary summary for this course experiment.",
            "- The weights reflect the project focus on F1, Recall, and reducing missed Yes cases.",
            "- The score is not a universal, objective, or statistically definitive model-selection criterion.",
        ]
    )
    (METRICS_DIR / "model_comparison_findings.txt").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    csv_hash_before = sha256_file(DATA_PATH)

    ensure_required_files_exist()
    all_df, tuned_df, changes, predictions, best_params = collect_results()
    ranking = make_ranking(tuned_df)

    all_df.to_csv(METRICS_DIR / "model_comparison_all.csv", index=False, encoding="utf-8")
    tuned_df.to_csv(METRICS_DIR / "tuned_model_comparison.csv", index=False, encoding="utf-8")
    changes.to_csv(METRICS_DIR / "baseline_tuning_changes.csv", index=False, encoding="utf-8")
    ranking.to_csv(METRICS_DIR / "model_ranking.csv", index=False, encoding="utf-8")
    write_findings(all_df, tuned_df, changes, ranking, best_params)

    plot_tuned_metrics(tuned_df)
    plot_roc_comparison(predictions, tuned_df)
    plot_baseline_vs_tuned(all_df)
    plot_confusion_counts(tuned_df)
    plot_cv_vs_test_f1(tuned_df)
    plot_ranking_reference(ranking)

    figure_paths = [
        FIGURES_DIR / "tuned_model_metrics_comparison.png",
        FIGURES_DIR / "tuned_model_roc_comparison.png",
        FIGURES_DIR / "baseline_vs_tuned_all_models.png",
        FIGURES_DIR / "confusion_counts_comparison.png",
        FIGURES_DIR / "cv_vs_test_f1.png",
        FIGURES_DIR / "model_ranking_comparison.png",
    ]
    for path in figure_paths:
        assert_figure_valid(path)

    csv_hash_after = sha256_file(DATA_PATH)
    if csv_hash_before != csv_hash_after:
        raise AssertionError("Original CSV SHA256 changed during comparison.")
    if len(all_df) != 8:
        raise AssertionError(f"model_comparison_all.csv should have 8 rows, got {len(all_df)}.")
    if len(tuned_df) != 4:
        raise AssertionError(f"tuned_model_comparison.csv should have 4 rows, got {len(tuned_df)}.")

    print("Model comparison completed successfully.")
    print("Rows generated: model_comparison_all=8, tuned_model_comparison=4.")
    print("Prediction consistency checks passed for sample_index, y_true, metrics, and confusion totals.")
    print(f"Original CSV SHA256 unchanged: {csv_hash_before == csv_hash_after}")
    print("Composite ranking:")
    print(ranking[["model", "composite_reference_score", "composite_rank"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
