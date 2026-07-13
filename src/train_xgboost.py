from pathlib import Path
import hashlib
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
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline

try:
    import xgboost
    from xgboost import XGBClassifier
except ImportError as exc:
    print(
        "Error: xgboost is not installed in the current Python environment.",
        file=sys.stderr,
    )
    raise

from data_preprocessing import (
    RANDOM_SEED,
    TARGET_COLUMN,
    build_tree_preprocessor,
    get_feature_sets,
    get_project_root,
    load_dataset,
    split_dataset,
    validate_dataset,
)


DATA_RELATIVE_PATH = Path("data") / "enterprise_phishing_simulation_2026.csv"
METRICS_DIR = Path("results") / "metrics"
FIGURES_DIR = Path("results") / "figures"
MODELS_DIR = Path("models")
MODEL_NAME = "XGBoost"
CLASS_LABELS = ["No", "Yes"]


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def ensure_directories(project_root):
    paths = {
        "metrics": project_root / METRICS_DIR,
        "figures": project_root / FIGURES_DIR,
        "models": project_root / MODELS_DIR,
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def split_features_by_type(df, features):
    numeric_features = [
        feature for feature in features if pd.api.types.is_numeric_dtype(df[feature])
    ]
    categorical_features = [
        feature for feature in features if feature not in numeric_features
    ]
    return numeric_features, categorical_features


def build_xgboost():
    return XGBClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=6,
        min_child_weight=1,
        gamma=0.0,
        subsample=1.0,
        colsample_bytree=1.0,
        reg_alpha=0.0,
        reg_lambda=1.0,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        importance_type="gain",
        random_state=RANDOM_SEED,
        n_jobs=-1,
        verbosity=1,
    )


def build_pipeline(numeric_features, categorical_features):
    return Pipeline(
        [
            ("preprocessor", build_tree_preprocessor(numeric_features, categorical_features)),
            ("model", build_xgboost()),
        ]
    )


def fit_preprocessor_for_metadata(split, numeric_features, categorical_features):
    preprocessor = build_tree_preprocessor(numeric_features, categorical_features)
    preprocessor.fit(split["X_train"])
    return preprocessor


def run_cross_validation(feature_set_name, features, df, split):
    numeric_features, categorical_features = split_features_by_type(df, features)
    pipeline = build_pipeline(numeric_features, categorical_features)
    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_SEED,
    )
    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc",
    }

    print(f"Starting 5-fold cross-validation for {feature_set_name} feature set.")
    cv_results = cross_validate(
        pipeline,
        split["X_train"],
        split["y_train"],
        cv=cv,
        scoring=scoring,
        n_jobs=1,
        error_score="raise",
        return_train_score=False,
    )
    print(f"Completed 5-fold cross-validation for {feature_set_name} feature set.")

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

    fold_df = pd.DataFrame(fold_rows)
    preprocessor = fit_preprocessor_for_metadata(
        split,
        numeric_features,
        categorical_features,
    )
    transformed_count = len(preprocessor.get_feature_names_out())

    summary = {
        "model": MODEL_NAME,
        "feature_set": feature_set_name,
        "raw_feature_count": len(features),
        "transformed_feature_count": transformed_count,
    }
    for metric in ["accuracy", "precision", "recall", "f1", "roc_auc", "fit_time", "score_time"]:
        summary[f"{metric}_mean"] = fold_df[metric].mean()
        summary[f"{metric}_std"] = fold_df[metric].std()

    print(
        f"{feature_set_name} CV mean/std: "
        f"accuracy={summary['accuracy_mean']:.4f}/{summary['accuracy_std']:.4f}, "
        f"precision={summary['precision_mean']:.4f}/{summary['precision_std']:.4f}, "
        f"recall={summary['recall_mean']:.4f}/{summary['recall_std']:.4f}, "
        f"f1={summary['f1_mean']:.4f}/{summary['f1_std']:.4f}, "
        f"roc_auc={summary['roc_auc_mean']:.4f}/{summary['roc_auc_std']:.4f}, "
        f"fit_time_mean={summary['fit_time_mean']:.2f}s"
    )

    return fold_df, summary


def choose_feature_set(summary_df):
    ranked = summary_df.sort_values(
        by=["f1_mean", "roc_auc_mean", "recall_mean", "raw_feature_count"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)
    selected = ranked.loc[0, "feature_set"]
    row = ranked.loc[0]
    reason = (
        f"Selected {selected} using training CV only: it ranked first by "
        f"mean F1 ({row['f1_mean']:.4f}); ties would be resolved by ROC-AUC, "
        f"then Yes-class Recall, then fewer raw features."
    )
    return selected, reason


def evaluate_final_model(model, split):
    y_pred = model.predict(split["X_test"])
    y_proba = model.predict_proba(split["X_test"])[:, 1]
    cm = confusion_matrix(split["y_test"], y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return {
        "accuracy": accuracy_score(split["y_test"], y_pred),
        "precision": precision_score(split["y_test"], y_pred, pos_label=1, zero_division=0),
        "recall": recall_score(split["y_test"], y_pred, pos_label=1, zero_division=0),
        "f1": f1_score(split["y_test"], y_pred, pos_label=1, zero_division=0),
        "roc_auc": roc_auc_score(split["y_test"], y_proba),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "confusion_matrix": cm,
        "y_proba": y_proba,
        "classification_report": classification_report(
            split["y_test"],
            y_pred,
            labels=[0, 1],
            target_names=CLASS_LABELS,
            digits=4,
            zero_division=0,
        ),
    }


def transformed_to_original_mapping(preprocessor, numeric_features, categorical_features):
    transformed_features = preprocessor.get_feature_names_out()
    mapping = {}

    numeric_names = [f"numeric__{feature}" for feature in numeric_features]
    for name, original in zip(numeric_names, numeric_features):
        mapping[name] = original

    if categorical_features:
        categorical_pipeline = preprocessor.named_transformers_["categorical"]
        encoder = categorical_pipeline.named_steps["onehot"]
        for original_feature, categories in zip(categorical_features, encoder.categories_):
            for category in categories:
                transformed_feature = f"categorical__{original_feature}_{category}"
                mapping[transformed_feature] = original_feature

    if len(mapping) != len(transformed_features):
        missing = sorted(set(transformed_features) - set(mapping))
        extra = sorted(set(mapping) - set(transformed_features))
        raise ValueError(
            f"Transformed feature mapping mismatch. Missing={missing}; extra={extra}"
        )
    if len(set(mapping)) != len(transformed_features):
        raise ValueError("Duplicate transformed feature names were found in mapping.")
    return mapping


def make_feature_importance(model, numeric_features, categorical_features):
    preprocessor = model.named_steps["preprocessor"]
    classifier = model.named_steps["model"]
    transformed_features = preprocessor.get_feature_names_out()
    importances = classifier.feature_importances_
    total_importance = importances.sum()
    if total_importance > 0:
        normalized_importances = importances / total_importance
    else:
        normalized_importances = importances

    importance_df = pd.DataFrame(
        {
            "transformed_feature": transformed_features,
            "importance": normalized_importances,
        }
    )
    importance_df["importance_percentage"] = importance_df["importance"] * 100
    importance_df = importance_df.sort_values("importance", ascending=False)
    importance_df.insert(0, "rank", range(1, len(importance_df) + 1))

    mapping = transformed_to_original_mapping(
        preprocessor,
        numeric_features,
        categorical_features,
    )
    original_df = importance_df.copy()
    original_df["original_feature"] = original_df["transformed_feature"].map(mapping)
    if original_df["original_feature"].isna().any():
        raise ValueError("Some transformed features did not map to original features.")
    original_df = (
        original_df.groupby("original_feature", as_index=False)["importance"]
        .sum()
        .sort_values("importance", ascending=False)
    )
    original_df["importance_percentage"] = (
        original_df["importance"] / original_df["importance"].sum() * 100
    )
    original_df.insert(0, "rank", range(1, len(original_df) + 1))
    return importance_df, original_df


def plot_confusion_matrix(cm, output_path):
    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(cm, cmap="Blues")
    ax.set_title("XGBoost Test Confusion Matrix")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(CLASS_LABELS)
    ax.set_yticklabels(CLASS_LABELS)
    labels = [["TN", "FP"], ["FN", "TP"]]
    threshold = cm.max() / 2
    for row in range(2):
        for column in range(2):
            color = "white" if cm[row, column] > threshold else "black"
            ax.text(
                column,
                row,
                f"{labels[row][column]}\n{cm[row, column]:,}",
                ha="center",
                va="center",
                color=color,
                fontsize=12,
            )
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_roc_curve(y_test, y_proba, auc_value, output_path):
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, label=f"XGBoost (AUC = {auc_value:.4f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random Classifier")
    ax.set_title("XGBoost Test ROC Curve")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_cv_comparison(summary_df, output_path):
    metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    x = np.arange(len(metrics))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 5))
    for offset, feature_set in [(-width / 2, "full"), (width / 2, "reduced")]:
        row = summary_df[summary_df["feature_set"] == feature_set].iloc[0]
        means = [row[f"{metric}_mean"] for metric in metrics]
        stds = [row[f"{metric}_std"] for metric in metrics]
        ax.bar(x + offset, means, width, yerr=stds, capsize=4, label=feature_set)
    ax.set_title("XGBoost 5-Fold CV Comparison")
    ax.set_xlabel("Metric")
    ax.set_ylabel("Mean Score")
    ax.set_xticks(x)
    ax.set_xticklabels(["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"])
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_feature_importance(original_importance_df, output_path):
    top_df = original_importance_df.head(15).sort_values("importance", ascending=True)
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(top_df["original_feature"], top_df["importance"], color="#4C78A8")
    ax.set_title("XGBoost Original Feature Gain Importance")
    ax.set_xlabel("Gain Importance")
    ax.set_ylabel("Original Feature")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def assert_figure_valid(path):
    if not path.exists() or path.stat().st_size == 0:
        raise ValueError(f"Figure was not created or is empty: {path}")
    image = mpimg.imread(path)
    if image.size == 0 or image.shape[0] < 10 or image.shape[1] < 10:
        raise ValueError(f"Figure appears invalid: {path}")


def fixed_parameter_text():
    return "\n".join(
        [
            "XGBClassifier(",
            "    n_estimators=200, learning_rate=0.1, max_depth=6,",
            "    min_child_weight=1, gamma=0.0, subsample=1.0,",
            "    colsample_bytree=1.0, reg_alpha=0.0, reg_lambda=1.0,",
            "    objective='binary:logistic', eval_metric='logloss',",
            "    tree_method='hist', importance_type='gain', random_state=42,",
            "    n_jobs=-1, verbosity=1",
            ")",
        ]
    )


def write_report(
    report_path,
    cv_summary,
    selected_feature_set,
    selection_reason,
    test_metrics,
    train_size,
    test_size,
    csv_hash_before,
    csv_hash_after,
    total_runtime,
):
    lines = []
    lines.append("XGBoost Baseline Experiment")
    lines.append("=" * 80)
    lines.append(f"XGBoost version: {xgboost.__version__}")
    lines.append(f"Target column: {TARGET_COLUMN}")
    lines.append("Positive class: Yes / 1")
    lines.append(f"Random seed: {RANDOM_SEED}")
    lines.append(f"Train samples: {train_size}")
    lines.append(f"Test samples: {test_size}")
    lines.append("")
    lines.append("Fixed baseline parameters")
    lines.append(fixed_parameter_text())
    lines.append("")
    lines.append("Method")
    lines.append("- CV: StratifiedKFold(n_splits=5, shuffle=True, random_state=42).")
    lines.append("- cross_validate n_jobs=1 and error_score='raise'.")
    lines.append("- XGBClassifier n_jobs=-1.")
    lines.append("- No GridSearchCV, RandomizedSearchCV, manual parameter trials, early stopping, scale_pos_weight, sample_weight, class_weight, SMOTE, or threshold tuning.")
    lines.append("- Test set was evaluated exactly once after selecting the feature set using training CV only.")
    lines.append("")
    lines.append("Cross-validation summary")
    for _, row in cv_summary.iterrows():
        lines.append(
            f"- {row['feature_set']}: "
            f"accuracy={row['accuracy_mean']:.4f} +/- {row['accuracy_std']:.4f}, "
            f"precision={row['precision_mean']:.4f} +/- {row['precision_std']:.4f}, "
            f"recall={row['recall_mean']:.4f} +/- {row['recall_std']:.4f}, "
            f"f1={row['f1_mean']:.4f} +/- {row['f1_std']:.4f}, "
            f"roc_auc={row['roc_auc_mean']:.4f} +/- {row['roc_auc_std']:.4f}, "
            f"fit_time={row['fit_time_mean']:.2f}s +/- {row['fit_time_std']:.2f}s."
        )
    lines.append("")
    lines.append(f"Selected feature set: {selected_feature_set}")
    lines.append(f"Selection basis: {selection_reason}")
    lines.append("")
    lines.append("Final test metrics")
    lines.append(f"- Accuracy: {test_metrics['accuracy']:.4f}")
    lines.append(f"- Precision (Yes): {test_metrics['precision']:.4f}")
    lines.append(f"- Recall (Yes): {test_metrics['recall']:.4f}")
    lines.append(f"- F1-score (Yes): {test_metrics['f1']:.4f}")
    lines.append(f"- ROC-AUC: {test_metrics['roc_auc']:.4f}")
    lines.append(
        f"- TN={test_metrics['tn']}, FP={test_metrics['fp']}, "
        f"FN={test_metrics['fn']}, TP={test_metrics['tp']}"
    )
    lines.append("")
    lines.append("Classification report")
    lines.append(test_metrics["classification_report"])
    lines.append("")
    lines.append("Feature importance notes")
    lines.append("- Gain importance reflects predictive contribution in tree splits.")
    lines.append("- It does not represent strict causality.")
    lines.append("- Highly correlated features may split importance across one another.")
    lines.append("")
    lines.append(f"Original CSV SHA256 before run: {csv_hash_before}")
    lines.append(f"Original CSV SHA256 after run:  {csv_hash_after}")
    lines.append(f"Original CSV unchanged: {csv_hash_before == csv_hash_after}")
    lines.append(f"Total runtime seconds: {total_runtime:.2f}")
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_checks(
    fold_df,
    cv_summary,
    test_metrics,
    transformed_importance_df,
    original_importance_df,
    selected_transformed_count,
    figure_paths,
    csv_hash_before,
    csv_hash_after,
):
    checks = []
    checks.append(("two_feature_sets_each_have_5_folds", fold_df.groupby("feature_set").size().to_dict() == {"full": 5, "reduced": 5}))
    checks.append(("total_cv_fits_is_10", len(fold_df) == 10))
    metric_columns = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    checks.append(("cv_metrics_between_0_and_1", bool(fold_df[metric_columns].apply(lambda col: col.between(0, 1).all()).all())))
    checks.append(("test_metrics_between_0_and_1", all(0 <= test_metrics[key] <= 1 for key in ["accuracy", "precision", "recall", "f1", "roc_auc"])))
    checks.append(("no_nan_inf_or_empty_results", not fold_df.replace([np.inf, -np.inf], np.nan).isna().any().any() and not cv_summary.replace([np.inf, -np.inf], np.nan).isna().any().any()))
    checks.append(("confusion_matrix_sum_is_20000", test_metrics["tn"] + test_metrics["fp"] + test_metrics["fn"] + test_metrics["tp"] == 20000))
    checks.append(("transformed_importance_count_matches", len(transformed_importance_df) == selected_transformed_count))
    checks.append(("transformed_importance_sum_close_to_1", np.isclose(transformed_importance_df["importance"].sum(), 1.0, atol=1e-8)))
    checks.append(("original_importance_sum_close_to_1", np.isclose(original_importance_df["importance"].sum(), 1.0, atol=1e-8)))
    checks.append(("no_duplicate_transformed_features", not transformed_importance_df["transformed_feature"].duplicated().any()))
    checks.append(("csv_sha256_unchanged", csv_hash_before == csv_hash_after))
    for figure_path in figure_paths:
        assert_figure_valid(figure_path)
    checks.append(("all_four_figures_created", all(path.exists() for path in figure_paths)))

    failed = [name for name, passed in checks if not passed]
    if failed:
        raise AssertionError(f"Post-run checks failed: {failed}")
    return checks


def main():
    start_time = time.perf_counter()
    project_root = get_project_root()
    paths = ensure_directories(project_root)
    data_path = project_root / DATA_RELATIVE_PATH
    csv_hash_before = sha256_file(data_path)

    print(f"XGBoost version: {xgboost.__version__}")

    df = load_dataset()
    validate_dataset(df)
    feature_sets = get_feature_sets(df)
    splits = split_dataset(df)

    same_train_index = splits["full"]["train_index"].equals(splits["reduced"]["train_index"])
    same_test_index = splits["full"]["test_index"].equals(splits["reduced"]["test_index"])
    if not same_train_index or not same_test_index:
        raise ValueError("Feature sets do not share identical train/test indices.")

    all_fold_results = []
    summaries = []
    for feature_set_name in ["full", "reduced"]:
        fold_df, summary = run_cross_validation(
            feature_set_name,
            feature_sets[feature_set_name],
            df,
            splits[feature_set_name],
        )
        all_fold_results.append(fold_df)
        summaries.append(summary)

    fold_results = pd.concat(all_fold_results, ignore_index=True)
    cv_summary = pd.DataFrame(summaries)

    selected_feature_set, selection_reason = choose_feature_set(cv_summary)
    print(f"Selected feature set: {selected_feature_set}")
    print(f"Selection reason: {selection_reason}")

    selected_features = feature_sets[selected_feature_set]
    selected_numeric, selected_categorical = split_features_by_type(df, selected_features)
    selected_split = splits[selected_feature_set]
    final_model = build_pipeline(selected_numeric, selected_categorical)
    final_model.fit(selected_split["X_train"], selected_split["y_train"])
    test_metrics = evaluate_final_model(final_model, selected_split)

    transformed_importance, original_importance = make_feature_importance(
        final_model,
        selected_numeric,
        selected_categorical,
    )
    selected_transformed_count = len(transformed_importance)

    fold_results.to_csv(
        paths["metrics"] / "xgboost_cv_fold_results.csv",
        index=False,
        encoding="utf-8",
    )
    cv_summary.to_csv(
        paths["metrics"] / "xgboost_cv_summary.csv",
        index=False,
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {
                "model": MODEL_NAME,
                "selected_feature_set": selected_feature_set,
                "accuracy": test_metrics["accuracy"],
                "precision": test_metrics["precision"],
                "recall": test_metrics["recall"],
                "f1": test_metrics["f1"],
                "roc_auc": test_metrics["roc_auc"],
                "tn": test_metrics["tn"],
                "fp": test_metrics["fp"],
                "fn": test_metrics["fn"],
                "tp": test_metrics["tp"],
            }
        ]
    ).to_csv(
        paths["metrics"] / "xgboost_test_metrics.csv",
        index=False,
        encoding="utf-8",
    )
    transformed_importance.to_csv(
        paths["metrics"] / "xgboost_feature_importance.csv",
        index=False,
        encoding="utf-8",
    )
    original_importance.to_csv(
        paths["metrics"] / "xgboost_original_feature_importance.csv",
        index=False,
        encoding="utf-8",
    )

    model_path = paths["models"] / f"xgboost_{selected_feature_set}_baseline.joblib"
    joblib.dump(final_model, model_path)

    confusion_path = paths["figures"] / "xgboost_confusion_matrix.png"
    roc_path = paths["figures"] / "xgboost_roc_curve.png"
    cv_comparison_path = paths["figures"] / "xgboost_cv_comparison.png"
    importance_path = paths["figures"] / "xgboost_feature_importance.png"
    plot_confusion_matrix(test_metrics["confusion_matrix"], confusion_path)
    plot_roc_curve(
        selected_split["y_test"],
        test_metrics["y_proba"],
        test_metrics["roc_auc"],
        roc_path,
    )
    plot_cv_comparison(cv_summary, cv_comparison_path)
    plot_feature_importance(original_importance, importance_path)

    total_runtime = time.perf_counter() - start_time
    csv_hash_after = sha256_file(data_path)
    write_report(
        paths["metrics"] / "xgboost_classification_report.txt",
        cv_summary,
        selected_feature_set,
        selection_reason,
        test_metrics,
        len(selected_split["y_train"]),
        len(selected_split["y_test"]),
        csv_hash_before,
        csv_hash_after,
        total_runtime,
    )

    checks = run_checks(
        fold_results,
        cv_summary,
        test_metrics,
        transformed_importance,
        original_importance,
        selected_transformed_count,
        [confusion_path, roc_path, cv_comparison_path, importance_path],
        csv_hash_before,
        csv_hash_after,
    )

    print("Final test metrics:")
    print(
        f"accuracy={test_metrics['accuracy']:.4f}, "
        f"precision={test_metrics['precision']:.4f}, "
        f"recall={test_metrics['recall']:.4f}, "
        f"f1={test_metrics['f1']:.4f}, "
        f"roc_auc={test_metrics['roc_auc']:.4f}"
    )
    print(
        f"TN={test_metrics['tn']}, FP={test_metrics['fp']}, "
        f"FN={test_metrics['fn']}, TP={test_metrics['tp']}"
    )
    print(f"Total runtime: {total_runtime:.2f} seconds")
    print("Post-run checks passed:")
    for name, _ in checks:
        print(f"- {name}")
    print(f"Saved final model: {model_path}")
    print(f"Original CSV SHA256 unchanged: {csv_hash_before == csv_hash_after}")

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
