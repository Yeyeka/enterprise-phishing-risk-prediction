from pathlib import Path
import math
import os
import sys
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "phishing_ml_matplotlib_cache"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype


RANDOM_SEED = 42
TARGET_COLUMN = "failed_phishing_simulation"
DATA_RELATIVE_PATH = Path("data") / "enterprise_phishing_simulation_2026.csv"
METRICS_DIR = Path("results") / "metrics"
FIGURES_DIR = Path("results") / "figures"
DPI = 150
TOP_CORRELATIONS = 10


def get_project_root():
    return Path(__file__).resolve().parents[1]


def ensure_directory(path):
    path.mkdir(parents=True, exist_ok=True)


def save_figure(fig, output_path):
    fig.tight_layout()
    fig.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def get_numeric_columns(df):
    return [column for column in df.columns if is_numeric_dtype(df[column])]


def get_categorical_columns(df):
    return [column for column in df.columns if not is_numeric_dtype(df[column])]


def make_numeric_summary(df, numeric_columns):
    summary = df[numeric_columns].describe().T[
        ["count", "mean", "std", "min", "25%", "50%", "75%", "max"]
    ]
    missing_count = df[numeric_columns].isna().sum()
    summary["missing_count"] = missing_count
    summary["missing_rate"] = (missing_count / len(df) * 100).round(2)
    return summary


def make_categorical_summary(df, categorical_columns):
    rows = []
    for column in categorical_columns:
        value_counts = df[column].value_counts(dropna=True)
        missing_count = int(df[column].isna().sum())
        if value_counts.empty:
            most_frequent_value = ""
            most_frequent_count = 0
        else:
            most_frequent_value = str(value_counts.index[0])
            most_frequent_count = int(value_counts.iloc[0])
        rows.append(
            {
                "feature": column,
                "unique_count": int(df[column].nunique(dropna=True)),
                "missing_count": missing_count,
                "missing_rate": round(missing_count / len(df) * 100, 2),
                "most_frequent_value": most_frequent_value,
                "most_frequent_count": most_frequent_count,
            }
        )
    return pd.DataFrame(rows)


def make_target_distribution(df):
    counts = df[TARGET_COLUMN].value_counts(dropna=False)
    distribution = counts.rename_axis("target_class").reset_index(name="count")
    distribution["proportion"] = (distribution["count"] / len(df) * 100).round(2)
    return distribution


def make_outlier_summary(df, numeric_columns):
    rows = []
    for column in numeric_columns:
        series = df[column].dropna()
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outlier_count = int(((series < lower_bound) | (series > upper_bound)).sum())
        rows.append(
            {
                "feature": column,
                "q1": q1,
                "q3": q3,
                "iqr": iqr,
                "lower_bound": lower_bound,
                "upper_bound": upper_bound,
                "outlier_count": outlier_count,
                "outlier_rate": round(outlier_count / len(df) * 100, 2),
            }
        )
    return pd.DataFrame(rows).sort_values("outlier_count", ascending=False)


def get_top_correlations(correlation_matrix):
    rows = []
    columns = correlation_matrix.columns.tolist()
    for i, feature_1 in enumerate(columns):
        for j in range(i + 1, len(columns)):
            feature_2 = columns[j]
            correlation = correlation_matrix.loc[feature_1, feature_2]
            if pd.isna(correlation):
                continue
            rows.append(
                {
                    "feature_1": feature_1,
                    "feature_2": feature_2,
                    "correlation": correlation,
                    "abs_correlation": abs(correlation),
                }
            )
    return pd.DataFrame(rows).sort_values("abs_correlation", ascending=False).head(TOP_CORRELATIONS)


def get_numeric_target_differences(df, numeric_columns):
    rows = []
    grouped = df.groupby(TARGET_COLUMN, dropna=False)
    for column in numeric_columns:
        if {"Yes", "No"}.issubset(set(df[TARGET_COLUMN].dropna().unique())):
            yes_mean = grouped[column].mean().get("Yes")
            no_mean = grouped[column].mean().get("No")
            pooled_std = df[column].std()
            absolute_difference = abs(yes_mean - no_mean)
            standardized_difference = absolute_difference / pooled_std if pooled_std else np.nan
            rows.append(
                {
                    "feature": column,
                    "yes_mean": yes_mean,
                    "no_mean": no_mean,
                    "absolute_mean_difference": absolute_difference,
                    "standardized_difference": standardized_difference,
                }
            )
    return pd.DataFrame(rows).sort_values("standardized_difference", ascending=False)


def get_categorical_target_rate_ranges(df, categorical_predictors):
    rows = []
    target_yes = df[TARGET_COLUMN].eq("Yes")
    for column in categorical_predictors:
        labels = df[column].astype("object").where(df[column].notna(), "Missing")
        rates = target_yes.groupby(labels).mean() * 100
        if rates.empty:
            continue
        rows.append(
            {
                "feature": column,
                "min_yes_rate": rates.min(),
                "max_yes_rate": rates.max(),
                "yes_rate_range": rates.max() - rates.min(),
            }
        )
    return pd.DataFrame(rows).sort_values("yes_rate_range", ascending=False)


def plot_target_distribution(target_distribution, output_path):
    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(target_distribution["target_class"].astype(str), target_distribution["count"], color=["#4C78A8", "#F58518"])
    ax.set_title("Target Distribution")
    ax.set_xlabel("Target Class")
    ax.set_ylabel("Count")
    for bar, count, proportion in zip(bars, target_distribution["count"], target_distribution["proportion"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{count:,}\n{proportion:.2f}%",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    save_figure(fig, output_path)


def plot_missing_values(df, output_path):
    missing_count = df.isna().sum()
    missing_count = missing_count[missing_count > 0].sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(10, 5))
    if missing_count.empty:
        ax.text(0.5, 0.5, "No Missing Values", ha="center", va="center", fontsize=16)
        ax.set_axis_off()
    else:
        missing_rate = missing_count / len(df) * 100
        positions = np.arange(len(missing_count))
        bars = ax.bar(positions, missing_count.values, color="#4C78A8", label="Missing Count")
        ax.set_title("Missing Values by Feature")
        ax.set_xlabel("Feature")
        ax.set_ylabel("Missing Count")
        ax.set_xticks(positions)
        ax.set_xticklabels(missing_count.index, rotation=35, ha="right")
        ax_rate = ax.twinx()
        ax_rate.plot(positions, missing_rate.values, color="#E45756", marker="o", label="Missing Rate (%)")
        ax_rate.set_ylabel("Missing Rate (%)")
        for bar, rate in zip(bars, missing_rate.values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{int(bar.get_height()):,}\n{rate:.2f}%",
                ha="center",
                va="bottom",
                fontsize=8,
            )
        lines, labels = ax.get_legend_handles_labels()
        lines_rate, labels_rate = ax_rate.get_legend_handles_labels()
        ax.legend(lines + lines_rate, labels + labels_rate, loc="upper right")
    save_figure(fig, output_path)


def plot_correlation_heatmap(correlation_matrix, output_path):
    fig, ax = plt.subplots(figsize=(12, 10))
    image = ax.imshow(correlation_matrix.values, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_title("Pearson Correlation Heatmap")
    ax.set_xticks(np.arange(len(correlation_matrix.columns)))
    ax.set_yticks(np.arange(len(correlation_matrix.index)))
    ax.set_xticklabels(correlation_matrix.columns, rotation=45, ha="right")
    ax.set_yticklabels(correlation_matrix.index)
    for i in range(len(correlation_matrix.index)):
        for j in range(len(correlation_matrix.columns)):
            value = correlation_matrix.iloc[i, j]
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=7, color="black")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="Correlation")
    save_figure(fig, output_path)


def plot_numeric_histograms(df, numeric_columns, output_path):
    columns = 3
    rows = math.ceil(len(numeric_columns) / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(15, 4 * rows))
    axes = np.array(axes).reshape(-1)
    for ax, column in zip(axes, numeric_columns):
        ax.hist(df[column].dropna(), bins=30, color="#4C78A8", edgecolor="white")
        ax.set_title(column)
        ax.set_xlabel("Value")
        ax.set_ylabel("Frequency")
    for ax in axes[len(numeric_columns) :]:
        ax.set_visible(False)
    fig.suptitle("Numeric Feature Histograms", y=1.01, fontsize=14)
    save_figure(fig, output_path)


def plot_numeric_boxplots(df, numeric_columns, output_path):
    columns = 3
    rows = math.ceil(len(numeric_columns) / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(15, 4 * rows))
    axes = np.array(axes).reshape(-1)
    for ax, column in zip(axes, numeric_columns):
        ax.boxplot(
            df[column].dropna(),
            orientation="vertical",
            patch_artist=True,
            boxprops={"facecolor": "#72B7B2"},
        )
        ax.set_title(column)
        ax.set_ylabel("Value")
        ax.set_xticks([1])
        ax.set_xticklabels([column], rotation=20, ha="right")
    for ax in axes[len(numeric_columns) :]:
        ax.set_visible(False)
    fig.suptitle("Numeric Feature Boxplots", y=1.01, fontsize=14)
    save_figure(fig, output_path)


def plot_numeric_by_target_boxplots(df, numeric_columns, output_path):
    columns = 3
    rows = math.ceil(len(numeric_columns) / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(15, 4 * rows))
    axes = np.array(axes).reshape(-1)
    target_order = ["No", "Yes"]
    for ax, column in zip(axes, numeric_columns):
        values = [df.loc[df[TARGET_COLUMN] == target_value, column].dropna() for target_value in target_order]
        ax.boxplot(
            values,
            tick_labels=target_order,
            patch_artist=True,
            boxprops={"facecolor": "#B279A2"},
        )
        ax.set_title(column)
        ax.set_xlabel("Target Class")
        ax.set_ylabel("Value")
    for ax in axes[len(numeric_columns) :]:
        ax.set_visible(False)
    fig.suptitle("Numeric Feature Boxplots by Target", y=1.01, fontsize=14)
    save_figure(fig, output_path)


def plot_categorical_distributions(df, categorical_predictors, output_path):
    columns = 2
    rows = math.ceil(len(categorical_predictors) / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(15, 4.5 * rows))
    axes = np.array(axes).reshape(-1)
    for ax, column in zip(axes, categorical_predictors):
        counts = df[column].astype("object").where(df[column].notna(), "Missing").value_counts()
        ax.bar(counts.index.astype(str), counts.values, color="#54A24B")
        ax.set_title(column)
        ax.set_xlabel("Category")
        ax.set_ylabel("Count")
        ax.tick_params(axis="x", rotation=35)
    for ax in axes[len(categorical_predictors) :]:
        ax.set_visible(False)
    fig.suptitle("Categorical Feature Distributions", y=1.01, fontsize=14)
    save_figure(fig, output_path)


def plot_categorical_target_rates(df, categorical_predictors, output_path):
    columns = 2
    rows = math.ceil(len(categorical_predictors) / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(15, 4.5 * rows))
    axes = np.array(axes).reshape(-1)
    target_yes = df[TARGET_COLUMN].eq("Yes")
    for ax, column in zip(axes, categorical_predictors):
        labels = df[column].astype("object").where(df[column].notna(), "Missing")
        rates = (target_yes.groupby(labels).mean() * 100).sort_values(ascending=False)
        ax.bar(rates.index.astype(str), rates.values, color="#F58518")
        ax.set_title(column)
        ax.set_xlabel("Category")
        ax.set_ylabel("Failure Rate (%)")
        ax.tick_params(axis="x", rotation=35)
        ax.set_ylim(0, min(100, max(rates.max() * 1.2, 5)))
    for ax in axes[len(categorical_predictors) :]:
        ax.set_visible(False)
    fig.suptitle("Target Yes Failure Rate by Category", y=1.01, fontsize=14)
    save_figure(fig, output_path)


def make_findings(
    df,
    target_distribution,
    missing_fields,
    numeric_summary,
    outlier_summary,
    top_correlations,
    categorical_summary,
    numeric_target_differences,
    categorical_target_rate_ranges,
):
    lines = []
    lines.append("EDA Findings")
    lines.append("=" * 80)
    lines.append(f"Data shape: {df.shape[0]} rows, {df.shape[1]} columns.")
    lines.append("")
    lines.append("Target class distribution:")
    for _, row in target_distribution.iterrows():
        lines.append(f"- {row['target_class']}: {int(row['count'])} samples ({row['proportion']:.2f}%).")
    lines.append("")
    lines.append("Missing value fields:")
    if missing_fields.empty:
        lines.append("- No missing value fields were found.")
    else:
        for feature, count in missing_fields.items():
            lines.append(f"- {feature}: {int(count)} missing values ({count / len(df) * 100:.2f}%).")
    lines.append("")
    lines.append("Numeric feature distribution notes:")
    for feature, row in numeric_summary.iterrows():
        lines.append(
            f"- {feature}: mean={row['mean']:.2f}, std={row['std']:.2f}, "
            f"min={row['min']:.2f}, median={row['50%']:.2f}, max={row['max']:.2f}."
        )
    lines.append("")
    lines.append("Potential extreme values based on IQR rule:")
    if outlier_summary["outlier_count"].sum() == 0:
        lines.append("- No IQR-based potential outliers were found.")
    else:
        for _, row in outlier_summary.head(10).iterrows():
            lines.append(
                f"- {row['feature']}: {int(row['outlier_count'])} potential outliers "
                f"({row['outlier_rate']:.2f}%)."
            )
    lines.append("")
    lines.append("Top absolute Pearson correlations among numeric features:")
    if top_correlations.empty:
        lines.append("- No numeric feature pairs were available for correlation analysis.")
    else:
        for _, row in top_correlations.iterrows():
            lines.append(
                f"- {row['feature_1']} vs {row['feature_2']}: "
                f"r={row['correlation']:.4f}, abs={row['abs_correlation']:.4f}."
            )
    lines.append("")
    lines.append("Categorical feature balance notes:")
    for _, row in categorical_summary[categorical_summary["feature"] != TARGET_COLUMN].iterrows():
        lines.append(
            f"- {row['feature']}: {int(row['unique_count'])} categories; "
            f"most frequent={row['most_frequent_value']} ({int(row['most_frequent_count'])} samples)."
        )
    lines.append("")
    lines.append("Numeric features with clearer target-class mean differences:")
    for _, row in numeric_target_differences.head(5).iterrows():
        lines.append(
            f"- {row['feature']}: Yes mean={row['yes_mean']:.2f}, No mean={row['no_mean']:.2f}, "
            f"standardized difference={row['standardized_difference']:.4f}."
        )
    lines.append("")
    lines.append("Categorical features with wider target Yes rate ranges:")
    for _, row in categorical_target_rate_ranges.head(5).iterrows():
        lines.append(
            f"- {row['feature']}: min Yes rate={row['min_yes_rate']:.2f}%, "
            f"max Yes rate={row['max_yes_rate']:.2f}%, range={row['yes_rate_range']:.2f} percentage points."
        )
    lines.append("")
    lines.append("Issues to address in preprocessing stage:")
    lines.append("- Missing values need an explicit strategy fitted without data leakage.")
    lines.append("- Categorical predictors need encoding after data splitting or inside a pipeline.")
    lines.append("- IQR-based potential outliers should be evaluated, but no rows were removed in this EDA stage.")
    lines.append("- Correlated numeric feature pairs should be documented; no feature was removed in this EDA stage.")
    lines.append("- Target imbalance is moderate and should be considered during model evaluation.")
    return "\n".join(lines) + "\n"


def main():
    np.random.seed(RANDOM_SEED)
    project_root = get_project_root()
    data_path = project_root / DATA_RELATIVE_PATH
    metrics_dir = project_root / METRICS_DIR
    figures_dir = project_root / FIGURES_DIR

    if not data_path.exists():
        print(f"Error: CSV file does not exist: {data_path}", file=sys.stderr)
        return 1

    ensure_directory(metrics_dir)
    ensure_directory(figures_dir)

    try:
        df = pd.read_csv(data_path)
    except Exception as exc:
        print(f"Error: failed to read CSV file: {data_path}; reason: {exc}", file=sys.stderr)
        return 1

    if df.empty:
        print(f"Error: dataset is empty: {data_path}", file=sys.stderr)
        return 1

    if TARGET_COLUMN not in df.columns:
        print(f"Error: target column does not exist: {TARGET_COLUMN}", file=sys.stderr)
        return 1

    numeric_columns = get_numeric_columns(df)
    categorical_columns = get_categorical_columns(df)
    categorical_predictors = [column for column in categorical_columns if column != TARGET_COLUMN]

    numeric_summary = make_numeric_summary(df, numeric_columns)
    categorical_summary = make_categorical_summary(df, categorical_columns)
    target_distribution = make_target_distribution(df)
    correlation_matrix = df[numeric_columns].corr(method="pearson")
    outlier_summary = make_outlier_summary(df, numeric_columns)
    top_correlations = get_top_correlations(correlation_matrix)
    missing_fields = df.isna().sum()
    missing_fields = missing_fields[missing_fields > 0].sort_values(ascending=False)
    numeric_target_differences = get_numeric_target_differences(df, numeric_columns)
    categorical_target_rate_ranges = get_categorical_target_rate_ranges(df, categorical_predictors)

    numeric_summary.to_csv(metrics_dir / "numeric_summary.csv", encoding="utf-8")
    categorical_summary.to_csv(metrics_dir / "categorical_summary.csv", index=False, encoding="utf-8")
    target_distribution.to_csv(metrics_dir / "target_distribution.csv", index=False, encoding="utf-8")
    correlation_matrix.to_csv(metrics_dir / "correlation_matrix.csv", encoding="utf-8")
    outlier_summary.to_csv(metrics_dir / "outlier_summary.csv", index=False, encoding="utf-8")

    findings = make_findings(
        df,
        target_distribution,
        missing_fields,
        numeric_summary,
        outlier_summary,
        top_correlations,
        categorical_summary,
        numeric_target_differences,
        categorical_target_rate_ranges,
    )
    (metrics_dir / "eda_findings.txt").write_text(findings, encoding="utf-8")

    plot_target_distribution(target_distribution, figures_dir / "target_distribution.png")
    plot_missing_values(df, figures_dir / "missing_values.png")
    plot_correlation_heatmap(correlation_matrix, figures_dir / "correlation_heatmap.png")
    plot_numeric_histograms(df, numeric_columns, figures_dir / "numeric_histograms.png")
    plot_numeric_boxplots(df, numeric_columns, figures_dir / "numeric_boxplots.png")
    plot_numeric_by_target_boxplots(df, numeric_columns, figures_dir / "numeric_by_target_boxplots.png")
    plot_categorical_distributions(df, categorical_predictors, figures_dir / "categorical_distributions.png")
    plot_categorical_target_rates(df, categorical_predictors, figures_dir / "categorical_target_rates.png")

    print("EDA analysis completed successfully.")
    print(f"Rows: {df.shape[0]}")
    print(f"Columns: {df.shape[1]}")
    print(f"Numeric features: {len(numeric_columns)}")
    print(f"Categorical features: {len(categorical_columns)}")
    print("Generated metric files:")
    for file_name in [
        "numeric_summary.csv",
        "categorical_summary.csv",
        "target_distribution.csv",
        "correlation_matrix.csv",
        "outlier_summary.csv",
        "eda_findings.txt",
    ]:
        print(f"- {metrics_dir / file_name}")
    print("Generated figures:")
    for file_name in [
        "target_distribution.png",
        "missing_values.png",
        "correlation_heatmap.png",
        "numeric_histograms.png",
        "numeric_boxplots.png",
        "numeric_by_target_boxplots.png",
        "categorical_distributions.png",
        "categorical_target_rates.png",
    ]:
        print(f"- {figures_dir / file_name}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
