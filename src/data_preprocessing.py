from pathlib import Path
import inspect
import sys

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from scipy import sparse
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


RANDOM_SEED = 42
TARGET_COLUMN = "failed_phishing_simulation"
TARGET_MAPPING = {"No": 0, "Yes": 1}
DATA_RELATIVE_PATH = Path("data") / "enterprise_phishing_simulation_2026.csv"
METRICS_DIR = Path("results") / "metrics"
REDUCED_EXCLUDED_FEATURE = "cybersecurity_awareness_score"
ID_KEYWORDS = ("id", "编号", "number", "code", "uuid")
NEAR_CONSTANT_THRESHOLD = 0.99
HIGH_CARDINALITY_THRESHOLD = 50
DENSE_MEMORY_LIMIT_MB = 1024


def get_project_root():
    return Path(__file__).resolve().parents[1]


def load_dataset():
    project_root = get_project_root()
    data_path = project_root / DATA_RELATIVE_PATH
    if not data_path.exists():
        raise FileNotFoundError(f"CSV file does not exist: {data_path}")
    return pd.read_csv(data_path)


def validate_dataset(df):
    if df.empty:
        raise ValueError("Dataset is empty.")
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column does not exist: {TARGET_COLUMN}")

    target_values = set(df[TARGET_COLUMN].dropna().astype(str).unique())
    expected_values = set(TARGET_MAPPING)
    unexpected_values = sorted(target_values - expected_values)
    if unexpected_values:
        raise ValueError(f"Unexpected target values found: {unexpected_values}")

    if df[TARGET_COLUMN].isna().any():
        raise ValueError("Target column contains missing values.")

    return True


def identify_feature_types(df):
    predictors = [column for column in df.columns if column != TARGET_COLUMN]
    numeric_features = [column for column in predictors if is_numeric_dtype(df[column])]
    categorical_features = [column for column in predictors if column not in numeric_features]
    return numeric_features, categorical_features


def check_columns(df, numeric_features, categorical_features):
    predictors = numeric_features + categorical_features

    constant_columns = [
        column for column in predictors if df[column].nunique(dropna=False) == 1
    ]

    near_constant_columns = []
    for column in predictors:
        top_frequency_ratio = df[column].value_counts(dropna=False, normalize=True).iloc[0]
        if top_frequency_ratio > NEAR_CONSTANT_THRESHOLD:
            near_constant_columns.append(
                {
                    "feature": column,
                    "top_frequency_ratio": top_frequency_ratio,
                }
            )

    high_cardinality_columns = [
        {
            "feature": column,
            "unique_count": int(df[column].nunique(dropna=True)),
        }
        for column in categorical_features
        if df[column].nunique(dropna=True) > HIGH_CARDINALITY_THRESHOLD
    ]

    suspected_id_columns = []
    row_count = len(df)
    for column in predictors:
        column_lower = str(column).lower()
        non_missing = df[column].dropna()
        unique_count = int(non_missing.nunique())
        unique_ratio = unique_count / len(non_missing) if len(non_missing) else 0
        reasons = []
        if any(keyword in column_lower for keyword in ID_KEYWORDS):
            reasons.append("name contains ID-like keyword")
        if row_count > 0 and unique_count > HIGH_CARDINALITY_THRESHOLD and unique_ratio > 0.99:
            reasons.append("nearly unique values")
        if reasons:
            suspected_id_columns.append(
                {
                    "feature": column,
                    "unique_count": unique_count,
                    "unique_ratio": unique_ratio,
                    "reason": "; ".join(reasons),
                }
            )

    return {
        "constant_columns": constant_columns,
        "near_constant_columns": near_constant_columns,
        "high_cardinality_columns": high_cardinality_columns,
        "suspected_id_columns": suspected_id_columns,
    }


def get_feature_sets(df):
    predictors = [column for column in df.columns if column != TARGET_COLUMN]
    full_features = predictors.copy()
    reduced_features = [
        column for column in full_features if column != REDUCED_EXCLUDED_FEATURE
    ]
    return {
        "full": full_features,
        "reduced": reduced_features,
    }


def split_dataset(df):
    validate_dataset(df)
    feature_sets = get_feature_sets(df)
    y = df[TARGET_COLUMN].map(TARGET_MAPPING)

    train_index, test_index = train_test_split(
        df.index,
        test_size=0.2,
        random_state=RANDOM_SEED,
        stratify=y,
    )

    splits = {}
    for name, features in feature_sets.items():
        splits[name] = {
            "X_train": df.loc[train_index, features],
            "X_test": df.loc[test_index, features],
            "y_train": y.loc[train_index],
            "y_test": y.loc[test_index],
            "train_index": pd.Index(train_index),
            "test_index": pd.Index(test_index),
        }

    return splits


def make_one_hot_encoder(sparse_output=True):
    params = {"handle_unknown": "ignore"}
    signature = inspect.signature(OneHotEncoder)
    if "sparse_output" in signature.parameters:
        params["sparse_output"] = sparse_output
    else:
        params["sparse"] = sparse_output
    return OneHotEncoder(**params)


def build_linear_preprocessor(numeric_features, categorical_features):
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", make_one_hot_encoder(sparse_output=True)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ]
    )


def build_naive_bayes_preprocessor(numeric_features, categorical_features):
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", make_one_hot_encoder(sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ],
        sparse_threshold=0.0,
    )


def build_tree_preprocessor(numeric_features, categorical_features):
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", make_one_hot_encoder(sparse_output=True)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ]
    )


def matrix_shape(matrix):
    return matrix.shape


def matrix_memory_mb(matrix):
    if sparse.issparse(matrix):
        total_bytes = matrix.data.nbytes + matrix.indices.nbytes + matrix.indptr.nbytes
    else:
        array = np.asarray(matrix)
        total_bytes = array.nbytes
    return round(total_bytes / (1024**2), 2)


def count_nan_and_infinite(matrix):
    if sparse.issparse(matrix):
        values = matrix.data
    else:
        values = np.asarray(matrix)
    nan_count = int(np.isnan(values).sum())
    infinite_count = int(np.isinf(values).sum())
    return nan_count, infinite_count


def feature_types_for_feature_set(features, numeric_features, categorical_features):
    feature_set = set(features)
    feature_numeric = [feature for feature in numeric_features if feature in feature_set]
    feature_categorical = [feature for feature in categorical_features if feature in feature_set]
    return feature_numeric, feature_categorical


def get_preprocessor_builders():
    return {
        "linear": build_linear_preprocessor,
        "naive_bayes": build_naive_bayes_preprocessor,
        "tree": build_tree_preprocessor,
    }


def validate_preprocessors(feature_sets, splits, numeric_features, categorical_features):
    validation_rows = []
    feature_name_rows = []

    for preprocessor_name, builder in get_preprocessor_builders().items():
        for feature_set_name, features in feature_sets.items():
            feature_numeric, feature_categorical = feature_types_for_feature_set(
                features, numeric_features, categorical_features
            )
            preprocessor = builder(feature_numeric, feature_categorical)
            split = splits[feature_set_name]
            x_train = split["X_train"]
            x_test = split["X_test"]

            x_train_transformed = preprocessor.fit_transform(x_train)
            x_test_transformed = preprocessor.transform(x_test)

            train_rows_after, train_columns_after = matrix_shape(x_train_transformed)
            test_rows_after, test_columns_after = matrix_shape(x_test_transformed)
            train_missing_after, train_infinite_after = count_nan_and_infinite(x_train_transformed)
            test_missing_after, test_infinite_after = count_nan_and_infinite(x_test_transformed)
            output_type = "sparse_matrix" if sparse.issparse(x_train_transformed) else "dense_array"
            estimated_memory_mb = round(
                matrix_memory_mb(x_train_transformed) + matrix_memory_mb(x_test_transformed),
                2,
            )

            feature_names_available = True
            duplicate_feature_name_count = 0
            try:
                transformed_feature_names = preprocessor.get_feature_names_out()
                duplicate_feature_name_count = int(
                    pd.Series(transformed_feature_names).duplicated().sum()
                )
                for feature_index, transformed_feature_name in enumerate(transformed_feature_names):
                    feature_name_rows.append(
                        {
                            "preprocessor": preprocessor_name,
                            "feature_set": feature_set_name,
                            "feature_index": feature_index,
                            "transformed_feature_name": transformed_feature_name,
                        }
                    )
            except Exception:
                feature_names_available = False
                transformed_feature_names = []

            dense_memory_ok = True
            if preprocessor_name == "naive_bayes":
                dense_memory_ok = estimated_memory_mb <= DENSE_MEMORY_LIMIT_MB
                if not dense_memory_ok:
                    raise MemoryError(
                        f"Naive Bayes dense preprocessing output is too large: "
                        f"{estimated_memory_mb} MB > {DENSE_MEMORY_LIMIT_MB} MB"
                    )

            validation_passed = all(
                [
                    train_rows_after == len(x_train),
                    test_rows_after == len(x_test),
                    train_columns_after == test_columns_after,
                    train_missing_after == 0,
                    test_missing_after == 0,
                    train_infinite_after == 0,
                    test_infinite_after == 0,
                    feature_names_available,
                    duplicate_feature_name_count == 0,
                    dense_memory_ok,
                ]
            )

            validation_rows.append(
                {
                    "preprocessor": preprocessor_name,
                    "feature_set": feature_set_name,
                    "raw_feature_count": len(features),
                    "train_rows_before": len(x_train),
                    "train_rows_after": train_rows_after,
                    "test_rows_before": len(x_test),
                    "test_rows_after": test_rows_after,
                    "transformed_feature_count": train_columns_after,
                    "train_missing_after": train_missing_after,
                    "test_missing_after": test_missing_after,
                    "train_infinite_after": train_infinite_after,
                    "test_infinite_after": test_infinite_after,
                    "output_type": output_type,
                    "estimated_memory_mb": estimated_memory_mb,
                    "feature_names_available": feature_names_available,
                    "duplicate_feature_name_count": duplicate_feature_name_count,
                    "validation_passed": validation_passed,
                }
            )

    return pd.DataFrame(validation_rows), pd.DataFrame(feature_name_rows)


def class_distribution(series):
    counts = series.value_counts().sort_index()
    proportions = (counts / len(series) * 100).round(2)
    rows = []
    for class_value, count in counts.items():
        label = "No" if class_value == 0 else "Yes"
        rows.append(
            {
                "class": label,
                "encoded_value": int(class_value),
                "count": int(count),
                "proportion": float(proportions.loc[class_value]),
            }
        )
    return rows


def format_list(items):
    if not items:
        return "- None"
    return "\n".join(f"- {item}" for item in items)


def format_dict_rows(rows, empty_message="- None"):
    if not rows:
        return empty_message
    return "\n".join(f"- {row}" for row in rows)


def make_feature_sets_table(df, numeric_features, categorical_features, feature_sets):
    rows = []
    missing_counts = df.isna().sum()
    all_predictors = [column for column in df.columns if column != TARGET_COLUMN]
    full_set = set(feature_sets["full"])
    reduced_set = set(feature_sets["reduced"])
    numeric_set = set(numeric_features)

    for feature in all_predictors:
        exclusion_reason = ""
        if feature not in reduced_set:
            exclusion_reason = "Removed from reduced set to test composite score redundancy"
        rows.append(
            {
                "feature": feature,
                "data_type": "numeric" if feature in numeric_set else "categorical",
                "missing_count": int(missing_counts.loc[feature]),
                "missing_rate": round(missing_counts.loc[feature] / len(df) * 100, 2),
                "included_in_full": feature in full_set,
                "included_in_reduced": feature in reduced_set,
                "exclusion_reason": exclusion_reason,
            }
        )
    return pd.DataFrame(rows)


def make_summary_text(
    df,
    numeric_features,
    categorical_features,
    checks,
    feature_sets,
    splits,
    validation_results=None,
):
    full_split = splits["full"]
    reduced_split = splits["reduced"]
    train_overlap = full_split["train_index"].intersection(full_split["test_index"])
    same_train_index = full_split["train_index"].equals(reduced_split["train_index"])
    same_test_index = full_split["test_index"].equals(reduced_split["test_index"])
    missing_counts = df.isna().sum()
    missing_counts = missing_counts[missing_counts > 0]

    lines = []
    lines.append("Stage 3: Data Preprocessing Design and Dataset Split")
    lines.append("=" * 80)
    lines.append(f"Original data shape: {df.shape[0]} rows, {df.shape[1]} columns")
    lines.append("Target mapping: No -> 0, Yes -> 1")
    lines.append(f"Random seed: {RANDOM_SEED}")
    lines.append("")
    lines.append("Numeric predictive features:")
    lines.append(format_list(numeric_features))
    lines.append("")
    lines.append("Categorical predictive features:")
    lines.append(format_list(categorical_features))
    lines.append("")
    lines.append("Fields with missing values:")
    if missing_counts.empty:
        lines.append("- None")
    else:
        for feature, count in missing_counts.items():
            lines.append(f"- {feature}: {int(count)} ({count / len(df) * 100:.2f}%)")
    lines.append("")
    lines.append("Column checks:")
    lines.append(f"- Constant columns: {checks['constant_columns'] or 'None'}")
    if checks["near_constant_columns"]:
        near_constant = [
            f"{row['feature']} ({row['top_frequency_ratio']:.2%})"
            for row in checks["near_constant_columns"]
        ]
        lines.append(f"- Near-constant columns: {near_constant}")
    else:
        lines.append("- Near-constant columns: None")
    if checks["high_cardinality_columns"]:
        high_cardinality = [
            f"{row['feature']} ({row['unique_count']} unique values)"
            for row in checks["high_cardinality_columns"]
        ]
        lines.append(f"- High-cardinality categorical columns: {high_cardinality}")
    else:
        lines.append("- High-cardinality categorical columns: None")
    if checks["suspected_id_columns"]:
        suspected_ids = [
            f"{row['feature']} ({row['reason']})"
            for row in checks["suspected_id_columns"]
        ]
        lines.append(f"- Suspected ID columns: {suspected_ids}")
    else:
        lines.append("- Suspected ID columns: None")
    lines.append("")
    lines.append(f"Feature set A - full feature set: {len(feature_sets['full'])} features")
    lines.append(format_list(feature_sets["full"]))
    lines.append("")
    lines.append(f"Feature set B - reduced feature set: {len(feature_sets['reduced'])} features")
    lines.append(f"Excluded feature: {REDUCED_EXCLUDED_FEATURE}")
    lines.append(format_list(feature_sets["reduced"]))
    lines.append("")
    lines.append("Dataset split:")
    lines.append(f"- Train samples: {len(full_split['y_train'])}")
    lines.append(f"- Test samples: {len(full_split['y_test'])}")
    lines.append(f"- Train/test index overlap count: {len(train_overlap)}")
    lines.append(f"- Same train indices for both feature sets: {same_train_index}")
    lines.append(f"- Same test indices for both feature sets: {same_test_index}")
    lines.append("")
    lines.append("Train target distribution:")
    for row in class_distribution(full_split["y_train"]):
        lines.append(
            f"- {row['class']} ({row['encoded_value']}): {row['count']} ({row['proportion']:.2f}%)"
        )
    lines.append("")
    lines.append("Test target distribution:")
    for row in class_distribution(full_split["y_test"]):
        lines.append(
            f"- {row['class']} ({row['encoded_value']}): {row['count']} ({row['proportion']:.2f}%)"
        )
    lines.append("")
    lines.append("Preprocessor design:")
    lines.append("- Linear models: numeric median imputation + StandardScaler; categorical most-frequent imputation + OneHotEncoder(handle_unknown='ignore').")
    lines.append("- Gaussian Naive Bayes: numeric median imputation + StandardScaler; categorical most-frequent imputation + dense OneHotEncoder(handle_unknown='ignore'); ColumnTransformer uses sparse_threshold=0.0.")
    lines.append("- Tree models: numeric median imputation only; categorical most-frequent imputation + OneHotEncoder(handle_unknown='ignore').")
    lines.append("")
    lines.append("Feature engineering scope:")
    lines.append("- Target binary mapping: No -> 0, Yes -> 1.")
    lines.append("- Numeric missing values are handled with median imputation inside preprocessing pipelines.")
    lines.append("- Categorical missing values are handled with most-frequent imputation inside preprocessing pipelines.")
    lines.append("- Categorical predictors are one-hot encoded with handle_unknown='ignore'.")
    lines.append("- Linear and GaussianNB preprocessing standardizes numeric features; tree preprocessing does not standardize numeric features.")
    lines.append("- Two raw feature sets are maintained: full features and a reduced feature set excluding cybersecurity_awareness_score.")
    lines.append("- No additional interaction or composite features are created because the source data already contains multiple behavior and awareness scores, several score fields are highly correlated, ad-hoc combinations could add redundancy, and later feature importance plus ablation testing is a cleaner way to judge feature value.")
    lines.append("- IQR-based potential outliers are not removed because their rates are low and may reflect plausible employee behavior differences.")
    lines.append("- SMOTE is not used because the positive class proportion is 37.31%, which is not severe class imbalance.")
    lines.append("")
    if validation_results is not None:
        lines.append("Preprocessor validation results:")
        for _, row in validation_results.iterrows():
            lines.append(
                f"- {row['preprocessor']} / {row['feature_set']}: "
                f"raw_features={int(row['raw_feature_count'])}, "
                f"train_shape=({int(row['train_rows_after'])}, {int(row['transformed_feature_count'])}), "
                f"test_shape=({int(row['test_rows_after'])}, {int(row['transformed_feature_count'])}), "
                f"output_type={row['output_type']}, "
                f"missing_after train/test={int(row['train_missing_after'])}/{int(row['test_missing_after'])}, "
                f"infinite_after train/test={int(row['train_infinite_after'])}/{int(row['test_infinite_after'])}, "
                f"estimated_memory_mb={row['estimated_memory_mb']:.2f}, "
                f"feature_names_available={row['feature_names_available']}, "
                f"duplicate_feature_names={int(row['duplicate_feature_name_count'])}, "
                f"validation_passed={row['validation_passed']}."
            )
        all_passed = bool(validation_results["validation_passed"].all())
        lines.append(f"- All six preprocessing validations passed: {all_passed}")
        naive_bayes_rows = validation_results[validation_results["preprocessor"] == "naive_bayes"]
        if not naive_bayes_rows.empty:
            memory_text = ", ".join(
                f"{row['feature_set']}={row['estimated_memory_mb']:.2f} MB"
                for _, row in naive_bayes_rows.iterrows()
            )
            lines.append(f"- GaussianNB dense output estimated memory: {memory_text}.")
        lines.append("")
    lines.append("Data leakage prevention:")
    lines.append("- This script never fits an imputer, scaler, or encoder on the full dataset.")
    lines.append("- Validation calls fit_transform only on X_train and transform only on X_test.")
    lines.append("- Fitted validation preprocessors are not saved.")
    lines.append("- Future model training should keep preprocessors inside model Pipelines fitted on training folds only.")
    lines.append("- Train/test split is performed before any future preprocessing fit.")
    lines.append("- The test set is not used for fill values, feature selection, parameter tuning, or scaling statistics.")
    lines.append("- The original CSV is read-only in this stage and is not modified.")
    return "\n".join(lines) + "\n"


def main():
    project_root = get_project_root()
    metrics_dir = project_root / METRICS_DIR
    metrics_dir.mkdir(parents=True, exist_ok=True)

    try:
        df = load_dataset()
        validate_dataset(df)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    numeric_features, categorical_features = identify_feature_types(df)
    checks = check_columns(df, numeric_features, categorical_features)
    feature_sets = get_feature_sets(df)
    splits = split_dataset(df)

    full_split = splits["full"]
    reduced_split = splits["reduced"]
    train_overlap_count = len(full_split["train_index"].intersection(full_split["test_index"]))
    same_split = (
        full_split["train_index"].equals(reduced_split["train_index"])
        and full_split["test_index"].equals(reduced_split["test_index"])
    )

    validation_results, transformed_feature_names = validate_preprocessors(
        feature_sets,
        splits,
        numeric_features,
        categorical_features,
    )

    feature_sets_table = make_feature_sets_table(
        df, numeric_features, categorical_features, feature_sets
    )
    feature_sets_table.to_csv(metrics_dir / "feature_sets.csv", index=False, encoding="utf-8")
    validation_results.to_csv(
        metrics_dir / "preprocessing_validation.csv", index=False, encoding="utf-8"
    )
    transformed_feature_names.to_csv(
        metrics_dir / "transformed_feature_names.csv", index=False, encoding="utf-8"
    )

    summary_text = make_summary_text(
        df,
        numeric_features,
        categorical_features,
        checks,
        feature_sets,
        splits,
        validation_results,
    )
    (metrics_dir / "preprocessing_summary.txt").write_text(summary_text, encoding="utf-8")

    print("Data preprocessing design completed successfully.")
    print(f"Original data shape: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"Numeric predictive features: {len(numeric_features)}")
    print(f"Categorical predictive features: {len(categorical_features)}")
    print(f"Feature set A size: {len(feature_sets['full'])}")
    print(f"Feature set B size: {len(feature_sets['reduced'])}")
    print(f"Train samples: {len(full_split['y_train'])}")
    print(f"Test samples: {len(full_split['y_test'])}")
    print(f"Train/test index overlap count: {train_overlap_count}")
    print(f"Same split for feature sets A and B: {same_split}")
    print("Train target distribution:")
    for row in class_distribution(full_split["y_train"]):
        print(f"- {row['class']} ({row['encoded_value']}): {row['count']} ({row['proportion']:.2f}%)")
    print("Test target distribution:")
    for row in class_distribution(full_split["y_test"]):
        print(f"- {row['class']} ({row['encoded_value']}): {row['count']} ({row['proportion']:.2f}%)")
    print(f"Generated: {metrics_dir / 'preprocessing_summary.txt'}")
    print(f"Generated: {metrics_dir / 'feature_sets.csv'}")
    print(f"Generated: {metrics_dir / 'preprocessing_validation.csv'}")
    print(f"Generated: {metrics_dir / 'transformed_feature_names.csv'}")
    print("Preprocessor validation summary:")
    print(validation_results.to_string(index=False))
    print(
        "Validation used fit_transform on X_train and transform on X_test only; "
        "no fitted preprocessors or transformed datasets were saved."
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
