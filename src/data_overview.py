from pathlib import Path
import sys

import pandas as pd
from pandas.api.types import is_numeric_dtype


TARGET_COLUMN = "failed_phishing_simulation"
DATA_RELATIVE_PATH = Path("data") / "enterprise_phishing_simulation_2026.csv"
OUTPUT_RELATIVE_PATH = Path("results") / "metrics" / "data_overview.txt"
TOP_N_CATEGORIES = 10
HIGH_CARDINALITY_RATIO = 0.9
HIGH_CARDINALITY_MIN_UNIQUE = 20
ID_KEYWORDS = ("id", "编号", "number", "code", "uuid")


def add_section(lines, title):
    lines.append("")
    lines.append("=" * 80)
    lines.append(title)
    lines.append("=" * 80)


def format_series(series):
    if series.empty:
        return "(无)"
    return series.to_string()


def detect_suspicious_columns(df):
    suspicious = []
    row_count = len(df)

    for column in df.columns:
        non_null = df[column].dropna()
        unique_count = non_null.nunique()
        unique_ratio = unique_count / len(non_null) if len(non_null) else 0
        column_lower = str(column).lower()

        reasons = []
        if any(keyword in column_lower for keyword in ID_KEYWORDS):
            reasons.append("字段名疑似ID/编号")
        if (
            row_count > 0
            and unique_count >= HIGH_CARDINALITY_MIN_UNIQUE
            and unique_ratio >= HIGH_CARDINALITY_RATIO
        ):
            reasons.append(f"唯一值比例较高({unique_ratio:.2%})")

        if reasons:
            suspicious.append(
                {
                    "field": column,
                    "unique_count": unique_count,
                    "unique_ratio": unique_ratio,
                    "reason": "；".join(reasons),
                }
            )

    return pd.DataFrame(suspicious)


def build_report(df, csv_path, output_path, output_dir_created):
    lines = []
    row_count, column_count = df.shape

    lines.append("阶段1：数据读取与数据质量检查")
    lines.append(f"项目根目录: {Path(__file__).resolve().parents[1]}")
    lines.append(f"原始数据路径: {csv_path}")
    lines.append(f"结果保存路径: {output_path}")
    lines.append("CSV文件存在: 是")
    lines.append("CSV读取成功: 是")
    lines.append(f"输出目录自动创建: {'是' if output_dir_created else '否'}")

    add_section(lines, "1. 数据集规模")
    lines.append(f"行数: {row_count}")
    lines.append(f"列数: {column_count}")

    add_section(lines, "2. 全部字段名称")
    for index, column in enumerate(df.columns, start=1):
        lines.append(f"{index}. {column}")

    add_section(lines, "3. 字段数据类型")
    lines.append(df.dtypes.astype(str).to_string())

    add_section(lines, "4. 非空数量")
    lines.append(df.notna().sum().to_string())

    add_section(lines, "5. 缺失值数量")
    missing_counts = df.isna().sum()
    lines.append(missing_counts.to_string())

    add_section(lines, "6. 缺失率(%)")
    missing_rates = (missing_counts / row_count * 100).round(2)
    lines.append(missing_rates.to_string())

    add_section(lines, "7. 完全重复记录")
    duplicate_count = int(df.duplicated().sum())
    lines.append(f"完全重复记录数量: {duplicate_count}")

    add_section(lines, "8. 目标字段检查")
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"目标字段不存在: {TARGET_COLUMN}")

    target = df[TARGET_COLUMN]
    target_unique = target.dropna().unique().tolist()
    target_counts = target.value_counts(dropna=False)
    target_proportions = (target_counts / row_count * 100).round(2)
    target_missing_count = int(target.isna().sum())

    lines.append(f"目标字段存在: 是")
    lines.append(f"目标字段: {TARGET_COLUMN}")
    lines.append(f"目标字段唯一类别: {target_unique}")
    lines.append("目标字段类别样本数量:")
    lines.append(target_counts.to_string())
    lines.append("目标字段类别样本比例(%):")
    lines.append(target_proportions.to_string())
    lines.append(f"目标字段缺失值数量: {target_missing_count}")
    lines.append(f"目标字段是否存在缺失值: {'是' if target_missing_count > 0 else '否'}")

    add_section(lines, "9. 数值型特征描述性统计")
    numeric_df = df.select_dtypes(include="number")
    lines.append(f"数值型字段数量: {numeric_df.shape[1]}")
    if numeric_df.empty:
        lines.append("(无数值型字段)")
    else:
        lines.append(
            numeric_df.describe().loc[
                ["count", "mean", "std", "min", "25%", "50%", "75%", "max"]
            ].to_string()
        )

    add_section(lines, "10. 类别型特征")
    categorical_columns = [column for column in df.columns if not is_numeric_dtype(df[column])]
    categorical_df = df[categorical_columns]
    lines.append(f"类别型字段数量: {len(categorical_columns)}")
    lines.append("类别型字段名称:")
    if categorical_columns:
        for column in categorical_columns:
            lines.append(f"- {column}")
    else:
        lines.append("(无类别型字段)")

    add_section(lines, "11. 类别型字段唯一取值数量")
    if categorical_columns:
        lines.append(categorical_df.nunique(dropna=True).to_string())
    else:
        lines.append("(无类别型字段)")

    add_section(lines, "12. 类别型字段各类别频数")
    if categorical_columns:
        for column in categorical_columns:
            value_counts = df[column].value_counts(dropna=False)
            lines.append("")
            lines.append(f"[{column}]")
            if len(value_counts) > TOP_N_CATEGORIES:
                lines.append(f"类别数较多，仅输出频数最高的前{TOP_N_CATEGORIES}项:")
                value_counts = value_counts.head(TOP_N_CATEGORIES)
            lines.append(format_series(value_counts))
    else:
        lines.append("(无类别型字段)")

    add_section(lines, "13. 疑似ID、编号或高基数字段检查")
    suspicious_df = detect_suspicious_columns(df)
    if suspicious_df.empty:
        lines.append("未发现疑似ID、编号或高基数字段。")
    else:
        lines.append("发现以下疑似ID、编号或高基数字段，本阶段不删除任何字段:")
        lines.append(
            suspicious_df.assign(
                unique_ratio=suspicious_df["unique_ratio"].map(lambda value: f"{value:.2%}")
            ).to_string(index=False)
        )

    add_section(lines, "14. 数据质量问题简要总结")
    issues = []
    if missing_counts.sum() > 0:
        missing_columns = missing_counts[missing_counts > 0]
        issues.append(f"存在缺失值字段 {len(missing_columns)} 个，缺失值总数 {int(missing_counts.sum())}。")
    else:
        issues.append("未发现缺失值。")

    if duplicate_count > 0:
        issues.append(f"存在完全重复记录 {duplicate_count} 条。")
    else:
        issues.append("未发现完全重复记录。")

    if target_missing_count > 0:
        issues.append(f"目标字段存在缺失值 {target_missing_count} 个。")
    else:
        issues.append("目标字段不存在缺失值。")

    expected_target_values = {"Yes", "No"}
    observed_target_values = set(target.dropna().astype(str).unique())
    unexpected_values = sorted(observed_target_values - expected_target_values)
    if unexpected_values:
        issues.append(f"目标字段存在预期外类别: {unexpected_values}。")
    else:
        issues.append("目标字段类别均在预期范围 Yes/No 内。")

    if suspicious_df.empty:
        issues.append("未发现疑似ID、编号或高基数字段。")
    else:
        issues.append(f"发现疑似ID、编号或高基数字段 {len(suspicious_df)} 个，本阶段未做删除处理。")

    for index, issue in enumerate(issues, start=1):
        lines.append(f"{index}. {issue}")

    return "\n".join(lines) + "\n"


def main():
    project_root = Path(__file__).resolve().parents[1]
    csv_path = project_root / DATA_RELATIVE_PATH
    output_path = project_root / OUTPUT_RELATIVE_PATH
    output_dir = output_path.parent

    if not csv_path.exists():
        print(f"错误: CSV文件不存在: {csv_path}", file=sys.stderr)
        return 1

    output_dir_created = False
    if not output_dir.exists():
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            output_dir_created = True
        except OSError as exc:
            print(f"错误: 输出目录不存在且无法创建: {output_dir}；原因: {exc}", file=sys.stderr)
            return 1

    try:
        df = pd.read_csv(csv_path)
    except Exception as exc:
        print(f"错误: CSV读取失败: {csv_path}；原因: {exc}", file=sys.stderr)
        return 1

    if df.empty:
        print(f"错误: 数据集为空: {csv_path}", file=sys.stderr)
        return 1

    if TARGET_COLUMN not in df.columns:
        print(f"错误: 目标字段不存在: {TARGET_COLUMN}", file=sys.stderr)
        return 1

    try:
        report = build_report(df, csv_path, output_path, output_dir_created)
    except Exception as exc:
        print(f"错误: 生成数据检查报告失败；原因: {exc}", file=sys.stderr)
        return 1

    print(report, end="")

    try:
        output_path.write_text(report, encoding="utf-8")
    except OSError as exc:
        print(f"错误: 写入结果文件失败: {output_path}；原因: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
