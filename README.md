# 基于员工行为与安全意识特征的企业网络钓鱼模拟失败风险预测

## 项目背景

本项目面向企业网络安全培训与钓鱼邮件模拟场景，基于员工行为特征和安全意识相关特征，预测员工是否存在网络钓鱼模拟失败风险。

## 任务说明

本项目是结构化表格数据上的二分类机器学习任务，用于预测员工在企业网络钓鱼模拟中是否失败。

## 目标变量

目标字段为 `failed_phishing_simulation`。

- `Yes`：员工在网络钓鱼模拟中失败。
- `No`：员工未在网络钓鱼模拟中失败。

## 数据文件位置

原始数据文件位于：

```text
data/enterprise_phishing_simulation_2026.csv
```

## 计划使用模型

- Logistic Regression
- Gaussian Naive Bayes
- Random Forest
- XGBoost

## 项目目录说明

```text
data/       原始数据文件
notebooks/  实验记录和探索性分析笔记本
src/        项目 Python 源代码
results/    实验输出结果
models/     训练后的模型文件
reports/    项目报告和说明文档
```

## 当前项目进度

项目初始化完成。
