# 企业员工网络钓鱼模拟失败风险预测

## 1. 项目简介

本项目利用员工基本信息、安全意识、邮件使用行为、密码安全习惯和安全防护措施等结构化表格特征，预测员工是否会在企业网络钓鱼模拟中失败。

- 任务类型：二分类
- 目标字段：`failed_phishing_simulation`
- `Yes / 1`：员工在网络钓鱼模拟中失败
- `No / 0`：员工未在网络钓鱼模拟中失败

本项目预测的是员工在模拟场景中的失败风险，不是钓鱼邮件识别任务，也不是直接预测员工安全意识高低。

## 2. 项目实际意义

本项目可用于探索如何提前识别更容易在钓鱼攻击中上当的员工，为差异化安全培训提供参考，并帮助企业优化安全培训和防护资源配置。

当前数据为合成模拟数据，实验结果主要用于验证机器学习方法在该类风险预测问题中的可行性，不能直接代表真实企业应用效果。

## 3. 数据集概况

数据文件位于：

```text
data/enterprise_phishing_simulation_2026.csv
```

数据来源信息将在最终报告中补充。

当前数据质量检查结果如下：

- 样本数量：100000
- 字段数量：20
- 预测特征数量：19
- 数值型预测特征：11
- 类别型预测特征：8
- 目标类别 `No`：62686，占 62.69%
- 目标类别 `Yes`：37314，占 37.31%
- 完全重复记录：0
- 存在缺失值的字段数量：6
- 每个缺失字段缺失 1000 条，缺失率为 1%

存在缺失值的字段：

- `password_manager_usage`
- `security_quiz_score`
- `verification_before_click`
- `reporting_suspicious_email`
- `antivirus_installed`
- `vpn_usage`

## 4. 数据预处理与特征工程

当前已完成数据预处理方案设计和预处理器验证：

- 目标变量映射：`No -> 0`，`Yes -> 1`
- 数值缺失值使用中位数填充
- 类别缺失值使用众数填充
- 类别特征使用 `OneHotEncoder`
- 逻辑回归和 Gaussian Naive Bayes 的数值特征使用 `StandardScaler` 标准化
- 随机森林和 XGBoost 的数值特征不进行标准化
- 使用 80% 训练集和 20% 测试集分层划分
- 固定随机种子为 42
- 预处理器只在训练集上拟合，测试集只执行 `transform`

两套特征方案将用于后续消融对比：

- 完整特征方案：19 个原始特征，预处理转换后 36 个特征
- 精简特征方案：删除 `cybersecurity_awareness_score`，剩余 18 个原始特征，预处理转换后 35 个特征

当前不额外构造新的组合特征；原始数据已经包含多项安全意识和行为评分，且多个评分字段之间相关性较高，后续将通过特征重要性和消融实验判断特征价值。

## 5. 计划使用的模型

后续计划训练以下模型：

1. Logistic Regression
2. Gaussian Naive Bayes
3. Random Forest
4. XGBoost

模型训练尚未开始，因此当前 README 不包含准确率、召回率、F1-score 或其他模型性能结果。

## 6. 评价指标

后续分类模型评价将至少包括：

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion Matrix

本项目重点关注目标类别 `Yes` 的 Recall 和 F1-score，因为该类别表示员工在网络钓鱼模拟中失败。

## 7. 项目结构

```text
data/              原始数据文件
models/            后续保存训练后的模型文件
notebooks/         实验记录和探索性分析笔记本
reports/           项目报告和答辩材料
results/figures/   EDA 和模型评价图表
results/metrics/   统计表格、文本结果和模型指标
src/               项目 Python 源代码
AGENTS.md          项目执行规则
README.md          项目说明文档
requirements.txt   项目依赖列表
```

## 8. 当前进度

- [x] 项目初始化
- [x] 数据读取与数据质量检查
- [x] 探索性数据分析
- [x] 数据预处理
- [x] 基础特征工程与预处理验证
- [ ] 基线模型训练
- [ ] 随机森林与XGBoost训练
- [ ] 超参数调优
- [ ] 模型对比
- [ ] 特征重要性与误分类分析
- [ ] 报告和答辩PPT整理

## 9. 运行环境

- Python 3.11
- pandas
- numpy
- scikit-learn
- matplotlib
- xgboost

## 10. 运行说明

运行前需要先激活项目对应的 Python 环境。各阶段脚本可从项目根目录运行：

```bash
python src/data_overview.py
python src/eda_analysis.py
python src/data_preprocessing.py
```

当前阶段尚未开始模型训练。后续训练脚本和模型评价脚本将在对应阶段补充。
