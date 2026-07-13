# AGENTS.md

## Project Rules

1. 项目是结构化表格数据二分类任务。
2. 目标字段为 `failed_phishing_simulation`。
3. `Yes` 表示员工在网络钓鱼模拟中失败，`No` 表示未失败。
4. 原始数据位于 `data/enterprise_phishing_simulation_2026.csv`。
5. 未经明确允许，不得修改原始 CSV 文件。
6. 使用 Conda 环境 `phishing_ml`。
7. 使用 Python 解释器：

   ```text
   C:\Users\35533\.conda\envs\phishing_ml\python.exe
   ```

   运行脚本时使用：

   ```powershell
   & "C:\Users\35533\.conda\envs\phishing_ml\python.exe" <脚本路径>
   ```

8. 固定随机种子为 `42`。
9. 不得编造实验结果。
10. 数据预处理必须避免数据泄漏。
11. 参数调优不得使用测试集。
12. 计划训练：
    - Logistic Regression
    - Gaussian Naive Bayes
    - Random Forest
    - XGBoost
13. 分类评价指标至少包括：
    - Accuracy
    - Precision
    - Recall
    - F1-score
    - ROC-AUC
    - Confusion Matrix
14. 重点关注目标类别 `Yes` 的 Recall 和 F1。
15. 图表保存到 `results/figures`。
16. 表格和文本结果保存到 `results/metrics`。
17. 模型文件保存到 `models`。
18. 代码必须可以从项目根目录独立运行。
19. 每个阶段完成后说明创建或修改的文件、运行命令和运行结果。
20. 必须按照数据检查、EDA、预处理、基线模型、集成模型、参数调优、模型评价和误差分析的顺序逐步完成，不要一次完成全部实验。
