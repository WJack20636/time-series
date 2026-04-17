# 时间序列异常检测可视化汇报示例（科学计算可视化）

本仓库提供一套可直接运行的 Python 示例代码，用公开数据集展示“时间序列异常检测”的可视化汇报流程，覆盖：

- 背景介绍
- 具体方法
- 用可视化展示研究难点
- 生成可用于课堂汇报的图
- 典型应用场景

---

## 1. 数据集（公开）

使用 **NAB（Numenta Anomaly Benchmark）** 公开数据集中的样例：

- 文件：`data/nab/realKnownCause/ambient_temperature_system_failure.csv`
- 标签：`data/nab/labels/combined_windows.json`

该样例是温度传感器时间序列，包含已标注异常时间窗口，适合教学演示。

---

## 2. 方法设计（用于汇报）

示例脚本 `visualize_ts_anomaly.py` 对同一数据执行两种方法并可视化对比：

1. **Robust Z-Score（基于滚动中位数 + MAD）**
   - 强调“可解释性强、实现简单”
   - 适合做传统统计基线

2. **Isolation Forest（机器学习方法）**
   - 基于多维时序特征（值、差分、滞后、局部统计）
   - 适合展示“从统计到学习方法”的升级路线

---

## 3. 可视化如何体现研究难点

脚本会输出 4 张图到 `figures/`：

1. `01_raw_with_ground_truth.png`
   - 原始序列 + 真实异常窗口（难点：异常稀疏、被趋势/噪声淹没）

2. `02_robust_zscore_detection.png`
   - 统计法检测结果 + 分数阈值图（难点：阈值敏感、误报漏报平衡）

3. `03_iforest_detection.png`
   - 机器学习法检测结果 + 分数图（难点：特征工程与污染率设定）

4. `04_method_comparison.png`
   - Precision/Recall/F1 对比 + 同图叠加预测点（难点：评价指标权衡）

---

## 4. 运行方式（直接出图）

在仓库根目录执行：

```bash
python -m pip install -r requirements.txt
python visualize_ts_anomaly.py
```

运行后会在 `figures/` 下生成全部示例图。

---

## 5. 汇报建议结构（可直接用）

1. 研究背景与意义：工业监控、运维、金融风控中的异常检测需求
2. 数据与任务定义：单变量时间序列 + 已标注异常窗口
3. 方法路线：统计基线（Robust Z）→ 学习方法（Isolation Forest）
4. 可视化展示难点：
   - 异常稀疏性
   - 阈值选择
   - 误报/漏报权衡
   - 评价指标多目标冲突
5. 应用场景：设备故障预警、云服务监控告警、环境传感器异常发现

---

## 6. 关于“论文截图”

你可以在课程 PPT 中补充相关论文中的示意图（务必标注来源与引用），与本仓库生成图形成“方法原理 + 实证示例”的组合展示。
