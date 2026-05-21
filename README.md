# 空气质量协同变化预测与分级决策模型

## 0. Question

在评价空气质量时，PM2.5、PM10、O₃、NO₂、SO₂、CO是主要的污染物，其浓度受污染源排放和气象条件共同影响。
比如，温度与湿度就会显著改变污染物浓度的时空分布。这些影响可能会立刻反映为污染物浓度变化，也可能会有一定滞后性，即此前一段时间的温度、湿度导致了后来的污染物浓度变化。

附件中给出了北京房山良乡地区2026年1月1日-3月31日逐小时的空气质量监测数据，包含PM2.5 (μg/m³)、PM10 (μg/m³)、O₃ (μg/m³)、NO₂ (μg/m³)、SO₂ (μg/m³)、CO (mg/m³)和同时段温度（℃）、相对湿度（%）。
请根据所给数据，如果需要可以结合其它数据(自己决定要不要补充数据、补充什么数据、如何补充)，建立模型回答如下问题：
1、分析单一污染物浓度与时间、温度、湿度之间的关系，并给出合理解释。
2、建立数学模型，基于时间、温度、湿度预测六种污染物浓度的协同变化规律。
3、请根据温湿条件与污染物变化规律，建立空气质量分级、预测模型。

## 1. Overview

本仓库包含了用于分析并预测空气中六种主要污染物（PM2.5、PM10、O₃、NO₂、SO₂、CO）协同变化规律的深度学习代码与探索性数据分析实验代码。

**主要方法与核心思路：**

- **特征挖掘阶段**：使用交叉相关函数 (CCF) 与 XGBoost 结合的 SHAP 值特征归因分析，衡量时间、温度、湿度等气象变量对单一污染物的即时或滞后影响。
- **时空预测机制**：构建了多输入多输出 (MIMO) 架构的深度学习网络（**LSTM + Attention**）。其中，LSTM 编码器捕捉历史气象与排放的序列时间依赖性，`Temporal Attention` 机制则能够提取能够极大影响未来浓度的特定关键历史时间步。注意力权重可视化热力图为论文的协同变化解释提供了可量化的支撑。
- **规则决策**：基于模型预测得到的未来浓度的具体数值，依照国家标准（HJ 633-2012）计算并返回 AQI，构建校园等场所的活动指南推荐引擎。

## 2. Requirements

复现本仓库的实验需配备以下 Python 配置（Python 3.12）：

```text
torch>=2.0.0
numpy
pandas
scikit-learn
matplotlib
seaborn
joblib
xgboost
shap
nbformat
jupyter
```

## 3. Files

目录结构与各文件作用如下：

```text
├── data/
│   ├── raw/                     # 原始数据集
│   └── processed/               # 切分后数据集
├── notebooks/
│   ├── Data_Preprocessing.ipynb # 数据清理与切分
│   └── CCF_and_SHAP.ipynb       # 交叉相关性与 SHAP 归因
├── src/                         # 深度学习源码
│   ├── config.py                # 全局训练超参数
│   ├── dataset_builder.py       # 归一化及滑动窗口切片
│   ├── model.py                 # LSTM 网络定义
│   ├── train.py
│   ├── test.py
│   └── aqi_decision_system.py   # 决策系统框架
├── results/                     # 结果输出存放
│   ├── figures/                 # SHAP 瀑布图、相关性柱状图及测试曲线、热力图
│   ├── models/                  # 验证集上最佳的权重文件
│   └── predictions/             # 测试集的污染物预测
└── paper/
```

## 4. Usage

复现实验并回答题目中三个问题的操作步骤如下：

### 环境准备

所有命令应当在一个独立的 Conda 环境中执行：

```bash
conda create -n pollution python=3.12
conda activate pollution
pip install -r requirements.txt
```

### 问题一：特征分析

1. 将原始监测数据放置入 `data/raw/` 文件夹中

2. 运行 `notebooks/Data_Preprocessing.ipynb`，生成清理好的 `train_set.csv`, `val_set.csv`, `test_set.csv`，并会自动落盘至 `data/processed/` 目录

3. 运行 `notebooks/CCF_and_SHAP.ipynb`，即可得出温湿条件对各污染物的最大滞后阶数，以及 SHAP 的特征重要性分布图（图片保存在 `results/figures/`）

### 问题二：协同预测规律构建

1. 调整 `src/config.py`

2. 开始模型训练：

   ```bash
   python src/train.py
   ```

   训练脚本将在每个 Epoch 输出 Loss，并在满足收敛条件后自动保存表现最好的模型状态至 `results/models/best_forecaster.pth`。

3. 执行模型评估与可解释性导出：
   ```bash
   python src/test.py
   ```
   读取最优权重在测试集上预测，并打印出各项评价指标，提取 Attention 权重并绘制出热力图

### 问题三：AQI 评级预测

1. 在获得了以上步长预测的污染物浓度后，执行或将其引入至 `src/aqi_decision_system.py` 中

2. 该引擎将根据六项主要污染物的最高分项（IAQI）敲定整体空气污染指数（AQI），结合温湿度为您输出最终的校园活动指引等级
