1. data/ (数据中心：存放所有输入和输出数据)

raw/
data_official.xslx: 提供的基本数据
data_stream-oper_stepType-accum.nc: 补充数据
data_stream-oper_stepType-instant.nc: 补充数据

processed/
data——merged.csv: 拼接合并、处理完缺失值后的全量完整数据集。
train_set.csv: 训练集（前70%的时间段，约1月1日-3月4日）。
val_set.csv: 验证集（中间15%的时间段，约3月5日-3月17日）。
test_set.csv: 测试集（最后15%的时间段，约3月18日-3月31日）。

2. notebooks/ (探索性分析与问题一实验)
   使用 Jupyter Notebook，方便随时查看图表和数据中间状态。

Data_Preprocessing.ipynb: 数据清洗流。负责读取原始数据，进行异常值检测、缺失值插补、特征对齐，并执行 70:15:15 的数据集切分，最后将文件保存到 data/processed/。

CCF_and_SHAP.ipynb: 针对第一问。负责计算单一污染物与温湿度的交叉相关系数（CCF），寻找最大滞后阶数；随后调用简单的基于树的回归模型（如 XGBoost），输出特征重要性和 SHAP 瀑布图并保存图片。

3. src/ (核心源代码：问题二与问题三的模型构建与训练)
   采用 Python 脚本形式，保证结构清晰，便于多次调参运行。

config.py: 全局配置文件。集中存放所有超参数（如滑动窗口大小、LSTM隐藏层维度、学习率、Epoch数量、文件路径等）。只需修改这里，即可控制整个训练流程。

dataset_builder.py: 数据集构造器。负责读取划分好的训练/验证/测试集，执行归一化（记录 Scaler），并利用滑动窗口（Sliding Window）技术将一维时间序列转换为深度学习模型需要的三维张量 (Samples, Time_Steps, Features)。

model.py: 第二问网络拓扑定义文件。仅负责搭建包含 Attention 机制的 LSTM 神经网络结构，定义输入层、LSTM层、注意力权重计算层和全连接输出层，不涉及训练逻辑。

train.py: 模型训练引擎。读取数据和模型架构，配置优化器（如 Adam）和损失函数，执行前向传播和反向传播；实现早停机制（Early Stopping），监控 val_set 的 Loss，保存表现最好的模型权重。

test.py: 模型测试与解释文件。在 test_set 上进行预测，执行反归一化；计算各项评价指标（RMSE, MAE）；最重要的是，提取 Attention 层的权重矩阵，将其可视化为热力图，用于解释第二问的“协同变化规律”。

(这部分暂且搁置) aqi_decision_system.py: 第三问专属逻辑文件。接收模型的预测输出，根据国家标准 HJ 633-2012 计算 IAQI 和最终 AQI 等级；根据预测等级和温湿度条件，输出对应的校园活动建议（规则引擎）。

4. results/ (结果输出：存放所有用于写进论文的图表和数据)
   figures/:

q1_ccf_lag.png: 第一问的滞后相关性图。

q1_shap_summary.png: 第一问的 SHAP 特征归因图。

q2_loss_curve.png: 训练集与验证集的 Loss 下降曲线（证明模型未过拟合）。

q2_attention_heatmap.png: 第二问提取的注意力机制热力图（证明协同规律）。

q2_prediction_vs_true.png: 六种污染物在测试集上的预测值与真实值折线对比图。

models/:

best_lstm_attn_model.pth (或 .h5): 训练好的最优模型权重文件。

data_scaler.pkl: 归一化缩放器对象（保证预测时使用的是与训练时一致的量纲）。

predictions/:

test_set_predictions.csv: 模型在测试集上预测出的未来污染物浓度具体数值，方便查阅或复制到论文附录中。

5. paper/ (论文撰写目录)
   main.tex: 论文正文。
