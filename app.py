import streamlit as st
import pandas as pd
import numpy as np
import torch
import joblib
import os
from datetime import datetime, timedelta

from src.config import Config
from src.model import SpatioTemporalForecaster


# 缓存加载函数
@st.cache_data
def load_data():
    if not os.path.exists(Config.FULL_DATA_PATH):
        st.error(f"找不到全量数据集文件 {Config.FULL_DATA_PATH}")
        st.stop()

    df = pd.read_csv(Config.FULL_DATA_PATH)
    if 'datetime' not in df.columns:
        st.error("数据集中未找到 'datetime' 列！请确保数据格式正确。")
        st.stop()
        
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.set_index('datetime')
    return df

@st.cache_resource
def load_model_and_scaler():
    device = torch.device(Config.DEVICE if torch.cuda.is_available() else 'cpu')
    model = SpatioTemporalForecaster(
        input_dim=Config.NUM_FEATURES,
        hidden_dim=Config.HIDDEN_DIM,
        num_layers=Config.NUM_LAYERS,
        pred_horizon=Config.PRED_HORIZON,
        num_targets=Config.NUM_TARGETS,
        seq_length=Config.SEQ_LENGTH,
        dropout_rate=Config.DROPOUT_RATE
    ).to(device)
    
    if not os.path.exists(Config.BEST_MODEL_PATH):
        st.error(f"找不到模型权重文件 {Config.BEST_MODEL_PATH}，请先运行训练脚本！")
        st.stop()
        
    model.load_state_dict(torch.load(Config.BEST_MODEL_PATH, map_location=device))
    model.eval()
        
    if not os.path.exists(Config.SCALER_PATH):
        st.error(f"找不到归一化处理器文件 {Config.SCALER_PATH}，请先运行训练脚本！")
        st.stop()
        
    scaler = joblib.load(Config.SCALER_PATH)
        
    return model, scaler, device


# 模糊综合评价 (FCE) 数学模块
def calc_pollutant(x, c1, c2, c3, c4):
    """偏小型隶属度函数"""
    u = [0.0, 0.0, 0.0, 0.0]
    # U1
    if x <= c1: u[0] = 1.0
    elif c1 < x <= c2: u[0] = (c2 - x) / (c2 - c1)
    
    # U2
    if c1 < x <= c2: u[1] = (x - c1) / (c2 - c1)
    elif c2 < x <= c3: u[1] = (c3 - x) / (c3 - c2)
    
    # U3
    if c2 < x <= c3: u[2] = (x - c2) / (c3 - c2)
    elif c3 < x <= c4: u[2] = (c4 - x) / (c4 - c3)
    
    # U4
    if c3 < x <= c4: u[3] = (x - c3) / (c4 - c3)
    elif x > c4: u[3] = 1.0
    
    return u




# 页面布局 - Sidebar
st.sidebar.title("⚙️ 预测引擎配置")
df_data = load_data()
min_date = df_data.index.min().date() + timedelta(days=1) # 预留24小时历史
max_date = df_data.index.max().date()

selected_date = st.sidebar.date_input("选择预测日期 T", value=max_date, min_value=min_date, max_value=max_date)
selected_hour = st.sidebar.slider("选择预测小时 T", 0, 23, 12)

# 计算绝对时间 T
T = datetime.combine(selected_date, datetime.min.time()) + timedelta(hours=selected_hour)

run_engine = st.sidebar.button("启动预测与评估")


# 页面布局 - Main Container
st.title("🍃 校园空气质量与活动协同决策系统")

if run_engine:
    model, scaler, device = load_model_and_scaler()
    
    # 截取历史 24 小时数据 [T-24h, T-1h]
    start_history = T - timedelta(hours=Config.SEQ_LENGTH)
    end_history = T - timedelta(hours=1)
    
    # 安全索引处理
    history_df = df_data.loc[start_history:end_history]
    if len(history_df) < Config.SEQ_LENGTH:
        st.error(f"时间截断错误：从 {start_history} 到 {end_history} 的数据不足 {Config.SEQ_LENGTH} 条，请选择更晚的时间！")
        st.stop()
        
    history_df = history_df.iloc[-Config.SEQ_LENGTH:]
    

        
    # 推理
    x_input = history_df[Config.FEATURE_NAMES].values
    x_scaled = scaler.transform(x_input)
    x_tensor = torch.tensor(x_scaled, dtype=torch.float32).unsqueeze(0).to(device)
    
    with torch.no_grad():
        pred_tensor = model(x_tensor)
        
    pred_numpy = pred_tensor.squeeze(0).cpu().numpy() # Shape: (PRED_HORIZON, NUM_TARGETS)
    
    # 反归一化
    dummy_pred = np.zeros((Config.PRED_HORIZON, Config.NUM_FEATURES))
    dummy_pred[:, :Config.NUM_TARGETS] = pred_numpy
    pred_inv = scaler.inverse_transform(dummy_pred)[:, :Config.NUM_TARGETS]
    
    # 提取 T 时刻 (第 1 个小时) 的预测浓度用于 FCE 计算
    pm25_val = pred_inv[0, 0]
    pm10_val = pred_inv[0, 1]
    o3_val = pred_inv[0, 2]
    

    # FCE 矩阵运算
    u_pm25 = calc_pollutant(pm25_val, 35, 75, 115, 150)
    u_pm10 = calc_pollutant(pm10_val, 50, 150, 250, 350)
    u_o3 = calc_pollutant(o3_val, 100, 160, 215, 265)
    R = np.array([u_pm25, u_pm10, u_o3]) # 3x4
    W = np.array([0.4, 0.3, 0.3])        # 1x3
    B = np.dot(W, R)                             # 1x4
    
    # 评价等级
    level_idx = np.argmax(B)
    levels = ["V1 (极度适宜/优)", "V2 (适宜/良)", "V3 (较不适宜/轻度污染)", "V4 (不适宜/中重度污染)"]
    final_level = levels[level_idx]
    
    # 首要污染物计算
    ratio_pm25 = pm25_val / 75.0
    ratio_pm10 = pm10_val / 150.0
    ratio_o3 = o3_val / 160.0
    ratios = {'PM2.5': ratio_pm25, 'PM10': ratio_pm10, 'O3': ratio_o3}
    primary_pollutant = max(ratios, key=ratios.get)
    primary_val = {'PM2.5': pm25_val, 'PM10': pm10_val, 'O3': o3_val}[primary_pollutant]
    
    
    # 核心结论 Dashboard
    st.markdown("### 📊 核心评价指标")
    c1, c2, c3 = st.columns(3)
    c1.metric("综合等级 (FCE评判)", final_level)
    c2.metric("首要污染物及预测浓度", f"{primary_pollutant} ({primary_val:.1f})")
    
    # PM2.5 峰值预警
    eval_horizon = min(12, Config.PRED_HORIZON)
    pm25_future = pred_inv[:eval_horizon, 0]
    c3.metric("未来 12H PM2.5 峰值", f"{np.max(pm25_future):.1f}")
    
    st.divider()
    
    
    # 智能活动决策
    st.markdown("### 💡 智能活动决策推荐")
    if level_idx == 0:
        st.success("✅ **极度适宜户外活动与开窗通风**：空气清新，各项指标优秀，鼓励校园师生积极开展户外体育活动。")
    elif level_idx == 1:
        st.info("ℹ️ **适宜常规活动**：空气质量良好，基本无健康风险，可正常进行室外课程及活动。")
    elif level_idx == 2:
        st.warning("⚠️ **减少剧烈运动 / 敏感人群防护**：空气呈轻度污染，儿童、呼吸道疾病等敏感人群应尽量减少长时间的高强度户外锻炼。")
    else:
        st.error("🚫 **不适宜 / 室内活动建议**：中重度污染预警，强烈建议停止所有户外体育课程，师生佩戴口罩，关闭教室门窗并开启空气净化器。")
        
    st.divider()
    

    # 时空协同趋势图
    st.markdown("### 📈 时空协同趋势 (历史 24H vs 未来 12H)")
    
    future_times = [T + timedelta(hours=i) for i in range(eval_horizon)]
    
    # 构建绘图用的 DataFrame
    plot_data_history = history_df[Config.TARGET_FEATURES].copy()
    plot_data_history['Type'] = 'History (True)'
    
    # 仅绘制 TARGET_FEATURES
    plot_data_future = pd.DataFrame(pred_inv[:eval_horizon, :], columns=Config.TARGET_FEATURES, index=future_times)
    plot_data_future['Type'] = 'Prediction (LSTM)'
    
    df_plot = pd.concat([plot_data_history, plot_data_future])
    
    # 利用 line_chart 展示各变量走势
    st.line_chart(df_plot.drop(columns=['Type']))
    st.caption("注：折线图前半段为过去 24 小时的实际监测值，后半段为模型输出的多步长协同预测值。")
    
    st.divider()


    # 数学模型展示
    with st.expander("📊 查看底层模糊合成数学细节 (FCE Matrix)"):
        st.write("根据模糊综合评价理论，合成评价向量由权重矩阵与隶属度评价矩阵点积得到：")
        st.latex(r"B = W \cdot R")
        
        st.markdown("**1. 权重向量 W (Weight)**")
        df_W = pd.DataFrame([W], columns=["PM2.5", "PM10", "O3"])
        st.dataframe(df_W)
        
        st.markdown("**2. 评价矩阵 R (隶属度)**")
        df_R = pd.DataFrame(R, index=["PM2.5", "PM10", "O3"], columns=["V1 (优)", "V2 (良)", "V3 (轻度)", "V4 (中重度)"])
        st.dataframe(df_R.style.format("{:.4f}"))
        
        st.markdown("**3. 合成评价结果 B**")
        df_B = pd.DataFrame([B], columns=["V1", "V2", "V3", "V4"])
        st.dataframe(df_B.style.format("{:.4f}"))
        st.write(f"最大值出现于 **V{level_idx+1}**，得分：`{np.max(B):.4f}`，由此确定最终等级。")