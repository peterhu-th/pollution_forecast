# -*- coding: utf-8 -*-
"""
模型测试与解释模块 (Model Testing and Interpretation)
在测试集上评估模型，输出 RMSE、MAE 等指标，并提取 Attention 权重生成热力图。
遵守学术界规范与代码架构要求。
"""

import os
import torch
import numpy as np
import pandas as pd
import joblib
# 假设有matplotlib等，如果未安装则可能会在画图时报错，提示用户安装
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
except ImportError:
    pass

from .config import Config
from .dataset_builder import build_dataloader
from .model import SpatioTemporalForecaster

def calculate_metrics(y_true, y_pred):
    """计算 RMSE 和 MAE"""
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2, axis=0))
    mae = np.mean(np.abs(y_true - y_pred), axis=0)
    return rmse, mae

def evaluate_model():
    device = torch.device(Config.DEVICE if torch.cuda.is_available() else 'cpu')
    print(f"[*] Evaluating on device: {device}")
    
    # 1. 加载 Scaler
    if not os.path.exists(Config.SCALER_PATH):
        raise FileNotFoundError(f"Scaler not found at {Config.SCALER_PATH}. Please train the model first.")
    scaler = joblib.load(Config.SCALER_PATH)
    
    # 2. 构建测试集数据加载器
    test_loader, _ = build_dataloader(
        file_path=Config.TEST_DATA_PATH,
        seq_length=Config.SEQ_LENGTH,
        pred_horizon=Config.PRED_HORIZON,
        batch_size=Config.BATCH_SIZE,
        is_train=False,
        scaler=scaler
    )
    
    # 3. 加载模型与权重
    model = SpatioTemporalForecaster(
        input_dim=Config.NUM_FEATURES,
        hidden_dim=Config.HIDDEN_DIM,
        num_layers=Config.NUM_LAYERS,
        pred_horizon=Config.PRED_HORIZON,
        num_targets=Config.NUM_TARGETS,
        dropout_rate=0.0 # 测试时不使用 Dropout
    ).to(device)
    
    if os.path.exists(Config.BEST_MODEL_PATH):
        model.load_state_dict(torch.load(Config.BEST_MODEL_PATH, map_location=device))
        print(f"[*] Loaded best model from {Config.BEST_MODEL_PATH}")
    else:
        print("[!] Warning: Best model weights not found. Using untrained weights.")
    
    model.eval()
    
    all_preds = []
    all_trues = []
    all_attentions = []
    
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            predictions, attn_weights = model(batch_x)
            
            all_preds.append(predictions.cpu().numpy())
            all_trues.append(batch_y.cpu().numpy())
            all_attentions.append(attn_weights.cpu().numpy())
            
    # 合并批次
    all_preds = np.concatenate(all_preds, axis=0)
    all_trues = np.concatenate(all_trues, axis=0)
    all_attentions = np.concatenate(all_attentions, axis=0)
    
    # 4. 反归一化 (Inverse Transform)
    # 注意: 模型预测形状为 (Samples, Horizon, Num_Targets)
    # 为了使用 Scaler反归一化，需要将其扩展到特征数维度
    samples, horizon, targets = all_preds.shape
    
    preds_flat = all_preds.reshape(-1, targets)
    trues_flat = all_trues.reshape(-1, targets)
    
    # 创建一个空的满特征数组来填充目标预测值
    dummy_preds = np.zeros((preds_flat.shape[0], Config.NUM_FEATURES))
    dummy_trues = np.zeros((trues_flat.shape[0], Config.NUM_FEATURES))
    
    dummy_preds[:, :targets] = preds_flat
    dummy_trues[:, :targets] = trues_flat
    
    preds_inv = scaler.inverse_transform(dummy_preds)[:, :targets]
    trues_inv = scaler.inverse_transform(dummy_trues)[:, :targets]
    
    # 5. 计算指标
    rmse, mae = calculate_metrics(trues_inv, preds_inv)
    print("\n--- Evaluation Results ---")
    for i, feature in enumerate(Config.TARGET_FEATURES):
        print(f"[{feature}] RMSE: {rmse[i]:.2f}, MAE: {mae[i]:.2f}")
        
    # 6. 保存预测结果
    os.makedirs(Config.FIGURE_DIR, exist_ok=True)
    
    # 提取某个样本的时间注意力权重进行热力图可视化 (如第一个样本)
    try:
        sample_attention = all_attentions[0].reshape(-1) # (Seq_Length,)
        plt.figure(figsize=(10, 2))
        sns.heatmap([sample_attention], cmap="YlGnBu", cbar=True, xticklabels=range(1, Config.SEQ_LENGTH+1))
        plt.title("Temporal Attention Weights for Sequence Context")
        plt.xlabel("Past Hours")
        plt.yticks([])
        heatmap_path = os.path.join(Config.FIGURE_DIR, 'q2_attention_heatmap.png')
        plt.savefig(heatmap_path, bbox_inches='tight')
        print(f"\n[*] Attention heatmap saved to: {heatmap_path}")
    except Exception as e:
        print(f"\n[!] Visualization skipped or failed: {e}")

if __name__ == '__main__':
    evaluate_model()
