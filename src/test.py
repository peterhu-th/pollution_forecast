import os
import torch
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from config import Config
from dataset_builder import build_dataloader
from model import SpatioTemporalForecaster


def calculate_metrics(y_true, y_pred):
    """计算 RMSE 和 MAE"""
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2, axis=0))
    mae = np.mean(np.abs(y_true - y_pred), axis=0)
    return rmse, mae

def evaluate_model():
    device = torch.device(Config.DEVICE if torch.cuda.is_available() else 'cpu')
    
    # 加载 Scaler
    if not os.path.exists(Config.SCALER_PATH):
        raise FileNotFoundError(f"Scaler not found at {Config.SCALER_PATH}. Please train the model first.")
    scaler = joblib.load(Config.SCALER_PATH)
    
    # 构建测试集数据加载器
    test_loader, _ = build_dataloader(
        file_path=Config.TEST_DATA_PATH,
        seq_length=Config.SEQ_LENGTH,
        pred_horizon=Config.PRED_HORIZON,
        batch_size=Config.BATCH_SIZE,
        is_train=False,
        scaler=scaler
    )
    
    # 加载模型与权重
    model = SpatioTemporalForecaster(
        input_dim=Config.NUM_FEATURES,
        hidden_dim=Config.HIDDEN_DIM,
        num_layers=Config.NUM_LAYERS,
        pred_horizon=Config.PRED_HORIZON,
        num_targets=Config.NUM_TARGETS,
        seq_length=Config.SEQ_LENGTH,
        dropout_rate=0.0
    ).to(device)
    
    if os.path.exists(Config.BEST_MODEL_PATH):
        model.load_state_dict(torch.load(Config.BEST_MODEL_PATH, map_location=device))
    else:
        print("Warning: Best model weights not found. Using untrained weights.")
    
    model.eval()
    all_preds, all_trues = [], []
    
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            predictions = model(batch_x)            
            all_preds.append(predictions.cpu().numpy())
            all_trues.append(batch_y.cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_trues = np.concatenate(all_trues, axis=0)
    
    # 反归一化 (Inverse Transform)
    # 预测形状 (Samples, Horizon, Num_Targets)
    samples, horizon, targets = all_preds.shape
    preds_flat = all_preds.reshape(-1, targets)
    trues_flat = all_trues.reshape(-1, targets)
    
    dummy_preds = np.zeros((preds_flat.shape[0], Config.NUM_FEATURES))
    dummy_trues = np.zeros((trues_flat.shape[0], Config.NUM_FEATURES))
    
    target_indices = list(range(Config.NUM_TARGETS)) 
    
    for i, target_idx in enumerate(target_indices):
        dummy_preds[:, target_idx] = preds_flat[:, i]
        dummy_trues[:, target_idx] = trues_flat[:, i]
    
    preds_inv = scaler.inverse_transform(dummy_preds)[:, target_indices]
    trues_inv = scaler.inverse_transform(dummy_trues)[:, target_indices]
    
    preds_inv = preds_inv.reshape(samples, horizon, targets)
    trues_inv = trues_inv.reshape(samples, horizon, targets)

    # 计算指标
    rmse, mae = calculate_metrics(trues_inv.reshape(-1, targets), preds_inv.reshape(-1, targets))
    for i, feature in enumerate(Config.TARGET_FEATURES):
        print(f"[{feature}] RMSE: {rmse[i]:.2f}, MAE: {mae[i]:.2f}")

    os.makedirs(Config.FIGURE_DIR_EVAL, exist_ok=True)

    # Evaluation 1: 计算随预测步长变化的 RMSE/MAE
    horizon_rmse = np.sqrt(np.mean((trues_inv - preds_inv)**2, axis=0)) # (Horizon, Targets)
    
    plt.figure(figsize=(10, 6))
    for i, feature in enumerate(Config.TARGET_FEATURES):
        plt.plot(range(1, horizon + 1), horizon_rmse[:, i], label=feature, marker='o', markersize=4)
        print(f"[{feature}] 1h RMSE: {horizon_rmse[0, i]:.2f}, {horizon}h RMSE: {horizon_rmse[-1, i]:.2f}")
    plt.xlabel('Future Time Steps (Hours)')
    plt.ylabel('RMSE')
    plt.title('Prediction Error Decay over Time Horizon')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(Config.FIGURE_DIR_EVAL, 'horizon_rmse_decay.png'), bbox_inches='tight')
    
    # Evaluation 2: 具体样本的时间序列折线图
    sample_idx = np.random.randint(0, samples)
    plt.figure(figsize=(15, 8))
    for i, feature in enumerate(Config.TARGET_FEATURES):
        plt.subplot(2, 3, i+1)
        plt.plot(range(horizon), trues_inv[sample_idx, :, i], label='True', marker='.')
        plt.plot(range(horizon), preds_inv[sample_idx, :, i], label='Pred', marker='x')
        plt.title(f"{feature} Forecast")
        plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(Config.FIGURE_DIR_EVAL, 'sample_forecast_series.png'))
    

if __name__ == '__main__':
    evaluate_model()
