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
        dropout_rate=0.0
    ).to(device)
    
    if os.path.exists(Config.BEST_MODEL_PATH):
        model.load_state_dict(torch.load(Config.BEST_MODEL_PATH, map_location=device))
    else:
        print("Warning: Best model weights not found. Using untrained weights.")
    
    model.eval()
    all_preds, all_trues, all_feat_attns = [], [], []
    
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            predictions, temp_attn, feat_attn = model(batch_x)            
            all_preds.append(predictions.cpu().numpy())
            all_trues.append(batch_y.cpu().numpy())
            all_feat_attns.append(feat_attn.cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_trues = np.concatenate(all_trues, axis=0)
    all_feat_attns = np.concatenate(all_feat_attns, axis=0)
    
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
    
    # 计算指标
    rmse, mae = calculate_metrics(trues_inv, preds_inv)
    for i, feature in enumerate(Config.TARGET_FEATURES):
        print(f"[{feature}] RMSE: {rmse[i]:.2f}, MAE: {mae[i]:.2f}")

    os.makedirs(Config.FIGURE_DIR, exist_ok=True)
    
    # 提取时间注意力权重可视化
    sample_attention = all_feat_attns[0].reshape(-1) # (NUM_FEATURES,)
    plt.figure(figsize=(12, 4))
    sns.heatmap([sample_attention], cmap="YlGnBu", cbar=True, xticklabels=range(1, Config.SEQ_LENGTH+1))
    plt.title("Temporal Attention Weights for Sequence Context")
    plt.xlabel("Past Hours")
    plt.yticks([])
    heatmap_path = os.path.join(Config.FIGURE_DIR, 'attention_heatmap.png')
    plt.savefig(heatmap_path, bbox_inches='tight')

    try:
        mean_feat_attn = np.mean(all_feat_attns, axis=0)
        
        plt.figure(figsize=(12, 4))
        sns.barplot(
            x=list(range(Config.NUM_FEATURES)), y=mean_feat_attn, hue=list(range(Config.NUM_FEATURES)),
            palette="viridis", legend=False)
        plt.title("Temporal Attention Weights for Sequence Context")
        plt.xlabel("Feature index")
        plt.ylabel("Attention Weight")
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.savefig(os.path.join(Config.FIGURE_DIR, 'feature_attention.png'), bbox_inches='tight')
        print("Feature co-weight graph saved.")
    except Exception as e:
        print(f"Failed: {e}")

if __name__ == '__main__':
    evaluate_model()
