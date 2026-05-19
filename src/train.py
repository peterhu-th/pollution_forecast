# -*- coding: utf-8 -*-
"""
模型训练引擎 (Model Training Engine)
实现基于 AdamW 优化器和 Huber Loss 的训练循环，并包含 Early Stopping 防止过拟合。
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from .config import Config
from .dataset_builder import build_dataloader
from .model import SpatioTemporalForecaster

class EarlyStopping:
    """早停机制 (Early Stopping Mechanism)"""
    def __init__(self, patience=7, delta=0, path='checkpoint.pth'):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float('inf')
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """当验证集 loss 降低时保存模型。"""
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

def train_model():
    """
    模型训练主程序 (Training Main Loop)。
    """
    device = torch.device(Config.DEVICE if torch.cuda.is_available() else 'cpu')
    print(f"[*] Training on device: {device}")
    
    # 1. 准备数据加载器
    train_loader, scaler = build_dataloader(
        file_path=Config.TRAIN_DATA_PATH,
        seq_length=Config.SEQ_LENGTH,
        pred_horizon=Config.PRED_HORIZON,
        batch_size=Config.BATCH_SIZE,
        is_train=True
    )
    
    val_loader, _ = build_dataloader(
        file_path=Config.VAL_DATA_PATH,
        seq_length=Config.SEQ_LENGTH,
        pred_horizon=Config.PRED_HORIZON,
        batch_size=Config.BATCH_SIZE,
        is_train=False,
        scaler=scaler
    )
    
    # 2. 实例化模型
    model = SpatioTemporalForecaster(
        input_dim=Config.NUM_FEATURES,
        hidden_dim=Config.HIDDEN_DIM,
        num_layers=Config.NUM_LAYERS,
        pred_horizon=Config.PRED_HORIZON,
        num_targets=Config.NUM_TARGETS,
        dropout_rate=Config.DROPOUT_RATE
    ).to(device)
    
    # 3. 定义损失函数与优化器
    criterion = nn.HuberLoss() # 对极端污染值鲁棒
    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY)
    
    early_stopping = EarlyStopping(patience=Config.PATIENCE, path=Config.BEST_MODEL_PATH)
    
    train_losses = []
    val_losses = []
    
    # 4. 训练循环
    for epoch in range(Config.EPOCHS):
        model.train()
        train_loss = 0.0
        
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            predictions, _ = model(batch_x)
            
            loss = criterion(predictions, batch_y)
            loss.backward()
            
            # 梯度裁剪防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            train_loss += loss.item() * batch_x.size(0)
            
        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)
        
        # 验证循环
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                predictions, _ = model(batch_x)
                loss = criterion(predictions, batch_y)
                val_loss += loss.item() * batch_x.size(0)
                
        val_loss /= len(val_loader.dataset)
        val_losses.append(val_loss)
        
        print(f"Epoch [{epoch+1}/{Config.EPOCHS}] | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        # Early Stopping 检查
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("[!] Early stopping triggered.")
            break
            
    print(f"[*] Training completed. Best model saved to: {Config.BEST_MODEL_PATH}")
    return train_losses, val_losses

if __name__ == '__main__':
    train_model()
