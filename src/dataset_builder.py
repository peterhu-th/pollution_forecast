# -*- coding: utf-8 -*-
"""
数据加载与预处理模块 (Data Loading and Preprocessing)
实现时间序列数据的窗口切分、归一化以及 PyTorch Dataset 的构建。
"""

import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import joblib
from .config import Config

class TimeSeriesDataset(Dataset):
    """
    针对多输入多输出 (MIMO) 时间序列预测任务的数据集类。
    """
    def __init__(self, data: np.ndarray, seq_length: int, pred_horizon: int):
        """
        :param data: 形状为 (Num_Samples, Num_Features) 的 numpy 数组
        :param seq_length: 历史输入窗口大小
        :param pred_horizon: 预测窗口大小
        """
        self.data = data
        self.seq_length = seq_length
        self.pred_horizon = pred_horizon
        
        # 计算有效样本数量
        self.num_samples = len(data) - seq_length - pred_horizon + 1
        
    def __len__(self):
        return max(0, self.num_samples)
    
    def __getitem__(self, idx):
        # 提取历史窗口作为输入 X: (seq_length, num_features)
        x = self.data[idx : idx + self.seq_length]
        
        # 提取未来窗口的目标作为标签 Y: (pred_horizon, num_targets)
        # 假设前 Config.NUM_TARGETS 列是我们要预测的目标污染物
        y = self.data[idx + self.seq_length : idx + self.seq_length + self.pred_horizon, :Config.NUM_TARGETS]
        
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

def build_dataloader(file_path: str, seq_length: int, pred_horizon: int, batch_size: int, 
                     is_train: bool = True, scaler: StandardScaler = None):
    """
    构建数据加载器。
    
    :param file_path: CSV 数据集路径
    :param seq_length: 历史步长
    :param pred_horizon: 预测步长
    :param batch_size: 批次大小
    :param is_train: 是否为训练集（训练集需拟合 Scaler）
    :param scaler: 如果非训练集，则必须传入在训练集上拟合好的 Scaler
    :return: DataLoader, Scaler
    """
    # 如果文件不存在，为了代码鲁棒性，生成假数据(仅供测试)
    if not os.path.exists(file_path):
        print(f"Warning: {file_path} not found. Using random dummy data for testing.")
        df_data = pd.DataFrame(np.random.randn(1000, Config.NUM_FEATURES))
    else:
        df_data = pd.read_csv(file_path)
        # 假定第一列为时间，去掉时间列
        if 'time' in df_data.columns or 'date' in df_data.columns.str.lower():
            df_data = df_data.select_dtypes(include=[np.number])
    
    data_array = df_data.values
    
    if is_train:
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data_array)
        # 保存 Scaler
        os.makedirs(os.path.dirname(Config.SCALER_PATH), exist_ok=True)
        joblib.dump(scaler, Config.SCALER_PATH)
    else:
        if scaler is None:
            if os.path.exists(Config.SCALER_PATH):
                scaler = joblib.load(Config.SCALER_PATH)
            else:
                raise ValueError("Scaler not provided and not found on disk. Train first.")
        data_scaled = scaler.transform(data_array)
        
    dataset = TimeSeriesDataset(data_scaled, seq_length, pred_horizon)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=is_train, drop_last=is_train)
    
    return dataloader, scaler
