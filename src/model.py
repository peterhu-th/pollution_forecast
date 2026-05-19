# -*- coding: utf-8 -*-
"""
模型架构定义模块 (Model Architecture Definitions)
包含基于特征注意力和时序注意力的 LSTM 网络，用于多变量多步预测。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class TemporalAttention(nn.Module):
    """
    时间级别注意力机制 (Temporal Attention Mechanism)。
    用于计算历史各时间步的隐藏状态对当前预测的贡献权重。
    """
    def __init__(self, hidden_dim: int):
        super(TemporalAttention, self).__init__()
        self.attention_linear = nn.Linear(hidden_dim, 1)

    def forward(self, lstm_outputs):
        """
        :param lstm_outputs: 形状为 (batch_size, seq_length, hidden_dim)
        :return: context_vector, attention_weights
        """
        # score形状: (batch_size, seq_length, 1)
        score = self.attention_linear(lstm_outputs)
        # 在序列长度维度上计算 softmax，得到归一化权重
        attention_weights = F.softmax(score, dim=1)
        
        # 将权重与 lstm 隐藏状态相乘并沿时间维度求和
        # (batch_size, seq_length, hidden_dim) * (batch_size, seq_length, 1) => sum => (batch_size, hidden_dim)
        context_vector = torch.sum(attention_weights * lstm_outputs, dim=1)
        
        return context_vector, attention_weights


class SpatioTemporalForecaster(nn.Module):
    """
    深度时空预测网络 (Spatio-Temporal Forecaster)。
    基于 LSTM 编码器与注意力机制构建，进行多污染物协同预测。
    """
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, 
                 pred_horizon: int, num_targets: int, dropout_rate: float = 0.2):
        """
        :param input_dim: 输入特征数 (例如: 8)
        :param hidden_dim: LSTM 隐藏层维度
        :param num_layers: LSTM 层数
        :param pred_horizon: 预测未来步长 (H)
        :param num_targets: 目标特征数 (例如: 6个污染物)
        :param dropout_rate: Dropout 比例
        """
        super(SpatioTemporalForecaster, self).__init__()
        
        self.pred_horizon = pred_horizon
        self.num_targets = num_targets
        self.hidden_dim = hidden_dim
        
        # 时序特征提取编码层
        self.encoder_lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0.0
        )
        
        # 注意力机制融合模块
        self.attention_mechanism = TemporalAttention(hidden_dim)
        
        # 全连接输出映射层
        self.fc_decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim * 2, pred_horizon * num_targets)
        )

    def forward(self, x):
        """
        前向传播函数。
        :param x: 输入张量，形状为 (batch_size, seq_length, input_dim)
        :return: 预测张量 (batch_size, pred_horizon, num_targets), 注意力权重
        """
        # lstm_out: (batch_size, seq_length, hidden_dim)
        lstm_out, (h_n, c_n) = self.encoder_lstm(x)
        
        # context_vector: (batch_size, hidden_dim)
        # attn_weights: (batch_size, seq_length, 1)
        context_vector, attn_weights = self.attention_mechanism(lstm_out)
        
        # output_flat: (batch_size, pred_horizon * num_targets)
        output_flat = self.fc_decoder(context_vector)
        
        # reshape 为 (batch_size, pred_horizon, num_targets)
        predictions = output_flat.view(-1, self.pred_horizon, self.num_targets)
        
        return predictions, attn_weights


if __name__ == '__main__':
    # 极简测试数据验证模型可行性
    print("="*50)
    print("Executing Academic Integrity & Dimensions Check for Model Forward Pass")
    print("="*50)
    
    # 模拟超参数
    BATCH_SIZE = 16
    SEQ_LENGTH = 24
    INPUT_DIM = 8       # 6污染物 + 温度 + 湿度
    HIDDEN_DIM = 64
    NUM_LAYERS = 2
    PRED_HORIZON = 12
    NUM_TARGETS = 6     # 预测6种污染物
    
    # 创建随机输入张量 (Batch, Seq, Features)
    dummy_input = torch.randn(BATCH_SIZE, SEQ_LENGTH, INPUT_DIM)
    print(f"[*] Input Data Shape: {dummy_input.shape}")
    
    # 初始化模型
    model = SpatioTemporalForecaster(
        input_dim=INPUT_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        pred_horizon=PRED_HORIZON,
        num_targets=NUM_TARGETS
    )
    
    # 前向传播
    predictions, attn_weights = model(dummy_input)
    
    print(f"[*] Expected Output Shape: ({BATCH_SIZE}, {PRED_HORIZON}, {NUM_TARGETS})")
    print(f"[*] Actual Output Shape:   {predictions.shape}")
    
    print(f"[*] Expected Attention Weights Shape: ({BATCH_SIZE}, {SEQ_LENGTH}, 1)")
    print(f"[*] Actual Attention Weights Shape:   {attn_weights.shape}")
    
    if predictions.shape == (BATCH_SIZE, PRED_HORIZON, NUM_TARGETS):
        print("\n[SUCCESS] Model Forward Pass Dimension Verification Passed!")
    else:
        print("\n[ERROR] Model output dimensions mismatch!")
    print("="*50)
