import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalAttention(nn.Module):

    """
    时间级别注意力机制 (Temporal Attention Mechanism)
    用于计算历史各时间步的隐藏状态对当前预测的贡献权重
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


class FeatureAttention(nn.Module):
    """
    特征级别注意力机制 (Feature Attention)
    用于计算不同输入特征(温湿度、风速等)对预测模型的相对重要性权重
    """
    def __init__(self, input_dim: int):
        super(FeatureAttention, self).__init__()
        # 对输入特征维度进行打分
        self.attention_linear = nn.Linear(input_dim, input_dim)

    def forward(self, x):
        # x 形状: (batch_size, seq_length, input_dim)
        score = self.attention_linear(x)
        # 在特征维度(dim=2)上计算 softmax
        attention_weights = F.softmax(score, dim=2)
        # 将权重与原始输入相乘
        weighted_x = x * attention_weights
        
        # 返回加权后的输入，以及特征权重（取时间维度的平均值用于可视化）
        return weighted_x, attention_weights.mean(dim=1)


class SpatioTemporalForecaster(nn.Module):

    """
    深度时空预测网络 (Spatio-Temporal Forecaster)
    基于 LSTM 编码器与注意力机制构建，进行多污染物协同预测
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
        self.feature_attention = FeatureAttention(input_dim)
        
        # 时序特征提取编码层
        self.encoder_lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0.0
        )
        
        # 注意力机制融合模块
        self.temporal_attention = TemporalAttention(hidden_dim)
        
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
        :return: 预测张量, 时间注意力权重, 特征注意力权重
        """
        # 特征注意力计算
        weighted_x, feat_attn_weights = self.feature_attention(x)
        
        # LSTM 提取时序特征
        lstm_out, (h_n, c_n) = self.encoder_lstm(weighted_x)
        
        # 时间注意力池化
        context_vector, temp_attn_weights = self.temporal_attention(lstm_out)
        
        # 全连接层输出
        output_flat = self.fc_decoder(context_vector)
        
        # 重塑维度
        predictions = output_flat.view(-1, self.pred_horizon, self.num_targets)
        
        return predictions, temp_attn_weights, feat_attn_weights


if __name__ == '__main__':
    print("Executing Academic Integrity & Dimensions Check for Model Forward Pass")
    
    BATCH_SIZE = 16
    SEQ_LENGTH = 24
    INPUT_DIM = 8
    HIDDEN_DIM = 64
    NUM_LAYERS = 2
    PRED_HORIZON = 12
    NUM_TARGETS = 6
    
    # 创建随机输入张量 (Batch, Seq, Features)
    dummy_input = torch.randn(BATCH_SIZE, SEQ_LENGTH, INPUT_DIM)
    print(f"Input Data Shape: {dummy_input.shape}")
    
    # 初始化模型
    model = SpatioTemporalForecaster(
        input_dim=INPUT_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        pred_horizon=PRED_HORIZON,
        num_targets=NUM_TARGETS
    )
    
    # 前向传播
    predictions, temp_attn, feat_attn = model(dummy_input)
    
    print(f"Expected Output Shape: ({BATCH_SIZE}, {PRED_HORIZON}, {NUM_TARGETS})")
    print(f"Actual Output Shape:   {predictions.shape}")
    
    print(f"Expected Temporal Attention Shape: ({BATCH_SIZE}, {SEQ_LENGTH}, 1)")
    print(f"Actual Temporal Attention Shape:   {temp_attn.shape}")
    
    print(f"Expected Feature Attention Shape: ({BATCH_SIZE}, {INPUT_DIM})")
    print(f"Actual Feature Attention Shape:   {feat_attn.shape}")
