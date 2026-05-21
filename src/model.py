import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatioTemporalForecaster(nn.Module):

    """
    深度时空预测网络 (Spatio-Temporal Forecaster)
    基于 LSTM 编码器构建，进行多污染物协同预测
    """

    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, 
                 pred_horizon: int, num_targets: int, seq_length: int, dropout_rate: float = 0.2):

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
        
        # 时序特征提取编码层 (直接接收原始输入)
        self.encoder_lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0.0
        )
        
        # 全连接输出映射层
        self.flatten = nn.Flatten()

        self.shared_fc = nn.Sequential(
            nn.Linear(seq_length * hidden_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )

        self.target_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, pred_horizon)
            ) for _ in range(num_targets)
        ])


    def forward(self, x):
        """
        前向传播函数。
        :param x: 输入张量，形状为 (batch_size, seq_length, input_dim)
        :return: 预测张量
        """
        # 提取共享时序特征
        lstm_out, _ = self.encoder_lstm(x)
        
        # 共享层映射
        flat_out = self.flatten(lstm_out)
        shared_features = self.shared_fc(flat_out)
        
        # 多头预测
        head_outputs = [head(shared_features) for head in self.target_heads]
        
        # 堆叠维度 -> (batch_size, pred_horizon, num_targets)
        predictions = torch.stack(head_outputs, dim=-1)
        
        return predictions


if __name__ == '__main__':
    print("Executing Dimensions Check for Pure LSTM Model")
    
    BATCH_SIZE = 16
    SEQ_LENGTH = 12
    INPUT_DIM = 14
    HIDDEN_DIM = 256
    NUM_LAYERS = 2
    PRED_HORIZON = 24
    NUM_TARGETS = 6
    
    dummy_input = torch.randn(BATCH_SIZE, SEQ_LENGTH, INPUT_DIM)
    model = SpatioTemporalForecaster(
        input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS,
        pred_horizon=PRED_HORIZON, num_targets=NUM_TARGETS, seq_length=SEQ_LENGTH
    )
    
    predictions = model(dummy_input)
    print(f"Input Shape: {dummy_input.shape}")
    print(f"Output Shape: {predictions.shape}")
