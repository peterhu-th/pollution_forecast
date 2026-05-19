# -*- coding: utf-8 -*-
"""
全局配置文件 (Global Configuration)
集中管理数据路径、模型超参数及训练策略。
遵守学术界代码规范。
"""

import os

class Config:
    # 路径配置
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    PROCESSED_DIR = os.path.join(DATA_DIR, 'processed')
    MODEL_DIR = os.path.join(BASE_DIR, 'results', 'models')
    FIGURE_DIR = os.path.join(BASE_DIR, 'results', 'figures')
    
    TRAIN_DATA_PATH = os.path.join(PROCESSED_DIR, 'train_set.csv')
    VAL_DATA_PATH = os.path.join(PROCESSED_DIR, 'val_set.csv')
    TEST_DATA_PATH = os.path.join(PROCESSED_DIR, 'test_set.csv')
    SCALER_PATH = os.path.join(MODEL_DIR, 'data_scaler.pkl')
    BEST_MODEL_PATH = os.path.join(MODEL_DIR, 'best_forecaster.pth')

    # 数据集和滑动窗口配置
    SEQ_LENGTH = 24       # 输入的历史时间步长 (例如：使用过去24小时数据)
    PRED_HORIZON = 12     # 预测未来的时间步长 (例如：预测未来12小时数据)
    TARGET_FEATURES = ['PM2.5', 'PM10', 'O3', 'NO2', 'SO2', 'CO']
    NUM_TARGETS = len(TARGET_FEATURES)
    NUM_FEATURES = 8      # 假设输入特征总数为8 (6个目标污染物 + 温度 + 湿度)，需要根据实际合并后的数据列数调整

    # 模型架构超参数
    HIDDEN_DIM = 64       # LSTM 隐藏层维度
    NUM_LAYERS = 2        # LSTM 层数
    DROPOUT_RATE = 0.2    # Dropout 概率

    # 训练配置
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4
    EPOCHS = 100
    PATIENCE = 10         # Early stopping 耐心值
    DEVICE = 'cuda'       # 或 'cpu'
