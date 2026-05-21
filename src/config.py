import os
import torch

class Config:
    # 路径配置
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    PROCESSED_DIR = os.path.join(DATA_DIR, 'processed')
    MODEL_DIR = os.path.join(BASE_DIR, 'results', 'models')
    FIGURE_DIR = os.path.join(BASE_DIR, 'results', 'figures')
    FIGURE_DIR_ANALYSIS = os.path.join(FIGURE_DIR, 'analysis')
    FIGURE_DIR_EVAL = os.path.join(FIGURE_DIR, 'evaluation')
    
    TRAIN_DATA_PATH = os.path.join(PROCESSED_DIR, 'train_set.csv')
    VAL_DATA_PATH = os.path.join(PROCESSED_DIR, 'val_set.csv')
    TEST_DATA_PATH = os.path.join(PROCESSED_DIR, 'test_set.csv')
    SCALER_PATH = os.path.join(MODEL_DIR, 'data_scaler.pkl')
    BEST_MODEL_PATH = os.path.join(MODEL_DIR, 'best_forecaster.pth')

    # 特征与维度配置
    FEATURE_NAMES = [
        'PM2.5', 'PM10', 'O3', 'NO2', 'SO2', 'CO', 
        'temperature', 'humidity', 'Wind_Speed(m/s)', 
        'Wind_U(m/s)', 'Wind_V(m/s)', 'PBL_Height(m)', 
        'Precipitation(m)', 'Solar_Radiation(J/m2)'
    ]
    
    TARGET_FEATURES = ['PM2.5', 'PM10', 'O3', 'NO2', 'SO2', 'CO']

    # 模型架构超参数
    NUM_FEATURES = len(FEATURE_NAMES)
    NUM_TARGETS = len(TARGET_FEATURES)

    SEQ_LENGTH = 24
    PRED_HORIZON = 24

    HIDDEN_DIM = 256
    NUM_LAYERS = 2
    DROPOUT_RATE = 0.2

    # 训练配置
    BATCH_SIZE = 16
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4
    EPOCHS = 100
    PATIENCE = 10
    DEVICE = 'cuda'
