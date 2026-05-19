# -*- coding: utf-8 -*-
"""
空气质量决策系统 (Air Quality Decision System)
接收预测模型的输出，根据国家标准 HJ 633-2012 计算 AQI，并提供预警建议。
"""

import numpy as np
import pandas as pd

class AQIDecisionEngine:
    def __init__(self):
        """
        初始化 AQI 计算所需的各项标准阈值。
        """
        # TODO: 填入真实的 IAQI 计算对应浓度阈值表 (HJ 633-2012)
        pass
        
    def calculate_iaqi(self, concentration: float, pollutant_type: str) -> float:
        """
        计算单项污染物的空气质量分指数 (IAQI)。
        """
        # TODO: 查表插值计算 IAQI
        return 0.0
        
    def calculate_aqi(self, concentrations: dict) -> int:
        """
        根据各项污染物的浓度，计算最终的 AQI。
        AQI = max(IAQI_1, IAQI_2, ..., IAQI_n)
        """
        iaqi_list = []
        for pollutant, conc in concentrations.items():
            iaqi_list.append(self.calculate_iaqi(conc, pollutant))
            
        if not iaqi_list:
            return 0
        return int(max(iaqi_list))
        
    def generate_advisory(self, aqi: int, temp: float, humidity: float) -> str:
        """
        结合 AQI 预测值与温湿度条件，输出校园活动建议。
        """
        if aqi <= 50:
            return "空气质量优，温湿适宜，建议开展户外活动。"
        elif aqi <= 100:
            return "空气质量良，可正常进行户外活动。"
        elif aqi <= 150:
            return "轻度污染，敏感人群应减少户外剧烈运动。"
        else:
            return "中重度污染，建议停止户外活动，关闭门窗。"

if __name__ == '__main__':
    # 极简框架测试
    engine = AQIDecisionEngine()
    test_conc = {"PM2.5": 35.0, "O3": 120.0}
    aqi = engine.calculate_aqi(test_conc)
    advice = engine.generate_advisory(aqi, temp=20.0, humidity=50.0)
    print(f"[*] Calculated AQI: {aqi}")
    print(f"[*] Activity Advisory: {advice}")
