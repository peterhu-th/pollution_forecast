import cdsapi
import zipfile
import os

# ================= 静态相对路径配置 =================
raw_dir = r"raw"
supp_dir = r"raw/supp"  # 专门存放补充数据的隔离目录，防止覆盖主数据
zip_path = r"raw/era5_supp.zip"

os.makedirs(supp_dir, exist_ok=True)

# ================= 网络增量下载逻辑 =================
print("正在向 ECMWF 发起增量请求：获取 2025年12月31日 (UTC 16:00-23:00) 的边界自旋数据...")
c = cdsapi.Client()

c.retrieve(
    'reanalysis-era5-single-levels',
    {
        'product_type': 'reanalysis',
        'format': 'netcdf',
        'variable': [
            '10m_u_component_of_wind',    
            '10m_v_component_of_wind',    
            'boundary_layer_height',      
            'total_precipitation',        
            'surface_solar_radiation_downwards', 
        ],
        'year': '2025',             # 注意：跨年回溯
        'month': '12',              # 注意：跨月回溯
        'day': '31',
        'time': [f"{str(i).zfill(2)}:00" for i in range(16, 24)], # 精准抓取 UTC 16-23 点
        'area': [40.0, 115.9, 39.6, 116.2],
    },
    zip_path
)
print(f"增量数据包下载完成！")

# ================= 解压逻辑 =================
try:
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(supp_dir)
        print("补充数据解压成功！现在可以执行时空无缝融合了。")
except Exception as e:
    print(f"解压发生错误：{e}")