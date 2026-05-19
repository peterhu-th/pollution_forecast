import cdsapi
import zipfile
import os

raw_dir = r"raw"
zip_path = r"raw/era5_fangshan.zip"
os.makedirs(raw_dir, exist_ok=True)

c = cdsapi.Client()

c.retrieve(
    'reanalysis-era5-single-levels',
    {
        'product_type': 'reanalysis',
        'format': 'netcdf',

        'variable': [
            '10m_u_component_of_wind',    # 10米U向风
            '10m_v_component_of_wind',    # 10米V向风
            'boundary_layer_height',      # 边界层高度
            'total_precipitation',        # 总降水量
            'surface_solar_radiation_downwards', # 太阳下行辐射
        ],
        'year': '2026',
        'month': ['01', '02', '03'],
        'day': [str(i).zfill(2) for i in range(1, 32)],
        'time': [f"{str(i).zfill(2)}:00" for i in range(24)],
        'area': [40.0, 115.9, 39.6, 116.2],
    },
    zip_path
)

try:
    with zipfile.ZipFile(fake_nc_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
        extracted_files = zip_ref.namelist()
        for f in extracted_files:
            print(f" -> {f}")
        
        real_nc_path = os.path.join(extract_dir, extracted_files[0])
        print(f"era5_file = r\"{real_nc_path}\"")
        
except zipfile.BadZipFile:
    print("文件不是 ZIP 格式")