import xarray as xr
import pandas as pd
import numpy as np
import os


era5_file_0 = r"raw/data_stream-oper_stepType-accum.nc"
era5_file_1 = r"raw/data_stream-oper_stepType-instant.nc"
supp_file_0 = r"raw/supp/data_stream-oper_stepType-accum.nc"
supp_file_1 = r"raw/supp/data_stream-oper_stepType-instant.nc"

original_file = r"raw/data_official.xlsx"
output_file = r"data_merged.csv"


def load_and_clean(accum_path, instant_path):
    """封装加载函数，自动消除 expver 版本冲突"""
    ds0 = xr.open_dataset(accum_path, engine='netcdf4')
    ds1 = xr.open_dataset(instant_path, engine='netcdf4')

    def resolve_expver(dataset):
        if 'expver' in dataset.dims:
            return dataset.sel(expver=1).combine_first(dataset.sel(expver=5))
        elif 'expver' in dataset.coords or 'expver' in dataset.variables:
            return dataset.drop_vars('expver', errors='ignore')
        return dataset

    return xr.merge([resolve_expver(ds0), resolve_expver(ds1)])

# 加载高维气象张量
ds_main = load_and_clean(era5_file_0, era5_file_1)
ds_supp = load_and_clean(supp_file_0, supp_file_1)

# 拼接增量边界数据
ds_combined = xr.concat([ds_supp, ds_main], dim='valid_time' if 'valid_time' in ds_main.dims else 'time')

point_data = ds_combined.sel(latitude=39.73, longitude=116.13, method='nearest')
df_era5 = point_data.to_dataframe().reset_index()
df_era5['wind_speed'] = np.sqrt(df_era5['u10']**2 + df_era5['v10']**2)

# 时区平移
time_col = 'valid_time' if 'valid_time' in df_era5.columns else 'time'
df_era5['datetime'] = df_era5[time_col] + pd.Timedelta(hours=8)

df_era5_final = df_era5[['datetime', 'wind_speed', 'u10', 'v10', 'blh', 'tp', 'ssrd']].copy()
df_era5_final.columns = ['datetime', 'Wind_Speed(m/s)', 'Wind_U(m/s)', 'Wind_V(m/s)', 
                         'PBL_Height(m)', 'Precipitation(m)', 'Solar_Radiation(J/m2)']

# 清洗污染物数据并进行时序匹配
df_temp = pd.read_excel(original_file, header=None, nrows=20)
header_row_index = df_temp[df_temp.apply(lambda row: 'datetime' in row.astype(str).values, axis=1)].index[-1]

df_original = pd.read_excel(original_file, header=header_row_index)
df_original = df_original[df_original['datetime'] != 'datetime']
df_original['datetime'] = df_original['datetime'].astype(str).str.replace('2 026', '2026').str.strip()

df_original['datetime'] = pd.to_datetime(df_original['datetime'], format='mixed', errors='coerce')
df_era5_final['datetime'] = pd.to_datetime(df_era5_final['datetime'], format='mixed', errors='coerce')

df_original = df_original.dropna(subset=['datetime'])
df_era5_final = df_era5_final.dropna(subset=['datetime'])

# 执行合并
df_merged = pd.merge(df_original, df_era5_final, on='datetime', how='left')

df_merged.to_csv(output_file, index=False, encoding='utf-8-sig')
print(f"Merged file saved：{output_file}")