import xarray as xr
import pandas as pd
import numpy as np
import os

era5_file_0 = r"raw/data_stream-oper_stepType-accum.nc"
era5_file_1 = r"raw/data_stream-oper_stepType-instant.nc"
original_file = r"raw/data_official.xlsx"
output_file = r"data_merged.csv"

os.makedirs(os.path.dirname(output_file), exist_ok=True)


ds0 = xr.open_dataset(era5_file_0, engine='netcdf4')
ds1 = xr.open_dataset(era5_file_1, engine='netcdf4')

def resolve_expver(dataset):
    if 'expver' in dataset.dims:
        return dataset.sel(expver=1).combine_first(dataset.sel(expver=5))
    elif 'expver' in dataset.coords or 'expver' in dataset.variables:
        return dataset.drop_vars('expver', errors='ignore')
    return dataset

ds0 = resolve_expver(ds0)
ds1 = resolve_expver(ds1)
ds = xr.merge([ds0, ds1])

point_data = ds.sel(latitude=39.73, longitude=116.13, method='nearest')
df_era5 = point_data.to_dataframe().reset_index()

if 'u10' in df_era5.columns and 'v10' in df_era5.columns:
    df_era5['wind_speed'] = np.sqrt(df_era5['u10']**2 + df_era5['v10']**2)
else:
    raise KeyError("未能识别到 u10 或 v10 风场变量。")

time_col = 'valid_time' if 'valid_time' in df_era5.columns else 'time'
df_era5['datetime'] = df_era5[time_col] + pd.Timedelta(hours=8)

era5_features = ['datetime', 'wind_speed', 'u10', 'v10', 'blh', 'tp', 'ssrd']
df_era5_final = df_era5[era5_features].copy()

df_era5_final.columns = ['datetime', 'Wind_Speed(m/s)', 'Wind_U(m/s)', 'Wind_V(m/s)', 
                         'PBL_Height(m)', 'Precipitation(m)', 'Solar_Radiation(J/m2)']

df_temp = pd.read_excel(original_file, header=None, nrows=20)
header_row_index = df_temp[df_temp.apply(lambda row: 'datetime' in row.values, axis=1)].index[0]

df_original = pd.read_excel(original_file, header=header_row_index)
df_original['datetime'] = pd.to_datetime(df_original['datetime'], format='mixed')
df_era5_final['datetime'] = pd.to_datetime(df_era5_final['datetime'], format='mixed')
df_merged = pd.merge(df_original, df_era5_final, on='datetime', how='left')
df_merged.to_csv(output_file, index=False, encoding='utf-8-sig')

print(f"新数据集保存至：\n{output_file}")