import cdsapi
import zipfile
import os

raw_dir = r"raw"
supp_dir = r"raw/supp"
main_zip_path = r"raw/era5_main.zip"
supp_zip_path = r"raw/era5_supp.zip"

os.makedirs(raw_dir, exist_ok=True)
os.makedirs(supp_dir, exist_ok=True)


c = cdsapi.Client()

print("\n Submitting request 1...")
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
        'year': '2026',
        'month': ['01', '02', '03'],
        'day': [str(i).zfill(2) for i in range(1, 32)],
        'time': [f"{str(i).zfill(2)}:00" for i in range(24)],
        'area': [40.0, 115.9, 39.6, 116.2],
    },
    main_zip_path
)

print("\n Submitting request 2...")
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
        'year': '2025',             
        'month': '12',              
        'day': '31',
        'time': [f"{str(i).zfill(2)}:00" for i in range(16, 24)], 
        'area': [40.0, 115.9, 39.6, 116.2],
    },
    supp_zip_path
)

print("\n Unzipping files...")

def extract_zip(zip_path, extract_target_dir, data_name):
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_target_dir)
            extracted_files = zip_ref.namelist()
            print(f"[{data_name}] extracted {len(extracted_files)} files to {extract_target_dir}")
    except zipfile.BadZipFile:
        print(f"Error: {data_name} broken")
    except Exception as e:
        print(f"{data_name} unknown error: {e}")

extract_zip(main_zip_path, raw_dir, "main_data")
extract_zip(supp_zip_path, supp_dir, "supp_data")
