import xarray as xr
import os

def inspect_nc_file(file_path):
    """
    检查单个 .nc 文件的详细信息
    """
    print("=" * 50)
    if not os.path.exists(file_path):
        print(f"❌ 找不到文件: {file_path}")
        return

    print(f"📂 正在检查文件: {file_path}")
    try:
        ds = xr.open_dataset(file_path)
        
        # 1. 检查包含的变量
        variables = list(ds.data_vars)
        print(f"\n✅ 包含的物理变量: {variables}")
        
        # 2. 检查数据的维度形状 (重点看 time)
        print(f"\n📏 数据的维度大小 (Shape):")
        for dim, size in ds.sizes.items():
            print(f"   - {dim}: {size}")
            
        # 3. 检查空间范围 (看是否贴合西北太平洋)
        lat_name = 'latitude' if 'latitude' in ds.coords else 'lat'
        lon_name = 'longitude' if 'longitude' in ds.coords else 'lon'
        
        if lat_name in ds.coords and lon_name in ds.coords:
            lat_min, lat_max = ds[lat_name].min().values, ds[lat_name].max().values
            lon_min, lon_max = ds[lon_name].min().values, ds[lon_name].max().values
            print(f"\n🌍 空间范围:")
            print(f"   - 纬度 (Lat): {lat_min:.2f}° 到 {lat_max:.2f}°")
            print(f"   - 经度 (Lon): {lon_min:.2f}° 到 {lon_max:.2f}°")
        
        # 4. 检查时间范围 (找出 T=1 的罪魁祸首)
        if 'time' in ds.coords:
            time_min = str(ds['time'].min().values)[:16] # 截取到分钟
            time_max = str(ds['time'].max().values)[:16]
            time_steps = ds.sizes['time']
            print(f"\n🕒 时间信息:")
            print(f"   - 起止时间: {time_min}  至  {time_max}")
            print(f"   - 总时间步数: {time_steps}")
            
            if time_steps <= 1:
                print("   ⚠️ 警告：时间步数过少！这会导致时空模型无法学习时间序列。")
        else:
            print("\n❌ 严重警告: 数据中没有找到 time 时间维度！")
            
    except Exception as e:
        print(f"读取文件时出错: {e}")
    print("=" * 50 + "\n")

# ==========================================
# 批量检查你下载的三个文件
# ==========================================
files_to_check = [
    './data/wave_data.nc',
    './data/currents_data.nc',
    './data/wind_data.nc'
]

for f in files_to_check:
    inspect_nc_file(f)
