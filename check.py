import xarray as xr
import os
import numpy as np


def _pick_coord_name(ds, candidates):
    for name in candidates:
        if name in ds.coords:
            return name
    return None


def _safe_float(v):
    try:
        return float(v)
    except Exception:
        return None


def _coord_resolution(coord):
    if coord is None or coord.size < 2:
        return None
    try:
        vals = coord.values
        # Use first two points as nominal resolution for regularly gridded products.
        return float(abs(vals[1] - vals[0]))
    except Exception:
        return None

def inspect_nc_file(file_path):
    """
    检查单个 .nc 文件的详细信息
    """
    print("=" * 50)
    if not os.path.exists(file_path):
        print(f"❌ 找不到文件: {file_path}")
        return

    print(f"正在检查文件: {file_path}")
    try:
        ds = xr.open_dataset(file_path)
        
        # 1. 检查包含的变量
        variables = list(ds.data_vars)
        print(f"\n✅ 包含的物理变量: {variables}")
        
        # 2. 检查数据的维度形状 (重点看 time)
        print(f"\n数据的维度大小 (Shape):")
        for dim, size in ds.sizes.items():
            print(f"   - {dim}: {size}")
            
        # 3. 检查空间范围 (看是否贴合西北太平洋)
        lat_name = _pick_coord_name(ds, ['latitude', 'lat', 'y'])
        lon_name = _pick_coord_name(ds, ['longitude', 'lon', 'x'])
        
        if lat_name is not None and lon_name is not None:
            lat = ds[lat_name]
            lon = ds[lon_name]
            lat_min, lat_max = _safe_float(lat.min().values), _safe_float(lat.max().values)
            lon_min, lon_max = _safe_float(lon.min().values), _safe_float(lon.max().values)
            lat_res = _coord_resolution(lat)
            lon_res = _coord_resolution(lon)

            print(f"\n空间范围:")
            if lat_min is not None and lat_max is not None:
                print(f"   - 纬度 (Lat): {lat_min:.2f}° 到 {lat_max:.2f}°")
            if lon_min is not None and lon_max is not None:
                print(f"   - 经度 (Lon): {lon_min:.2f}° 到 {lon_max:.2f}°")
            print(f"   - 纬度坐标名: {lat_name}, 点数: {lat.size}")
            print(f"   - 经度坐标名: {lon_name}, 点数: {lon.size}")
            if lat_res is not None and lon_res is not None:
                print(f"   - 网格分辨率(约): dLat={lat_res:.4f}°, dLon={lon_res:.4f}°")
        else:
            print("\n⚠️ 未检测到完整的经纬度坐标（支持: latitude/lat/y, longitude/lon/x）")
        
        # 4. 检查时间范围 (找出 T=1 的罪魁祸首)
        if 'time' in ds.coords:
            time_min = str(ds['time'].min().values)[:16] # 截取到分钟
            time_max = str(ds['time'].max().values)[:16]
            time_steps = ds.sizes['time']
            print(f"\n时间信息:")
            print(f"   - 起止时间: {time_min}  至  {time_max}")
            print(f"   - 总时间步数: {time_steps}")

            if time_steps > 1:
                try:
                    tvals = ds['time'].values
                    delta_hours = (tvals[1] - tvals[0]) / np.timedelta64(1, 'h')
                    print(f"   - 时间分辨率(约): {float(delta_hours):.2f} 小时")
                except Exception:
                    pass
            
            if time_steps <= 1:
                print("   ⚠️ 警告：时间步数过少！这会导致时空模型无法学习时间序列。")
        else:
            print("\n❌ 错误: 数据中没有找到 time 时间维度！")
        ds.close()
            
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
