import xarray as xr
import numpy as np
import os

print("正在加载合并后的海洋数据...")
ds = xr.open_dataset('./data/final_merged_ocean_data.nc')

# 1. 剥离多余的深度维度（安全做法，不用 squeeze）
if 'depth' in ds.dims:
    print("检测到 depth 维度，正在提取表层数据...")
    ds = ds.isel(depth=0)
elif 'elevation' in ds.dims:
    ds = ds.isel(elevation=0)

# 2. 检查并强制保留 time 维度
if 'time' not in ds.dims:
    print("⚠️ 警告：数据中没有 time 维度！正在自动为您补全 1 维时间轴...")
    ds = ds.expand_dims('time')

# 3. 兼容不同 nc 文件对经纬度的命名规范，并强制排列维度顺序
lat_name = 'latitude' if 'latitude' in ds.dims else 'lat'
lon_name = 'longitude' if 'longitude' in ds.dims else 'lon'
ds = ds.transpose('time', lat_name, lon_name)

# 提取特征
features = ['VHM0', 'VMDR', 'VTM02', 'uo', 'vo']
data_list = []

print("正在检查各变量维度形状...")
for feat in features:
    arr = ds[feat].values  # 现在一定是严格的 (Time, Lat, Lon)
    print(f" -> 变量 {feat} 的形状: {arr.shape}")
    data_list.append(arr)

# 4. 组合并展平
print("正在组合张量...")
raw_data = np.stack(data_list, axis=-1)
T, Lat, Lon, C = raw_data.shape  # 这次绝对能完美解包 4 个值！

nodes = Lat * Lon
print(f"✅ 时间步(T): {T}, 空间节点(Nodes): {nodes}, 特征数(C): {C}")
reshaped_data = raw_data.reshape(T, nodes, C)

# 5. 保存数据
save_dir = './data/ocean/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

print("正在写入本地...")
np.savez_compressed(f'{save_dir}/ocean_data.npz', data=reshaped_data)
print(f"🎉 数据转换成功！最终张量形状为: {reshaped_data.shape} (Time x Nodes x Features)")
