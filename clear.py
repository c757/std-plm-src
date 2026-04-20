import xarray as xr
import numpy as np
import os
import pandas as pd
import gc
import json

print("启动【全海域 0.25° 分辨率】数据融合与格式转换脚本（内置 VMDR->sin/cos，直接产出8维）...")

# ==========================================
# 第一部分：基础参数与目标网格设定
# ==========================================
print("正在读取网格基准与时间边界...")
# 仅读取元数据，绝对不加载实体数据
ds_wave_base = xr.open_dataset('./data/wave_data.nc')
lat_name = 'latitude' if 'latitude' in ds_wave_base.coords else 'lat'
lon_name = 'longitude' if 'longitude' in ds_wave_base.coords else 'lon'

lat_min, lat_max = float(ds_wave_base[lat_name].min()), float(ds_wave_base[lat_name].max())
lon_min, lon_max = float(ds_wave_base[lon_name].min()), float(ds_wave_base[lon_name].max())
ds_wave_base.close()

# 目标网格
lat_target = np.arange(lat_min, lat_max, 0.25)
lon_target = np.arange(lon_min, lon_max, 0.25)
n_lat, n_lon = len(lat_target), len(lon_target)
n_nodes = n_lat * n_lon

# 目标时间轴
time_start, time_end = '2023-12-31T00:00:00', '2025-12-01T21:00:00'
TIME_STEP = '9h'
target_time = pd.date_range(start=time_start, end=time_end, freq=TIME_STEP)
T = len(target_time)

variables = ["uo", "vo", "VHM0", "VMDR", "VTM02", "eastward_wind", "northward_wind"]
N_FEAT = len(variables)

print(f"目标网格: Lat({lat_min:.2f}~{lat_max:.2f}), Lon({lon_min:.2f}~{lon_max:.2f})")
print(f"时间轴: T={T}步, 节点 N={n_nodes}")

# ==========================================
# 第二部分：时间分批插值并填充 (防 38GB 原文件 OOM 核心)
# ==========================================
N_FEAT_NEW = 8
print(f"\n正在内存中预分配 {T * n_lat * n_lon * N_FEAT_NEW * 4 / (1024**3):.2f} GB 二维图像张量...")
data_grid = np.zeros((T, n_lat, n_lon, N_FEAT_NEW), dtype=np.float32)

# 定义每次处理的时间步数量，如果机器卡顿可以调小到 50，M4 100 没问题
BATCH_SIZE = 100


def _to_time_lat_lon(da, var_name):
    # Currents may include a singleton depth dim; collapse it explicitly.
    for d in ['depth', 'depthu', 'depthv', 'lev']:
        if d in da.dims:
            if da.sizes[d] != 1:
                raise ValueError(f"变量 {var_name} 的维度 {d} 大小为 {da.sizes[d]}，当前脚本仅支持单层深度")
            da = da.isel({d: 0})

    expected_dims = ['time', lat_name, lon_name]
    missing = [d for d in expected_dims if d not in da.dims]
    if missing:
        raise ValueError(f"变量 {var_name} 缺少必要维度: {missing}, 当前维度: {list(da.dims)}")

    da = da.transpose('time', lat_name, lon_name)
    return da.values.astype(np.float32)


def process_in_batches(file_path, var_keys, feature_start_idx):
    print(f"打开文件 {file_path} 进行分批处理...")
    with xr.open_dataset(file_path, chunks={'time': 200}) as ds:
        for i in range(0, T, BATCH_SIZE):
            t_slice = target_time[i: i + BATCH_SIZE]
            actual_batch_len = len(t_slice)

            ds_batch = ds.interp(time=t_slice, method='linear') \
                .interp({lat_name: lat_target, lon_name: lon_target}, method='linear') \
                .compute()

            for j, var_name in enumerate(var_keys):
                v = _to_time_lat_lon(ds_batch[var_name], var_name)
                # 核心改动：如果是波向数据，直接在这一步拆成 Sin 和 Cos！
                if var_name == "VMDR":
                    vmdr_values = v
                    # 填入索引 3 (Sin) 和 4 (Cos)
                    data_grid[i: i + actual_batch_len, :, :, 3] = np.sin(np.radians(vmdr_values))
                    data_grid[i: i + actual_batch_len, :, :, 4] = np.cos(np.radians(vmdr_values))
                else:
                    feat_idx = feature_start_idx + j
                    # 遇到 VMDR 之后的变量，索引要往后顺延一位（因为VMDR占了俩坑）
                    if var_name in ["VTM02", "eastward_wind", "northward_wind"]:
                        feat_idx += 1

                        # 核心改动：直接赋值二维矩阵，绝对不准用 reshape 展平！
                    data_grid[i: i + actual_batch_len, :, :, feat_idx] = v

            del ds_batch
            gc.collect()

# 目标变量排在前面 (索引 0, 1)
print("\n[1/3] 正在插值并处理洋流数据 (Currents)...")
process_in_batches('./data/currents_data.nc', ["uo", "vo"], 0)

# 其他条件变量排在后面
print("\n[2/3] 正在插值并处理波浪数据 (Wave)...")
process_in_batches('./data/wave_data.nc', ["VHM0", "VMDR", "VTM02"], 2) # 从 2 开始

print("\n[3/3] 正在插值并处理风场数据 (Wind)...")
process_in_batches('./data/wind_data.nc', ["eastward_wind", "northward_wind"], 5) # 从 5 开始

print("\n多模态数据分批整合完毕！")

# ==========================================
# 第三部分：防泄漏的 Z-score 归一化与掩码生成
# ==========================================
print("\n正在进行全局 Z-score 归一化 (仅基于训练集统计)...")
SAMPLE_LEN = 9
PREDICT_LEN = 3
WINDOW = SAMPLE_LEN + PREDICT_LEN
TRAIN_RATIO = 0.6
VAL_RATIO = 0.2

# 【修复1】：先定义好长度和文件夹
T = data_grid.shape[0]
train_t_end = int(T * TRAIN_RATIO)
out_dir = './data/stdplm_input_025'
os.makedirs(out_dir, exist_ok=True) # 先建好文件夹！

# 先在原始插值结果上提取海洋掩码，避免被 nan_to_num 污染
ocean_mask_2d = np.isfinite(data_grid[0, :, :, 0])
if not np.any(ocean_mask_2d):
    raise ValueError("ocean_mask_2d 全为 False，请检查插值范围或变量映射是否正确")

# 【修复2】：仅基于训练集 + 海洋区域统计，避免陆地 0 值泄漏
train_data = data_grid[:train_t_end]
ocean_mask_4d = ocean_mask_2d[None, :, :, None]
train_data_ocean = np.where(ocean_mask_4d, train_data, np.nan)
mean = np.nanmean(train_data_ocean, axis=(0, 1, 2), keepdims=True)
std  = np.nanstd(train_data_ocean, axis=(0, 1, 2), keepdims=True)
std  = np.where(std < 1e-8, 1.0, std)

# 保存二维海洋掩码（陆地还是海洋？）和时间戳
np.save(f'{out_dir}/ocean_mask_2d.npy', ocean_mask_2d)
np.save(f'{out_dir}/timestamps.npy', target_time.values)
np.save(f'{out_dir}/norm_mean.npy', mean)
np.save(f'{out_dir}/norm_std.npy',  std)

# 【修复3】：只做一次全局归一化，并把陆地强制置 0 方便模型计算
data_grid = (data_grid - mean) / std
data_grid = np.where(ocean_mask_4d, data_grid, 0.0)
data_grid = np.nan_to_num(data_grid, nan=0.0)

# ==========================================
# 第四部分：生成动态读取索引 (零拷贝防 OOM 核心)
# ==========================================
print("正在生成滑动窗口索引 (Zero-copy 模式)...")
N_samples = T - WINDOW + 1
val_t_end = int(T * (TRAIN_RATIO + VAL_RATIO))

train_idx = [i for i in range(N_samples) if (i + WINDOW) <= train_t_end]
val_idx   = [i for i in range(N_samples) if i >= train_t_end and (i + WINDOW) <= val_t_end]
test_idx  = [i for i in range(N_samples) if i >= val_t_end]

print("正在保存唯一的连续二维时空张量 data_grid.npy (需等待磁盘写入)...")
np.save(f'{out_dir}/data_grid.npy', data_grid) # 文件名改为 data_grid.npy
np.savez_compressed(f'{out_dir}/indices.npz', train_idx=train_idx, val_idx=val_idx, test_idx=test_idx)

del data_grid # 释放正确的变量
gc.collect()

# ==========================================
# 第五部分：构建图邻接矩阵与元数据 (极速版)
# ==========================================
print("正在构建稀疏邻接矩阵 CSV（极速向量化 4-邻接）...")
# 【修复4】：用矩阵切片代替 for 循环，瞬间完成 6 万节点的连线
grid = np.arange(n_nodes).reshape(n_lat, n_lon)
rows = np.concatenate([grid[:, :-1].flatten(), grid[:, 1:].flatten(), grid[:-1, :].flatten(), grid[1:, :].flatten()])
cols = np.concatenate([grid[:, 1:].flatten(), grid[:, :-1].flatten(), grid[1:, :].flatten(), grid[:-1, :].flatten()])
dist = np.ones_like(rows, dtype=np.float32)

pd.DataFrame({'from': rows, 'to': cols, 'cost': dist}).to_csv(f'{out_dir}/ocean_adj.csv', index=False)

meta = {
    'n_nodes': n_nodes, 'n_lat': n_lat, 'n_lon': n_lon, 'n_features': N_FEAT_NEW,
    'variables': ["uo", "vo", "VHM0", "VMDR_sin", "VMDR_cos", "VTM02", "eastward_wind", "northward_wind"],
    'tensor_layout': 'T,H,W,C',
    'sample_len': SAMPLE_LEN, 'predict_len': PREDICT_LEN,
    'time_steps': T, 'train_samples': len(train_idx), 'val_samples': len(val_idx), 'test_samples': len(test_idx),
    'resolution': 0.25, 'time_step': TIME_STEP
}
with open(f'{out_dir}/meta.json', 'w') as f:
    json.dump(meta, f, indent=2)
print("转换完成！")
