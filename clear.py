import xarray as xr
import numpy as np
import os
import pandas as pd
import gc
import json

print("启动【全海域 0.25° 分辨率】数据融合与格式转换脚本...")

# ==========================================
# 第一部分：基础参数与目标网格设定
# ==========================================
print("⏳ 正在读取网格基准与时间边界...")
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
print(f"\n正在内存中预分配 {T * n_nodes * N_FEAT * 4 / (1024**3):.2f} GB 最终特征张量...")
data_nodes = np.zeros((T, n_nodes, N_FEAT), dtype=np.float32)

# 定义每次处理的时间步数量，如果机器卡顿可以调小到 50，M4 100 没问题
BATCH_SIZE = 100

def process_in_batches(file_path, var_keys, feature_start_idx):
    """通用的分批读取、插值与填充函数"""
    print(f"打开文件 {file_path} 进行分批处理...")
    # chunks={'time': 200} 告诉底层按块读取，千万别一口吞
    with xr.open_dataset(file_path, chunks={'time': 200}) as ds:
        for i in range(0, T, BATCH_SIZE):
            t_slice = target_time[i : i + BATCH_SIZE]
            actual_batch_len = len(t_slice)
            
            # 仅截取当前需要的这一小段时间进行插值！这是省内存的关键
            ds_batch = ds.interp(time=t_slice, method='linear') \
                         .interp({lat_name: lat_target, lon_name: lon_target}, method='linear') \
                         .compute() # 这里的 compute 只会计算这一小块，瞬间完成
            
            # 将处理好的小块填入预分配的巨型数组
            for j, var_name in enumerate(var_keys):
                feat_idx = feature_start_idx + j
                data_nodes[i : i + actual_batch_len, :, feat_idx] = \
                    ds_batch[var_name].values.reshape(actual_batch_len, n_nodes).astype(np.float32)
            
            # 强制清理这一小块的缓存
            del ds_batch
            gc.collect()
            print(f"  -> 已完成时间进度: {min(i + BATCH_SIZE, T)} / {T}")

# 目标变量排在前面 (索引 0, 1)
print("\n[1/3] 正在插值并处理洋流数据 (Currents)...")
process_in_batches('./data/currents_data.nc', ["uo", "vo"], 0)

# 其他条件变量排在后面
print("\n[2/3] 正在插值并处理波浪数据 (Wave)...")
process_in_batches('./data/wave_data.nc', ["VHM0", "VMDR", "VTM02"], 2) # 从 2 开始

print("\n[3/3] 正在插值并处理风场数据 (Wind)...")
process_in_batches('./data/wind_data.nc', ["eastward_wind", "northward_wind"], 5) # 从 5 开始

# 填补可能存在的缺失值
data_nodes = np.nan_to_num(data_nodes, nan=0.0)
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
T = data_nodes.shape[0]
train_t_end = int(T * TRAIN_RATIO)
out_dir = './data/stdplm_input_025'
os.makedirs(out_dir, exist_ok=True) # 先建好文件夹！

# 【修复2】：忽略陆地的 NaN，计算真实海洋区域的平均值
train_data = data_nodes[:train_t_end]
mean = np.nanmean(train_data, axis=(0, 1), keepdims=True)
std  = np.nanstd(train_data, axis=(0, 1), keepdims=True)
std  = np.where(std < 1e-8, 1.0, std)

# 保存海洋掩码（陆地还是海洋？）和时间戳
ocean_mask = ~np.isnan(data_nodes[0, :, 0])
np.save(f'{out_dir}/ocean_mask.npy', ocean_mask)
np.save(f'{out_dir}/timestamps.npy', target_time.values)
np.save(f'{out_dir}/norm_mean.npy', mean)
np.save(f'{out_dir}/norm_std.npy',  std)

# 【修复3】：只做一次归一化，并把陆地的 NaN 变成 0 方便模型计算
data_nodes = (data_nodes - mean) / std
data_nodes = np.nan_to_num(data_nodes, nan=0.0)

# ==========================================
# 第四部分：生成动态读取索引 (零拷贝防 OOM 核心)
# ==========================================
print("正在生成滑动窗口索引 (Zero-copy 模式)...")
N_samples = T - WINDOW + 1
val_t_end = int(T * (TRAIN_RATIO + VAL_RATIO))

train_idx = [i for i in range(N_samples) if (i + WINDOW) <= train_t_end]
val_idx   = [i for i in range(N_samples) if i >= train_t_end and (i + WINDOW) <= val_t_end]
test_idx  = [i for i in range(N_samples) if i >= val_t_end]

print("正在保存唯一的连续时空张量 data_nodes.npy (需等待磁盘写入)...")
np.save(f'{out_dir}/data_nodes.npy', data_nodes)
np.savez_compressed(f'{out_dir}/indices.npz', train_idx=train_idx, val_idx=val_idx, test_idx=test_idx)

del data_nodes
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
    'n_nodes': n_nodes, 'n_lat': n_lat, 'n_lon': n_lon, 'n_features': N_FEAT,
    'variables': variables, 'sample_len': SAMPLE_LEN, 'predict_len': PREDICT_LEN,
    'time_steps': T, 'train_samples': len(train_idx), 'val_samples': len(val_idx), 'test_samples': len(test_idx),
    'resolution': 0.25, 'time_step': TIME_STEP
}
with open(f'{out_dir}/meta.json', 'w') as f:
    json.dump(meta, f, indent=2)
print("转换完成！")
