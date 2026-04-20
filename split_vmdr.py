import gc
import json
import os
import shutil
import numpy as np

print("启动【波向三角函数拆分】迁移脚本...")

data_dir = './data/stdplm_input_025'
backup_dir = './data/stdplm_input_025_vol0'


def _find_data_file(base_dir):
    node_path = os.path.join(base_dir, 'data_nodes.npy')
    grid_path = os.path.join(base_dir, 'data_grid.npy')
    if os.path.exists(grid_path):
        return grid_path
    if os.path.exists(node_path):
        return node_path
    raise FileNotFoundError(f'未找到数据文件: {grid_path} 或 {node_path}')


def _load_stats(base_dir):
    mean = np.load(os.path.join(base_dir, 'norm_mean.npy'))
    std = np.load(os.path.join(base_dir, 'norm_std.npy'))
    return mean.astype(np.float32), std.astype(np.float32)


def _reshape_stats_for_data(mean, std, data_ndim):
    if data_ndim == 3:
        return mean.reshape(1, 1, -1), std.reshape(1, 1, -1)
    if data_ndim == 4:
        return mean.reshape(1, 1, 1, -1), std.reshape(1, 1, 1, -1)
    raise ValueError(f'不支持的数据维度: {data_ndim}, 预期 3 或 4')


def _convert_7_to_8(data, mean, std):
    # data: (..., 7), mean/std flattened to (7,)
    if data.shape[-1] != 7:
        raise ValueError(f'预期最后一维为 7，实际为 {data.shape[-1]}')

    mean7 = mean.reshape(-1)
    std7 = std.reshape(-1)
    if mean7.shape[0] != 7 or std7.shape[0] != 7:
        raise ValueError(f'norm_mean/norm_std 不是 7 维: mean={mean7.shape}, std={std7.shape}')

    # 反归一化真实 VMDR（原索引 3）
    vmdr_true = data[..., 3] * std7[3] + mean7[3]
    vmdr_sin = np.sin(np.radians(vmdr_true)).astype(np.float32)
    vmdr_cos = np.cos(np.radians(vmdr_true)).astype(np.float32)

    sin_mean = float(np.nanmean(vmdr_sin))
    sin_std = float(np.nanstd(vmdr_sin))
    cos_mean = float(np.nanmean(vmdr_cos))
    cos_std = float(np.nanstd(vmdr_cos))
    sin_std = 1.0 if sin_std < 1e-8 else sin_std
    cos_std = 1.0 if cos_std < 1e-8 else cos_std

    vmdr_sin_norm = (vmdr_sin - sin_mean) / sin_std
    vmdr_cos_norm = (vmdr_cos - cos_mean) / cos_std

    new_shape = list(data.shape)
    new_shape[-1] = 8
    new_data = np.zeros(new_shape, dtype=np.float32)
    new_data[..., 0:3] = data[..., 0:3]
    new_data[..., 3] = vmdr_sin_norm
    new_data[..., 4] = vmdr_cos_norm
    new_data[..., 5:8] = data[..., 4:7]

    new_mean = np.zeros((8,), dtype=np.float32)
    new_std = np.zeros((8,), dtype=np.float32)
    new_mean[0:3] = mean7[0:3]
    new_mean[3] = sin_mean
    new_mean[4] = cos_mean
    new_mean[5:8] = mean7[4:7]

    new_std[0:3] = std7[0:3]
    new_std[3] = sin_std
    new_std[4] = cos_std
    new_std[5:8] = std7[4:7]

    return new_data, new_mean, new_std


data_path = _find_data_file(data_dir)
print(f"检测到数据文件: {data_path}")
data = np.load(data_path)
mean, std = _load_stats(data_dir)

if data.shape[-1] == 8:
    print('✅ 当前数据已是 8 维，split_vmdr 无需处理。')
    print('提示：clear.py 已内置 VMDR -> sin/cos，一步产出 8 维数据。')
    raise SystemExit(0)

if data.shape[-1] != 7:
    raise ValueError(f'不支持的特征维度: {data.shape[-1]}，仅支持 7->8 迁移或已是 8 维')

if not os.path.exists(backup_dir):
    print(f"⏳ 正在备份数据目录至: {backup_dir}")
    shutil.copytree(data_dir, backup_dir)
    print('✅ 备份完成')
else:
    print(f"⚠️ 备份目录已存在，跳过备份: {backup_dir}")

print('📐 检测到旧版 7 维数据，开始执行迁移...')
new_data, new_mean_flat, new_std_flat = _convert_7_to_8(data.astype(np.float32), mean, std)

# 统计形状与数据布局对齐（兼容 data_nodes/data_grid）
data_ndim = new_data.ndim
mean_ref, std_ref = _reshape_stats_for_data(new_mean_flat, new_std_flat, data_ndim)

print('💾 写回数据与统计参数...')
tmp_file = data_path + '.tmp.npy'
np.save(tmp_file, new_data)
os.replace(tmp_file, data_path)
np.save(os.path.join(data_dir, 'norm_mean.npy'), mean_ref)
np.save(os.path.join(data_dir, 'norm_std.npy'), std_ref)

meta_path = os.path.join(data_dir, 'meta.json')
with open(meta_path, 'r') as f:
    meta = json.load(f)

meta['n_features'] = 8
meta['variables'] = ["uo", "vo", "VHM0", "VMDR_sin", "VMDR_cos", "VTM02", "eastward_wind", "northward_wind"]
with open(meta_path, 'w') as f:
    json.dump(meta, f, indent=2)

del data
del new_data
gc.collect()

print('🎉 迁移完成：数据已从 7 维升级为 8 维。')
