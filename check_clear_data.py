import numpy as np
import json
import os

print(" 开始验证生成的 STD-PLM 数据集...\n")

out_dir = './data/stdplm_input_025'
EXPECTED_VARS = [
    "uo", "vo", "VHM0", "VMDR_sin", "VMDR_cos", "VTM02", "eastward_wind", "northward_wind"
]


def _pick_data_file(base_dir):
    grid_path = os.path.join(base_dir, 'data_grid.npy')
    node_path = os.path.join(base_dir, 'data_nodes.npy')
    if os.path.exists(grid_path):
        return grid_path
    if os.path.exists(node_path):
        return node_path
    raise FileNotFoundError(f"找不到核心张量文件: {grid_path} 或 {node_path}")


def _pick_mask_file(base_dir):
    mask2d = os.path.join(base_dir, 'ocean_mask_2d.npy')
    mask1d = os.path.join(base_dir, 'ocean_mask.npy')
    if os.path.exists(mask2d):
        return mask2d
    if os.path.exists(mask1d):
        return mask1d
    return None

# ==========================================
# 1. 检查元数据
# ==========================================
print(" [1/5] 加载元数据 (meta.json)...")
try:
    with open(f'{out_dir}/meta.json', 'r') as f:
        meta = json.load(f)
    print(f"  -> 预期维度: T={meta.get('time_steps')}, N={meta.get('n_nodes')}, F={meta.get('n_features')}")
    print(f"  -> tensor_layout: {meta.get('tensor_layout', '未声明(将自动推断)')}")
    vars_meta = meta.get('variables', [])
    if vars_meta:
        print(f"  -> variables: {vars_meta}")
        if vars_meta == EXPECTED_VARS:
            print("  -> ✅ 变量顺序与 8 维目标一致")
        else:
            print("  -> ⚠️ 变量顺序与当前 8 维目标不完全一致，请确认通道映射")
except Exception as e:
    print(f"  -> ❌ 加载元数据失败: {e}")
    raise SystemExit(1)

# ==========================================
# 2. 检查核心张量 (零拷贝模式)
# ==========================================
print("\n [2/5] 加载核心张量 (data_grid.npy / data_nodes.npy)...")
data_path = _pick_data_file(out_dir)
data = np.load(data_path, mmap_mode='r')
file_size_gb = os.path.getsize(data_path) / (1024**3)

print(f"  -> 文件路径: {data_path}")
print(f"  -> 实际维度: {data.shape}")
print(f"  -> 数据类型: {data.dtype}")
print(f"  -> 文件大小: {file_size_gb:.2f} GB")

# 兼容新旧布局
layout = meta.get('tensor_layout', None)
shape_ok = False
if data.ndim == 4:
    # (T,H,W,F)
    t, h, w, f = data.shape
    shape_ok = (
        t == meta.get('time_steps') and
        h == meta.get('n_lat') and
        w == meta.get('n_lon') and
        f == meta.get('n_features')
    )
    if layout not in [None, 'T,H,W,C']:
        print(f"  -> ⚠️ meta.tensor_layout={layout} 与实际 4D 数据不一致")
elif data.ndim == 3:
    # (T,N,F)
    t, n, f = data.shape
    shape_ok = (
        t == meta.get('time_steps') and
        n == meta.get('n_nodes') and
        f == meta.get('n_features')
    )
    if layout not in [None, 'T,N,C']:
        print(f"  -> ⚠️ meta.tensor_layout={layout} 与实际 3D 数据不一致")
else:
    print(f"  -> ❌ 不支持的数据维度: {data.ndim}")

if shape_ok:
    print("  -> ✅ 维度验证匹配")
else:
    print("  -> ❌ 维度不匹配，请检查 clear.py 生成步骤")

# ==========================================
# 3. 抽样检查数据健康度
# ==========================================
print("\n [3/5] 抽样检查数据健康度 (NaN / Inf)...")
# 为了速度和省内存，我们只抽取第 1 个和最后 1 个时间步的数据进行检查
sample_data = np.concatenate([data[0].reshape(-1, data.shape[-1]), data[-1].reshape(-1, data.shape[-1])], axis=0)

if np.isnan(sample_data).any() or np.isinf(sample_data).any():
    print("  -> ⚠️ 警告：抽样数据中包含 NaN 或 Inf 值！")
else:
    print("  -> ✅ 数据抽样健康，无 NaN 或异常值。")
    print(f"  -> 抽样均值: {sample_data.mean():.4f}, 最大值: {sample_data.max():.4f}, 最小值: {sample_data.min():.4f}")

# ==========================================
# 4. 检查滑动窗口索引
# ==========================================
print("\n [4/5] 检查滑动窗口索引 (indices.npz)...")
try:
    indices = np.load(f'{out_dir}/indices.npz')
    train_idx = indices['train_idx']
    val_idx = indices['val_idx']
    test_idx = indices['test_idx']
    print(f"  -> 训练集样本数: {len(train_idx)}")
    print(f"  -> 验证集样本数: {len(val_idx)}")
    print(f"  -> 测试集样本数: {len(test_idx)}")
    print(f"  -> 总样本数: {len(train_idx) + len(val_idx) + len(test_idx)}")
    
    # 验证索引是否越界
    max_idx = max(train_idx.max(), val_idx.max(), test_idx.max())
    window_size = meta['sample_len'] + meta['predict_len']
    if max_idx + window_size <= meta['time_steps']:
        print("  -> ✅ 索引边界安全，无越界风险。")
    else:
        print("  -> ❌ 警告：部分索引可能导致读取越界！")
except Exception as e:
    print(f"  -> ❌ 加载索引失败: {e}")

# ==========================================
# 5. 验证归一化是否正确 (防静默错误)
# ==========================================
print("\n [5/5] 验证归一化数学属性...")
mean_stat = np.load(f'{out_dir}/norm_mean.npy')
std_stat = np.load(f'{out_dir}/norm_std.npy')
print(f"  -> norm_mean 形状: {mean_stat.shape}")
print(f"  -> norm_std 形状: {std_stat.shape}")

if mean_stat.reshape(-1).shape[0] == meta.get('n_features'):
    print("  -> ✅ 统计参数特征维度匹配")
else:
    print("  -> ❌ 统计参数特征维度不匹配")

# 掩码兼容检查
mask_path = _pick_mask_file(out_dir)
if mask_path is None:
    print("  -> ⚠️ 未找到 ocean_mask_2d.npy/ocean_mask.npy，归一化抽样将用非零近似")
    ocean_mask_flat = None
else:
    ocean_mask = np.load(mask_path)
    ocean_mask_flat = ocean_mask.reshape(-1).astype(bool)
    print(f"  -> 掩码文件: {mask_path}, 海洋点占比: {ocean_mask_flat.mean():.4f}")

# 抽样验证 uo (通道0) 的归一化状态
if data.ndim == 4:
    # (T,H,W,F)
    uo_sample = np.concatenate([data[0, :, :, 0].reshape(-1), data[-1, :, :, 0].reshape(-1)], axis=0)
else:
    # (T,N,F)
    uo_sample = np.concatenate([data[0, :, 0], data[-1, :, 0]], axis=0)

if ocean_mask_flat is not None:
    ocean_mask_sample = np.concatenate([ocean_mask_flat, ocean_mask_flat], axis=0)
    vals = uo_sample[ocean_mask_sample]
else:
    vals = uo_sample[uo_sample != 0]

if vals.size == 0:
    print("  -> ⚠️ 没有可用海洋样本用于归一化检查")
else:
    print(f"  -> 海洋样本归一化后均值(抽样): {vals.mean():.4f} (越接近0越好)")
    print(f"  -> 海洋样本归一化后方差(抽样): {vals.std():.4f} (越接近1越好)")

if meta.get('n_features') == 8:
    print("  -> ✅ 当前数据为 8 维流程")
else:
    print(f"  -> ⚠️ 当前 n_features={meta.get('n_features')}，非目标 8 维")

print("\n 验证完成！")
