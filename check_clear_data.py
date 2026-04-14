import numpy as np
import json
import os

print(" 开始验证生成的 STD-PLM 数据集...\n")

out_dir = './data/stdplm_input_025'

# ==========================================
# 1. 检查元数据
# ==========================================
print(" [1/4] 加载元数据 (meta.json)...")
try:
    with open(f'{out_dir}/meta.json', 'r') as f:
        meta = json.load(f)
    print(f"  -> 预期维度: T={meta['time_steps']}, N={meta['n_nodes']}, F={meta['n_features']}")
except Exception as e:
    print(f"  -> ❌ 加载元数据失败: {e}")

# ==========================================
# 2. 检查核心张量 (零拷贝模式)
# ==========================================
print("\n [2/4] 加载核心张量 (data_nodes.npy)...")
# mmap_mode='r' 是保护你 24GB 内存的法宝
data_nodes = np.load(f'{out_dir}/data_nodes.npy', mmap_mode='r')
file_size_gb = os.path.getsize(f'{out_dir}/data_nodes.npy') / (1024**3)

print(f"  -> 实际维度: {data_nodes.shape}")
print(f"  -> 数据类型: {data_nodes.dtype}")
print(f"  -> 文件大小: {file_size_gb:.2f} GB")

if data_nodes.shape == (meta['time_steps'], meta['n_nodes'], meta['n_features']):
    print("  -> ✅ 维度验证完美匹配！")
else:
    print("  -> ❌ 维度不匹配，请检查生成步骤。")

# ==========================================
# 3. 抽样检查数据健康度
# ==========================================
print("\n [3/4] 抽样检查数据健康度 (NaN / Inf)...")
# 为了速度和省内存，我们只抽取第 1 个和最后 1 个时间步的数据进行检查
sample_data = np.concatenate([data_nodes[0], data_nodes[-1]])

if np.isnan(sample_data).any() or np.isinf(sample_data).any():
    print("  -> ⚠️ 警告：抽样数据中包含 NaN 或 Inf 值！")
else:
    print("  -> ✅ 数据抽样健康，无 NaN 或异常值。")
    print(f"  -> 抽样均值: {sample_data.mean():.4f}, 最大值: {sample_data.max():.4f}, 最小值: {sample_data.min():.4f}")

# ==========================================
# 4. 检查滑动窗口索引
# ==========================================
print("\n [4/4] 检查滑动窗口索引 (indices.npz)...")
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
print(f"  -> 检查 norm_mean 形状: {mean_stat.shape} (预期为 1,1,7)")

# 抽取洋流 uo (通道0) 的数据，剔除等于 0 的部分（因为那是陆地）
uo_vals = data_nodes[:, :, 0]
nonzero = uo_vals[uo_vals != 0]
print(f"  -> 真实海洋数据归一化后均值: {nonzero.mean():.4f} (越接近0越好)")
print(f"  -> 真实海洋数据归一化后方差: {nonzero.std():.4f} (越接近1越好)")
print("\n 验证完成！")
