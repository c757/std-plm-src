import numpy as np
import os
import json
import shutil

print("启动【波向三角函数拆分】数据重构脚本...")

data_dir = './data/stdplm_input_025'
backup_dir = './data/stdplm_input_025_vol0'

# ==========================================
# 1. 安全备份
# ==========================================
if not os.path.exists(backup_dir):
    print(f"⏳ 正在备份原始数据集至: {backup_dir} (这可能需要一些时间)")
    shutil.copytree(data_dir, backup_dir)
    print("✅ 备份完成！")
else:
    print(f"⚠️ 备份目录 {backup_dir} 已存在，跳过备份步骤。")

# ==========================================
# 2. 读取原始数据
# ==========================================
print("\n⏳ 正在加载原始归一化数据和统计参数...")
data_nodes = np.load(f'{data_dir}/data_nodes.npy') # 原始 shape: (T, N, 7)
mean = np.load(f'{data_dir}/norm_mean.npy')        # 原始 shape: (1, 1, 7)
std = np.load(f'{data_dir}/norm_std.npy')          # 原始 shape: (1, 1, 7)

T, N, F = data_nodes.shape
if F != 7:
    raise ValueError(f"预期原始特征数为 7，但当前读取为 {F}，请确认数据集状态！")

# ==========================================
# 3. 反归一化提取真实的 VMDR (Index 3)
# ==========================================
print("📐 正在提取波向特征 (VMDR) 并反归一化...")
vmdr_norm = data_nodes[..., 3]
vmdr_mean = mean[0, 0, 3]
vmdr_std = std[0, 0, 3]

# 还原成真实的度数 (0 ~ 360)
vmdr_true = vmdr_norm * vmdr_std + vmdr_mean 

# ==========================================
# 4. 三角函数拆分 (转换为弧度后计算)
# ==========================================
print("🔄 正在执行 Sine / Cosine 拆分投影...")
vmdr_sin = np.sin(np.radians(vmdr_true))
vmdr_cos = np.cos(np.radians(vmdr_true))

# 对新的 sin/cos 特征也进行 Z-score 归一化 (保持与其他变量分布统一)
sin_mean, sin_std = np.nanmean(vmdr_sin), np.nanstd(vmdr_sin)
cos_mean, cos_std = np.nanmean(vmdr_cos), np.nanstd(vmdr_cos)

sin_std = 1.0 if sin_std < 1e-8 else sin_std
cos_std = 1.0 if cos_std < 1e-8 else cos_std

vmdr_sin_norm = (vmdr_sin - sin_mean) / sin_std
vmdr_cos_norm = (vmdr_cos - cos_mean) / cos_std

# ==========================================
# 5. 构建全新的 8 维数据集
# ==========================================
print("\n🔨 正在内存中拼接全新的 8 维时空张量...")
new_data_nodes = np.zeros((T, N, 8), dtype=np.float32)

# [0, 1] uo, vo | [2] VHM0
new_data_nodes[..., 0:3] = data_nodes[..., 0:3]
# [3, 4] 新的波向 sin 和 cos
new_data_nodes[..., 3] = vmdr_sin_norm
new_data_nodes[..., 4] = vmdr_cos_norm
# [5] VTM02 (原索引 4) | [6, 7] wind_u, wind_v (原索引 5, 6)
new_data_nodes[..., 5:8] = data_nodes[..., 4:7]

# 更新均值和标准差数组
new_mean = np.zeros((1, 1, 8), dtype=np.float32)
new_std = np.zeros((1, 1, 8), dtype=np.float32)

new_mean[0, 0, 0:3] = mean[0, 0, 0:3]
new_mean[0, 0, 3] = sin_mean
new_mean[0, 0, 4] = cos_mean
new_mean[0, 0, 5:8] = mean[0, 0, 4:7]

new_std[0, 0, 0:3] = std[0, 0, 0:3]
new_std[0, 0, 3] = sin_std
new_std[0, 0, 4] = cos_std
new_std[0, 0, 5:8] = std[0, 0, 4:7]

# ==========================================
# 6. 安全覆盖保存与元数据更新 (绕过 Windows 文件锁)
# ==========================================
import gc

print("🧹 正在清理内存并释放原始文件句柄...")
# 强制删除原始数据变量，解除可能的底层文件关联
del data_nodes 
gc.collect()

print("💾 正在安全写入磁盘中 (先存为临时文件)...")
tmp_file = f'{data_dir}/data_nodes_tmp.npy'
real_file = f'{data_dir}/data_nodes.npy'

# 先保存到临时文件
np.save(tmp_file, new_data_nodes)

# 安全替换原始文件（即使原始文件有一点系统残留，这种方式也更安全）
import os
os.replace(tmp_file, real_file)

np.save(f'{data_dir}/norm_mean.npy', new_mean)
np.save(f'{data_dir}/norm_std.npy', new_std)

with open(f'{data_dir}/meta.json', 'r') as f:
    meta = json.load(f)

meta['n_features'] = 8
if 'variables' in meta:
    old_vars = meta['variables']
    # 插入 sin 和 cos
    new_vars = old_vars[:3] + ['VMDR_sin', 'VMDR_cos'] + old_vars[4:]
    meta['variables'] = new_vars

with open(f'{data_dir}/meta.json', 'w') as f:
    json.dump(meta, f, indent=2)

print("\n🎉 数据重构大功告成！总特征数已正式从 7 维跃升至 8 维。")