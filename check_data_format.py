import numpy as np
import json
import os

EXPECTED_VARS = [
    "uo", "vo", "VHM0", "VMDR_sin", "VMDR_cos", "VTM02", "eastward_wind", "northward_wind"
]


def _pick_data_file(data_dir):
    grid_path = os.path.join(data_dir, 'data_grid.npy')
    node_path = os.path.join(data_dir, 'data_nodes.npy')
    if os.path.exists(grid_path):
        return grid_path
    if os.path.exists(node_path):
        return node_path
    return None

def diagnose_ocean_data(data_dir):
    print("="*50)
    print(f"🔍 正在診斷數據目錄: {data_dir}")
    print("="*50)

    # 1. 檢查 meta.json
    meta_path = os.path.join(data_dir, 'meta.json')
    if os.path.exists(meta_path):
        with open(meta_path, 'r') as f:
            meta = json.load(f)
            print(f"✅ 找到 meta.json")
            print(f"   - 總節點數: {meta.get('n_nodes')}")
            print(f"   - 網格大小: n_lat={meta.get('n_lat')}, n_lon={meta.get('n_lon')}")
            print(f"   - 特徵維度: {meta.get('n_features')}")
            print(f"   - 張量布局: {meta.get('tensor_layout', '未聲明')}")
            vars_meta = meta.get('variables', [])
            if vars_meta:
                print(f"   - 預定義特徵名: {vars_meta}")
                if vars_meta == EXPECTED_VARS:
                    print("   - ✅ 8維變量定義與當前流程一致")
                else:
                    print("   - ⚠️ 變量順序與當前8維流程不完全一致，請核對映射")
    else:
        print("❌ 未找到 meta.json")
        return

    data_path = _pick_data_file(data_dir)
    if data_path is None:
        print("❌ 未找到 data_grid.npy 或 data_nodes.npy")
    else:
        data = np.load(data_path, mmap_mode='r')
        print(f"   - 核心數據文件: {os.path.basename(data_path)}")
        print(f"   - 實際形狀: {data.shape}")
        if data.ndim == 4:
            ok = (
                data.shape[0] == meta.get('time_steps') and
                data.shape[1] == meta.get('n_lat') and
                data.shape[2] == meta.get('n_lon') and
                data.shape[3] == meta.get('n_features')
            )
            print(f"   - 布局推斷: T,H,W,C | 形狀匹配: {'✅' if ok else '❌'}")
        elif data.ndim == 3:
            ok = (
                data.shape[0] == meta.get('time_steps') and
                data.shape[1] == meta.get('n_nodes') and
                data.shape[2] == meta.get('n_features')
            )
            print(f"   - 布局推斷: T,N,C | 形狀匹配: {'✅' if ok else '❌'}")
        else:
            print(f"   - ⚠️ 不支持的數據維度: {data.ndim}")

    # 2. 讀取統計參數 (這是判斷量綱的核心)
    mean_path = os.path.join(data_dir, 'norm_mean.npy')
    std_path = os.path.join(data_dir, 'norm_std.npy')

    if not os.path.exists(mean_path) or not os.path.exists(std_path):
        print("❌ 錯誤：找不到 norm_mean.npy 或 norm_std.npy")
        return

    means = np.load(mean_path) # 通常是 (1, 1, F)
    stds = np.load(std_path)
    
    # 壓縮維度到 (F,)
    means_vec = means.flatten()
    stds_vec = stds.flatten()

    if len(means_vec) != meta.get('n_features'):
        print(f"\n⚠️ 統計向量長度不匹配: len(mean)={len(means_vec)}, n_features={meta.get('n_features')}")

    print("\n📊 各特徵維度統計特性分析：")
    print("-" * 70)
    print(f"{'索引':<6} | {'特徵名':<16} | {'均值 (Mean)':<12} | {'標準差 (Std)':<12} | {'推測物理量'}")
    print("-" * 70)

    for i in range(len(means_vec)):
        m = means_vec[i]
        s = stds_vec[i]
        feat_name = meta.get('variables', [f'f{i}' for i in range(len(means_vec))])[i] if i < len(meta.get('variables', [])) else f'f{i}'
        
        # 啟發式邏輯判斷
        logic = "未知"
        if feat_name in ['uo', 'vo', 'eastward_wind', 'northward_wind']:
            logic = "速度分量 (m/s)"
        elif feat_name == 'VHM0':
            logic = "有效波高"
        elif feat_name == 'VTM02':
            logic = "波浪周期"
        elif feat_name in ['VMDR_sin', 'VMDR_cos']:
            logic = "波向三角分量 (無量綱)"
        elif -1 < m < 1 and 0 < s < 2:
            logic = "可能已標準化物理量"

        print(f"{i:<8} | {feat_name:<16} | {m:<14.4f} | {s:<14.4f} | {logic}")

    print("-" * 70)
    
    # 3. 针对当前 8 维流程的关键检查
    if meta.get('n_features') != 8:
        print(f"\n⚠️ 當前 n_features={meta.get('n_features')}，與目標 8 維不一致")
    if np.any(stds_vec <= 0):
        bad = np.where(stds_vec <= 0)[0].tolist()
        print(f"⚠️ 存在非正標準差通道: {bad}")
    else:
        print("\n✅ norm_std 全通道為正，統計參數有效")

    print("\n💡 修復建議：")
    print("1. clear.py 已一步產出 8 維，優先使用 clear.py 生成數據。")
    print("2. 若是歷史 7 維數據，僅用 split_vmdr.py 做一次遷移。")
    print("3. 確保 main.py 的 OceanScaler.var_indices 與 variables 順序一致。")

if __name__ == "__main__":
    # 指向你的數據路徑
    target_dir = './data/stdplm_input_025'
    diagnose_ocean_data(target_dir)