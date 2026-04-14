import numpy as np
import json
import os

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
            print(f"   - 特徵維度: {meta.get('n_features')}")
            if 'feature_names' in meta:
                print(f"   - 預定義特徵名: {meta['feature_names']}")
    else:
        print("❌ 未找到 meta.json")

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

    print("\n📊 各特徵維度統計特性分析：")
    print("-" * 70)
    print(f"{'索引':<6} | {'均值 (Mean)':<12} | {'標準差 (Std)':<12} | {'推測物理量'}")
    print("-" * 70)

    for i in range(len(means_vec)):
        m = means_vec[i]
        s = stds_vec[i]
        
        # 啟發式邏輯判斷
        logic = "未知"
        if -1 < m < 1 and 0 < s < 2:
            logic = "洋流 (uo/vo) 或 歸一化後的風"
        elif -20 < m < 20 and 1 < s < 15:
            logic = "海風 (u10/v10) - 典型值"
        elif 90000 < m < 110000:
            logic = "❗ 氣壓 (MSL/PSFC) - 量級極大"
        elif 270 < m < 310:
            logic = "🌡️ 溫度 (SST/T2) - 開爾文單位"
        elif 0 <= m < 10 and 0 < s < 5:
            logic = "🌊 波浪高度 (SWH) 或 潮位"

        print(f"{i:<8} | {m:<14.4f} | {s:<14.4f} | {logic}")

    print("-" * 70)
    
    # 3. 專門檢查第四個特徵 (索引 3)
    if len(means_vec) > 3:
        m3 = means_vec[3]
        s3 = stds_vec[3]
        print(f"\n📢 針對「第四個特徵 (Index 3)」的特別檢查：")
        if s3 > 50:
            print(f"⚠️  警告：索引 3 的標準差為 {s3:.2f}，這遠超正常風速範圍！")
            print(f"    這極大機率是【氣壓】或其他高變量物理量。")
            print(f"    如果你將其強行當作『風速』反歸一化，誤差會被放大 {s3:.0f} 倍。")
        else:
            print(f"✅ 索引 3 的數值特徵符合風速或小量綱物理量。")

    print("\n💡 修復建議：")
    print("1. 如果診斷顯示索引 2,3 不是風，請修改 main.py 中 OceanScaler 的 var_indices 映射。")
    print("2. 確保 model.py 中 forward 的切片 x_reshaped[..., 2:4] 對應的是真正的風速索引。")

if __name__ == "__main__":
    # 指向你的數據路徑
    target_dir = './data/stdplm_input_025'
    diagnose_ocean_data(target_dir)