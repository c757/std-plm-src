import numpy as np
import os

# 注意：请将这里的路径替换为你实际的最新测试文件夹名称
log_dir = "./logs/2026-03-15 15-19-10_phi2_s_token_O4BnoM" 
file_path = os.path.join(log_dir, "test.npz")

def calc_error(real, pred):
    """计算 MAE 和 MAPE (百分比误差)"""
    mae = np.abs(real - pred).mean()
    # 加上 1e-5 极小值防止真实流速为 0 时发生除零报错
    mape = (np.abs(real - pred) / (np.abs(real) + 1e-5)).mean() * 100
    return mae, mape

if not os.path.exists(file_path):
    print(f"找不到文件: {file_path}，请检查文件夹名称是否正确。")
else:
    print("成功加载测试结果文件...\n")
    data = np.load(file_path)
    
    targets = data['targets']   
    predicts = data['predicts'] 
    mask = data['mask']         
    
    # 获取第一个测试样本，第一个预测时间步的有效海洋节点（排除陆地）
    valid_nodes = np.where(mask[0, 0, :, 0])[0]
    
    if len(valid_nodes) >= 10:
        # 1. 随机抽取 10 个海洋节点 (无放回抽样)
        sampled_nodes = np.random.choice(valid_nodes, size=10, replace=False)
        
        # 提取这 10 个节点的真实值与预测值
        sampled_real_uo = targets[0, 0, sampled_nodes, 0]
        sampled_pred_uo = predicts[0, 0, sampled_nodes, 0]
        sampled_real_vo = targets[0, 0, sampled_nodes, 1]
        sampled_pred_vo = predicts[0, 0, sampled_nodes, 1]
        
        # 提取整体所有有效海洋节点的真实值与预测值
        all_real_uo = targets[0, 0, valid_nodes, 0]
        all_pred_uo = predicts[0, 0, valid_nodes, 0]
        all_real_vo = targets[0, 0, valid_nodes, 1]
        all_pred_vo = predicts[0, 0, valid_nodes, 1]
        
        # 2. 计算抽样误差
        samp_uo_mae, samp_uo_mape = calc_error(sampled_real_uo, sampled_pred_uo)
        samp_vo_mae, samp_vo_mape = calc_error(sampled_real_vo, sampled_pred_vo)
        
        # 3. 计算整体误差
        all_uo_mae, all_uo_mape = calc_error(all_real_uo, all_pred_uo)
        all_vo_mae, all_vo_mape = calc_error(all_real_vo, all_pred_vo)
        
        # 打印展示
        print("=" * 60)
        print("🎯 随机抽取的 10 个节点流速明细 (单位: m/s):")
        for i, node in enumerate(sampled_nodes):
            print(f"  节点 [{node:5d}] | uo(东向) 真实: {sampled_real_uo[i]:6.3f}, 预测: {sampled_pred_uo[i]:6.3f} | vo(北向) 真实: {sampled_real_vo[i]:6.3f}, 预测: {sampled_pred_vo[i]:6.3f}")
        
        print("\n" + "=" * 60)
        print(f"📊 误差统计对比 (抽样 10 个节点 vs 整体 {len(valid_nodes)} 个节点)")
        print(f"【东向洋流 uo】")
        print(f"   抽样 MAE: {samp_uo_mae:.4f} m/s  |  整体 MAE: {all_uo_mae:.4f} m/s")

        
        print(f"\n【北向洋流 vo】")
        print(f"   抽样 MAE: {samp_vo_mae:.4f} m/s  |  整体 MAE: {all_vo_mae:.4f} m/s")

        print("=" * 60 + "\n")