import copernicusmarine
import os

# ==========================================
# 第一部分：基础设置
# ==========================================
# 1. 设置保存数据的文件夹
output_dir = "./data"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 2. 设定研究区域 (西北太平洋)
lon_min, lon_max = 100.5, 179.5
lat_min, lat_max = 0.5, 50.5

print("初始化 Copernicus Marine 多数据集下载任务...")
print(f"目标区域: 经度[{lon_min}, {lon_max}], 纬度[{lat_min}, {lat_max}]")

# ==========================================
# 第二部分：定义下载任务清单 (核心部分)
# ==========================================
download_tasks = [
    {
        # 任务 1：下载【风场】相关数据
        "task_name": "风场数据 (Wave & Wind)",
        "dataset_id": "cmems_obs-wind_glo_phy_nrt_l4_0.125deg_PT1H",
        "variables": ["eastward_wind", "northward_wind"],
        "start_time": "2023-12-31 00:00:00",
        "end_time": "2025-12-01 23:59:59",
        "output_file": "wind_data.nc"   # 保存的文件名
    },
    {
        # 任务 2：下载【洋流】相关数据
        "task_name": "洋流数据 (Ocean Currents)",
        "dataset_id": "cmems_mod_glo_phy_anfc_0.083deg_PT1H-m",
        "variables": ["uo", "vo"],           # uo 是东向流, vo 是北向流
        "start_time": "2023-12-31 00:00:00",
        "end_time": "2025-12-01 23:59:59",
        "output_file": "currents_data.nc"    # 保存的文件名
    },
        {
        # 任务 3：下载【洋流】相关数据
        "task_name": "波浪数据 (Ocean Currents)",
        "dataset_id": "cmems_mod_glo_wav_anfc_0.083deg_PT3H-i",
        "variables": ["VHM0", "VMDR","VTM02"],
        "start_time": "2023-12-31 00:00:00",
        "end_time": "2025-12-01 23:59:59",
        "output_file": "wave_data.nc"    # 保存的文件名
    }
]

# ==========================================
# 第三部分：自动循环执行下载
# ==========================================
for task in download_tasks:
    print("-" * 50)
    print(f"开始执行任务: {task['task_name']}")
    print(f"目标数据集ID: {task['dataset_id']}")
    print(f"下载变量: {task['variables']}")
    print(f"时间范围: {task['start_time']} 至 {task['end_time']}")
    
    try:
        # 调用官方 API 进行裁剪和下载
        copernicusmarine.subset(
            dataset_id=task["dataset_id"],
            variables=task["variables"],
            minimum_longitude=lon_min,
            maximum_longitude=lon_max,
            minimum_latitude=lat_min,
            maximum_latitude=lat_max,
            start_datetime=task["start_time"],
            end_datetime=task["end_time"],
            output_filename=task["output_file"],
            output_directory=output_dir,
            force_download=True  # 如果遇到同名文件则覆盖
        )
        print(f"✅ {task['task_name']} 下载成功！文件保存在: {output_dir}/{task['output_file']}")
        
    except Exception as e:
        print(f"❌ {task['task_name']} 下载失败！")
        print(f"错误信息: {e}")
        print("请检查：1. Dataset ID 是否正确； 2. 该数据集是否包含你填写的变量名。")
        continue # 遇到错误跳过，继续执行下一个任务

print("-" * 50)
print("所有下载任务执行完毕！请去 data 文件夹查看 .nc 文件。")
