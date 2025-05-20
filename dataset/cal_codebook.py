import torch
import glob
import os

from tqdm import tqdm

# 配置参数
data_dir = "dataset/HumanML3D/vqvae_code24"  # 替换为存放 .pth 文件的文件夹路径
excluded_codes = {512, 513}               # 不参与统计的 code
total_codes = 514                        # Codebook 总大小（根据实际情况调整）

# 初始化统计容器
num_parts = 6
part_utilizations = [[] for _ in range(num_parts)]  # 每个部分单独存储利用率
file_list = glob.glob(os.path.join(data_dir, "*.pth"))
# 遍历所有 .pth 文件
for file_path in tqdm(file_list):
    code_idx = torch.load(file_path)  # 加载张量，形状应为 [6, 196]
    
    # 对每个部分单独统计
    for part in range(num_parts):
        # 提取当前部分的 code 并展平
        codes = code_idx[part].flatten().tolist()
        
        # 过滤排除的 code 并统计唯一值
        unique_codes = set(codes)
        unique_codes -= excluded_codes
        valid_unique = len(unique_codes)
        
        
        # 计算当前部分的利用率
        valid_total = total_codes - len(excluded_codes)
        utilization = valid_unique / valid_total
        part_utilizations[part].append(utilization)

# 计算每个部分的平均利用率
part_avg = [sum(utils)/len(utils) for utils in part_utilizations]

# 输出每个部分的结果
for part, avg in enumerate(part_avg):
    print(f"Part {part+1} 平均利用率: {avg:.2%}")

# 计算整体平均利用率
overall_avg = sum(part_avg) / num_parts
print(f"\n整体平均利用率: {overall_avg:.2%}")
