import os

# 原始的 ground truth 文件路径
original_gt_path = 'D:/Road Obstacle Detection/data/StixelsGroundTruth.txt'
# 再次过滤后的 ground truth 文件路径
filtered_gt_path = 'D:/Road Obstacle Detection/data/StixelsGroundTruth_selected_drives.txt'

# 要保留的行程列表，这里对应 drive_id
selected_drives = [2, 9, 13, 64]

# 读取原始文件
with open(original_gt_path, 'r') as f:
    lines = f.readlines()

# 过滤出指定行程的数据
filtered_lines = []
for line in lines:
    parts = line.strip().split('\t')
    series_date = parts[0]
    series_id = int(parts[1])
    if series_date == '09_26' and series_id in selected_drives:
        filtered_lines.append(line)

# 将过滤后的数据写入新文件
with open(filtered_gt_path, 'w') as f:
    f.writelines(filtered_lines)