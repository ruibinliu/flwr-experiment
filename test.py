import os.path
import shutil

# 源文件路径
source_file = 'source.txt'

# 目标文件路径
target_file = 'target.txt'

path1 = r'C:\Users\ruibi\WPSDrive\1264330518\WPS云盘\MUST\2402-资讯科技\资讯科技2402\学生作业\D12期末作业-Moodle系统下载\151399790'
path2 = r'C:\Users\ruibi\WPSDrive\1264330518\WPS云盘\MUST\2402-资讯科技\资讯科技2402\学生作业\D12期末作业-Moodle系统下载\新建文件夹'

# 复制文件
for file in source_file:
    shutil.copyfile(os.path.join(path1), os.path.join(path2, file))

print("文件复制完成！")