import os
import shutil

# 文件夹移动
path = r'C:/ML/train'  # 源文件夹
target_path = r'C:\ML\t_train'  # 目标文件夹

dirs_name = os.listdir(path)  # 获取path文件加下文件夹名称
print(dirs_name)
for dir_name in dirs_name:
    dir_path = path + '\\' + dir_name  # 拼接path下文件夹的路径，如停机坪路径
    files_name = os.listdir(dir_path)  # 获取停机坪等文件夹下的文件名
    print(dir_name)
    os.mkdir(target_path + '\\' + dir_name)  # 在目标文件夹下新建停机坪等文件夹
    for file_name in files_name[:50]:  # 取停机坪等文件夹下的前50个文件
        file_path = dir_path + '\\' + file_name
        # print(file_path)
        # print(target_path + '\\' + dir_name + '\\' + file_name)
        shutil.copy(file_path, target_path + '\\' + dir_name)   # 目标文件，移动后的文件