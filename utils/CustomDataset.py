import os
import shutil
from utils.FileDeal import delete_and_create_dir


# 用于创建自己的小量数据集
def custom_dataset(path, target_path, number=50):
    dirs_name = os.listdir(path)  # 获取path文件加下文件夹名称
    for dir_name in dirs_name:
        dir_path = path + '\\' + dir_name  # 拼接path下文件夹的路径，如停机坪路径
        files_name = os.listdir(dir_path)  # 获取停机坪等文件夹下的文件名
        print(dir_name)
        os.mkdir(target_path + '\\' + dir_name)  # 在目标文件夹下新建停机坪等文件夹
        for file_name in files_name[:number]:  # 取停机坪等文件夹下的前50个文件
            file_path = dir_path + '\\' + file_name
            shutil.copy(file_path, target_path + '\\' + dir_name)  # 目标文件，移动后的文件
    print("运行完成")


# 自定义训练集
def custom_train_dataset(number=50):
    target_path = r'C:\ML\t_train'
    delete_and_create_dir(target_path)
    custom_dataset(path=r'C:\ML\train', target_path=target_path, number=number)


# 自定义测试集
def custom_test_dataset(number=50):
    target_path = r'C:\ML\t_test'
    delete_and_create_dir(target_path)
    custom_dataset(path=r'C:\ML\test', target_path=target_path, number=number)

