# 数据可视化
import os.path
import shutil
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from torchvision.transforms import ToPILImage
from torchvision import utils as v_utils

project_path = r'C:\ML'
error_image_path = project_path + r'\error_result_image'
image_path = project_path + r'\save_image'
contrast_image_path = project_path + r'\image'
loss_image_path = project_path + r'\loss_image'

show_image = ToPILImage()  # 把Tensor转变为Image
labels_name = ['停机坪', '停车场', '公园', '公路', '冰岛', '商业区', '墓地', '太阳能发电厂', '居民区', '山地', '岛屿',
               '工厂', '教堂', '旱地', '机场跑道', '林地', '桥梁', '梯田', '棒球场', '水田', '沙漠', '河流', '油田',
               '油罐区', '海滩', '温室', '港口', '游泳池', '湖泊', '火车站', '直升机场', '石质地', '矿区', '稀疏灌木地',
               '立交桥', '篮球场', '网球场', '草地', '裸地', '足球场', '路边停车区', '转盘', '铁路', '风力发电站',
               '高尔夫球场']


# 创建文件夹并返回路径
def create_dir(path):
    if os.path.isdir(path):
        shutil.rmtree(path)
    os.mkdir(path)
    return str(path)


# 创建log.txt文件并记录训练模型的超参数
def create_log_model(model_path, dic):
    file = open(model_path + r'\log.txt', 'w')
    for x, y in dic.items():
        file.write(str(x) + " = " + str(y) + "\n")
    file.close()


# 往log.txt文件写入预测准确度
def write_log_model(model_path, accuracy):
    file = open(model_path + r'\log.txt', 'a')
    file.write("accuracy = %.2f%%" % accuracy)
    file.close()


# 训练时的loss显示
def show_train_loss(log_train_loss, model_path):
    iterations = len(log_train_loss)
    index = np.arange(0, iterations, 1)
    fig = plt.figure()
    plt.plot(index, log_train_loss, c='blue', linestyle='solid', label='train')
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))  # 用于只显示整数
    plt.savefig(model_path + r'\loss.jpg')
    # plt.show()


# 以epoch为迭代次数
def show_train_test_loss(log_train_loss, log_test_loss):
    iterations = len(log_train_loss)
    index = np.arange(0, iterations, 1)
    plt.plot(index, log_train_loss, c='blue', linestyle='solid', label='train')
    plt.plot(index, log_test_loss, c='red', linestyle='dashed', label='test')
    plt.legend()
    plt.xlabel("iteration")
    plt.ylabel("Loss")
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))  # 用于只显示整数
    plt.show()


# 获取相应序号文件名
def get_file_name(path, index):
    files_name = os.listdir(path)
    return files_name[index]


def get_model_last_file_name(path):
    num_files = os.listdir(path)
    return num_files[len(num_files) - 1]


# 显示预测错误图片
def show_predicted(predicted, labels, images, prefix_num, total, test_path):
    length = len(predicted)
    base = total - labels.size(0)
    index = 0
    for num in prefix_num:
        if num > base:
            break
        index += 1

    for i in range(length):
        if predicted[i] != labels[i]:
            image_number = base + i
            if image_number >= prefix_num[index]:
                num = image_number - prefix_num[index]  # 图片编号
                index += 1
            else:
                if index - 1 < 0:
                    num = image_number
                else:
                    num = image_number - prefix_num[index - 1]
            path = test_path + '\\' + labels_name[labels[i]]
            file_name = get_file_name(path, num)
            shutil.copy(path + '\\' + file_name, error_image_path + '\\' + labels_name[labels[i]] + str(num + 1)
                        + "-" + labels_name[predicted[i]] + '.jpg')
            # v_utils.save_image(images[i], image_path + '\\' + file_name + '-' + labels_name[labels[i]] + str(num + 1)
            #                    + "-" + labels_name[predicted[i]] + '.jpg')


# 获取路径下每个文件夹中文件数量
def get_files_num(path):
    files_num = []
    for name in labels_name:
        files_num.append(len(os.listdir(path + "\\" + name)))
    prefix_num = [0]  # 前缀和
    prefix_num[0] = files_num[0]
    for i in range(1, len(files_num)):
        prefix_num.append(prefix_num[i - 1] + files_num[i])
    return files_num, prefix_num


# 向文件中追加数据
def save_data(path, data):
    with open(path, 'a') as f:
        f.write(str(data) + '\n')
        f.write("\n")
