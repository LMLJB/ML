# 数据可视化
import os.path
import shutil
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from torchvision.transforms import ToPILImage
from torchvision import utils as v_utils

image_path = r'C:\ML\save_image'
show_image = ToPILImage()  # 把Tensor转变为Image
label_name = ['停机坪', '停车场', '公园', '公路', '冰岛', '商业区', '墓地', '太阳能发电厂', '居民区', '山地', '岛屿', '工厂', '教堂', '旱地', '机场跑道', '林地', '桥梁', '梯田', '棒球场', '水田', '沙漠', '河流', '油田', '油罐区', '海滩', '温室', '港口', '游泳池', '湖泊', '火车站', '直升机场', '石质地', '矿区', '稀疏灌木地', '立交桥', '篮球场', '网球场', '草地', '裸地', '足球场', '路边停车区', '转盘', '铁路', '风力发电站', '高尔夫球场']


def load_data():
    log_train_loss = np.random.randn(20)
    log_test_loss = np.random.randn(20)
    return log_train_loss, log_test_loss


# 单条线画图实现
def show_one_line(data, name, x_label_name, y_label_name):
    it = len(data)
    index = np.arange(0, it, 1)
    plt.plot(index, data, c='blue', linestyle='solid', label=name)
    plt.legend()
    plt.xlabel(x_label_name)
    plt.ylabel(y_label_name)
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))  # 用于只显示整数
    plt.show()


# 训练时的loss显示
def show_train_loss(log_train_loss):
    iterations = len(log_train_loss)
    index = np.arange(0, iterations, 1)
    plt.plot(index, log_train_loss, c='blue', linestyle='solid', label='train')
    plt.legend()
    plt.xlabel("Batch")
    plt.ylabel("Loss")
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))  # 用于只显示整数
    plt.show()


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


# 创建文件夹
def create_save_images_dir():
    if os.path.isdir(image_path):
        shutil.rmtree(image_path)
    os.mkdir(image_path)


# 显示预测错误图片
def show_predicted(predicted, labels, images):
    length = len(predicted)
    for i in range(length):
        if predicted[i] != labels[i]:
            print(image_path + '\\' + label_name[labels[i]] + str(i) + '.jpg')
            v_utils.save_image(images[i], image_path + '\\' + label_name[labels[i]] + str(i) + '.jpg')


# train_loss, test_loss = load_data()
# show_train_test_loss(train_loss, test_loss)
# show_train_loss(train_loss)
