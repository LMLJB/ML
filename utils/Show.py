# 数据可视化
import os.path
import shutil
import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import MaxNLocator
from torchvision.transforms import ToPILImage
from torchvision import utils as v_utils

project_path = r'C:\ML'
predicted_image_path = project_path + r'\predicated_image'  # 预测图片放置位置
loss_image_path = project_path + r'\loss_image'
default_path = r'C:\ML\model_parameter.csv'
show_image = ToPILImage()  # 把Tensor转变为Image
labels_name = ['停机坪', '停车场', '公园', '公路', '冰岛', '商业区', '墓地', '太阳能发电厂', '居民区', '山地', '岛屿',
               '工厂', '教堂', '旱地', '机场跑道', '林地', '桥梁', '梯田', '棒球场', '水田', '沙漠', '河流', '油田',
               '油罐区', '海滩', '温室', '港口', '游泳池', '湖泊', '火车站', '直升机场', '石质地', '矿区', '稀疏灌木地',
               '立交桥', '篮球场', '网球场', '草地', '裸地', '足球场', '路边停车区', '转盘', '铁路', '风力发电站',
               '高尔夫球场']
colors = ['black', 'darkred', 'red', 'sienna', 'chocolate', 'darkorange', 'olive', 'sage',
          'green', 'darkcyan', 'steelblue', 'slategray', 'navy', 'blue', 'mediumpurple',
          'm', 'deeppink', 'dodgerblue']


# 训练时的loss显示
def show_train_loss(log_train_loss, model_path):
    iterations = len(log_train_loss)
    index = np.arange(0, iterations, 1)
    # fig = plt.figure()
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


# 显示所有模型的loss
def show_all_model_loss(path=default_path):
    data = pd.read_csv(path)
    data = np.array(data)
    data_dic = {}
    for line in data:
        data_dic['model-' + str(line[0])] = {}
    for line in data:
        data_dic['model-' + str(line[0])][str(line[3])] = line[5]
    print(data_dic)
    keys = data_dic.keys()
    print(keys)
    i = 0
    for key in keys:
        epoch_keys = data_dic[key].keys()
        loss = []
        for epoch_key in epoch_keys:
            loss.append(data_dic[key][epoch_key])
        print(loss)
        index = np.arange(1, len(loss) + 1)
        print(index)
        plt.plot(index, loss, c=colors[i], linestyle='solid', label=key)
        i += 1
        i %= len(colors)
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("Loss")
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))  # 用于只显示整数
    plt.show()


# show_all_model_loss()
