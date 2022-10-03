# 数据可视化
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import MaxNLocator

project_path = r'C:\ML'
predicted_image_path = project_path + r'\predicated_image'  # 预测图片放置位置
loss_image_path = project_path + r'\loss_image'
model_loss_path = project_path + r"\model and log\model "
default_path = r'C:\ML\model_parameter.csv'
labels_name = ['停机坪', '停车场', '公园', '公路', '冰岛', '商业区', '墓地', '太阳能发电厂', '居民区', '山地', '岛屿',
               '工厂', '教堂', '旱地', '机场跑道', '林地', '桥梁', '梯田', '棒球场', '水田', '沙漠', '河流', '油田',
               '油罐区', '海滩', '温室', '港口', '游泳池', '湖泊', '火车站', '直升机场', '石质地', '矿区', '稀疏灌木地',
               '立交桥', '篮球场', '网球场', '草地', '裸地', '足球场', '路边停车区', '转盘', '铁路', '风力发电站',
               '高尔夫球场']
colors = ['black', 'darkred', 'red', 'sienna', 'chocolate', 'darkorange', 'olive',
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


# 显示所有模型的train_loss
def show_all_model_loss(path=default_path):
    data = pd.read_csv(path)
    data = np.array(data)
    data_dic = {}
    for line in data:
        data_dic['model ' + str(line[0])] = {}  # 每个模型建立一个字典
    for line in data:
        # 记录train_loss data_dic[模型名称][epoch] = train_loss
        data_dic['model ' + str(line[0])][str(line[4])] = line[5]
    keys = data_dic.keys()  # 所有模型名称
    i = 0
    for key in keys:
        epoch_keys = data_dic[key].keys()
        loss = []
        for epoch_key in epoch_keys:
            loss.append(data_dic[key][epoch_key])
        index = np.arange(1, len(loss) + 1)
        plt.plot(index, loss, c=colors[i], linestyle='solid', label=key)
        i += 1
        i %= len(colors)
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("Loss")
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))  # 用于只显示整数
    plt.savefig(project_path + r'\loss.jpg')
    plt.show()


# 显示model_number模型的train_loss与test_loss的比较
def show_model_train_test_loss(model_number, path=default_path):
    data = pd.read_csv(path)
    data = np.array(data)
    data_dic = {"train_loss": {}, "test_loss": {}}
    for line in data:
        if line[0] == model_number:
            data_dic["train_loss"][str(line[4])] = line[5]  # 记录train_loss
            data_dic["test_loss"][str(line[4])] = line[6]  # 记录test_loss
    keys = data_dic.keys()
    i = 0
    for key in keys:
        epoch_keys = data_dic[key].keys()
        loss = []
        for epoch_key in epoch_keys:
            loss.append(data_dic[key][epoch_key])  # loss列表
        index = np.arange(1, len(loss) + 1)
        plt.plot(index, loss, c=colors[i], linestyle='solid', label=key)
        i += 1
        i %= len(colors)
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("Loss")
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))  # 用于只显示整数
    plt.savefig(model_loss_path + str(model_number) + r'\loss.jpg')
    plt.show()


if __name__ == '__main__':
    show_all_model_loss()
    show_model_train_test_loss(11)
