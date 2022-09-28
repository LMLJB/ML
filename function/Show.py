# 数据可视化
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


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


# train_loss, test_loss = load_data()
# show_train_test_loss(train_loss, test_loss)
# show_train_loss(train_loss)
