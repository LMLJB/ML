# 预测函数
import torch
import torchvision
import os
import torch.nn as nn
from tqdm import tqdm
from function.Show import show_train_loss
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision import datasets
from pretreatment import test_transform
from function.Show import deal_predicted, create_save_images_dir

BATCH_SIZE = 50
NUM_WORKERS = 0
test_path = r'C:\ML\t_test'
test_dataset = datasets.ImageFolder(test_path, transform=test_transform)  # 载入训练集
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE,   # 训练集的数据加载器
                         shuffle=False, num_workers=NUM_WORKERS)


def predict():
    net = torchvision.models.resnet18()  # 获取cnn网络
    net.load_state_dict(torch.load('test_gpu.pkl'))  # 加载模型
    net.eval()  # 设置为推理模式
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = net.to(device)  # 模型加载到gpu中
    correct = 0
    total = 0
    create_save_images_dir()
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            # 数据加载到gpu中
            images = images.to(device)
            labels = labels.to(device)
            outputs = net(images)
            # 数据加载回cpu
            outputs = outputs.to(device)
            labels = labels.to(device)
            _, predicted = torch.max(outputs.data, 1)  # 取行最大值
            total += labels.size(0)
            # print("images: ", images)
            # print("predicted: ", predicted)
            # print("labels: ", labels)
            deal_predicted(predicted, labels, images)
            correct += (predicted == labels).sum().item()
    print('Accuracy of the network on the 10000 test images: %d %%' % (
            100 * correct / total))


# 预测
predict()
