# 预测函数
import torch
import torchvision
import os
import torch.nn as nn
from tqdm import tqdm

from function.Show import show_train_loss
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from pretreatment import test_transform

BATCH_SIZE = 96
NUM_WORKERS = 0
test_path = r'C:\ML\test'
test_dataset = datasets.ImageFolder(test_path, transform=test_transform)  # 载入训练集

# 训练集的数据加载器
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)


def predict(loader):
    # 获取cnn网络
    net = torchvision.models.resnet18()
    # 加载模型
    net.load_state_dict(torch.load('test_gpu.pkl'))
    # 设置为推理模式
    net.eval()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # 模型加载到gpu中
    net = net.to(device)
    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in loader:
            images, labels = data
            # 数据加载到gpu中
            images = images.to(device)
            labels = labels.to(device)
            # calculate outputs by running images through the network
            outputs = net(images)
            # 数据加载回cpu
            outputs = outputs.to(device)
            labels = labels.to(device)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('Accuracy of the network on the 10000 test images: %d %%' % (
            100 * correct / total))


predict(test_loader)
