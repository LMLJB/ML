import torch
import torchvision
import os
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms.functional as F
# from tqdm import tqdm
from torchvision import datasets
from torch.utils.data import DataLoader
from pretreatment import train_transform

# from model import ResNet18

# 超参数
LR = 0.001  # 学习率
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # 运行模型选择的设备
BATCH_SIZE = 16  # 一次输入训练的批量
# WEIGHT_DECAY = 0
EPOCH = 2  # 训练次数
NUM_WORKERS = 0
# PIN_MEMORY = True
# LOAD_MODEL = False  # 要绘图就True，否则就False
dataset_dir = r'C:\ML'  # 数据集路径
train_path = os.path.join(dataset_dir, 'train')  # 训练集路径
train_dataset = datasets.ImageFolder(train_path, transform=train_transform)  # 载入训练集

# 训练集的数据加载器
train_loader = DataLoader(train_dataset,
                          batch_size=BATCH_SIZE,
                          shuffle=True,  # shuffle是否打乱数据
                          num_workers=NUM_WORKERS)


## 每个batch输入进行的操作，用于for epoch中的for batch
# def train_one_batch()


def train_model():
    model = torchvision.models.resnet18()  # 设置模型
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)  # 优化器
    loss_func = nn.CrossEntropyLoss()  # 定义损失函数
    # 开始训练
    for epoch in range(EPOCH):
        running_loss = 0.0
        for step, (inputs, labels) in enumerate(train_loader):  # 分配batch data
            # 未加载到GPU中
            output = model(inputs)  # 将数据放入cnn中计算输出
            loss = loss_func(output, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ### 梯度下降算法 ###
            # 数据加载到cpu中
            loss = loss.to('cpu')
            running_loss += loss.item()
            if step % 10 == 0:
                print('[%d, %5d] loss: %.3f' % (epoch + 1, step + 1, running_loss / 10))
                running_loss = 0.0
                # 保存模型
    torch.save(model.state_dict(), 'test_cifar_gpu.pkl')
