import torch
import torchvision
import os
import torch.nn as nn
from tqdm import tqdm

from function.Show import show_train_loss
from torchvision import datasets
from torch.utils.data import DataLoader
from pretreatment import train_transform

# from model import ResNet18

# 超参数
LR = 0.001  # 学习率
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # 运行模型选择的设备
BATCH_SIZE = 50  # 一次输入训练的批量
# WEIGHT_DECAY = 0
EPOCH = 1  # 训练次数
NUM_WORKERS = 0
# PIN_MEMORY = True
# LOAD_MODEL = False  # 要绘图就True，否则就False
dataset_dir = r'C:\ML'  # 数据集路径
train_path = os.path.join(dataset_dir, 't_train')  # 训练集路径
train_dataset = datasets.ImageFolder(train_path, transform=train_transform)  # 载入训练集
# 种子seed

# 训练集的数据加载器
train_loader = DataLoader(train_dataset,
                          batch_size=BATCH_SIZE,
                          shuffle=True,  # shuffle是否打乱数据
                          num_workers=NUM_WORKERS)


# 每个batch输入进行的操作，用于for epoch中的for batch
def train_model():
    model = torchvision.models.resnet18()  # 设置模型
    model = model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)  # 优化器
    loss_func = nn.CrossEntropyLoss()  # 定义损失函数
    log_epoch_history = {}
    # 开始训练
    for epoch in range(EPOCH):
        # 单个batch训练
        running_loss = 0.0
        log_train_loss = []  # 记录训练时每个batch的损失值
        loop = tqdm(train_loader)
        for step, (inputs, labels) in enumerate(loop):  # 分配batch data
            # 未加载到GPU中
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)
            output = model(inputs)  # 将数据放入cnn中计算输出
            loss = loss_func(output, labels)
            optimizer.zero_grad()  # 清空过往梯度
            loss.backward()  # 反向传播，计算当前梯度；
            optimizer.step()  # 根据梯度更新网络参数
            loss = loss.to(DEVICE)  # loss.to('cpu')
            running_loss += loss.item()
            # TODO 记录训练过程中的正确率
            # print('[%d, %5d] loss: %.3f' % (epoch, step + 1, running_loss))
            log_train_loss.append(running_loss)
            running_loss = 0.0
        show_train_loss(log_train_loss)  # 显示训练的loss变化过程
        log_epoch_history[str(epoch + 1)] = log_train_loss  # 记录整个训练历史
    # 保存单个模型
    # TODO 注意，保存模型的文件名已经更改，预测函数读取的文件名未更改，记得更改。更改后可以把该条注释删除
    torch.save(model.state_dict(), 'test_gpu.pkl')
    return log_epoch_history


train_model()
