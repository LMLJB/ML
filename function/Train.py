import torch
# import torchvision
import os
import torch.nn as nn
from tqdm import tqdm
from function.Show import show_train_loss
from torchvision import datasets
from torch.utils.data import DataLoader
from Pretreatment import train_transform
from function.Show import save_data
from Model import resnet18, resnet50  # resnet18/resnet50


# 超参数
LR = 0.00001      # 学习率
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"  # 运行模型选择的设备
BATCH_SIZE = 128  # 一次输入训练的批量
EPOCH = 1         # 训练次数
NUM_WORKERS = 2
LOSS_FUNC = nn.CrossEntropyLoss()  # 定义损失函数
SEED = 1          # 固定种子，以保证获取相同的训练结果
MODEL = resnet18()
dataset_dir = r'C:\ML'  # 数据集路径
train_path = os.path.join(dataset_dir, 'train')  # 训练集路径
train_history_path = r'C:\ML\history\train.txt'

train_dataset = datasets.ImageFolder(train_path, transform=train_transform)  # 载入训练集
torch.manual_seed(SEED)  # 使用随机化种子使神经网络的初始化每次都相同
# 训练集的数据加载器
train_loader = DataLoader(train_dataset,
                          batch_size=BATCH_SIZE,
                          shuffle=True,             # shuffle为是否打乱数据
                          num_workers=NUM_WORKERS)


# 模型训练模型
def train_model():
    model = MODEL
    model = model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)  # 优化器
    loss_func = LOSS_FUNC
    log_all_epoch_history = []
    # 开始训练
    for epoch in range(EPOCH):
        log_train_loss = []  # 记录训练时每个batch的损失值
        DESC = "EPOCH " + str(epoch+1)  # tqdm进度条左侧内容
        UNIT = "Batches"                # tqdm进度条中处理量单位it改为Batches
        COLOUR = "blue"                 # tqdm进度条的颜色
        loop = tqdm(iterable=train_loader, desc=DESC, unit=UNIT, colour=COLOUR)
        for inputs, labels in loop:  # 分配batch data -> inputs为输入图片, labels为输入图片的类型（标签）
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)
            output = model(inputs)            # 将数据放入resnet中计算输出
            loss = loss_func(output, labels)
            optimizer.zero_grad()             # 清空过往梯度
            loss.backward()                   # 反向传播，计算当前梯度
            optimizer.step()                  # 根据梯度更新网络参数
            loss = loss.item()
            log_train_loss.append(loss)
            log_all_epoch_history.append(loss)
            loop.set_postfix({"loss": loss})
    history_data = {"batch_size": BATCH_SIZE, "epoc": EPOCH, "train_loss_change": log_all_epoch_history}
    save_data(train_history_path, history_data)  # 将训练数据保存到文件中
    torch.save(model.state_dict(), 'test_gpu.pkl')  # 保存单个模型
    show_train_loss(log_all_epoch_history)  # 显示所有loss变化过程


# 训练训练集
if __name__ == '__main__':
    train_model()
