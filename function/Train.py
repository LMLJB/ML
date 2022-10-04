import torch
# import torchvision
import os
import torch.nn as nn
import time
from tqdm import tqdm
from utils.Show import show_model_train_test_loss
from torchvision import datasets
from torch.utils.data import DataLoader
from function.Pretreatment import train_transform
from utils.FileDeal import save_data, create_dir, create_log_model
from utils.CSVDeal import model_parameter_save
from Model import resnet18  # resnet18/resnet50
from utils.Forecast import encapsulation


# 超参数
LR = 0.00001      # 学习率
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"  # 运行模型选择的设备
BATCH_SIZE = 150  # 一次输入训练的批量
EPOCH = 100       # 训练次数
NUM_WORKERS = 2
LOSS_FUNC = nn.CrossEntropyLoss()  # 定义损失函数
SEED = 1          # 固定种子，以保证获取相同的训练结果
MODEL = resnet18()
project_path = r'C:\ML'  # 项目路径
dataset_dir = r'C:\ML'   # 数据集路径
model_and_log_path = project_path + r"\model and log"
train_path = os.path.join(dataset_dir, 'train')  # 训练集路径

DIC = dict(time=time.strftime('%Y-%m-%d_%H:%M'),  # 记录模型的超参数
           learning_rate=LR,
           batch_size=BATCH_SIZE,
           epoch=EPOCH,
           num_workers=NUM_WORKERS,
           loss_function=LOSS_FUNC,
           seed=SEED,
           model=MODEL
           )

train_dataset = datasets.ImageFolder(train_path, transform=train_transform)  # 载入训练集
torch.manual_seed(SEED)  # 使用随机化种子使神经网络的初始化每次都相同
# 训练集的数据加载器
train_loader = DataLoader(train_dataset,
                          batch_size=BATCH_SIZE,
                          shuffle=True,             # shuffle为是否打乱数据
                          num_workers=NUM_WORKERS)


# 训练模型
def train_model():
    model, model_name = MODEL  # 获取模型与模型名，
    model = model.to(DEVICE)   # 将模型加载到DEVICE中（默认gpu）
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)  # 优化器
    loss_func = LOSS_FUNC
    # 准备数据
    log_all_epoch_history = []
    create_dir(model_and_log_path)  # 创建保存模型的文件夹
    model_parameter_dic = {}  # 保存模型参数
    model_number = len(os.listdir(model_and_log_path))  # 训练的第几个模型
    model_path = create_dir(model_and_log_path + '\\' + "model %d" % model_number)  # 创建model的文件夹地址
    # 开始训练
    for epoch in range(EPOCH):
        DESC = "EPOCH " + str(epoch+1)  # tqdm进度条左侧内容
        UNIT = "Batches"                # tqdm进度条中处理量单位it改为Batches
        COLOUR = "blue"                 # tqdm进度条的颜色
        loss = 0
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
            loop.set_postfix({"loss": loss})
        log_all_epoch_history.append(loss)
        # 模型参数采用字典形式保存
        model_parameter_dic['train_loss'] = loss
        model_parameter_dic['model_number'] = model_number
        model_parameter_dic['model_name'] = model_name
        model_parameter_dic.update(DIC)  # 字典合并
        model_parameter_dic['epoch'] = epoch + 1
        model_parameter_save(model_parameter_dic)  # 将模型参数保存到CSV文件中

        # 每进行一次Epoch，调用测试函数对测试集测试
        encapsulation(model.state_dict(), model_path + r'\model' + str(epoch + 1) + '.pkl', epoch + 1)

    history_data = {"train_loss_change": log_all_epoch_history}
    train_log_loss_path = model_path + r'\loss.txt'  # 本个model的loss记录的地址
    save_data(train_log_loss_path, history_data)  # 将训练数据保存到文件中
    create_log_model(model_path, dic=DIC)  # 记录模型超参数的日志
    show_model_train_test_loss(model_number)  # 显示所有loss变化过程


# 训练训练集
if __name__ == '__main__':
    train_model()
