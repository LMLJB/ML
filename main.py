import torch
from tqdm import tqdm
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
##
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision import models
import torch.optim as optim
from torch.optim import lr_scheduler

from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from Pretreatment import data_loader
import warnings
warnings.filterwarnings("ignore")

device = torch.device('cuda')

# dataset_dir = r'D:\Project\遥感图像场景分类'  # 数据集路径
# train_path = os.path.join(dataset_dir, 'train')  # 训练集路径
# test_path = os.path.join(dataset_dir, 'test')  # 测试集路径
# train_dataset = datasets.ImageFolder(train_path, transform.train_transform)  # 载入训练集
# test_dataset = datasets.ImageFolder(test_path, transform.test_transform)  # 载入测试集

class_names = data_loader.train_dataset.classes  # 各类别名称构成的一个列表
n_class = len(class_names)
class_to_idx = data_loader.train_dataset.class_to_idx  # 调用函数，达到映射关系：类别到索引号
idx_to_labels = {y: x for x, y in class_to_idx.items()}  # 反转上行代码的映射关系：索引号到类别
np.save('labels_to_idx.npy', class_to_idx)  # 保存为本地的npy文件
np.save('idx_to_labels.npy', idx_to_labels)  # 保存为本地的npy文件

# BATCH_SIZE = 10
# # 训练集的数据加载器
# train_loader = DataLoader(train_dataset,
#                           batch_size=BATCH_SIZE,
#                           shuffle=True,
#                           num_workers=0
#                           )
#
# # 测试集的数据加载器
# test_loader = DataLoader(test_dataset,
#                          batch_size=BATCH_SIZE,
#                          shuffle=False,
#                          num_workers=0
#                          )

model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)  # 使用ResNet18的默认权重
model.fc = nn.Linear(model.fc.in_features, n_class)
optimizer = optim.Adam(model.parameters())

model = model.to(device)
criterion = nn.CrossEntropyLoss()  # 交叉熵损失函数
EPOCHS = 2  # 训练轮次 Epoch
lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)  # 学习率降低策略


def train_one_batch(images, labels):
    """
    运行一个 batch 的训练，返回当前 batch 的训练日志
    """

    # 获得一个 batch 的数据和标注
    images = images.to(device)
    labels = labels.to(device)

    outputs = model(images)  # 输入模型，执行前向预测
    loss = criterion(outputs, labels)  # 计算当前 batch 中，每个样本的平均交叉熵损失函数值

    # 优化更新权重
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # 获取当前 batch 的标签类别和预测类别
    _, preds = torch.max(outputs, 1)  # 获得当前 batch 所有图像的预测类别
    preds = preds.cpu().numpy()
    loss = loss.detach().cpu().numpy()
    outputs = outputs.detach().cpu().numpy()
    labels = labels.detach().cpu().numpy()

    log_train = {'epoch': epoch, 'batch': batch_idx, 'train_loss': loss,
                 'train_accuracy': accuracy_score(labels, preds)}
    # log_train['train_precision'] = precision_score(labels, preds, average='macro')
    # log_train['train_recall'] = recall_score(labels, preds, average='macro')
    # log_train['train_f1-score'] = f1_score(labels, preds, average='macro')

    return log_train


def evaluate_testset():
    """
    在整个测试集上评估，返回分类评估指标日志
    """

    loss_list = []
    labels_list = []
    preds_list = []

    with torch.no_grad():
        for images, labels in data_loader.test_loader:  # 生成一个 batch 的数据和标注
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)  # 输入模型，执行前向预测

            # 获取整个测试集的标签类别和预测类别
            _, preds = torch.max(outputs, 1)  # 获得当前 batch 所有图像的预测类别
            preds = preds.cpu().numpy()
            loss = criterion(outputs, labels)  # 由 logit，计算当前 batch 中，每个样本的平均交叉熵损失函数值
            loss = loss.detach().cpu().numpy()
            outputs = outputs.detach().cpu().numpy()
            labels = labels.detach().cpu().numpy()

            loss_list.append(loss)
            labels_list.extend(labels)
            preds_list.extend(preds)

    log_test = {'epoch': epoch, 'test_loss': np.mean(loss), 'test_accuracy': accuracy_score(labels_list, preds_list),
                'test_precision': precision_score(labels_list, preds_list, average='macro'),
                'test_recall': recall_score(labels_list, preds_list, average='macro'),
                'test_f1-score': f1_score(labels_list, preds_list, average='macro')}

    return log_test


epoch = 0
batch_idx = 0
best_test_accuracy = 0

# 训练日志-训练集
# df_train_log = pd.DataFrame()
log_train = {}
log_train['epoch'] = 0
log_train['batch'] = 0
images, labels = next(iter(data_loader.train_loader))
log_train.update(train_one_batch(images, labels))
# df_train_log = df_train_log.append(log_train, ignore_index=True)
# 训练日志-测试集
# df_test_log = pd.DataFrame()
# log_test = {}
# log_test['epoch'] = 0
# log_test.update(evaluate_testset())
# df_test_log = df_test_log.append(log_test, ignore_index=True)


if __name__ == '__main__':
    for epoch in range(1, EPOCHS + 1):

        print(f'Epoch {epoch}/{EPOCHS}')

        ## 训练阶段
        model.train()
        for images, labels in tqdm(data_loader.train_loader):  # 获得一个 batch 的数据和标注
            batch_idx += 1
            log_train = train_one_batch(images, labels)
            # df_train_log = df_train_log.append(log_train, ignore_index=True)

        lr_scheduler.step()

        ## 测试阶段
        model.eval()
        log_test = evaluate_testset()
        # df_test_log = df_test_log.append(log_test, ignore_index=True)

        保存最新的最佳模型文件
        if log_test['test_accuracy'] > best_test_accuracy:
            # 删除旧的最佳模型文件(如有)
            old_best_checkpoint_path = 'checkpoints/best-{:.3f}.pth'.format(best_test_accuracy)
            if os.path.exists(old_best_checkpoint_path):
                os.remove(old_best_checkpoint_path)
            # 保存新的最佳模型文件
            new_best_checkpoint_path = 'checkpoints/best-{:.3f}.pth'.format(log_test['test_accuracy'])
            torch.save(model, new_best_checkpoint_path)
            print('保存新的最佳模型', 'checkpoints/best-{:.3f}.pth'.format(best_test_accuracy))
            best_test_accuracy = log_test['test_accuracy']

    df_train_log.to_csv('训练日志-训练集.csv', index=False)
    df_test_log.to_csv('训练日志-测试集.csv', index=False)

    model = torch.load('checkpoints/best-{:.3f}.pth'.format(best_test_accuracy))

    model.eval()
    print(evaluate_testset())
