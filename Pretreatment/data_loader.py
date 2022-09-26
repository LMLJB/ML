import os
from torchvision import datasets
from torch.utils.data import DataLoader
from Pretreatment import transform

dataset_dir = r'D:\Project\遥感图像场景分类'  # 数据集路径
train_path = os.path.join(dataset_dir, 'train')  # 训练集路径
test_path = os.path.join(dataset_dir, 'test')  # 测试集路径
train_dataset = datasets.ImageFolder(train_path, transform.train_transform)  # 载入训练集
test_dataset = datasets.ImageFolder(test_path, transform.test_transform)  # 载入测试集

BATCH_SIZE = 10

# 训练集的数据加载器
train_loader = DataLoader(train_dataset,
                          batch_size=BATCH_SIZE,
                          shuffle=True,
                          num_workers=0
                          )

# 测试集的数据加载器
test_loader = DataLoader(test_dataset,
                         batch_size=BATCH_SIZE,
                         shuffle=False,
                         num_workers=0
                         )
