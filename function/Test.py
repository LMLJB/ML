# 预测函数
import torch
# import torchvision
import os

from torch import nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import datasets
from function.Pretreatment import test_transform
from utils.FileDeal import show_predicted, create_predicted_dir, get_files_num, \
    get_model_last_file_name, write_log_model
from Model import resnet18  # resnet18/resnet50
# from Train import model_and_log_path, get_epoch
from utils.CSVDeal import model_parameter_add_acc

# 超参数
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # 运行模型选择的设备
BATCH_SIZE = 128
NUM_WORKERS = 2
MODEL = resnet18()
project_path = r'C:\ML'  # 项目路径
dataset_dir = r'C:\ML'  # 数据集路径
test_path = os.path.join(dataset_dir, 't_test')  # 测试集路径
predicated_image_path = project_path + r'\predicted_image'
test_history_path = project_path + r'\history\test.txt'
model_and_log_path = r'C:\ML\model and log'
test_dataset = datasets.ImageFolder(test_path, transform=test_transform)  # 载入训练集
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE,  # 测试集的数据加载器
                         shuffle=False, num_workers=NUM_WORKERS)
LOSS_FUNC = nn.CrossEntropyLoss()  # 定义损失函数


def predict(epoch):
    model, model_name = MODEL
    model_path = model_and_log_path + '\\' + get_model_last_file_name(model_and_log_path)  # 模型地址可能需要常改
    print("epoch: ", epoch)
    model.load_state_dict(torch.load(model_path + '\\' + 'model.pkl'))  # 加载模型
    model.eval()  # 设置为评估模式，关闭训练时用于优化的一些功能
    model = model.to(DEVICE)  # 模型加载到gpu中
    correct = 0
    total = 0
    create_predicted_dir(predicated_image_path)  # 创建保存预测图片的文件夹
    _, files_prefix_num = get_files_num(test_path)
    with torch.no_grad():
        DESC = "PREDICT"  # tqdm进度条左侧内容
        UNIT = "Batches"  # tqdm进度条中处理量单位it改为Batches
        COLOUR = "blue"  # tqdm进度条的颜色
        loop = tqdm(iterable=test_loader, desc=DESC, unit=UNIT, colour=COLOUR)
        test_loss_value = 0
        count = 0
        for images, labels in loop:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            outputs = model(images)
            test_loss = LOSS_FUNC(outputs, labels)
            test_loss_value += test_loss.item()  # 记录测试loss
            outputs = outputs.to(DEVICE)
            labels = labels.to(DEVICE)
            _, predicted = torch.max(outputs.data, 1)  # 取行最大值
            show_predicted(predicted, labels, files_prefix_num, total, test_path)  # 将预测图片分配到预测类中
            total += labels.size(0)
            count += 1  # 记录轮次
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    test_loss_value /= count
    model_number = get_model_last_file_name(model_and_log_path).split(' ')[1]  # 获取最新文件名的number
    model_parameter_add_acc(model_number, epoch, test_loss_value, accuracy)
    print('Accuracy of the network on the test_dataset images: %.2f%%' % accuracy)
    write_log_model(model_path, accuracy)


# 预测测试集
# if __name__ == '__main__':
#     predict()
