# 预测函数
import torch
# import torchvision
import os
from tqdm import tqdm
from function.Show import show_train_loss
from torch.utils.data import DataLoader
from torchvision import datasets
from pretreatment import test_transform
from function.Show import show_predicted, create_save_images_dir, get_files_num, save_data
from Model import resnet18, resnet50  # resnet18/resnet50

# 超参数
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # 运行模型选择的设备
BATCH_SIZE = 50
NUM_WORKERS = 0
MODEL = resnet18()
dataset_dir = r'C:\ML'  # 数据集路径
test_path = os.path.join(dataset_dir, 'test')  # 测试集路径
error_image_path = r'C:\ML\error_result_image'
test_history_path = r'C:\ML\history\test.txt'
test_dataset = datasets.ImageFolder(test_path, transform=test_transform)  # 载入训练集
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE,             # 测试集的数据加载器
                         shuffle=False, num_workers=NUM_WORKERS)


def predict():
    model = MODEL
    model.load_state_dict(torch.load('test_gpu.pkl'))  # 加载模型
    model.eval()  # 设置为评估模式，关闭训练时用于优化的一些功能
    model = model.to(DEVICE)  # 模型加载到gpu中
    correct = 0
    total = 0
    create_save_images_dir(error_image_path)  # 创建保存预测错误图片的文件夹
    _, files_prefix_num = get_files_num(test_path)
    with torch.no_grad():
        loop = tqdm(test_loader)
        for images, labels in loop:
            # 数据加载到gpu中
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            outputs = model(images)
            outputs = outputs.to(DEVICE)
            labels = labels.to(DEVICE)
            _, predicted = torch.max(outputs.data, 1)  # 取行最大值
            total += labels.size(0)
            show_predicted(predicted, labels, images, files_prefix_num, total, test_path)
            correct += (predicted == labels).sum().item()
    print('Accuracy of the network on the 10000 test images: %d %%' % (
            100 * correct / total))
    history_data = {"batch_size": BATCH_SIZE, "test_accuracy": 100 * correct / total}
    save_data(test_history_path, history_data)  # 将训练数据保存到文件中


# 预测
predict()
