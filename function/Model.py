import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

LR = 0.001  # 学习率
EPOCH = 1  # 训练次数
BATCH_SIZE = 10
DOWNLOAD_MNIST = True  # 表示还没有下载数据集，如果数据集下载好了就写False

# 预测函数
def predict():
    # 数据处理
    test_dir = r'D:\Homework\project4\test'
    transform_train_test = torchvision.transforms.Compose(
        [transforms.Resize((512, 512)), torchvision.transforms.ToTensor(),
         torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    test_set = datasets.ImageFolder(test_dir, transform=transform_train_test)
    test_data = DataLoader(test_set, batch_size=10, shuffle=False, num_workers=0)
    testloader = torch.utils.data.DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    # 获取cnn网络
    net = torchvision.models.resnet18()
    # 加载模型
    net.load_state_dict(torch.load('test_cifar_gpu.pkl'))
    # 设置为推理模式
    net.eval()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # 模型加载到gpu中
    net = net.to(device)
    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            # 数据加载到gpu中
            images = images.to(device)
            labels = labels.to(device)
            # calculate outputs by running images through the network
            outputs = net(images)
            # 数据加载回cpu
            outputs = outputs.to('cpu')
            labels = labels.to('cpu')
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('Accuracy of the network on the 10000 test images: %d %%' % (
            100 * correct / total))


def train_model():
    # 读取数据
    train_dir = r'D:\Homework\project4\train'
    transform_train_test = torchvision.transforms.Compose(
        [transforms.Resize((512, 512)), torchvision.transforms.ToTensor(),
         torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    train_set = datasets.ImageFolder(train_dir, transform=transform_train_test)
    train_data = DataLoader(train_set, batch_size=10, shuffle=True, num_workers=0)  # shuffle是否打乱数据
    net = torchvision.models.resnet18()  # 设置模型
    optimizer = torch.optim.Adam(net.parameters(), lr=LR)  # 优化器
    loss_func = nn.CrossEntropyLoss()  # 定义损失函数
    # 开始训练
    for epoch in range(EPOCH):
        running_loss = 0.0
        for step, (inputs, labels) in enumerate(train_data):  # 分配batch data
            # 未加载到GPU中
            output = net(inputs)  # 将数据放入cnn中计算输出
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
    torch.save(net.state_dict(), 'test_cifar_gpu.pkl')


# 无用
def dataloader():
    transforms = torchvision.transforms.Compose(
        [torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # 下载CIFAR10数据集
    train_data = torchvision.datasets.CIFAR10(
        root='./data/',  # 保存或提取的位置  会放在当前文件夹中
        train=True,  # true说明是用于训练的数据，false说明是用于测试的数据
        transform=transforms,  # 转换PIL.Image or numpy.ndarray
        download=DOWNLOAD_MNIST,  # 已经下载了就不需要下载了
    )

    test_data = torchvision.datasets.CIFAR10(
        root='./data',
        train=False,
        download=True,
        transform=transforms,
    )

    return train_data, test_data


# train_data, test_data = dataloader()
train_model()
predict()


