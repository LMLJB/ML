import torch
from torch import nn
import torch.nn.functional as F


# 用class类来建立CNN模型
class Net(nn.Module):  # 我们建立的Net继承nn.Module这个模块

    def __init__(self):
        super(Net, self).__init__()  # 第二、三行都是python类继承的基本操作
        self.conv1 = nn.Conv2d(3, 6, 5)  # 添加第一个卷积层,调用了nn里面的Conv2d（）
        self.pool = nn.MaxPool2d(2, 2)  # 最大池化层
        self.conv2 = nn.Conv2d(6, 16, 5)  # 同样是卷积层
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # 接着三个全连接层
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):  # 这里定义前向传播的方法，为什么没有定义反向传播的方法呢？这其实就涉及到torch.autograd模块了，
        # 但说实话这部分网络定义的部分还没有用到autograd的知识，所以后面遇到了再讲
        x = self.pool(torch.nn.functional.relu(self.conv1(x)))  # F是torch.nn.functional的别名，这里调用了relu函数 F.relu()
        x = self.pool(torch.nn.functional.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)  # .view( )是一个tensor的方法，使得tensor改变size但是元素的总数是不变的。
        #  第一个参数-1是说这个参数由另一个参数确定， 比如矩阵在元素总数一定的情况下，确定列数就能确定行数。
        #  那么为什么这里只关心列数不关心行数呢，因为马上就要进入全连接层了，而全连接层说白了就是矩阵乘法，
        #  你会发现第一个全连接层的首参数是16*5*5，所以要保证能够相乘，在矩阵乘法之前就要把x调到正确的size
        # 更多的Tensor方法参考Tensor: http://pytorch.org/docs/0.3.0/tensors.html
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        x = self.fc3(x)

        return x


# ResNet模型
class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=[1, 1], padding=1) -> None:
        super(BasicBlock, self).__init__()
        # 残差部分
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride[0], padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),  # 替代原来变量
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride[1], padding=padding, bias=False),
            nn.BatchNorm2d(out_channels)
        )

        # shortcut 部分
        # 由于存在维度不一致的情况 所以分情况
        self.shortcut = nn.Sequential()
        if stride[0] != 1 or in_channels != out_channels:  # TODO stride != 1条件是否有用？？？
            self.shortcut = nn.Sequential(
                # 卷积核为1 进行升降维
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride[0], bias=False),  #
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.layer(x)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


# 采用bn的网络中，卷积层的输出并不加偏置
class ResNet18(nn.Module):
    def __init__(self, BasicBlock, num_classes=10) -> None:
        super(ResNet18, self).__init__()
        self.in_channels = 64
        # 第一层作为单独
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        # conv2_x
        self.conv2 = self._make_layer(BasicBlock, 64, [[1, 1], [1, 1]])
        # self.conv2_2 = self._make_layer(BasicBlock,64,[1,1])

        # conv3_x
        self.conv3 = self._make_layer(BasicBlock, 128, [[2, 1], [1, 1]])
        # self.conv3_2 = self._make_layer(BasicBlock,128,[1,1])

        # conv4_x
        self.conv4 = self._make_layer(BasicBlock, 256, [[2, 1], [1, 1]])
        # self.conv4_2 = self._make_layer(BasicBlock,256,[1,1])

        # conv5_x
        self.conv5 = self._make_layer(BasicBlock, 512, [[2, 1], [1, 1]])
        # self.conv5_2 = self._make_layer(BasicBlock,512,[1,1])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    # 这个函数用来重复同一个残差块
    def _make_layer(self, block, out_channels, strides):
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)

        # out = F.avg_pool2d(out,7)
        out = self.avgpool(out)
        out = out.reshape(x.shape[0], -1)
        out = self.fc(out)
        return out

