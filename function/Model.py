import torch
from torch import nn


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
