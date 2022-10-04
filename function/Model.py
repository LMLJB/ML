# import torch
from torch import nn
import torch.nn.functional as F


# ResNet18的残差模块
class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, strides=[1, 1], padding=1):
        super(BasicBlock, self).__init__()
        # 残差部分
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=strides[0], padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),  # 替代原来变量
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=strides[1], padding=padding, bias=False),
            nn.BatchNorm2d(out_channels)
        )

        # shortcut部分  （由于存在维度或通道不一致的情况 所以分情况）
        self.shortcut = nn.Sequential()
        if strides[0] != 1 or in_channels != out_channels:  # 若一开始就下采样（即stride[0]!=1）或前后通道数不同，则将输入的x改为输出后的shape
            self.shortcut = nn.Sequential(
                # 卷积核为1 进行升降维
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=strides[0], bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.layer(x)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


# ResNet50的残差模块
class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, strides=[1, 1, 1]):
        super(Bottleneck, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=strides[0], padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=strides[1], padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * 4, kernel_size=1, stride=strides[2], padding=0, bias=False),
            nn.BatchNorm2d(out_channels * 4)  # 结合ResNet50的图，可知输出通道数为输入通道数的1/4
        )

        # shortcut部分  （由于存在维度或通道不一致的情况 所以分情况）
        out_channels = out_channels * 4
        self.shortcut = nn.Sequential()
        if strides[0] != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=strides[1], padding=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.layer(x)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


# ResNet18
class ResNet18(nn.Module):
    def __init__(self, basic_block, num_classes=45):
        super(ResNet18, self).__init__()
        self.in_channels = 64
        # conv1_x （第一层作为单独）
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        # conv2_x
        self.conv2 = self._make_layer(basic_block, 64, [[1, 1], [1, 1]])

        # conv3_x
        self.conv3 = self._make_layer(basic_block, 128, [[2, 1], [1, 1]])

        # conv4_x
        self.conv4 = self._make_layer(basic_block, 256, [[2, 1], [1, 1]])

        # conv5_x
        self.conv5 = self._make_layer(basic_block, 512, [[2, 1], [1, 1]])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    # 这个函数用来重复同一个残差块
    def _make_layer(self, block, out_channels, strides_list):
        layers = []
        for strides in strides_list:
            layers.append(block(self.in_channels, out_channels, strides))
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
        out = out.reshape(x.shape[0], -1)  # torch.flatten(out, 1)
        out = self.fc(out)
        return out


# ResNet50
class ResNet50(nn.Module):
    def __init__(self, bottleneck, num_classes=45):
        super(ResNet50, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        # conv2_x
        self.conv2 = self._make_layer(bottleneck, 64, [[1, 1, 1], [1, 1, 1]])

        # conv3_x
        self.conv3 = self._make_layer(bottleneck, 128, [[1, 2, 1], [1, 1, 1]])

        # conv4_x
        self.conv4 = self._make_layer(bottleneck, 256, [[1, 2, 1], [1, 1, 1]])

        # conv5_x
        self.conv5 = self._make_layer(bottleneck, 512, [[1, 2, 1], [1, 1, 1]])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2048, num_classes)

    # 这个函数用来重复同一个残差块
    def _make_layer(self, block, out_channels, strides_list):
        layers = []
        for strides in strides_list:
            layers.append(block(self.in_channels, out_channels, strides))
            self.in_channels = out_channels * 4
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)

        out = self.avgpool(out)
        out = out.reshape(x.shape[0], -1)  # torch.flatten(out, 1)
        out = self.fc(out)
        return out


def resnet18():
    model = ResNet18(BasicBlock, 45)
    return model, 'ResNet18'


def resnet50():
    model = ResNet50(Bottleneck, 45)
    return model, 'ResNet50'
