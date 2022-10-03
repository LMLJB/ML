import torch
from function.Test import predict


# 单纯用于封装下面两个函数
def encapsulation(model_state, path, epoch):
    torch.save(model_state, path)  # 保存模型
    predict(epoch)  # 训练完一个epoch，测试一次
