import os
import shutil

from torchvision.transforms import ToPILImage

project_path = r'C:\ML'
predicted_image_path = project_path + '\\predicted_image'  # 预测图片放置位置
loss_image_path = project_path + r'\loss_image'

show_image = ToPILImage()  # 把Tensor转变为Image
labels_name = ['停机坪', '停车场', '公园', '公路', '冰岛', '商业区', '墓地', '太阳能发电厂', '居民区', '山地', '岛屿',
               '工厂', '教堂', '旱地', '机场跑道', '林地', '桥梁', '梯田', '棒球场', '水田', '沙漠', '河流', '油田',
               '油罐区', '海滩', '温室', '港口', '游泳池', '湖泊', '火车站', '直升机场', '石质地', '矿区', '稀疏灌木地',
               '立交桥', '篮球场', '网球场', '草地', '裸地', '足球场', '路边停车区', '转盘', '铁路', '风力发电站',
               '高尔夫球场']


# 创建文件夹并返回路径，每次删除原文件夹
def delete_and_create_dir(path):
    if os.path.isdir(path):
        shutil.rmtree(path)
    os.mkdir(path)
    return str(path)


# 只需要创建
def create_dir(path):
    if not os.path.isdir(path):
        os.mkdir(path)
    return str(path)


# 创建预测图片放置的文件夹
def create_predicted_dir(path):
    delete_and_create_dir(path)
    for name in labels_name:
        create_dir(path + '\\' + name)


# 创建log.txt文件并记录训练模型的超参数
def create_log_model(model_path, dic):
    file = open(model_path + r'\log.txt', 'w')
    for x, y in dic.items():
        file.write(str(x) + " = " + str(y) + "\n")
    file.close()


# 往log.txt文件写入预测准确度
def write_log_model(model_path, accuracy):
    file = open(model_path + r'\log.txt', 'a')
    file.write("accuracy = %.2f%%" % accuracy)
    file.close()


# 获取相应序号文件名
def get_file_name(path, index):
    files_name = os.listdir(path)
    return files_name[index]


def get_model_last_file_name(path):
    num_files = os.listdir(path)
    return num_files[len(num_files) - 1]


# 将预测图片分配到预测类中
def show_predicted(predicted, labels, prefix_num, base, test_path):
    length = len(predicted)
    index = 0
    for num in prefix_num:
        if num > base:
            break
        index += 1

    for i in range(length):
        image_number = base + i
        if image_number >= prefix_num[index]:
            num = image_number - prefix_num[index]  # 图片编号
            index += 1
        else:
            if index - 1 < 0:
                num = image_number
            else:
                num = image_number - prefix_num[index - 1]
        path = test_path + '\\' + labels_name[labels[i]]
        file_name = get_file_name(path, num)
        shutil.copy(path + '\\' + file_name, predicted_image_path + '\\' + labels_name[predicted[i]] +
                    '\\' + file_name.split('.')[0] + "-" + labels_name[labels[i]] + '.jpg')


# 获取路径下每个文件夹中文件数量
def get_files_num(path):
    files_num = []
    for name in labels_name:
        files_num.append(len(os.listdir(path + "\\" + name)))
    prefix_num = [0]  # 前缀和
    prefix_num[0] = files_num[0]
    for i in range(1, len(files_num)):
        prefix_num.append(prefix_num[i - 1] + files_num[i])
    return files_num, prefix_num


# 向文件中追加数据
def save_data(path, data):
    with open(path, 'a') as f:
        f.write(str(data) + '\n')
        f.write("\n")
