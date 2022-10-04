import os
import shutil
from utils.Show import labels_name
from utils.Path import project_path

# project_path = r'D:\ML'
predicted_image_path = project_path + '/predicted_image'  # 预测图片放置位置
loss_image_path = project_path + '/loss_image'


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
        create_dir(path + '/' + name)


# 创建log.txt文件并记录训练模型的超参数
def create_log_model(model_path, dic):
    file = open(model_path + '/log.txt', 'w')
    for x, y in dic.items():
        file.write(str(x) + " = " + str(y) + "\n")
    file.close()


# 往log.txt文件写入预测准确度
def write_log_model(model_path, accuracy):
    file = open(model_path + '/log.txt', 'a')
    file.write("accuracy = %.2f%%" % accuracy)
    file.close()


# 获取相应序号文件名
def get_file_name(path, index):
    files_name = os.listdir(path)
    return files_name[index]


# 获取最新模型文件夹的名字
def get_model_last_file_name(path):
    num_files = os.listdir(path)
    length = len(num_files)
    return "model " + str(length - 1)


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
        path = test_path + '/' + labels_name[labels[i]]
        file_name = get_file_name(path, num)
        shutil.copy(path + '/' + file_name, predicted_image_path + '/' + labels_name[predicted[i]] +
                    '/' + labels_name[labels[i]] + "-" + file_name.split('.')[0] + '.jpg')


# 获取路径下每个文件夹中文件数量
def get_files_num(path):
    files_num = []
    for name in labels_name:
        files_num.append(len(os.listdir(path + "/" + name)))
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
