import csv
import os.path

import numpy as np
import pandas as pd

columns_name = ['model_number', 'model_name', 'learning_rate', 'batch_size', 'epoch',  'train_loss',
                'predicted_loss', 'accuracy', 'num_workers']
default_path = r'C:\ML\model_parameter.csv'  # 模型参数所存文件名


# 模型数据保存到CSV文件中
def model_parameter_save(data, file_path=default_path):
    if not os.path.exists(file_path):  # 文件还未创建
        with open(file_path, 'w', newline='', encoding='utf-8-sig') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(columns_name)

    with open(file_path, 'a', newline='', encoding='utf-8-sig') as csv_file:
        writer = csv.writer(csv_file)
        save_data = []
        keys = data.keys()
        for key in columns_name:
            if keys.__contains__(key):
                save_data.append(str(data[key]))
            else:
                save_data.append('0')  # key对应没有值，默认为0
        writer.writerow(save_data)  # 追加写入文件


# 将acc和test_loss值加入到相应数据中
def model_parameter_add_acc(model_number, epoch, test_loss, acc, file_path=default_path):
    history_model = pd.read_csv(file_path)
    history_model = np.array(history_model)
    for i in range(history_model.shape[0]):
        # model_number与epoch的数值都匹配成功，model_number和epoch一起才能决定数据所在行数
        if str(model_number) == str(history_model[i][0]) and str(epoch) == str(history_model[i][4]):
            history_model[i][7] = str(acc)
            history_model[i][6] = str(test_loss)
            history_model = pd.DataFrame(history_model, columns=columns_name)
            history_model.to_csv(file_path, index=None)
            print('成功添加，行数为：', i + 1)
            break


# def csv_to_dic_loss(file_path=default_path):
#     history_model = pd.read_csv(file_path)
#     history_model = np.array(history_model)
