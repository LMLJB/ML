import csv
import os.path

import numpy as np
import pandas as pd

columns_name = ['model_number', 'model_name', 'learning_rate', 'epoch', 'batch_size', 'loss', 'accuracy', 'num_workers', 'loss_function']
default_path = r'C:\ML\model_parameter.csv'


# 模型数据保存
def model_parameter_save(data, file_path=default_path):
    if not os.path.exists(file_path):
        with open(file_path, 'w', newline='', encoding='utf-8-sig') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(columns_name)

    with open(file_path, 'a', newline='', encoding='utf-8-sig') as csv_file:
        writer = csv.writer(csv_file)
        save_data = []
        keys = data.keys()
        for key in columns_name:
            # print("key: ", key)
            if keys.__contains__(key):
                save_data.append(data[key])
            else:
                save_data.append('0')
        writer.writerow(save_data)


def model_parameter_add_acc(model_number, epoch, acc, file_path=default_path):
    history_model = pd.read_csv(file_path)
    history_model = np.array(history_model)
    print(history_model)
    for i in range(history_model.shape[0]):
        print(history_model.shape[0])
        if model_number == history_model[i][0] and\
                epoch == history_model[i][3]:
            history_model[i][6] = str(acc)
            history_model = pd.DataFrame(history_model, columns=columns_name)
            history_model.to_csv(file_path, index=None)
            print('成功添加，行数为：', i + 1)
            break


# model_parameter_add_acc(3, 2, 70)
