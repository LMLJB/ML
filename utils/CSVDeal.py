import csv
import os.path
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
            print("key: ", key)
            if keys.__contains__(key):
                save_data.append(data[key])
            else:
                save_data.append('0')
        writer.writerow(save_data)


def model_parameter_add_acc(model_number, acc, file_path=default_path):
    history_model = pd.csv(file_path)
    history_model_number = history_model['model_number']
    for i in range(len(history_model_number)):
        if model_number == history_model[i]:
            history_model['accuracy'][i] = acc
            history_model.to_csv(file_path)
            print('成功添加，行数为：', i + 1)
            break



# path = r'C:\ML\model and log\model_parameter.csv'
# model_parameter_save(path, 0)
