# -*- coding:utf-8 -*-
import numpy as np
import pandas as pd
import random

def filter_zero(data_file, out_file, zero_rate = 0.3, length = 100):
    '''
    对于0值的点, 使用左右的平均值做为测量值
    :param data_file:
    :param out_file:
    :zero_rate: 0.3
    :param length: the length of consistent zero, eg:50, 100
    :return:
    '''
    writer = open(out_file, 'w+')
    zero_writer = open(out_file + '.zero_count', 'w+')
    print('zero_rate = %f' % zero_rate)
    print('zero_length = %d' % length)
    with open(data_file, 'r') as f:
        line = f.readline()
        writer.write(line)

        total_row = 0
        bad_row = 0
        zero_row = 0
        origin_row = 0
        while True:
            line = f.readline().rstrip()
            # print line
            if not line:
                break
            origin_row += 1
            values = line.split(',')
            zero_count = 0
            cons_zero_count = 0
            bad_row_flag = False
            for i in np.arange(1, len(values) - 1):
                if int(values[i]) == 0:
                    zero_count += 1
                    cons_zero_count += 1
                else:
                    if cons_zero_count > length:
                        bad_row_flag = True
                    else:
                        cons_zero_count = 0
            zero_writer.write('%s,%d\n' % (values[0], zero_count))
            # 连续0值超过length的, 剔除
            if bad_row_flag == True:
                bad_row += 1
                continue;

            # 0值缺失严重的, 剔除
            if zero_count >= zero_rate * 2400:
                zero_row += 1
                continue
  
            total_row += 1
            writer.write(values[0])
            max_len = len(values)
            for i in np.arange(1, max_len):
                if int(values[i]) == 0:
                    # 用相邻的均值替代0值
                    left_index = i
                    right_index = i
                    while left_index > 1 and int(values[left_index]) == 0:
                        left_index -= 1
                    while right_index < max_len - 1 and int(values[right_index]) == 0:
                        right_index += 1

                    val = (int(values[left_index]) + int(values[right_index])) / 2
                else:
                    val = values[i]
                writer.write(',' + str(val))
            writer.write('\n')

        writer.close()
        print('origin_row = %d.' % origin_row)
        print('total_row = %d.' % total_row)
        print('bad_row = %d.' % bad_row)
        print('zero_row = %d.' % zero_row)

def join_data_label(data_file='../data/data_zero_filter_03_50.csv', label_file='../data/info.csv'):
    data = pd.read_csv(data_file)

    df = pd.read_csv(label_file)
    label = df.loc[:, ['id', 'nst_result']]
    data_label = pd.merge(data, label, how='left', left_index=True, left_on='id', right_on='id')

    # 把'异常型' 归入 '可疑型'
    data_label.loc[data_label['nst_result'] == 3, 'nst_result'] = 2
    # 剔除'无法判读'型
    data_label.loc[data_label['nst_result'] == 4, 'nst_result'] = np.nan
    data_label.dropna(inplace=True)
    data_label.drop('id', axis=1, inplace=True)
    # label = data_label['nst_result']
    # data_label.drop(['nst_result'], axis=1, inplace=True)

    np.save('../data/fetal.npy', data_label)#, dataset=data_label, label=label)
    return

def load_data(path='../data/fetal.npz'):
    """Loads the fetal dataset.

    # Arguments
        path: path where to cache the dataset locally
            (relative to ~/.keras/datasets).

    # Returns
        Tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`.
    """
    f = np.load(path)
    # shuffle the dataset
    np.random.shuffle(f)
    x, y = f[:, 0:-1], f[:, -1]
    trainset_size = np.floor(f.shape[0] * 0.7)
    # testset_size = f.shape[0] - trainset_size
    x_train, y_train = x[ :trainset_size], y[ :trainset_size]

    x_test, y_test = x[trainset_size: ], y[trainset_size:]
    f.close()
    print('Successfully load data...')
    return (x_train, y_train), (x_test, y_test)

if __name__ == '__main__':
    join_data_label('../data/data_zero_filter_03_50.csv', '../data/info.csv')
    #filter_zero('../data/data_gzip.csv', '../data/data_zero_filter_03_50.csv', 0.3, 50)
