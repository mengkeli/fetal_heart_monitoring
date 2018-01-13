# -*- coding:utf-8 -*-
import numpy as np
import pandas as pd
import random

def filter_zero(data_file, out_file, zero_rate, length):
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

    with open(data_file, 'r') as f:
        line = f.readline()
        writer.write(line)

        total_row = 0
        while True:
            line = f.readline()
            # print line
            if not line:
                break
            values = line.split(',')
            zero_count = 0
            for i in np.arange(1, len(values) - 1):
                if int(values[i]) != 0:
                    if cons_zero_count >= length:
                        cons_zero_count = 0
                        zero_count = 0
                        continue
                    else:
                        cons_zero_count = 0
                else:
                    zero_count += 1
            zero_writer.write('%s,%d\n' % (values[0], zero_count))
            # print 'id: %s, zero count: %d' % (values[0], zero_count)

            # 0值缺失严重的, 忽略
            if zero_count >= zero_rate * 2400:
                continue

            total_row += 1
            # 找出0的个数小于300的
            writer.write(values[0])
            max_len = len(values)
            for i in np.arange(1, max_len):
                if int(values[i]) == 0:
                    # 用相邻的均值替代0值
                    left_index = i
                    right_index = i
                    while left_index >= 0 and int(values[left_index]) == 0:
                        left_index -= 1
                    while right_index < max_len - 1 and int(values[right_index]) == 0:
                        right_index += 1

                    val = (int(values[left_index]) + int(values[right_index])) / 2
                else:
                    val = values[i]
                writer.write(',' + str(val))
            writer.write('\n')

        writer.close()
        print 'filter at zero_rate = %d. total_row: %d' % zero_rate, total_row

def load_data(path='./data/fetal.npz'):
    """Loads the fetal dataset.

    # Arguments
        path: path where to cache the dataset locally
            (relative to ~/.keras/datasets).

    # Returns
        Tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`.
    """
    f = np.load(path)
    x, y = f['dataset'], f['label']
    # shuffle the dataset
#    permutation = np.random.permutation(y.shape[0])
#    shuffled_dataset = x[permutation, :, :]
#    shuffled_label = y[permutation]

    x_train, y_train = x[ :4000], y[ :4000]
    x_test, y_test = x[4000 :5000], y[4000:5000]
    f.close()
    print('Successfully load data...')
    return (x_train, y_train), (x_test, y_test)

