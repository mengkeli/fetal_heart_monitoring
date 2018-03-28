# -*- coding:utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
from scipy.fftpack import fft,ifft

data_path = '../data/'

raw_data_file = data_path + 'data_gzip.csv'
zero_filter_file = data_path + 'data_zero_filter_02_50.csv'
label_file = data_path + 'info.csv'

series_file = data_path + 'fetal_series_02_50.npy'
image_file = data_path + 'fetal_image.npy'
fft_file = data_path + 'fetal_fft.npy'

def filter_zero(raw_data_file, zero_filter_file, zero_rate = 0.3, length = 100):
    '''
    对于0值的点, 使用左右的平均值做为测量值
    :param raw_data_file:
    :param zero_filter_file:
    :zero_rate: 0.3
    :param length: the length of consistent zero, eg:50, 100
    :return:
    '''
    writer = open(zero_filter_file, 'w+')
    zero_writer = open(zero_filter_file + '.zero_count', 'w+')
    print('zero_rate = %f' % zero_rate)
    print('zero_length = %d' % length)
    with open(raw_data_file, 'r') as f:
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

                    val = np.floor_divide(int(values[left_index]) + int(values[right_index]), 2)
                else:
                    val = values[i]
                writer.write(',' + str(val))
            writer.write('\n')

        writer.close()
        print('origin_row = %d.' % origin_row)
        print('total_row = %d.' % total_row)
        print('bad_row = %d.' % bad_row)
        print('zero_row = %d.' % zero_row)

def join_data_label(zero_filter_file, label_file):
    data = pd.read_csv(zero_filter_file)
    df = pd.read_csv(label_file)

    label = df.loc[:, ['id', 'nst_result']]
    data_label = pd.merge(data, label, how='left', left_index=True, left_on='id', right_on='id')

    # 把'可疑型' 归入 '异常型'
    data_label.loc[data_label['nst_result'] == 3, 'nst_result'] = 2

    # 剔除'无法判读'型
    data_label.loc[data_label['nst_result'] == 4, 'nst_result'] = np.nan
    data_label.loc[data_label['nst_result'] == 5, 'nst_result'] = np.nan
    data_label.loc[data_label['nst_result'] == 0, 'nst_result'] = np.nan
    data_label.dropna(inplace=True)

    # 将1，2类，转化为0、1类
    data_label.loc[data_label['nst_result'] == 1, 'nst_result'] = 0
    data_label.loc[data_label['nst_result'] == 2, 'nst_result'] = 1
    data_label.drop('id', axis=1, inplace=True)

    # 规范数据，合理数据范围在80~199(120个值)
    label = data_label.loc[:, ['nst_result']]
    data = data_label.drop('nst_result', axis=1)
    data[data < 80] = 80
    data[data >= 200] = 199
    data['nst_result'] = label
    data = data.astype('uint8')

    np.save(series_file, data.values[:,:])
    return

def generate_imgdata(series_file):
    '''
    生成与时间序列对应的图像数据 1 * 2402 -> 120 * 2402
    :param path:
    :return:
    '''
    f = np.load(series_file)
    x, y = f[:, 0:-1], f[:, -1:]
    num_data = x.shape[0]
    cols = x.shape[1]
    rows = 200 - 80  # 图像的y轴刻度
    data_mat = np.zeros((num_data, rows * cols), dtype=np.uint8)

    for i in range(num_data):
        if (i % 1000 == 0):
            print('i = %s' % i)
        image_mat = np.zeros((rows, cols), dtype=np.uint8)
        for j in range(cols):
            image_mat[x[i][j]-60][j] = 1
        data_mat[i][:] = np.reshape(image_mat, (1, rows * cols))
    data_label_mat = np.hstack([data_mat, y])
   # data_mat_df = pd.DataFrame(data_label_mat)

    np.save(image_file, data_label_mat)
    return

def transfer_fft(series_file):
    '''
    对波形图像进行傅里叶变换
    :param path: 
    :return: 
    '''
    f = np.load(series_file)
    # 采样点选择100个，因为设置的信号频率分量最高为600赫兹，根据采样定理知采样频率要大于信号频率2倍（一秒内有2个采样点）
    x = np.linspace(0, 1, 100)
    y = f[:, 0:-1]
    # 快速傅里叶变换
    yy = fft(y)
    yreal = yy.real  # 获取实数部分
    yimag = yy.imag  # 获取虚数部分

    yf = abs(fft(y))  # 取绝对值
    yf1 = abs(fft(y)) / len(x)  # 归一化处理
    yf2 = yf1[range(int(len(x) / 2))]  # 由于对称性，只取一半区间

    xf = np.arange(len(y))  # 频率
    xf1 = xf
    xf2 = xf[range(int(len(x) / 2))]  # 取一半区间

    plt.subplot(221)
    plt.plot(x[0:50], y[0:50])
    plt.title('Original wave')

    plt.subplot(222)
    plt.plot(xf, yf, 'r')
    plt.title('FFT of Mixed wave(two sides frequency range)', fontsize=7, color='#7A378B')  # 注意这里的颜色可以查询颜色代码表

    plt.subplot(223)
    plt.plot(xf1, yf1, 'g')
    plt.title('FFT of Mixed wave(normalization)', fontsize=9, color='r')

    plt.subplot(224)
    plt.plot(xf2, yf2, 'b')
    plt.title('FFT of Mixed wave)', fontsize=10, color='#F08080')

    plt.show()

def load_data(file = series_file):
    """Loads the fetal dataset.

    # Arguments
        path: path where to cache the dataset locally
            (relative to ~/.keras/datasets).

    # Returns
        Tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`.
    """
    f = np.load(file)
    # shuffle the dataset
    np.random.shuffle(f)

    x, y = f[:, 0:-1], f[:, -1]
    trainset_size = int(np.floor(f.shape[0] * 0.8)) # 0.7 -> 15533, 6658; 0.8 ->
    x_train, y_train = x[ :trainset_size], y[ :trainset_size]
    x_test, y_test = x[trainset_size:], y[trainset_size:]
    print('Successfully load data...')
    return (x_train, y_train), (x_test, y_test)

if __name__ == '__main__':
    filter_zero(raw_data_file, zero_filter_file, 0.2, 50)
    # join_data_label(zero_filter_file, label_file)
    # generate_imgdata(series_file)

