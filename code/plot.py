# -*- coding:utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import base
import func

######################################## init ########################################
plt.style.use('ggplot')  # Good looking plots
np.set_printoptions(precision=4, threshold=10000, linewidth=100, edgeitems=999, suppress=True)

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 100)
pd.set_option('expand_frame_repr', False)
pd.set_option('precision', 6)


def plot(list, savename):
    '''
    绘图
    :param list: 所有文件的数据
    :param savename: 保存的文件名
    :return:
    '''
    plt.rcParams['figure.figsize'] = [25, len(list) * 5]
    j = 1
    for i in list:
        X, y, file = i
        plt.subplot(len(list), 1, j)
        plt.plot(X, y, '-')
        plt.minorticks_on()
        plt.grid(True, which='both')
        plt.title(file)
        j += 1
    plt.savefig(base.plot_dir + savename)


original_list = func.load_files(base.data_dir)
filter_zero_list = func.load_files_filter_zero(base.data_dir)
mean_per_second_list = func.load_files_mean_per_second(base.data_dir)
mean_per_second_filter_zero_list = func.load_files_mean_per_second_filter_zero(base.data_dir)
# plot(original_list, 'original_plot.pdf')
# plot(filter_zero_list, 'filter_zero_plot.pdf')
# plot(mean_per_second_list, 'mean_per_second_plot.pdf')
plot(mean_per_second_filter_zero_list, 'mean_per_second_filter_zero_plot.pdf')
