# -*- coding:utf-8 -*-

import json, os, gzip
from cStringIO import StringIO

import numpy as np


def load_file(file):
    '''
    导入数据, 返回X与y
    :param file: 文件名
    :return: X, y
    '''
    y = []
    with open(file, 'r') as f:
        data = json.load(f)
        for item in data:
            y.append(item['y'])
    X = np.arange(0, len(y), 1)
    return X, y


def load_files(dir):
    '''
    导入所有文件, 并返回list
    :param dir: 目录
    :return: list
    '''
    files = os.listdir(dir)
    list = []
    for file in files:
        X, y = load_file(dir + file)
        item = X, y, file
        list.append(item)
    return list


def load_file_filter_zero(file):
    '''
    导入数据, 将y里面的所有0用平均值替代之后返回X与y
    :param file: 文件名
    :return: X, y
    '''
    X, y = load_file(file)
    zero_num = len([i for i in y if i == 0])
    mean = np.sum(y) / (len(y) - zero_num)
    # 返回新的用平均值替代0之后的y
    list = []
    for i in y:
        list.append(mean if i == 0 else i)
    return X, list


def load_files_filter_zero(dir):
    '''
    导入所有文件, 并返回list, 结果为用平均值替代0之后的结果
    :param dir: 目录
    :return: list
    '''
    files = os.listdir(dir)
    list = []
    for file in files:
        X, y = load_file_filter_zero(dir + file)
        item = X, y, file
        list.append(item)
    return list


def load_file_mean_per_second(file):
    '''
    导入数据, 由于是每秒测到了2个值, 因此将原始数据从0开始连续的2个数值
    的平均值作为该秒的测量值
    :param file: 文件名
    :return: X, y
    '''
    ox, y = load_file(file)
    list = []
    for i in np.arange(0, len(y), 2):
        mean = (y[i] + y[i + 1]) / 2
        list.append(mean)

    X = np.arange(0, len(list), 1)
    return X, list


def load_files_mean_per_second(dir):
    '''
    导入所有文件, 并返回list, 结果为每秒的平均值
    :param dir: 目录
    :return: list
    '''
    files = os.listdir(dir)
    list = []
    for file in files:
        X, y = load_file_mean_per_second(dir + file)
        item = X, y, file
        list.append(item)
    return list


def load_file_mean_per_second_filter_zero(file):
    '''
    导入数据, 由于是每秒测到了2个值, 因此将原始数据从0开始连续的2个数值
    的平均值作为该秒的测量值, 最后为0的点用平均值替代
    :param file: 文件名
    :return: X, y
    '''
    X, y = load_file_mean_per_second(file)
    zero_num = len([i for i in y if i == 0])
    mean = np.sum(y) / (len(y) - zero_num)
    # 返回新的用平均值替代0之后的y
    list = []
    for i in y:
        list.append(mean if i == 0 else i)
    return X, list


def load_files_mean_per_second_filter_zero(dir):
    '''
    导入所有文件, 导入数据, 由于是每秒测到了2个值, 因此将原始数据从0开始连续的2个数值
    的平均值作为该秒的测量值, 最后为0的点用平均值替代
    :param dir: 目录
    :return: list
    '''
    files = os.listdir(dir)
    list = []
    for file in files:
        X, y = load_file_mean_per_second_filter_zero(dir + file)
        item = X, y, file
        list.append(item)
    return list


def get_attributes(file):
    '''
    获得一个文件的短平滑、长平滑、加速、减速等指标
    :param file: 文件
    :return:
    '''
    X, y = load_file(file)
    # TODO


def pca(dataMat, topNfeat=999999):
    '''
    PAC降纬
    :param dataMat:
    :param topNfeat:
    :return:
    '''
    meanVals = np.mean(dataMat, axis=0)
    DataAdjust = dataMat - meanVals  # 减去平均值
    covMat = np.cov(DataAdjust, rowvar=0)
    eigVals, eigVects = np.linalg.eig(np.mat(covMat))  # 计算特征值和特征向量
    # print eigVals
    eigValInd = np.argsort(eigVals)
    eigValInd = eigValInd[:-(topNfeat + 1):-1]  # 保留最大的前K个特征值
    redEigVects = eigVects[:, eigValInd]  # 对应的特征向量
    lowDDataMat = DataAdjust * redEigVects  # 将数据转换到低维新空间
    reconMat = (lowDDataMat * redEigVects.T) + meanVals  # 重构数据，用于调试
    return lowDDataMat, reconMat


def gzip_uncompress(c_data):
    buf = StringIO(c_data)
    f = gzip.GzipFile(mode='rb', fileobj=buf)
    try:
        r_data = f.read()
    finally:
        f.close()
    return r_data


file = '../data/76416.txt'
# load_file_filter_zero(file)
# load_file(file)
# dir = '../data/'
# load_files_filter_zero(dir)
# load_files(dir)
# load_file_mean_per_second(file)
