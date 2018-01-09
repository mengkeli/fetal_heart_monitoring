# -*- coding:utf-8 -*-

"""Functions for reading fetal_heart_rate_monitoring data."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pandas as pd

from sklearn import cross_validation
from tensorflow.contrib.learn.python.learn.datasets import base
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import random_seed

datapath = '../data/'
rawdata = pd.read_csv(datapath + 'data_gzip.csv')
info = pd.read_csv(datapath + 'info.csv')
info_features = ['jianceshichang', 'jixian', 'jixianbianyi', 'taidongjiasutime', 'taidongjiasufudu', 'taidongcishu',
                 'bianyizhouqi', 'jiasu', 'jiasuzhenfu', 'jiasutimes', 'jiansu', 'jiansuzhenfu', 'jiansuzhouqi',
                 'jiansutimes', 'wanjiansu', 'wanjiansuzhenfu', 'wanjiansuzhouqi', 'wanjiansutimes', 'jixian',
                 'jixianbianyi', 'taidongjiasutime', 'taidongjiasufudu', 'taidongcishu']
merge = pd.merge(rawdata, info, on='id')
merge.drop(info_features, axis=1, inplace=True)
merge.to_csv(datapath + 'rawdata.csv', index=False)

# 剔除0值大于50%的数据
zero_rows = []
for i in range(merge.shape[0]):
  zero_cnt = merge.iloc[i].tolist().count(0)
  if zero_cnt*1.0/2402.0 > 0.5:
    zero_rows.append(i)
merge.drop(merge.index[zero_rows], inplace=True)
merge.drop('id', axis=1, inplace=True)
merge.to_csv(datapath + 'rawdata_zero_filter.csv', index=False)

# 规范数据，合理数据范围在60~210
merge_value = merge.drop('nst_result', axis=1)
merge_value[merge_value<60] = 60
merge_value[merge_value>210] = 210

# generate labels.csv file
labels = merge['nst_result']
labels.to_csv(datapath + 'labels.csv', index=False)
#generate data.csv file
merge_value.to_csv(datapath + 'rawdata_value_filter.csv', index=False)

def extract_data(data_file):
  '''
  从data里提取数据，转换原始数据为数组[index, y, x, depth]
  :param f:
  :return:
  '''
  print('Extracting', data_file.name)
  data = pd.read_csv(data_file)
  num_data = data.shape[0]
  cols = data.shape[1]
  rows = 210-60+1  # 图像的y轴刻度
  data_mat = np.zeros((num_data, rows*cols), dtype=np.int8)
  for i in range(num_data):
    if (i%1000==0):
      print('i:%s'%i)
    image_mat = np.zeros((rows, cols), dtype=np.int8)
    for j in range(cols):
      image_mat[data.iloc[i][j]-60][j] = 1
      # print('i:%s, j:%s, data[i][j]-60:%s' % (i, j, data.iloc[i][j] - 60))
    image_mat_reshape = np.reshape(image_mat, -1)
    data_mat[i] = image_mat_reshape
  data_mat_df = pd.DataFrame(data_mat)
  data_mat_df.to_csv(datapath + 'data.csv')
  data = data_mat.reshape(num_data, rows, cols, 1)
  return data

def dense_to_one_hot(labels_dense, num_classes):
  '''
  转换数据标量标签为one-hot向量
  :param labels_dense: 
  :param num_classes: 
  :return: 
  '''
  num_labels = labels_dense.shape[0]
  index_offset = np.arange(num_labels) * num_classes
  labels_one_hot = np.zeros((num_labels, num_classes))
  labels_one_hot.flat[index_offset , labels_dense.loc[:,'2'].ravel()] = 1
  return labels_one_hot


def extract_labels(labels_file, one_hot=False, num_classes=3):
  '''
  提取标签，生成一维数组 [index]
  :param labels_file: 
  :param one_hot: 
  :param num_classes: 类别数目
  :return: labels
  '''
  print('Extracting', labels_file.name)
  labels = pd.read_csv(labels_file)
  if one_hot:
    return dense_to_one_hot(labels, num_classes)
  return labels

class DataSet(object):
  def __init__(self,
               data,
               labels,
               one_hot=False,
               dtype=dtypes.uint8,
               reshape=True,
               seed=None):
    '''
    构建数据集
    :param data: 
    :param labels:  
    :param one_hot:
    :param dtype: dtype的作用是将图像像素点的灰度值从[0, 255]转变为[0.0, 1.0]。can be 'uint8' -> '[0, 255]', or 'float32' -> '[0,1]'
    :param reshape: 将图像的形状从[num examples, rows, columns, depth]转变为[num examples, rows*columns] （对于二维图片，depth为1）
    :param seed: 
    '''
    seed1, seed2 = random_seed.get_seed(seed)
    # If op level seed is not set, use whatever graph level seed is returned
    np.random.seed(seed1 if seed is None else seed2)
    dtype = dtypes.as_dtype(dtype).base_dtype
    self._data = data
    self._labels = labels
    self._epochs_completed = 0
    self._index_in_epoch = 0

  @property
  def data(self):
    return self._data

  @property
  def labels(self):
    return self._labels

  @property
  def num_examples(self):
    return self._num_examples

  @property
  def epochs_completed(self):
    return self._epochs_completed

  def next_batch(self, batch_size, shuffle=True):
    '''
    从数据集中取出下一个batch_size数目的数据
    :param batch_size: 
    :param shuffle: 
    :return: 
    '''
    start = self._index_in_epoch
    # Shuffle for the first epoch
    if self._epochs_completed == 0 and start == 0 and shuffle:
      perm0 = np.arange(self._num_examples)
      np.random.shuffle(perm0)
      self._data = self.data[perm0]
      self._labels = self.labels[perm0]
    # Go to the next epoch
    if start + batch_size > self._num_examples:
      # Finished epoch
      self._epochs_completed += 1
      # Get the rest examples in this epoch
      rest_num_examples = self._num_examples - start
      data_rest_part = self._data[start:self._num_examples]
      labels_rest_part = self._labels[start:self._num_examples]
      # Shuffle the data
      if shuffle:
        perm = np.arange(self._num_examples)
        np.random.shuffle(perm)
        self._data = self.data[perm]
        self._labels = self.labels[perm]
      # Start next epoch
      start = 0
      self._index_in_epoch = batch_size - rest_num_examples
      end = self._index_in_epoch
      data_new_part = self._data[start:end]
      labels_new_part = self._labels[start:end]
      return np.concatenate((data_rest_part, data_new_part), axis=0) , np.concatenate((labels_rest_part, labels_new_part), axis=0)
    else:
      self._index_in_epoch += batch_size
      end = self._index_in_epoch
      return self._data[start:end], self._labels[start:end]


def read_data_sets(datapath,
                   one_hot=False,
                   dtype=dtypes.int8,
                   reshape=True,
                   validation_size=2000,
                   seed=None):

  data = pd.read_csv(datapath + 'data.csv')
  labels = extract_labels(datapath + 'labels.csv', one_hot=one_hot)
  train_data, test_data, train_labels, test_labels = cross_validation.train_test_split(data, labels, test_size=0.2)

  if not 0 <= validation_size <= len(train_data):
    raise ValueError(
        'Validation size should be between 0 and {}. Received: {}.'
        .format(len(train_data), validation_size))

  validation_data = train_data[:validation_size]
  validation_labels = train_labels[:validation_size]
  train_data = train_data[validation_size:]
  train_labels = train_labels[validation_size:]

  train = DataSet(train_data, train_labels, dtype=dtype, reshape=reshape, seed=seed)
  validation = DataSet(
      validation_data,
      validation_labels,
      dtype=dtype,
      reshape=reshape,
      seed=seed)
  test = DataSet(
      test_data, test_labels, dtype=dtype, reshape=reshape, seed=seed)

  return base.Datasets(train=train, validation=validation, test=test)