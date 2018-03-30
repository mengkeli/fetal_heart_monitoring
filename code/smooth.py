# -*- coding:utf-8 -*-
'''
 fixing the missing values and smoothing the line
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib.pyplot import savefig, axis
from scipy import interpolate
import random


datapath = '../data/'
series_file = datapath + 'fetal_series_02_50.npy'
image_file = datapath + 'fetal_image_02_50_bold.npy'

f = np.load(series_file)
arr_100 = f[0:100, 0:-1]
label_100 = f[0:100, -1]

y = arr_100[0, :]
x = np.arange(0, 2402, 1)

xnew = np.arange(0, 2401, 0.5)

"""
kind方法：
nearest、zero、slinear、quadratic、cubic
实现函数func
"""
func = interpolate.interp1d(x, y, kind='cubic')
# 利用xnew和func函数生成ynew，xnew的数量等于ynew数量
ynew = func(xnew)


# 原图
plt.plot(x, y, 'ro-')
savefig(datapath+'images/'+'raw.jpg')
# 拟合之后的平滑曲线图
plt.plot(xnew, ynew)
savefig(datapath+'images/'+'smooth.jpg')