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
np.savetxt(datapath+'94th_sample.csv', arr_100[94, :], fmt = '%d')

