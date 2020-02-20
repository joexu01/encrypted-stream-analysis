# -*- coding: UTF-8 -*-
import math
import copy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

plt.rcParams['font.sans-serif'] = ['STXihei']  # （替换sans-serif字体）显示中文

data_set = [
    [1, 4, 6, 6, 8],
    [2, 2, 3, 6, 9],
    [6, 8, 5, 7, 4],
]

labels = ['Group1', 'Group2', 'Group3']

plt.figure(1)

for i in range(0, 3):
   plt.plot(data_set[i], )
