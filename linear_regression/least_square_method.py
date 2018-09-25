#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @File    : a.py
# @Time    : 2018/09/18
# @Author  : sunxiaodong (sunxiaodong@360.cn)

from __future__ import division
import numpy as np
import time
import random
from matplotlib import pyplot as plt

random.seed(int(time.time()))

np.nan


def matrix_lstsqr(x, y):
    X = np.vstack([x, np.ones(len(x))]).T
    return (np.linalg.inv(X.T.dot(X)).dot(X.T)).dot(y)


def classic_lstsqr(x_list, y_list):
    N = len(x_list)
    x_avg = sum(x_list) / N
    y_avg = sum(y_list) / N
    var_x, cov_xy = 0, 0
    for x, y in zip(x_list, y_list):
        temp = x - x_avg
        var_x += temp**2
        cov_xy += temp * (y - y_avg)
    slope = cov_xy / var_x
    y_interc = y_avg - slope * x_avg
    return (slope, y_interc)


x = [x_i + random.randrange(8, 12) for x_i in range(50)]
y = [y_i + random.randrange(20, 50) for y_i in range(50)]

slope, intercept = classic_lstsqr(x, y)
slope, intercept = matrix_lstsqr(x, y)

line_x = [round(min(x)) - 1, round(max(x)) + 1]
line_y = [slope * x_i + intercept for x_i in line_x]

plt.figure(figsize=(8, 8))
plt.scatter(x, y)
plt.plot(line_x, line_y, color='red', lw='2')

plt.ylabel('y')
plt.xlabel('x')
plt.title('Linear regression via least squares fit')

ftext = 'y = ax + b = {:.3f} + {:.3f}x'.format(slope, intercept)
plt.figtext(.15, .8, ftext, fontsize=11, ha='left')

plt.show()
