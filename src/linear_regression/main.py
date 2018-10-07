#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @File    : main.py
# @Time    : 2018/10/04
# @Author  : spxcds (spxcds@gmail.com)

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
import time
import threading


class Ploter(object):
    def __init__(self, title=None):
        self.fig, self.ax = plt.subplots()
        plt.title(title)
        plt.grid(True)
        self.xs = [0]
        self.ys = [0]
        self.line = None

        self.text = plt.figtext(0.7, 0.95, '', fontsize=10, ha='left')

        plt.legend()
        self.ani = animation.FuncAnimation(self.fig, self.update, interval=200)

    def set_scatter(self, xs, ys):
        self.ax.scatter(xs, ys)

    def set_lines(self, xs, ys):
        self.xs = xs
        self.ys = ys

    def set_figtext(self, s):
        self.text.set_text(s)

    def update(self, frame, *fargs):
        if not self.line:
            self.line = self.ax.plot(self.xs, self.ys, color='red', linewidth=3)[0]
        else:
            self.line.set_xdata(self.xs)
            self.line.set_ydata(self.ys)


def get_data(low=5, high=50, data_size=50):
    y = np.random.randint(low=low, high=high, size=data_size)
    y.sort()
    y = y.reshape(y.shape[0], 1)
    X = 1 + np.arange(y.shape[0]).reshape(y.shape[0], 1)
    X = np.hstack((X, np.ones((y.shape[0], 1))))

    return X, y


def gradient_descent(X, y, W):
    # delta_y = np.dot(X, W) - y
    return 1.0 / X.shape[0] * np.dot(X.T, (np.dot(X, W) - y))


def linear_regression(p):

    # J(W) = 1/2m∑(XW - y)^2
    # J'(W) = 1/m∑(XW - y)X
    """numpy
    J(W) = 1/2m(XW - y)^T(XW - y)
    J'(W) = 1/m(X^T)(XW - y)
    """

    X, y = get_data()
    p.set_scatter(X[:, 0], y.flatten())
    W = np.random.rand(X.shape[1], 1)

    alpha = 0.001
    gradient = gradient_descent(X, y, W)
    pre_gradient = gradient
    while not np.all(np.absolute(gradient) <= 1e-5):
        W = W - alpha * gradient

        gradient = gradient_descent(X, y, W)
        x0 = np.arange(50)
        y0 = np.dot(np.vstack((x0, np.ones(50))).T, W)

        p.set_figtext('y = {}x + {}'.format(W[0], W[1]))
        p.set_lines(x0, y0.flatten())


def main():
    p = Ploter('linear_regression')

    t = threading.Thread(target=linear_regression, args=(p, ))
    t.setDaemon(True)
    t.start()

    plt.show()


if __name__ == '__main__':
    main()
