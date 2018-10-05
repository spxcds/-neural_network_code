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
        self.xs = [0]
        self.ys = [0]

        self.line = None
        plt.legend()
        self.ani = animation.FuncAnimation(self.fig, self.update, interval=500)

    def set_scatter(self, xs, ys):
        self.ax.scatter(xs, ys)

    def set_lines(self, xs, ys):
        self.xs = xs
        self.ys = ys

    def update(self, frame, *fargs):
        if not self.line:
            self.line = self.ax.plot(self.xs, self.ys)[0]
        else:
            self.line.set_xdata(self.xs)
            self.line.set_ydata(self.ys)


def get_data():
    y = np.array([3, 4, 5, 5, 2, 4, 7, 8, 11, 8, 12, 11, 13, 13, 16, 17, 18, 17, 19, 21])
    y = y.reshape(y.shape[0], 1)
    X = 1 + np.arange(y.shape[0]).reshape(y.shape[0], 1)
    X = np.hstack((X, np.ones((y.shape[0], 1))))

    return X, y


def gradient_descent(X, y, W):
    # delta_y = np.dot(X, W) - y
    return 1. / X.shape[0] * np.dot(X.T, (np.dot(X, W) - y))


def linear_regression(p):
    X, y = get_data()
    p.set_scatter(X[:, 0], y.flatten())
    W = np.random.rand(X.shape[1], 1)
    alpha = 0.01
    gradient = gradient_descent(X, y, W)
    # print(gradient)
    while not np.all(np.absolute(gradient) <= 1e-5):
        time.sleep(1)
        W = W - alpha * gradient

        gradient = gradient_descent(X, y, W)
        x0 = np.arange(20)
        y0 = np.dot(np.vstack((x0, np.ones(20))).T, W)
        # print(x0, y0)
        p.set_lines(x0, y0.flatten())
        # print(gradient)
        # print(W)
        # print(X[:, 0], y)

    print('gradient: {}'.format(gradient))
    print('Weights: {}'.format(W))


def main():
    p = Ploter()

    t = threading.Thread(target=linear_regression, args=(p, ))
    t.setDaemon(True)
    t.start()

    plt.show()


if __name__ == '__main__':
    main()
