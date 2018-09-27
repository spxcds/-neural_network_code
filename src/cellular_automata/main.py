#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @File    : main.py
# @Time    : 2018/09/25
# @Author  : sunxiaodong (sunxiaodong@360.cn)

import os
import time
import numpy as np
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, wait, as_completed
from multiprocessing import cpu_count


class CellularAutomataUpdateRules(object):
    def __init__(self):
        self.rule_no = 0

    def set_rule_no(self, rule_no):
        self.rule_no = rule_no

    def default_update_rule(self, neighbor):
        """
        neighbor is a list type
        """
        rule_no = self.rule_no
        num = np.sum([2**i for i in range(len(neighbor)) if neighbor[-(i + 1)]])
        return num in [i for i in range(len(bin(rule_no))) if bin(rule_no)[::-1][i] == '1']


class MnistCellularAutomata(object):
    def __init__(self, cells, update_rule, neighbor_radius=1):
        """
        cells: a list of int number
        update_rule: update_rule
        """
        self.cells = cells
        self.last_cells = cells
        self.neighbor_radius = neighbor_radius
        self.cells_len = len(cells)
        self.cells_procedure = [cells]
        self.update_rule = update_rule
        self.steps = 0

    def update_state(self):
        buf = np.zeros(self.cells_len)
        cells = self.cells

        for i in range(self.cells_len):
            left_idx = i - self.neighbor_radius if i - self.neighbor_radius > 0 else 0
            right_idx = i + self.neighbor_radius if i + self.neighbor_radius < self.cells_len else self.cells_len

            buf[i] = self.update_rule(self.last_cells[left_idx:right_idx + 1])

        self.last_cells = buf
        self.cells_procedure.append(self.last_cells)

    @staticmethod
    def img_save(array, file_name):
        img = Image.fromarray(array)
        img = img.convert('L')
        img.save(file_name)

    def update_and_save(self, steps, file_name):
        for i in range(steps):
            self.update_state()
        ndarray = np.array(self.cells_procedure)
        ndarray[ndarray > 0] = 255

        self.img_save(ndarray, file_name)


def load_data(data_path):
    data = np.loadtxt(data_path, delimiter=',', dtype=np.uint8)
    return data


pool = ThreadPoolExecutor(max_workers=cpu_count())
tasks = []
MAX_TASKS = 200


def draw_img_by_rules(img_data, rule_list, steps, directory_name):
    """
    img_data: a list indicates the img data
    rule_list: a list indicates the rule
    directory_name: save ca picture in this directory_name
    """
    if not os.path.exists(directory_name):
        os.makedirs(directory_name)

    MnistCellularAutomata.img_save(array=img_data.reshape(28, 28), file_name=os.path.join(directory_name, 'pic.jpeg'))

    rule = CellularAutomataUpdateRules()
    for rule_no in rule_list:
        rule.set_rule_no(rule_no)
        mca = MnistCellularAutomata(img_data, rule.default_update_rule, neighbor_radius=1)
        # mca.update_and_save(steps=steps, file_name=os.path.join(directory_name, str(rule_no)) + '.jpeg')
        tasks.append(
            pool.submit(
                mca.update_and_save, steps=steps, file_name=os.path.join(directory_name, str(rule_no)) + '.jpeg'))
        while len(tasks) >= MAX_TASKS:
            for task in as_completed(tasks):
                tasks.remove(task)
            time.sleep(1)


if __name__ == '__main__':

    root_path = '../../dataset/mnist'
    num = 1  # each kind of pic num

    for i in range(10):
        data_num = i
        data_path = os.path.join(root_path, str(data_num) + '.csv')
        data = load_data(data_path)

        for j in range(num):
            print('Digit num: {}'.format(i))
            draw_img_by_rules(
                img_data=data[j], rule_list=range(256), steps=100, directory_name=os.path.join('build', str(i), str(j)))
    wait(tasks)
