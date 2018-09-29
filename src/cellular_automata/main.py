#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @File    : main.py
# @Time    : 2018/09/28
# @Author  : spxcds (spxcds@gmail.com)

import os
import time
import numpy as np
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, wait, as_completed

# from multiprocessing import cpu_count


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
        num = int(np.sum(2**i for i, j in enumerate(neighbor) if j))
        return (1 << num) & rule_no


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


# pool = ThreadPoolExecutor(max_workers=cpu_count())
pool = ThreadPoolExecutor(max_workers=4)
tasks = []
MAX_TASKS = 32
ROOT_DIR = 'build'


def draw_img_by_rules(img_data, rule_list, steps, file_name):
    """
    img_data: a list indicates the img data
    rule_list: a list indicates the rule
    directory_name: save ca picture in this directory_name
    """
    # MnistCellularAutomata.img_save(array=img_data.reshape(28, 28), file_name=os.path.join(file_name, 'pic.jpeg'))

    for rule_no in rule_list:
        # create dirname
        dirname = os.path.join(ROOT_DIR, str(rule_no), os.path.dirname(file_name))
        if not os.path.exists(dirname):
            os.makedirs(dirname)

        rule = CellularAutomataUpdateRules()
        rule.set_rule_no(rule_no)
        mca = MnistCellularAutomata(img_data, rule.default_update_rule, neighbor_radius=1)
        # mca.update_and_save(steps=steps, file_name=os.path.join(directory_name, str(rule_no)) + '.jpeg')
        tasks.append(
            pool.submit(mca.update_and_save, steps=steps, file_name=os.path.join(ROOT_DIR, str(rule_no), file_name)))
        while len(tasks) >= MAX_TASKS:
            for task in as_completed(tasks):
                tasks.remove(task)
            time.sleep(1)


if __name__ == '__main__':
    root_path = '../../dataset/mnist'
    num = 50  # each kind of pic num

    for digit_num in range(10):
        if digit_num not in [1, 8]:
            continue
        data_path = os.path.join(root_path, str(digit_num) + '.csv')
        data = load_data(data_path)

        for j in range(num):
            print('Digit num: {}'.format(digit_num))
            draw_img_by_rules(
                img_data=data[j],
                rule_list=range(256),
                steps=1000,
                file_name=os.path.join(str(digit_num),
                                       str(j) + '.jpeg'))
    wait(tasks)
