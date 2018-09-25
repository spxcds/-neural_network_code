import numpy as np
import matplotlib.pyplot as plt


class CellularAutomataUpdateRules(object):
    @classmethod
    def sigmod(self):
        pass


class MnistCellularAutomata(object):
    def __init__(self, cells, update_rule, neighbor_radius=1):
        """
        cells: a list of int number
        update_rule: update_rule
        """
        self.cells = cells
        self.neighbor_radius = neighbor_radius
        self.cells_len = len(cells)
        self.cells_procedure = [cells]
        self.update_rule = update_rule
        self.steps = 0

    def update_state(self):
        buf = np.zeros(self.cells_len)
        cells = self.cells

        for i in range(self.cells_len):
            neighbor_sum = 

        for i in range(1, cells.shape[0] - 1):
            for j in range(1, cells.shape[0] - 1):
                # 计算该细胞周围的存活细胞数
                neighbor = cells[i - 1:i + 2, j - 1:j + 2].reshape((-1, ))
                neighbor_num = np.convolve(self.mask, neighbor, 'valid')[0]
                if neighbor_num == 3:
                    buf[i, j] = 1
                elif neighbor_num == 2:
                    buf[i, j] = cells[i, j]
                else:
                    buf[i, j] = 0
        self.cells = buf
        self.timer += 1

    def plot_state(self):
        plt.title('Iter :{}'.format(self.timer))
        plt.imshow(self.cells)
        plt.show()

    def update_and_plot(self, n_iter):
        plt.ion()
        for _ in range(n_iter):
            plt.title('Iter :{}'.format(self.timer))
            plt.imshow(self.cells)
            self.update_state()
            plt.pause(0.2)
        plt.ioff()


if __name__ == '__main__':
    game = MnistCellularAutomata(cells_shape=784)
    game.update_and_plot(200)