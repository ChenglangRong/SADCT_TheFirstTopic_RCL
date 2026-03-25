import os
import random
import numpy as np
import collections
import pandas as pd
import matplotlib.pyplot as plt
import torch


def moving_average(a, window_size):
    if window_size <= 1:
        return np.array(a)

    # 计算中间部分的滑动平均
    cumulative_sum = np.cumsum(np.insert(a, 0, 0))
    middle = (cumulative_sum[window_size:] - cumulative_sum[:-window_size]) / window_size

    # 计算前半部分的滑动平均（处理边界）
    begin_len = (window_size - 1) // 2
    begin = []
    for i in range(1, begin_len + 1):
        begin.append(np.mean(a[:2 * i]))
    begin = np.array(begin)

    # 计算后半部分的滑动平均（处理边界）
    end_len = (window_size - 1) - begin_len
    end = []
    for i in range(1, end_len + 1):
        end.append(np.mean(a[-2 * i:]))
    end = np.array(end)[::-1]  # 反转以保持顺序

    return np.concatenate((begin, middle, end))

def scatter_method(x_list,  y_list, title, ylabel, figure_file):
    plt.figure()
    plt.scatter(x_list,  y_list, color='r')
    plt.title(title)
    plt.xlabel('episode')
    plt.ylabel(ylabel)
    plt.savefig(figure_file, dpi=300)  # dpi=300调节分辨率，默认为100
    plt.show()

def plot_method(episodes, records, title, ylabel, figure_file):
    plt.figure()
    plt.plot(episodes, records, linestyle='-', color='r')
    plt.title(title)
    plt.xlabel('episode')
    plt.ylabel(ylabel)
    plt.savefig(figure_file, dpi=300)  # dpi=300调节分辨率，默认为100
    plt.show()

def save_data(data,data_file_path):
    df = pd.DataFrame(data)
    # 导出到 Excel 文件
    df.to_excel(data_file_path, index=False)

def create_directory(path: str, sub_dirs: list):
    for sub_dir in sub_dirs:
        if os.path.exists(path + sub_dir):
            print(path + sub_dir + ' is already exist!')
        else:
            os.makedirs(path + sub_dir, exist_ok=True)
            print(path + sub_dir + ' create successfully!')

def compute_advantage(gamma, lmbda, td_delta):
    td_delta = td_delta.detach().numpy()
    advantage_list = []
    advantage = 0.0
    for delta in td_delta[::-1]:
        advantage = gamma * lmbda * advantage + delta
        advantage_list.append(advantage)
    advantage_list.reverse()
    return torch.tensor(advantage_list, dtype=torch.float)


class RunningMeanStd:
    def __init__(self, shape, epsilon=1e-4, dtype=np.float32):
        self.mean = np.zeros(shape, dtype=dtype)
        self.var = np.ones(shape, dtype=dtype)
        self.count = epsilon  # 避免除零
        self.dtype = dtype

    def update(self, x):
        """更新均值和方差（x为单个状态或批量状态）"""
        x = np.asarray(x, dtype=self.dtype)
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]

        # 合并新的统计量
        self.mean, self.var, self.count = self._merge(
            self.mean, self.var, self.count,
            batch_mean, batch_var, batch_count
        )

    def normalize(self, x):
        """将状态x归一化"""
        x = np.asarray(x, dtype=self.dtype)
        return (x - self.mean) / np.sqrt(self.var + 1e-8)

    @staticmethod
    def _merge(mean1, var1, count1, mean2, var2, count2):
        """合并两个分布的均值、方差和计数"""
        total_count = count1 + count2
        mean = (count1 * mean1 + count2 * mean2) / total_count
        var = (count1 * var1 + count2 * var2 +
               count1 * (mean1 - mean) ** 2 +
               count2 * (mean2 - mean) ** 2) / total_count
        return mean, var, total_count