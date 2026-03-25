import os
import random
import numpy as np
import collections
import pandas as pd
import matplotlib.pyplot as plt



def moving_average(a, window_size):
    cumulative_sum = np.cumsum(np.insert(a, 0, 0))
    middle = (cumulative_sum[window_size:] - cumulative_sum[:-window_size]) / window_size
    r = np.arange(1, window_size - 1, 2)
    begin = np.cumsum(a[:window_size - 1])[::2] / r
    end = (np.cumsum(a[:-window_size:-1])[::2] / r)[::-1]
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