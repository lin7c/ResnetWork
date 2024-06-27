import numpy as np
import random
import math


def load_data(gauss_1, gauss_2, num=100):
    """
    随机生成两个高斯分布的数据
    :param gauss_1: 第一个高斯分布参数
    :param gauss_2: 第二个高斯分布参数
    :param num: 生成数据个数
    :return: 生成的数据集, 标签
    """
    x = []
    y = []
    while len(x) < num:
        if random.random() > 0.5:
            x.append(random.gauss(gauss_1[0], gauss_1[1]))
            y.append(0)
        else:
            x.append(random.gauss(gauss_2[0], gauss_2[1]))
            y.append(1)
    return np.array(x), y


def train(train_x, train_y):
    """
    训练函数，计算两个高斯分布参数及类别概率
    :param train_x: 训练集
    :param train_y: 标签
    :return: 所有参数
    """
    m = train_x.shape[0]
    phi = train_y.count(0) / float(m)   # 是高斯分布 0 的概率
    total_mean_0 = 0.0
    total_mean_1 = 0.0
    for i in range(len(train_y)):
        if not train_y[i]:
            total_mean_0 += train_x[i]
        else:
            total_mean_1 += train_x[i]
    miu_0 = total_mean_0 / float(train_y.count(0))  # 高斯分布 0 的均值
    miu_1 = total_mean_1 / float(train_y.count(1))  # 高斯分布 1 的均值
    total_variance_0 = 0.0
    total_variance_1 = 0.0
    for j in range(len(train_y)):
        if not train_y[j]:
            total_variance_0 = (train_x[j] - miu_0) ** 2
        else:
            total_variance_1 = (train_x[j] - miu_1) ** 2
    sigma_0 = total_variance_0 / float(train_y.count(0))    # 高斯分布 0 的方差
    sigma_1 = total_variance_1 / float(train_y.count(1))    # 高斯分布 1 的方差
    return phi, [miu_0, sigma_0], [miu_1, sigma_1]


def predict(x, gauss):
    """
    预测函数
    :param x: 预测值
    :param gauss: 高斯分布参数及类别概率参数
    :return: 概率值
    """
    p_y_0 = gauss[0] * math.exp(-1 * (x - gauss[1][0]) ** 2 / (2 * gauss[1][1]))
    p_y_1 = (1 - gauss[0]) * math.exp(-1 * (x - gauss[2][0]) ** 2 / (2 * gauss[2][1]))
    return p_y_0, p_y_1



train_x, train_y = load_data([10, 3], [20, 3], 100)
gauss_para = train(train_x, train_y)
p0, p1 = predict(12, gauss_para)
print('是高斯分布 0 的概率 ： ' + str(p0 / (p1 + p0)))
print('是高斯分布 1 的概率 ： ' + str(p1 / (p1 + p0)))