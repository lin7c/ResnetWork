# -*- coding: utf-8 -*-
# Created on: 2021/6/24
# Function: 逻辑回归，是一种分类算法， 二分类， 梯度下降法

import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from sklearn import datasets
from sklearn.model_selection import train_test_split

import pdb
pdb.set_trace()
iris = datasets.load_iris()
# 将data和target合并成一个ndarray
iris = np.c_[iris.data, iris.target]
# 删除重复行
iris = np.unique(iris, axis=0)
# 只取标签为0，1的数据
iris = iris[iris[:,-1] != 2]


class LogisticRegression:
    """使用python语言实现逻辑回归算法"""

    def __init__(self, alpha, times):
        """
        初始化
        :param alpha: float，学习率（步长）
        :param times: int，迭代次数
        """
        self.alpha = alpha
        self.times = times

    def sigmoid(self, z):
        """
        函数，将连续的值转换成离散的值，[0, 1]
        :param z: float，自变量，值为 z=w.T * x
        :return: p, float, 值为[0,1]之间，返回样本属于类别1的概率值，用来作为结果的预测
                当 z >= 0.5,判定为类别1，否则判定为类别0
        """
        return 1. / (1.+np.exp(-z))

    def fit(self, X, y):
        """
        根据提供的训练数据，对模型进行训练
        :param X: 类数组类型, 形状为 [样本数量，特征数量]，待训练的样本特征属性
        :param y: 类数组类型，形状为 [样本数量]，每个样本的目标值（标签）
        :return:
        """
        X = np.asarray(X)
        y = np.asarray(y)
        # 创建权重的向量。初始值为0， 长度比特征数多1，多出来的一个值作为截距
        self.w_ = np.zeros(1+X.shape[1])
        # 创建损失列表，用来保存每次迭代的损失值
        self.loss_ = []

        for i in range(self.times):
            z = np.dot(X, self.w_[1:]) + self.w_[0]
            # 计算概率值（结果判定为1的概率值）
            p = self.sigmoid(z)
            # 根据逻辑回归的代价函数（目标函数 or 损失函数）
            # 逻辑回归的目标函数： J(w) = -sum(y(i)*log(sigmoid(z(i))) + (1-y(i))*log(1-sigmoid(z(i))))
            cost = -np.sum(y * np.log(p) + (1-y) * np.log(1-p))
            self.loss_.append(cost)

            # 调整权重值，根据公式： 权重（j列） = 权重（j列）+ 学习率 * sum((y-sigmoid(z)) * x(j))
            self.w_[0] += self.alpha * np.sum((y-p) * 1)
            self.w_[1:] += self.alpha * np.dot(X.T, y-p)

    def predict_proba(self, X):
        """
        根据参数传递的样本，对样本数据进行预测
        :param X: 类数组类型，形状为 [样本数量，特征数量]， 待测试的样本特征（属性）
        :return: result 数组类型， 预测的结果（概率值）
        """
        X = np.asarray(X)
        z = np.dot(X, self.w_[1:]) + self.w_[0]
        p = self.sigmoid(z)
        # 将预测结果变成二维结构
        p = p.reshape(-1, 1)
        # 将两个数组进行拼接，方向为横向,前者为接近0的概率，后者为接近1的概率
        return np.c_[1-p, p]

    def preddict(self, X):
        """
        根据参数传递的样本，对样本数据进行预测
        :param X: 类数组类型，形状为 [样本数量，特征数量]， 待测试的样本特征（属性）
        :return: result 数组类型，预测的结果（分类标签）
        """
        # 取概率大的索引即为预测的分类标签
        return np.argmax(self.predict_proba(X), axis=1)


def create_meshgrid_pic(plt, predict, X, Y, step=0.01):
    """
    画分类网格
    :param plt: 画图对象
    :param predict: 预测对象
    :param X:
    :param Y:
    :param step:
    :return:
    """
    # 确认训练集的边界
    x_min, x_max = X[:].min() - .5, X[:].max() + .5
    y_min, y_max = Y[:].min() - .5, Y[:].max() + .5
    # 生成网络数据, xx所有网格点的x坐标,yy所有网格点的y坐标
    xx, yy = np.meshgrid(np.arange(x_min, x_max, step),
                         np.arange(y_min, y_max, step))
    # xx,yy的扁平化成一串坐标点（密密麻麻的网格点平摊开来）
    d = np.c_[xx.ravel(), yy.ravel()]
    # 对网格点进行预测
    Z = predict(d)
    # 预测完之后重新变回网格的样子，因为后面pcolormesh接受网格形式的绘图数据
    Z = Z.reshape(xx.shape)
    # class_size = np.unique(Z).size
    # classes_color = ['#FFAAAA', '#AAFFAA', '#AAAAFF'][:class_size]
    # cmap_light = ListedColormap(classes_color)
    # # 接受网络化的x,y,z
    # plt.pcolormesh(xx, yy, Z, cmap=cmap_light)
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral) #等位线

pdb.set_trace()
data = iris[:, :-1]
target = iris[:, -1]
# 对标签的数据类型修改为整数
target = np.asarray(target, dtype=int)

X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2)

lr = LogisticRegression(alpha=0.0005, times=500)
lr.fit(X_train[:, :2], y_train)
result = lr.preddict(X_test[:, :2])

err = sum(result == y_test)/len(result)
print(err)


# 可视化
# 由于鸢尾花数据是4维的，我们画图只取两位即可

import matplotlib.pyplot as plt

plt.figure(figsize=(10,10))

# 画分类网格
create_meshgrid_pic(plt, lr.preddict, X_train[:, 0], X_train[:, 1])
# 画预测值
plt.scatter(X_test[:, 0], X_test[:, 1], c=result)
# 画训练数据
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train)

plt.show()
