import numpy as np


def load_data():
    """
    功能：获取训练集(数据来自《机器学习实战》)
    返回：训练集
    """
    postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                   ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                   ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]

    labels = [0, 1, 0, 1, 0, 1]  # 1为侮辱性文字，0为正常言论
    return postingList, labels


def create_vocabulary_list(dataSet):
    """
    功能：生成单词库
    参数：训练集
    返回：单词库列表
    """
    vocabList = []
    for statement in dataSet:
        for word in statement:
            if word not in vocabList:
                vocabList.append(word)
    return vocabList


def words2vector(statement, vocabList):
    """
    功能：语句转化为向量
    参数：语句，单词库
    返回：向量
    """
    vector = []
    for word in vocabList:
        if word in statement:
            vector.append(1)
        else:
            vector.append(0)
    return vector


def train(dataSet, labels, vocabList):
    """
    功能：训练
    参数：训练集，标签，单词库
    返回：参数值
    """
    m = len(dataSet)
    num_y_1 = labels.count(1)   # 侮辱性言论的个数
    num_y_0 = labels.count(0)   # 正常言论的个数
    phi_y_1 = num_y_1 / float(m)    # P(y=1)的概率
    phi_y_0 = num_y_0 / float(m)    # P(y=0)的概率
    phi_j_y_1 = []                  # p(xj=1|y=1)的概率
    phi_j_y_0 = []                  # p(xj=1|y=0)的概率
    for word in vocabList:
        num_j_y_1 = 0
        num_j_y_0 = 0
        for i in range(len(labels)):
            print(word)
            if labels[i] == 1 and word in dataSet[i]:
                num_j_y_1 += 1
            if labels[i] == 0 and word in dataSet[i]:
                num_j_y_0 += 1
        phi_j_y_1.append((num_j_y_1+1) / (float(num_y_1)+2))    # Laplace Smoothing
        phi_j_y_0.append((num_j_y_0+1) / (float(num_y_0)+2))    # 某个词是好词/坏词的概率即这个词在好词/坏词中是否出现和出现多少
    return phi_j_y_1, phi_j_y_0, phi_y_1, phi_y_0


def predict(words, vocabList, phi):
    """
    功能：预测
    参数：预测言论序列，单词库，参数值
    返回：参数值
    """
    vect = words2vector(words, vocabList)
    p_y_1 = phi[2]
    p_y_0 = phi[3]
    for i in range(len(vocabList)):
        if vect[i] == 1: #如果这个单词出现了
            p_y_1 *= phi[0][i] #就乘以认为这个词是好词的概率
            p_y_0 *= phi[1][i]
        else:
            p_y_1 *= (1 - phi[0][i]) #就认为他没有说好词
            p_y_0 *= (1 - phi[1][i])
    return p_y_1, p_y_0

'''判断一个句子的好坏，把这个句子划分成几个单词，每个单词都有一个是好词还是坏词的概率'''


DataSet, Labels = load_data()
VocabList = create_vocabulary_list(DataSet)
phi = train(DataSet, Labels, VocabList)
p1, p0 = predict(['i', 'licks', 'ate', 'my', 'dog', 'food', 'stupid'], VocabList, phi)
print("是侮辱性言论的概率：" + str(p1 / (p1 + p0)))
print("是正常言论的概率：" + str(p0 / (p1 + p0)))