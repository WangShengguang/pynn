# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 14:34:10 2020
@author: zhangjuefei
"""

import numpy as np
from sklearn.datasets import make_classification

from pynn.graph import default_graph
from pynn.layer import fc
from pynn.loss import Logistic, LogLoss
from pynn.node import Variable
from pynn.ops import MatMul, Add, Multiply, Reshape
from pynn.optimizer import Adam

# 特征维数
dimension = 60

# 构造二分类样本，有用特征占20维
X, y = make_classification(600, dimension, n_informative=20)
y = y * 2 - 1

# 嵌入向量维度
k = 20

# 一次项
x1 = Variable(shape=(dimension, 1), init=False, trainable=False)

# 标签
label = Variable(shape=(1, 1), init=False, trainable=False)

# 一次项权值向量
w = Variable(shape=(1, dimension), init=True, trainable=True)

# 嵌入矩阵
E = Variable(shape=(k, dimension), init=True, trainable=True)

# 偏置
b = Variable(shape=(1, 1), init=True, trainable=True)

# Wide部分
wide = MatMul(w, x1, name='MatMul wide')

embedding = MatMul(E, x1, name='MatMul embedding')  # 用嵌入矩阵与特征向量相乘，得到嵌入向量

fm = Add(wide,
         MatMul(Reshape(embedding, shape=(1, k)), embedding)
         )

# Deep部分

# 第一隐藏层
hidden_1 = fc(embedding, k, 8, "ReLU")

# 第二隐藏层
hidden_2 = fc(hidden_1, 8, 4, "ReLU")

# 输出层
deep = fc(hidden_2, 4, 1, None)

# 输出
output = Add(deep, fm, b)

# 预测概率
predict = Logistic(output)

# 损失函数
loss = LogLoss(Multiply(label, output))

learning_rate = 0.005
optimizer = Adam(default_graph, loss, learning_rate)

batch_size = 16

for epoch in range(200):

    batch_count = 0
    for i in range(len(X)):

        x1.set_value(np.mat(X[i]).T)
        label.set_value(np.mat(y[i]))

        optimizer.one_step()

        batch_count += 1
        if batch_count >= batch_size:
            optimizer.update()
            batch_count = 0

    pred = []
    for i in range(len(X)):
        x1.set_value(np.mat(X[i]).T)

        predict.forward()
        pred.append(predict.value[0, 0])

    pred = (np.array(pred) > 0.5).astype(np.int) * 2 - 1
    accuracy = (y == pred).astype(np.int).sum() / len(X)

    print("epoch: {:d}, accuracy: {:.3f}".format(epoch + 1, accuracy))
