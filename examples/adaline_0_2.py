import numpy as np

from pynn.graph import default_graph
from pynn.loss import PerceptionLoss
from pynn.node import Variable
from pynn.ops import Add, MatMul, Step

np.random.seed(25)

male_heights = np.random.normal(171, 6, 500)
female_heights = np.random.normal(158, 5, 500)

male_weights = np.random.normal(70, 10, 500)
female_weights = np.random.normal(57, 8, 500)
# 体脂率 Bodlabel fat Ratio (percentage)
male_bfrs = np.random.normal(16, 2, 500)
female_bfrs = np.random.normal(22, 2, 500)

male_labels = [1] * 500
female_labels = [-1] * 500
train_set = np.array([
    np.concatenate((male_heights, female_heights)),
    np.concatenate((male_weights, female_weights)),
    np.concatenate((male_bfrs, female_bfrs)),
    np.concatenate((male_labels, female_labels))
]).T
np.random.shuffle(train_set)

x = Variable(shape=(3, 1), init=False, trainable=False)
# label = Variable(shape=(1, 1), init=False, trainable=False)
# #
# w = Variable(shape=(1, 3), init=True, trainable=True)
# b = Variable(shape=(1, 1), init=True, trainable=True)

# 类别标签，1男，-1女
label = Variable(shape=(1, 1), init=False, trainable=False)

# 权重向量，是一个1x3矩阵，需要初始化，参与训练
w = Variable(shape=(1, 3), init=True, trainable=True)

# 阈值，是一个1x1矩阵，需要初始化，参与训练
b = Variable(shape=(1, 1), init=True, trainable=True)
#
# output = Add(MatMul(w, x), b)
# predict = Step(output)
# loss = PerceptionLoss(MatMul(label, output))
# ADALINE的预测输出
output = Add(MatMul(w, x), b)
predict = Step(output)

# 损失函数
loss = PerceptionLoss(MatMul(label, output))

#
learning_rate = 0.0001

for epoch in range(50):
    for i in range(len(train_set)):
        features = np.mat(train_set[i, :-1]).T
        l = np.mat(train_set[i, -1])
        x.set_value(features)
        # breakpoint()
        label.set_value(l)
        #
        loss.forward()  # 前向传播，计算损失
        # print('loss: {:.05f}'.format(loss.value.mean()))
        #
        w.backward(loss)  # 反向传播，计算loss值对w的雅克比矩阵
        b.backward(loss)  # 反向传播，计算loss值对b的雅克比矩阵
        #
        w.set_value(
            w.value - learning_rate * w.jacobi.T.reshape(w.shape))
        b.set_value(
            b.value - learning_rate * b.jacobi.T.reshape(b.shape))
        default_graph.clear_jacobi()

    pred = []
    for i in range(len(train_set)):
        features = np.mat(train_set[i, :-1]).T
        x.set_value(features)
        predict.forward()
        pred.append(predict.value[0, 0])
    pred = np.array(pred) * 2 - 1
    accuracy = (train_set[:, -1] == pred).astype(int).sum() / len(train_set)
    print(accuracy)
