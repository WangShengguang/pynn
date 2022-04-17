# 该数据集一共包含4个特征变量，1个类别变量。共有150个样本，iris是鸢尾植物，这里存储了其萼片和花瓣的长宽，共4个属性
import numpy as np
from sklearn.datasets import load_iris  # 导入数据集iris
from sklearn.preprocessing import OneHotEncoder

from pynn.graph import default_graph
from pynn.loss import SoftMax, CrossEntropyWithSoftmax
from pynn.node import Variable
from pynn.ops import Add, MatMul
from pynn.optimizer import Adam

np.random.seed(41)

iris = load_iris()  # 载入数据集

# train_set = np.concatenate([iris.data, iris.target], axis=1)
# np.random.shuffle(train_set)

x = Variable(shape=(4, 1), init=False, trainable=False)
onehot_label = OneHotEncoder(sparse=False).fit_transform(iris.target.reshape(-1, 1))
label = Variable(shape=(3, 1), init=False, trainable=False)
#
w = Variable(shape=(3, 4), init=True, trainable=True)
b = Variable(shape=(3, 1), init=True, trainable=True)
#
output = Add(MatMul(w, x), b)
predict = SoftMax(output)
loss = CrossEntropyWithSoftmax(output, label)
learning_rate = 0.02
optimizer = Adam(default_graph, loss, learning_rate=learning_rate)
num_epochs = 500
batch_size = 16
for epoch in range(num_epochs):
    batch_count = 0
    for i in range(len(iris.data)):
        features = np.mat(iris.data[i]).T
        l = np.mat(onehot_label[i]).T
        x.set_value(features)
        label.set_value(l)
        optimizer.one_step()
        batch_count += 1
        if batch_count % batch_size == 0:
            optimizer.update()
            batch_count = 0

    pred = []
    for i in range(len(iris.data)):
        features = np.mat(iris.data[i]).T
        x.set_value(features)
        predict.forward()
        pred.append(predict.value.A.ravel())
    pred = np.array(pred).argmax(axis=-1)
    accuracy = (iris.target == pred).astype(int).sum() / len(onehot_label)
    print(accuracy)
