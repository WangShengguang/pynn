import numpy as np

from pynn.graph import default_graph
from pynn.loss import PerceptionLoss
from pynn.node import Variable
from pynn.ops import Add, MatMul, Step, ScalarMultiply, Multiply

np.random.seed(41)


def get_samples():
    male_heights = np.random.normal(171, 6, 500)
    female_heights = np.random.normal(158, 5, 500)

    male_weights = np.random.normal(70, 10, 500)
    female_weights = np.random.normal(57, 8, 500)
    # 体脂率 Body fat Ratio (percentage)
    male_bfrs = np.random.normal(16, 2, 500)
    female_bfrs = np.random.normal(22, 2, 500)

    male_labels = [1] * 500
    female_labels = [-1] * 500
    train_samples = np.array([
        np.concatenate((male_heights, female_heights)),
        np.concatenate((male_weights, female_weights)),
        np.concatenate((male_bfrs, female_bfrs)),
        np.concatenate((male_labels, female_labels))
    ]).T
    np.random.shuffle(train_samples)
    return train_samples


batch_size = 8


class Adaline(object):
    def __init__(self):
        #
        self.X = Variable(shape=(batch_size, 3), init=False, trainable=False)
        self.Y = Variable(shape=(batch_size, 1), init=False, trainable=False)
        #
        self.W = Variable(shape=(3, 1), init=True, trainable=True)
        self.b = Variable(shape=(1, 1), init=True, trainable=True)
        ones = Variable(shape=(batch_size, 1), init=False, trainable=True)
        ones.set_value(np.mat(np.ones(batch_size)).T)
        bias = ScalarMultiply(self.b, ones)
        #
        self.output = Add(MatMul(self.X, self.W), bias)
        self.predict = Step(self.output)
        self.loss = PerceptionLoss(Multiply(self.Y, self.output))
        B = Variable(shape=(1, batch_size), init=False, trainable=False)
        B.set_value(1 / batch_size * np.mat(np.ones(batch_size)))
        self.mean_loss = MatMul(B, self.loss)


class Trainer(object):
    def __init__(self, model: Adaline, learning_rate=0.0001):
        self.learning_rate = learning_rate
        self.model = model

    def fit(self, x, y, num_epochs=50):
        # label = np.mat(y)
        for epoch in range(num_epochs):
            for i in range(0, len(x), batch_size):
                features = np.mat(x[i:i + batch_size])
                l = np.mat(y[i:i + batch_size]).T
                self.model.X.set_value(features)
                self.model.Y.set_value(l)
                #
                self.model.mean_loss.forward()  # 前向传播，计算损失
                # print('train loss: {:.05f}'.format(self.model.loss.value.mean()))
                #
                self.model.W.backward(self.model.mean_loss)  # 反向传播，计算loss值对W的雅克比矩阵
                self.model.b.backward(self.model.mean_loss)  # 反向传播，计算loss值对b的雅克比矩阵
                #
                self.model.W.set_value(
                    self.model.W.value - self.learning_rate * self.model.W.jacobi.T.reshape(self.model.W.shape))
                self.model.b.set_value(
                    self.model.b.value - self.learning_rate * self.model.b.jacobi.T.reshape(self.model.b.shape))
                default_graph.clear_jacobi()
            self.evaluate(x, y)

    def evaluate(self, x, y):
        pred = []
        for i in range(0, len(x), batch_size):
            features = np.mat(x[i:i + batch_size])
            self.model.X.set_value(features)
            self.model.predict.forward()
            pred.extend(self.model.predict.value.A.ravel())
        pred = np.array(pred) * 2 - 1
        accuracy = (y == pred).astype(int).sum() / len(x)
        print(accuracy)


def train():
    samples = get_samples()
    model = Adaline()
    trainer = Trainer(model=model)
    trainer.fit(samples[:, :-1], samples[:, -1])


def main():
    train()


if __name__ == '__main__':
    main()
