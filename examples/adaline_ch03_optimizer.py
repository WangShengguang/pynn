import numpy as np

from pynn.graph import default_graph
from pynn.loss import PerceptionLoss
from pynn.node import Variable
from pynn.ops import Add, MatMul, Step
from pynn.optimizer import GradientDescent, AdaGrad, Momentum, RMSProb, Adam

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


class Adaline(object):
    def __init__(self):
        #
        self.X = Variable(shape=(3, 1), init=False, trainable=False)
        self.Y = Variable(shape=(1, 1), init=False, trainable=False)
        #
        self.W = Variable(shape=(1, 3), init=True, trainable=True)
        self.b = Variable(shape=(1, 1), init=True, trainable=True)
        #
        self.output = Add(MatMul(self.W, self.X), self.b)
        self.predict = Step(self.output)
        self.loss = PerceptionLoss(MatMul(self.Y, self.output))


class Trainer(object):
    def __init__(self, model: Adaline, learning_rate=0.0001, batch_size=8):
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.model = model

    def fit(self, x, y, num_epochs=50):
        optimizer = GradientDescent(default_graph, self.model.loss, self.learning_rate)  # 0.834
        optimizer = Momentum(default_graph, self.model.loss, self.learning_rate)  # 0.87
        # optimizer = AdaGrad(default_graph, self.model.loss, self.learning_rate) # 0.965
        optimizer = RMSProb(default_graph, self.model.loss, self.learning_rate)  # 0.844
        optimizer = Adam(default_graph, self.model.loss, self.learning_rate)  # 0.962 总体好

        for epoch in range(num_epochs):
            batch_count = 0
            for i in range(len(x)):
                features = np.mat(x[i]).T
                self.model.X.set_value(features)
                # breakpoint()
                self.model.Y.set_value(np.mat(y[i]).T)
                #
                optimizer.one_step()
                batch_count += 1
                if batch_count % self.batch_size == 0:
                    optimizer.update()
                    batch_count = 0

            self.evaluate(x, y)

    def evaluate(self, x, y):
        pred = []
        for i in range(len(x)):
            features = np.mat(x[i]).T
            self.model.X.set_value(features)
            self.model.predict.forward()
            pred.append(self.model.predict.value[0, 0])
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
