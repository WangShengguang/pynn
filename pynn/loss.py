import numpy as np

from pynn.node import Node


class LossFunction(Node):
    pass


class PerceptionLoss(LossFunction):
    """
    感知机损失，输入为正时为0，输入为负时为输入的相反数
    """

    def compute(self):
        self.value = np.mat(np.where(self.parents[0].value >= 0.0, 0.0, -self.parents[0].value))

    def get_jacobi(self, parent):
        jacobi = np.where(parent.value >= 0.0, 0.0, -1)
        return np.diag(jacobi.ravel())


class LogLoss(LossFunction):
    def compute(self):
        assert len(self.parents) == 1
        x = self.parents[0].value
        self.value = np.log(1 + np.power(np.e, np.where(-x > 1e2, 1e2, -x)))

    def get_jacobi(self, parent):
        x = parent.value
        diag = -1 / (1 + np.power(np.e, np.where(x > 1e2, 1e2, x)))
        return np.diag(diag.ravel())


class Logistic(LossFunction):
    def compute(self):
        x = self.parents[0].value
        self.value = 1.0 / (1 + np.power(np.e, np.where(-x > 1e2, 1e2, -x)))

    def get_jacobi(self, parent):
        return np.diag(np.mat(np.multiply(self.value, 1 - self.value)).A1)


class SoftMax(LossFunction):
    @staticmethod
    def softmax(x):
        x[x > 1e2] = 1e2
        ep = np.power(np.e, x)
        return ep / np.sum(ep)

    def compute(self):
        self.value = SoftMax.softmax(self.parents[0].value)

    def get_jacobi(self, parent):
        """不使用softmax做损失，使用CrossEntropyWithSoftmax"""
        raise NotImplementedError


class CrossEntropyWithSoftmax(LossFunction):
    def compute(self):
        prob = SoftMax.softmax(self.parents[0].value)
        self.value = np.mat(-np.sum(np.multiply(self.parents[1].value, np.log(prob + 1e-10))))

    def get_jacobi(self, parent):
        prob = SoftMax.softmax(self.parents[0].value)
        if parent is self.parents[0]:
            return (prob - self.parents[1].value).T
        else:
            return (-np.log(prob)).T
