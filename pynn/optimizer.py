import abc

import numpy as np

from pynn.graph import Graph
from pynn.node import Node, Variable


class Optimizer(object):
    def __init__(self, graph, target, learning_rate=0.01):
        assert isinstance(target, Node) and isinstance(graph, Graph)
        self.graph = graph
        self.target = target
        self.learning_rate = learning_rate
        self.acc_gradient = dict()
        self.acc_no = 0

    def one_step(self):
        self.forward_backward()
        self.acc_no += 1

    def get_gradient(self, node):
        """返回样本的平均梯度"""
        assert node in self.acc_gradient
        return self.acc_gradient[node] / self.acc_no

    @abc.abstractmethod
    def _update(self):
        """抽象方法，执行具体梯度更新算法，由子类具体实现"""

    def apply_gradients(self, node_gradients_dict, summarize=False, acc_no=None):
        pass

    def update(self, var_gradients=None):
        if var_gradients is not None:
            self.apply_gradients(var_gradients)
        self._update()
        self.acc_gradient.clear()
        self.acc_no = 0

    def forward_backward(self):
        self.graph.clear_jacobi()
        self.target.forward()

        for node in self.graph.nodes:
            if isinstance(node, Variable) and node.trainable:
                node.backward(self.target)
                gradient = node.jacobi.T.reshape(node.shape)
                if node not in self.acc_gradient:
                    self.acc_gradient[node] = gradient
                else:
                    self.acc_gradient[node] += gradient


class GradientDescent(Optimizer):
    """梯度下降法优化器"""

    def __init__(self, graph, target, learning_rate=0.01):
        super().__init__(graph, target)
        self.learning_rate = learning_rate

    def _update(self):
        for node in self.graph.nodes:
            if isinstance(node, Variable) and node.trainable:
                gradient = self.get_gradient(node)
                node.set_value(node.value - self.learning_rate * gradient)


class Momentum(Optimizer):

    def __init__(self, graph, target, learning_rate=0.01, momentum=0.9):
        super().__init__(graph, target)
        self.learning_rate = learning_rate
        #
        self.momentum = momentum
        self.v = dict()

    def _update(self):
        for node in self.graph.nodes:
            if isinstance(node, Variable) and node.trainable:
                gradient = self.get_gradient(node)
                if node not in self.v:
                    self.v[node] = -self.learning_rate * gradient
                else:
                    self.v[node] = self.momentum * self.v[node] - self.learning_rate * gradient
                node.set_value(node.value + self.v[node])


class AdaGrad(Optimizer):
    def __init__(self, graph, target, learning_rate=0.01, momentum=0.9):
        super().__init__(graph, target)
        self.learning_rate = learning_rate
        #
        self.momentum = momentum
        self.s = dict()

    def _update(self):
        for node in self.graph.nodes:
            if isinstance(node, Variable) and node.trainable:
                gradient = self.get_gradient(node)
                if node not in self.s:
                    self.s[node] = np.power(gradient, 2)
                else:
                    self.s[node] += np.power(gradient, 2)
                node.set_value(node.value - self.learning_rate * gradient / np.sqrt(self.s[node] + 1e-10))


class RMSProb(Optimizer):
    def __init__(self, graph, target, learning_rate=0.01, beta=0.9):
        super().__init__(graph, target)
        self.learning_rate = learning_rate
        #
        assert 0.0 < beta < 1.0
        self.beta = beta
        self.s = dict()

    def _update(self):
        for node in self.graph.nodes:
            if isinstance(node, Variable) and node.trainable:
                gradient = self.get_gradient(node)
                if node not in self.s:
                    self.s[node] = np.power(gradient, 2)
                else:
                    self.s[node] = self.beta * self.s[node] + (1 - self.beta) * np.power(gradient, 2)
                node.set_value(node.value - self.learning_rate * gradient / np.sqrt(self.s[node] + 1e-10))


class Adam(Optimizer):
    def __init__(self, graph, target, learning_rate=0.01,
                 beta_1=0.9, beta_2=0.99):
        super().__init__(graph, target)
        self.learning_rate = learning_rate
        #
        assert 0.0 < beta_1 < 1.0 and 0.0 < beta_2 < 1.0
        self.beta_1 = beta_1
        self.v = dict()
        self.beta_2 = beta_2
        self.s = dict()

    def _update(self):
        for node in self.graph.nodes:
            if isinstance(node, Variable) and node.trainable:
                gradient = self.get_gradient(node)
                if node not in self.s:
                    self.v[node] = gradient
                    self.s[node] = np.power(gradient, 2)
                else:
                    self.v[node] = self.beta_1 * self.v[node] + (1 - self.beta_1) * gradient
                    self.s[node] = self.beta_2 * self.s[node] + (1 - self.beta_2) * np.power(gradient, 2)
                node.set_value(node.value - self.learning_rate * self.v[node] / np.sqrt(self.s[node] + 1e-10))
