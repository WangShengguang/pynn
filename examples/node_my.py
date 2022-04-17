import abc
from typing import List

import numpy as np

from pynn.graph import default_graph


class Node(object):
    """
    计算图节点类基类
    """

    def __init__(self, *parents, **kwargs):
        self.graph = kwargs.get('graph', default_graph)  # 计算图，默认为全局计算图
        self.gen_node_name(**kwargs)
        #
        self.parents: List[Node] = list(parents)  # 父节点列表
        self.children: List[Node] = []  # 子节点列表
        self.value: np.matrix = None  # 本节点的值
        self.jacobi: np.matrix = None  # 结果节点对本节点的雅克比矩阵
        # append都是self
        for node in self.parents:
            node.children.append(self)  # 节点添加到父节点的
        self.graph.add_node(self)  # 节点添加到计算图

    def get_parents(self):
        return self.parents

    def get_children(self):
        return self.children

    def gen_node_name(self, **kwargs):
        default_name = '{}:{}'.format(self.__class__.__name__, self.graph.node_count)
        self.name = kwargs.get('name', default_name)
        if self.graph.name_scope:
            self.name = '{}/{}'.format(self.graph.name_scope, self.name)

    def forward(self):
        for node in self.parents:
            if node.value is None:
                node.forward()
        self.compute()

    @abc.abstractmethod
    def compute(self):
        """抽象方法，根据父节点的值计算本节点的值"""

    @abc.abstractmethod
    def get_jacobi(self, parent):
        """抽象方法，计算本节点对某个父节点的雅克比矩阵"""

    def backward(self, result):
        """反向传播，计算结果节点对本节点的雅克比矩阵"""
        if self.jacobi is None:
            if self is result:
                self.jacobi = np.mat(np.eye(self.dimension))
            else:
                self.jacobi = np.mat(np.zeros((result.dimension, self.dimension)))
                for child in self.get_children():
                    if child.value is not None:
                        # breakpoint()
                        self.jacobi += child.backward(result) * child.get_jacobi(self)
        return self.jacobi

    def clear_jacobi(self):
        self.jacobi = None

    @property
    def shape(self):
        return self.value.shape

    @property
    def dimension(self):
        return self.value.shape[0] * self.value.shape[1]

    def reset_value(self, recursive=True):
        """重置本节点的值，递归重置本节点的下游节点的值"""
        self.value = None
        if recursive:
            for child in self.children:
                child.reset_value()


class Variable(Node):
    def __init__(self, shape, init=False, trainable=True, **kwargs):
        # Node.__init__(self, **kwargs)
        super(Variable, self).__init__(**kwargs)
        self.dim = shape
        if init:
            self.value = np.mat(np.random.normal(0, 0.001, size=self.dim))
        self.trainable = trainable

    def set_value(self, value):
        # assert isinstance(value, np.matrix) and value.shape == self.dim, breakpoint()
        if not (isinstance(value, np.matrix) and value.shape == self.dim):
            # breakpoint()
            # print(value)
            raise ValueError
        self.reset_value(recursive=True)  # 本节点被改变， 重置下游所有节点的值
        self.value = value
