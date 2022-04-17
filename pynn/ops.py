import numpy as np

from pynn.node import Node


def fill_diagonal(to_be_filled, filler):
    """
    将 filler 矩阵填充在 to_be_filled 的对角线上
    """
    assert to_be_filled.shape[0] / \
           filler.shape[0] == to_be_filled.shape[1] / filler.shape[1]
    n = int(to_be_filled.shape[0] / filler.shape[0])

    r, c = filler.shape
    for i in range(n):
        to_be_filled[i * r:(i + 1) * r, i * c:(i + 1) * c] = filler

    return to_be_filled


class Operator(Node):
    def __init__(self, *parents, **kwargs):
        super(Operator, self).__init__(*parents, **kwargs)


class MatMul(Operator):
    """矩阵乘法"""

    def compute(self):
        assert len(self.parents) == 2 and self.parents[0].shape[1] == self.parents[1].shape[0]
        self.value = self.parents[0].value * self.parents[1].value

    def get_jacobi(self, parent):
        """"""
        zeros = np.mat(np.zeros((self.dimension, parent.dimension)))
        if parent is self.parents[0]:
            return fill_diagonal(zeros, self.parents[1].value.T)
        else:  # parent[1]
            jacobi = fill_diagonal(zeros, self.parents[0].value)
            # breakpoint()
            row_sort = np.arange(self.dimension).reshape(self.shape[::-1]).T.ravel()
            col_sort = np.arange(parent.dimension).reshape(parent.shape[::-1]).T.ravel()
            return jacobi[row_sort, :][:, col_sort]


class Add(Operator):
    """矩阵加法"""

    def compute(self):
        self.value = np.mat(np.zeros(self.parents[0].shape))
        for parent in self.parents:
            self.value += parent.value

    def get_jacobi(self, parent):
        return np.mat(np.eye(self.dimension))  # 矩阵之和对其中任一个矩阵的雅可比矩阵是单位矩阵


class Step(Operator):
    def compute(self):
        self.value = np.mat(np.where(self.parents[0].value > 0.0, 1, 0))

    def get_jacobi(self, parent):
        assert parent in self.parents
        return np.mat(np.where(self.parents[0].value.A1 > 0.0, 0.0, -1.0))


class ScalarMultiply(Operator):
    def compute(self):
        assert self.parents[0].shape == (1, 1)
        self.value = np.multiply(self.parents[0].value, self.parents[1].value)

    def get_jacobi(self, parent):
        assert parent in self.parents
        if parent is self.parents[0]:
            return self.parents[1].value.flatten().T
        else:
            return np.mat(np.eye(self.parents[1].dimension)) * self.parents[0].value[0, 0]


class Multiply(Operator):
    def compute(self):
        self.value = np.multiply(self.parents[0].value, self.parents[1].value)

    def get_jacobi(self, parent):
        if parent is self.parents[0]:
            return np.diag(self.parents[1].value.A1)
        else:
            return np.diag(self.parents[0].value.A1)


class Reshape(Operator):
    """
    改变父节点的值（矩阵）的形状
    """

    def __init__(self, *parent, **kargs):
        Operator.__init__(self, *parent, **kargs)

        self.to_shape = kargs.get('shape')
        assert isinstance(self.to_shape, tuple) and len(self.to_shape) == 2

    def compute(self):
        self.value = self.parents[0].value.reshape(self.to_shape)

    def get_jacobi(self, parent):
        assert parent is self.parents[0]
        return np.mat(np.eye(self.dimension ))


class Relu(Operator):
    def compute(self):
        self.value = np.mat(np.where(self.parents[0].value > 0.0, self.parents[0].value, 0.0))

    def get_jacobi(self, parent):
        return np.diag(np.where(self.parents[0].value.A1 > 0.0, 1.0, 0))


class LeakyRelu(Operator):
    def compute(self):
        self.value = np.mat(np.where(
            self.parents[0].value > 0.0,
            self.parents[0].value,
            0.1 * self.parents[0].value))

    def get_jacobi(self, parent):
        return np.diag(np.where(self.parents[0].value.A1 > 0.0, 1.0, 0.1))
