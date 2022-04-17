from pynn.loss import Logistic
from pynn.node import Variable
from pynn.ops import MatMul, Add, Relu


def fc(input, input_size, size, activation):
    weights = Variable((size, input_size), init=True, trainable=True)
    bias = Variable((size, 1), init=True, trainable=True)
    affine = Add(MatMul(weights, input), bias)
    if activation == 'Relu':
        return Relu(affine)
    elif activation == 'Logistic':
        return Logistic(affine)
    else:
        return affine
