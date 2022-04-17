"""
JVP (Jacobian vector product)

"""
import jax.numpy as np
from jax import grad

# WARNING:absl:No GPU/TPU found, falling back to CPU. (Set TF_CPP_MIN_LOG_LEVEL=0 and rerun for more info.)

x = np.array([1, 2, 3])
print(x)


# print(grad(x))

def func(x, y):
    return x * y


print(grad(func)(1.0, 2.0))
