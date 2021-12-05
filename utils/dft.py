import numpy as np
from misc import *

def dft(x):
    """
        Discrete fourier transform of x
    """
    x = np.array(x, copy = False, dtype=float)
    size = x.shape[0]
    n = np.arange(size)
    return np.dot(getComplexRoots(n.reshape((size, 1)), n, size), x)

def idft2(y):
    size = len(y)

    if isPowerOfTwo(size) == False:
        raise ValueError(f"size must be a power of 2.")
        return

    y = np.array(y, copy=False)
    y.reshape((size, 1))
    return np.matmul(np.array([[(1/getComplexRoots(i, j, size))/size for j in range(size)] for i in range(size)]), y).flatten()


