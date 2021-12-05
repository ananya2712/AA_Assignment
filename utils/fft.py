import numpy as np
from misc import *

def fft(x):
    """
        Fast fourier transform of x
    """
    x = np.array(x, copy = False, dtype=float)
    size = x.shape[0]
    if isPowerOfTwo(x.shape[0]) == False:
        raise ValueError(f"size must be a power of 2.")
        return
    if (size == 1):
        return x
    x_even, x_odd = fft(x[::2]), fft(x[1::2])
    complexRoots = getComplexRoots(1, np.arange(size), size)
    return np.concatenate([x_even + complexRoots[:size//2] * x_odd, x_even + complexRoots[size//2:] * x_odd])

def ifft(y):
    size = len(y)
    if (size) == 1:
        return y
    if isPowerOfTwo(size) == False:
        raise ValueError(f"size must be a power of 2.")
        return
    w, y_even, y_odd, A = 1/getComplexRoots(1, 1, size), ifft(y[::2]), ifft(y[1::2]), [0]*size
    for i in range(size//2):
        A[i] = (y_even[i] + (w**i)*y_odd[i])/2
        A[i+size//2] = (y_even[i] - (w**i)*y_odd[i])/2
    return A

def multiply(a,b):
    """
        CR to CR multiplication using FFT
    """
    a_p2 = np.pad(a, (0, len(a)), 'constant')
    b_p2 = np.pad(b, (0, len(b)), 'constant')
    y_c = np.multiply(fft(a_p2), fft(b_p2))

    C = [i.real for i in ifft(y_c)]
    return C

def fft_test(arr_a,arr_b):
    """
        fft based polynomial multiplication using numpy functions
    """
    length = 4
    arr_a1=np.pad(arr_a,(0,length),'constant')
    arr_b1=np.pad(arr_b,(0,length),'constant')
    a_f=np.fft.fft(arr_a1)
    b_f=np.fft.fft(arr_b1)

    c_f=[0]*(2*length)

    for i in range( len(a_f) ):
        c_f[i]=a_f[i]*b_f[i]

    C = np.fft.ifft(c_f)
    C = [i.real for i in C]
    return C

def fft2(A):
    """
        2 d fft
    """
    A = np.matrix(A)

    if len(A.shape) != 2:
        raise ValueError("Input must be of 2 dimensions")
        return

    for size in A.shape:
        if isPowerOfTwo(size)==False:
            raise ValueError("Dimensions must be a power of 2")
            return


    y_r = np.array([fft(row) for row in A])
    y_rc = np.array([fft(row) for row in y_r.T]).T

    return y_rc


def ifft2(y):
    """
        2 d inverse fft
    """
    y = np.matrix(y)

    if len(y.shape) != 2:
        raise ValueError("Input must be of 2 dimensions")
        return

    for size in y.shape:
        if isPowerOfTwo(size)==False:
            raise ValueError("Dimensions must be a power of 2")
            return

    A_r = np.array([ifft(row) for row in y])
    A_rc = np.array([ifft(row) for row in A_r.T]).T
    A_rc = np.reshape(A_rc, (A_rc.shape[0], A_rc.shape[2]))
    return A_rc
