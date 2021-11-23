import numpy as np
import rsa
from numpy import polymul, pad
import warnings
warnings.simplefilter("ignore", np.ComplexWarning)

def getComplexRoots(k, n, m):
    """
        Returns complex roots of 1
    """
    return np.exp(-2j * np.pi * k * n / m)

def isPowerOfTwo(x):
    """
        Checks if x is a perfect power of 2
    """
    return ((x & (x - 1)) == 0)

def dft(x):
    """
        Discrete fourier transform of x
    """
    x = np.array(x, copy = False, dtype=float)
    size = x.shape[0]
    n = np.arange(size)
    return np.dot(getComplexRoots(n.reshape((size, 1)), n, size), x)

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

def fft_test(arr_a,arr_b):  #fft based polynomial multiplication
    length = 4
    arr_a1=np.pad(arr_a,(0,length),'constant')
    arr_b1=np.pad(arr_b,(0,length),'constant')
    a_f=np.fft.fft(arr_a1)
    b_f=np.fft.fft(arr_b1)

    c_f=[0]*(2*length)

    for i in range( len(a_f) ):
        c_f[i]=a_f[i]*b_f[i]

    return np.fft.ifft(c_f)


def polynomial_multiplication(P, Q):
    m = len(P)
    n = len(Q)
    result = [0]*(m+n-1)
    for i in range(m):
        for j in range(n):
            result[i+j] += P[i]*Q[j]
    return result

# print(fft_test([1, 2, 3, 4], [1, 2, 3, 4]))
# print(multiply([1, 2, 3, 4], [1, 2, 3, 4]))
# print(polynomial_multiplication([1, 2, 3, 4], [1, 2, 3, 4]))


BITS = 1024
publicKey, privateKey = rsa.newkeys(BITS)

def rsa_encryption(message, publicKey):
    return rsa.encrypt(message.encode(),publicKey)
def rsa_decryption(message, privateKey):
    return rsa.decrypt(message, privateKey).decode()

# m = "Hello world"
# print(m)
# em = rsa_encryption(m, publicKey)
# print(em)
# print(rsa_decryption(em, privateKey))

def fft2(A):
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
    y = np.matrix(y)

    if len(y.shape) != 2:
        raise ValueError("Input must be of 2 dimensions")
        return

    for size in A.shape:
        if isPowerOfTwo(size)==False:
            raise ValueError("Dimensions must be a power of 2")
            return

    A_r = np.array([ifft(row) for row in y])
    A_rc = np.array([ifft(row) for row in A_r.T]).T

    return A_rc

# print(ifft2(fft2([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]])))
