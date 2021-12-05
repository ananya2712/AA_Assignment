import numpy as np
import random

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

def genPolynomial(n):
    x = [random.randint(1, 100) for i in range(n)]
    return x

def horner(a, x):
    result = 0
    for i in range(len(a)-1, -1, -1):
        result = a[i] + (x * result)
    return result

def getPvForm(A):
    pv = []
    for x in range(1, len(A) + 1):
        y = horner(A, x)
        pv.append((x, y))
    return pv

def writeToFile(path, text):
    fo = open(path, 'w')
    for x, y in text:
        fo.write(f"{str(x)} : {str(y)} \n")
    fo.close()

def generateImage():
    return np.reshape(np.random.random(32*64),(32,64))

def getImageToPowerOfTwo(img):
    return img[:2**int(np.log2(img.shape[0])),:2**int(np.log2(img.shape[1]))]

def binary_exponentiation(base, power, mod):
    result = 1
    while (power):
        if (power & 1):
            if (mod != -1):
                result = (result * base) % mod
            else:
                result *= base
        if (mod != -1):
            base = (base * base) % mod
        else:
            base *= base
        power >>= 1
    return result

def xgcd(a, b):
    x, old_x = 0, 1
    y, old_y = 1, 0

    while (b != 0):
        quotient = a // b
        a, b = b, a - quotient * b
        old_x, x = x, old_x - quotient * x
        old_y, y = y, old_y - quotient * y

    return a, old_x, old_y

def polynomial_multiplication(P, Q):
    """
        Brute force multiplication
    """
    m = len(P)
    n = len(Q)
    result = [0]*(m+n-1)
    for i in range(m):
        for j in range(n):
            result[i+j] += P[i]*Q[j]
    return result
