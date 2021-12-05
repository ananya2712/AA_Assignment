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
    x = [random.randint(1, 10) for i in range(n)]
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

first_primes_list = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29,
                     31, 37, 41, 43, 47, 53, 59, 61, 67,
                     71, 73, 79, 83, 89, 97, 101, 103,
                     107, 109, 113, 127, 131, 137, 139,
                     149, 151, 157, 163, 167, 173, 179,
                     181, 191, 193, 197, 199, 211, 223,
                     227, 229, 233, 239, 241, 251, 257,
                     263, 269, 271, 277, 281, 283, 293,
                     307, 311, 313, 317, 331, 337, 347, 349]

def nBitRandom(n):
    start = binary_exponentiation(2, n - 1, -1)
    return random.randrange(start+1, 2*start - 1)

def getLowLevelPrime(n):
    while True:
        pc = nBitRandom(n)
        for divisor in first_primes_list:
            if pc % divisor == 0 and divisor**2 <= pc:
                break
        else: return pc

def isMillerRabinPassed(mrc):
    maxDivisionsByTwo = 0
    ec = mrc-1
    while ec % 2 == 0:
        ec >>= 1
        maxDivisionsByTwo += 1
    assert(2**maxDivisionsByTwo * ec == mrc-1)

    def trialComposite(round_tester):
        if pow(round_tester, ec, mrc) == 1:
            return False
        for i in range(maxDivisionsByTwo):
            if pow(round_tester, 2**i * ec, mrc) == mrc-1:
                return False
        return True
    numberOfRabinTrials = 20
    for i in range(numberOfRabinTrials):
        round_tester = random.randrange(2, mrc)
        if trialComposite(round_tester):
            return False
    return True

def getPrime(n):
    while(True):
        prime_candidate = getLowLevelPrime(n)
        if not isMillerRabinPassed(prime_candidate):
            continue
        else:
            return prime_candidate
