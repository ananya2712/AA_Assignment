import math
from Crypto.Util import number


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

def chooseE(totient):
    for i in range(3, totient, 2):
        if (math.gcd(i, totient) == 1):
            return i

def chooseKeys():
    bits = int(input("Enter number of bits in RSA key: "))
    bits >>= 1
    prime1 = number.getPrime(bits)
    prime2 = number.getPrime(bits)
    while (prime2 == prime1):
        prime2 = number.getPrime(bits)
    n = prime1 * prime2
    totient = (prime1 - 1) * (prime2 - 1)
    e = chooseE(totient)
    _, x, y = xgcd(e, totient)
    d = ((x + totient) % totient)

    return {'public_key': (e, n), 'private_key': (d, n)}

def encrypt(file_read, file_write, key):
    fo = open(file_read, 'r')
    message = fo.read()
    fo.close()

    e, n = key
    encrypted_blocks = []

    for i in message:
        encrypted_blocks.append(str(binary_exponentiation(ord(i), e, n)))

    encrypted_message = " ".join(encrypted_blocks)

    fo = open(file_write, 'w')
    fo.write(encrypted_message)
    fo.close()

def decrypt(file_read, file_write, key):
    d, n = key

    fo = open(file_read, 'r')
    blocks = fo.read()
    fo.close()

    list_blocks = blocks.split(' ')
    message = ""
    for i in range(len(list_blocks)):
        message += chr(binary_exponentiation(int(list_blocks[i]), d, n))

    fo = open(file_write, 'w')
    fo.write(message)
    fo.close()
