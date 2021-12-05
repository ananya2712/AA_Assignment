import math
from Crypto.Util import number
from misc import *

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
