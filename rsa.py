import math

def xgcd(a, b):
    x, old_x = 0, 1
    y, old_y = 1, 0

    while (b != 0):
        quotient = a // b
        a, b = b, a - quotient * b
        old_x, x = x, old_x - quotient * x
        old_y, y = y, old_y - quotient * y

    return a, old_x, old_y

def binary_exponentiation(base, power, mod):
    result = 1
    while (power):
        if (power & 1):
            result = (result * base) % mod
        base = (base * base) % mod
        power >>= 1
    return result

def chooseE(totient):
    for i in range(3, totient, 2):
        if (math.gcd(i, totient) == 1):
            return i

def chooseKeys():
    prime1 = 100000007
    prime2 = 998244353
    n = prime1 * prime2
    totient = (prime1 - 1) * (prime2 - 1)
    e = chooseE(totient)
    _, x, y = xgcd(e, totient)
    d = ((x + totient) % totient)

    return {'public_key': (e, n), 'private_key': (d, n)}

def encrypt(message, key):
    e, n = key

    encrypted_blocks = []
    ciphertext = ord(message[0])

    for i in range(len(message)):
        ciphertext = ord(message[i])
        encrypted_blocks.append(str(binary_exponentiation(ciphertext, e, n)))

    encrypted_message = " ".join(encrypted_blocks)
    return encrypted_message

def decrypt(blocks, key, block_size = 1):
    d, n = key

    list_blocks = blocks.split(' ')
    message = ""
    for i in range(len(list_blocks)):
        message += chr(binary_exponentiation(int(list_blocks[i]), d, n))

    return message

keys = chooseKeys()
print(decrypt(encrypt("Hello world.", keys['public_key']), keys['private_key']))
