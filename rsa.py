def gcd(a, b):
    if (b == 0):
        return a
    else:
        return gcd(b, a % b)

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
        if (gcd(i, totient) == 1):
            return i

def chooseKeys():
    prime1 = 100000007
    prime2 = 998244353
    n = prime1 * prime2
    totient = (prime1 - 1) * (prime2 - 1)
    e = chooseE(totient)
    gcd, x, y = xgcd(e, totient)
    d = ((x + totient) % totient)

    return {'public_key': (e, n), 'private_key': (d, n)}

def encrypt(message, key, block_size = 2):
    n = key[1]
    e = key[0]

    encrypted_blocks = []
    ciphertext = -1

    if (len(message) > 0):
        ciphertext = ord(message[0])

    for i in range(1, len(message)):
        if (i % block_size == 0):
            encrypted_blocks.append(ciphertext)
            ciphertext = 0
        ciphertext = ciphertext * 1000 + ord(message[i])
    encrypted_blocks.append(ciphertext)
    for i in range(len(encrypted_blocks)):
        encrypted_blocks[i] = str(binary_exponentiation(encrypted_blocks[i], e, n))
    encrypted_message = " ".join(encrypted_blocks)
    return encrypted_message

def decrypt(blocks, key, block_size = 2):
    n = key[1]
    d = key[0]
    list_blocks = blocks.split(' ')
    int_blocks = []

    for s in list_blocks:
        int_blocks.append(int(s))
    message = ""

    for i in range(len(int_blocks)):
        int_blocks[i] = binary_exponentiation(int_blocks[i], d, n)
        tmp = ""
        for c in range(block_size):
            tmp = chr(int_blocks[i] % 1000) + tmp
            int_blocks[i] //= 1000
        message += tmp
    return message
keys = chooseKeys()

print(decrypt(encrypt("Hello world.", keys['public_key']), keys['private_key']))
