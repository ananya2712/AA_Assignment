import sys
sys.path.append('./utils/')

from dft import *
from fft import *
from rsa_custom import *
from misc import *
from image_functions import *

import random
import numpy as np
from filecmp import cmp
import argparse
import warnings
warnings.simplefilter("ignore", np.ComplexWarning)

parser = argparse.ArgumentParser(description='Set verbosity.')
parser.add_argument('--v', default = 1, help = 'increase verbosity of program')
args = parser.parse_args()
verbosity = (args.v == 1)

size = int(input("Enter number of elements in point value representation: "))
assert isPowerOfTwo(size) == True

A = genPolynomial(size)
B = genPolynomial(size)

print("A = ", A)
print("B = ", B)

################################################################################
print("\n\n\n")
print("\033[1mRunning DFT: \033[0m")
dft_A = dft(A)
dft_B = dft(B)

if (verbosity):
    print("dft_A = ", dft_A)
    print("dft_B = ", dft_B)

if (np.allclose(dft_A, np.fft.fft(A)) and np.allclose(dft_B, np.fft.fft(B))):
    print("\033[92mPASSED\033[0m DFT")
else:
    print("\033[91mFAILED\033[0m DFT")
################################################################################
print("\n\n\n")
print("\033[1mRunning FFT: \033[0m")
fft_A = fft(A)
fft_B = fft(B)

if (verbosity):
    print("fft_A = ", fft_A)
    print("fft_B = ", fft_B)

if (np.allclose(fft_A, np.fft.fft(A)) and np.allclose(fft_B, np.fft.fft(B))):
    print("\033[92mPASSED\033[0m FFT")
else:
    print("\033[91mFAILED\033[0m FFT")
################################################################################
print("\n\n\n")
print("\033[1mRunning Pointwise Multiplication: \033[0m")

print("\033[96mNumpy: \033[0m")
numpy_multiply = fft_test(A, B)
if (verbosity):
    print(numpy_multiply)
print("Done")

print("\033[96mCustom Implementation: \033[0m")
custom_multiply = multiply(A, B)
if (verbosity):
    print(custom_multiply)
print("Done")

if (np.allclose(numpy_multiply, custom_multiply)):
    print("\033[92mPASSED\033[0m Multiplication test with FFT")
else:
    print("\033[91mFAILED\033[0m Multiplication test with FFT")
################################################################################
print("\n\n\n")
print("\033[1mRunning Encryption and Decryption: \033[0m")

pv = getPvForm(custom_multiply[:-1])
writeToFile('pv.txt', pv)

keys = chooseKeys()
encrypt('pv.txt', 'encrypted_pv.txt', keys['public_key'])
decrypt('encrypted_pv.txt', 'decrypted_pv.txt', keys['private_key'])

if (cmp('pv.txt', 'decrypted_pv.txt')):
    print("\033[92mPASSED\033[0m File Encryption")
else:
    print("\033[91mFAILED\033[0m File Encryption")
################################################################################
print("\n\n\n")
print("\033[1mRunning Multiplication test with conventional for loop: \033[0m")

print("\033[96mBrute force multiplication: \033[0m")
brute = polynomial_multiplication(A, B)
if (verbosity):
    print(brute)
print("Done")

if (np.allclose(brute, custom_multiply[:-1])):
    print("\033[92mPASSED\033[0m Multiplication test with conventional for loop")
else:
    print("\033[91mFAILED\033[0m Multiplication test with conventional for loop")
################################################################################
print("\n\n\n")
print("\033[1mRunning 2D FFT and IFFT on random image: \033[0m")

original_image, fft_image = fftOnImage()
# TODO: make it grayscale image

if (np.allclose(original_image, fft_image)):
    print("\033[92mPASSED\033[0m 2D FFT and IFFT")
else:
    print("\033[91mFAILED\033[0m 2D FFT and IFFT")
################################################################################
print("\n\n\n")
print("\033[1mRunning Image Compression: \033[0m")

compression_ratio = int(input("Enter compression ratio: "))
path = input("Enter path to image: ")
imgToFFT(path, compression_ratio)
# TODO: get image which is black and white to begin with
print(f"\033[96mCheck {path} and compare with 'converted.jpg'\033[0m")
################################################################################
