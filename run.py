import sys
sys.path.append('./utils/')

from dft import *
from fft import *
from rsa_custom import *
from misc import *
from image_functions import *
import time
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

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

size = int(input("Enter number of elements in point value representation: "))
assert isPowerOfTwo(size) == True

A = genPolynomial(size)
B = genPolynomial(size)

if (verbosity):
    print("A = ", A)
    print("B = ", B)


################################################################################
print("\n\n\n")
print(bcolors.HEADER + "Running DFT: " + bcolors.ENDC)
dft_A = dft(A)
dft_B = dft(B)

if (verbosity):
    print("dft_A = ", dft_A)
    print("dft_B = ", dft_B)

if (np.allclose(dft_A, np.fft.fft(A)) and np.allclose(dft_B, np.fft.fft(B))):
    print(bcolors.OKGREEN + "PASSED" + bcolors.ENDC + " DFT")
else:
    print(bcolors.FAIL + "FAILED" + bcolors.ENDC + " DFT")
################################################################################
print("\n\n\n")
print(bcolors.HEADER + "Running FFT: " + bcolors.ENDC)
fft_A = fft(A)
fft_B = fft(B)

if (verbosity):
    print("fft_A = ", fft_A)
    print("fft_B = ", fft_B)

if (np.allclose(fft_A, np.fft.fft(A)) and np.allclose(fft_B, np.fft.fft(B))):
    print(bcolors.OKGREEN + "PASSED" + bcolors.ENDC + " FFT")
else:
    print(bcolors.FAIL + "FAILED" + bcolors.ENDC + " FFT")
################################################################################
print("\n\n\n")
print(bcolors.HEADER + "Running Pointwise Multiplication: " + bcolors.ENDC)

print(bcolors.OKCYAN + bcolors.UNDERLINE + "Numpy:" + bcolors.ENDC + bcolors.ENDC)
numpy_multiply = fft_test(A, B)
if (verbosity):
    print(numpy_multiply)
print("Done")

print(bcolors.OKCYAN + bcolors.UNDERLINE + "Custom Implementation:" + bcolors.ENDC + bcolors.ENDC)
custom_multiply = multiply(A, B)
if (verbosity):
    print(custom_multiply)
print("Done")

if (np.allclose(numpy_multiply, custom_multiply)):
    print(bcolors.OKGREEN + "PASSED" + bcolors.ENDC + " Multiplication test with FFT")
else:
    print(bcolors.FAIL + "FAILED" + bcolors.ENDC + " Multiplication test with FFT")
################################################################################
print("\n\n\n")
print(bcolors.HEADER + "Running Encryption and Decryption: " + bcolors.ENDC)

pv = getPvForm(custom_multiply[:-1])
writeToFile('pv.txt', pv)

keys = chooseKeys()
encrypt('pv.txt', 'encrypted_pv.txt', keys['public_key'])
decrypt('encrypted_pv.txt', 'decrypted_pv.txt', keys['private_key'])

if (cmp('pv.txt', 'decrypted_pv.txt')):
    print(bcolors.OKGREEN + "PASSED" + bcolors.ENDC + " File encryption")
else:
    print(bcolors.FAIL + "FAILED" + bcolors.ENDC + " File encryption")
################################################################################
print("\n\n\n")
print(bcolors.HEADER + "Running Multiplication test with conventional for loop: " + bcolors.ENDC)

print(bcolors.OKCYAN + bcolors.UNDERLINE + "Brute force multiplication:" + bcolors.ENDC + bcolors.ENDC)
brute = polynomial_multiplication(A, B)
if (verbosity):
    print(brute)
print("Done")

if (np.allclose(brute, custom_multiply[:-1])):
    print(bcolors.OKGREEN + "PASSED" + bcolors.ENDC + " Multiplication test with conventional for loop")
else:
    print(bcolors.FAIL + "FAILED" + bcolors.ENDC + " Multiplication test with conventional for loop")
################################################################################
print("\n\n\n")
print(bcolors.HEADER + "Running 2D FFT and IFFT on random image: " + bcolors.ENDC)

original_image, fft_image = fftOnImage()
# TODO: make it grayscale image

if (np.allclose(original_image, fft_image)):
    print(bcolors.OKGREEN + "PASSED" + bcolors.ENDC + " 2D FFT and IFFT")
else:
    print(bcolors.FAIL + "FAILED" + bcolors.ENDC + " 2D FFT and IFFT")
################################################################################
print("\n\n\n")
print(bcolors.HEADER + "Running Image Compression: " + bcolors.ENDC)

compression_ratio = int(input("Enter compression ratio: "))
path = input("Enter path to image: ")
compressImage(path, compression_ratio)
# TODO: get image which is black and white to begin with
print(bcolors.OKCYAN + bcolors.UNDERLINE + f"Check {path} and compare with 'converted.jpg'" + bcolors.ENDC + bcolors.ENDC)
################################################################################
print("\n\n\n")
print(bcolors.HEADER + "Brute VS FFT VS DFT: " + bcolors.ENDC)

print(bcolors.OKCYAN + bcolors.UNDERLINE + "Brute force multiplication:" + bcolors.ENDC + bcolors.ENDC)
start_time = time.time()
brute = polynomial_multiplication(A, B)
print("--- %s seconds ---" % (time.time() - start_time))

print(bcolors.OKCYAN + bcolors.UNDERLINE + "FFT Numpy:" + bcolors.ENDC + bcolors.ENDC)
start_time = time.time()
fft_numpy = fft_test(A, B)
print("--- %s seconds ---" % (time.time() - start_time))

print(bcolors.OKCYAN + bcolors.UNDERLINE + "FFT custom:" + bcolors.ENDC + bcolors.ENDC)
start_time = time.time()
custom_multiply = multiply(A, B)
print("--- %s seconds ---" % (time.time() - start_time))

print(bcolors.OKCYAN + bcolors.UNDERLINE + "DFT:" + bcolors.ENDC + bcolors.ENDC)
start_time = time.time()
dft_A = dft(A)
dft_B = dft(B)
dft_C = np.multiply(dft_A, dft_B)
dft_y = idft(dft_C)
print("--- %s seconds ---" % (time.time() - start_time))
################################################################################
