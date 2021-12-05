from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
from misc import *
from fft import *

def fftOnImage():
    imarray = getImageToPowerOfTwo(generateImage())
    img = ifft2(fft2(imarray))
    plt.imshow(imarray)
    plt.show()
    plt.imshow(img)
    plt.show()
    return (imarray, img)

class ImageCompressor:

    def __init__(self,img,compression_ratio):
        self.img = np.array(img)
        assert compression_ratio < 100 and compression_ratio >= 0
        self.compression_ratio = compression_ratio

    def getThreshold(self,A, compression_ratio):
        self.y_A = np.fft.fft2(A)
        flat = self.y_A.flatten()
        flat.sort()
        return flat[int(len(self.y_A)*(100-self.compression_ratio)/100)]

    def render(self):
        threshold = self.getThreshold(self.img, self.compression_ratio)

        for r in range(len(self.y_A)):
            for c in range(len(self.y_A[r])):
                if (self.y_A[r][c].real < threshold):
                    self.y_A[r][c] = 0

        A = np.fft.ifft2(self.y_A).real

        return A



def compressImage(path, compression_ratio):
    img = Image.open(path)
    img = img.convert('L')
    img = np.array(img)
    fft_img = ImageCompressor(img, compression_ratio)
    fft_img_c = fft_img.render()
    cv2.imwrite("converted.jpg", fft_img_c)
