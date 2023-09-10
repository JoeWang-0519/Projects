import numpy as np
import scipy
from PIL import Image
import argparse
from scipy.signal import convolve2d
from matplotlib import pyplot as plt
import math

### firstly, write function for 1-level 2d haar DWT(Discrete Wavelet Transform)
def haar2dlvl1(im):
    # Parameters:
    # im: (2**N) * (2**N) array, image grey array
    # Returens:
    # out: (2**N) * (2**N) array, Haar wavelet coefficients with level 1

    # Main function:
    ## Here, no need to do padding since our assumption is, the image is (2**N) * (2**N)

    a, _ = im.shape
    out = np.zeros((a, a))
    H = np.array([[1, 1], [1, 1]]) / 2
    G1 = np.array([[-1, -1], [1, 1]]) / 2
    G2 = np.array([[-1, 1], [-1, 1]]) / 2
    G3 = np.array([[1, -1], [-1, 1]]) / 2

    H_rev, G1_rev, G2_rev, G3_rev = H[::-1, ::-1], G1[::-1, ::-1], G2[::-1, ::-1], G3[::-1, ::-1]

    out[0: int(a / 2), 0: int(a / 2)] = convolve2d(im, H_rev)[1::2, 1::2]
    out[int(a / 2):, 0: int(a / 2)] = convolve2d(im, G1_rev)[1::2, 1::2]
    out[0: int(a / 2), int(a / 2):] = convolve2d(im, G2_rev)[1::2, 1::2]
    out[int(a / 2):, int(a / 2):] = convolve2d(im, G3_rev)[1::2, 1::2]

    return out


def haar2d(im, lvl):
    # Computing 2D discrete Haar wavelet transform of a given ndarray im.
    # Parameters:
    #   im: ndarray.    An array representing image
    #   lvl: integer.   An integer representing the level of wavelet decomposition
    #  Returns:
    #   out: ndarray.   An array representing Haar wavelet coefficients with lvl level. It has the same shape as im

    # ----
    # Insert your code here
    a, _ = im.shape

    ### we cannot directly use 'out=im' since it will assign 'uint8' data type to each entry

    # out = np.zeros((a,a))
    # out = im.copy()

    for i in range(lvl):
        tmp = 2 ** (i)
        scale = int(a / tmp)
        # if i == 0:
        #    input_haarlvl1 = im
        # else:
        #    input_haarlvl1 = out[0:scale, 0:scale]
        # out[0:scale, 0:scale] = haar2dlvl1(input_haarlvl1)
        if i == 0:
            out = haar2dlvl1(im)
        else:
            out[0:scale, 0:scale] = haar2dlvl1(out[0:scale, 0:scale])
    # ----
    return out


def ihaar2dlvl1(coef):
    a, _ = coef.shape

    H = np.array([[1, 1], [1, 1]]) / 2
    G1 = np.array([[-1, -1], [1, 1]]) / 2
    G2 = np.array([[-1, 1], [-1, 1]]) / 2
    G3 = np.array([[1, -1], [-1, 1]]) / 2

    tmp = np.zeros((a, a))
    tmp[0::2, 0::2] = coef[0:int(a / 2), 0:int(a / 2)]
    out = convolve2d(tmp, H)[0:-1, 0:-1]
    tmp[0::2, 0::2] = coef[int(a / 2):, 0:int(a / 2)]
    out = out + convolve2d(tmp, G1)[0:-1, 0:-1]
    tmp[0::2, 0::2] = coef[0:int(a / 2), int(a / 2):]
    out = out + convolve2d(tmp, G2)[0:-1, 0:-1]
    tmp[0::2, 0::2] = coef[int(a / 2):, int(a / 2):]
    out = out + convolve2d(tmp, G3)[0:-1, 0:-1]

    return out


def ihaar2d(coef, lvl):
    # Computing an image in the form of ndarray from the ndarray coef which represents its DWT coefficients.
    # Parameters:
    #   coef: ndarray   An array representing 2D Haar wavelet coefficients
    #   lvl: integer.   An integer representing the level of wavelet decomposition
    #  Returns:
    #   out: ndarray.   An array representing the image reconstructed from its Haar wavelet coefficients.

    # ----
    # Insert your code here
    a, _ = coef.shape
    out = coef.copy()
    for i in range(lvl - 1, -1, -1):
        tmp = 2 ** (i)
        scale = int(a / tmp)
        out[0:scale, 0:scale] = ihaar2dlvl1(out[0:scale, 0:scale])
    # ----
    return np.uint8(out)

if __name__ == "__main__":
# Code for testing.
# Please modify the img_path to the path stored image and the level of wavelet decomposition.
# Feel free to revise the codes for your convenience
# If you have any question, please send email to e0444157@u.nus.edu for help
# As the hw_1.pdf mentioned, you can also write the test codes on other .py file.

    parser = argparse.ArgumentParser(description="wavelet")
    parser.add_argument("--img_path",  type=str, default='./test.png',  help='The test image path')
    parser.add_argument("--level", type=int, default=4, help="The level of wavelet decomposition")
    parser.add_argument("--save_pth", type=str, default='./recovery.png', help="The save path of reconstructed image ")
    opt = parser.parse_args()

    img_path = opt.img_path # The test image path
    level = opt.level # The level of wavelet decomposition
    save_pth = opt.save_pth

    img = np.array(Image.open(img_path).convert('L'))
    haar2d_coef = haar2d(img,level)
    recovery =  Image.fromarray(ihaar2d(haar2d_coef,level),mode='L')
    recovery.save(save_pth)
    np.save('./haar2_coeff.npy',haar2d_coef)
    
    