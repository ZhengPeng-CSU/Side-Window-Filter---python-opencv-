#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:Pengzheng.1997

import cv2
import numpy as np
from scipy import signal


def swfilter(im, radius, iter):
    k = np.ones((2*radius+1, 1))/(2*radius+1)
    k_L = k.copy()
    k_L[radius+1:] = 0
    k_L = k_L/sum(k_L)
    k_R =k_L[::-1]
    m = im.shape[0] + 2*radius
    n = im.shape[1] + 2*radius
    total = m*n
    col, row = np.meshgrid(np.arange(n), np.arange(m))
    offset = row + m*col
    im = np.array(im, dtype=np.float)
    result = im.copy()
    for ch in range(3):
        U = cv2.copyMakeBorder(im[:, :, ch], radius, radius, radius, radius, cv2.BORDER_REPLICATE)
        for i in range(iter):
            d = np.zeros((m, n, 8), dtype=np.float)
            d[:, :, 0] = conv2d(k_L, k_L, U, 'same') - U
            d[:, :, 1] = conv2d(k_L, k_R, U, 'same') - U
            d[:, :, 2] = conv2d(k_R, k_L, U, 'same') - U
            d[:, :, 3] = conv2d(k_R, k_R, U, 'same') - U
            d[:, :, 4] = conv2d(k_L, k, U, 'same') - U
            d[:, :, 5] = conv2d(k_R, k, U, 'same') - U
            d[:, :, 6] = conv2d(k, k_L, U, 'same') - U
            d[:, :, 7] = conv2d(k, k_R, U, 'same') - U
            tmp = abs(d)
            d = d.flatten('F')
            ind = tmp.argmin(2)
            index = offset + total*ind
            dm = d[index]
            U = U + dm
        result[:, :, ch] = U[radius:-1-radius+1, radius:-1-radius+1]
    out = np.array(result, dtype=np.uint8)
    return out

def conv2d(k_x, k_y, im, mode):
    tmp = signal.convolve2d(im, k_x, mode=mode)
    out = signal.convolve2d(tmp, k_y.T, mode=mode)
    return out


if __name__ == '__main__':
    im = cv2.imread('panda_noise.jpg')
    out1 = swfilter(im, 3, 3)
    out2 = cv2.blur(im, (7, 7))
    out = np.vstack([out1, out2])

    cv2.imshow('figure1', out)
    cv2.waitKey()
    cv2.destroyAllWindows()
