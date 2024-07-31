import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import random
import ipywidgets as widgets

plt.rcParams['figure.figsize'] = (5, 5)

img = cv.imread('lena.png')
img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# cv.imshow('Image', img)
# cv.waitKey(0)

def foo(a, minv, maxv):
    if a > maxv:
        return maxv
    if a < minv:
        return minv
    return a

def addNoise(origImage: np.array, s=60):
    image = origImage.copy()
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            noise = np.random.randint(-s, s)
            new_val = image[i, j] + noise
            image[i, j] = foo(new_val, 0, 255)
    return image

noisy = addNoise(img)
#cv.imshow('Image', noisy)
#cv.waitKey(0)

sum = np.zeros_like(img)

def makeNoisyImage(origImage: np.array, N: int):
    #print(N)
    for k in range(N):
        #print(0)
        noisyImages = addNoise(origImage)
        #print(noisyImages.shape)
        #cv.imshow('Image', noisyImages)
        #cv.waitKey(0)
        for i in range(len(img)):
            for j in range(len(img)):
                sum[i][j] += noisyImages[i][j] / N
                #print(type(sum[i][j]))
N=5
makeNoisyImage(img, N)
# print(sum)

img1 = cv.imread('lena.png', cv.IMREAD_GRAYSCALE)
print(len(sum[0]))
print(img1.shape)
print(type(img1[1][2]))
for n in range(len(img1)):
    for m in range(len(img1)):
        img1[n][m] = sum[n][m]
print(sum/N)
# filtered = np.mean(LennaImages, axis=0)
cv.imshow('Image', sum)
cv.waitKey(0)
