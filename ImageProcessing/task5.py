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

def sp_noise(image,prob):
    output = np.zeros(image.shape, np.uint8)
    thres = 1 - prob
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = random.random()
            if rdn < prob:
                output[i][j] = 0
            elif rdn > thres:
                output[i][j] = 255
            else:
                output[i][j] = image[i][j]
    return output
noisyLena = sp_noise(img, 0.07)

# cv.imshow('Image', noisyLena)
# cv.waitKey(0)

@widgets.interact(ksizeGauss=(1, 11, 2), ksizeMedian=(1, 11, 2), ksizeNorm=(1, 11, 2))
def filterImage(ksizeGauss=3, ksizeMedian=3, ksizeNorm=3):
    cv.imshow('Noisy Lena', noisyLena)

    GaussFilter = cv.GaussianBlur(img, (ksizeGauss, ksizeGauss), 2)
    cv.imshow('Gaussian Filter', GaussFilter)

    medianFilter = cv.medianBlur(img, ksizeMedian)
    cv.imshow('Median Filter', medianFilter)

    kernel = np.ones((ksizeNorm, ksizeNorm), dtype=np.float32)
    kernel /= (ksizeNorm * ksizeNorm)
    normalizerBlur = cv.filter2D(img, -1, kernel)

    cv.imshow('Normalized Filter', normalizerBlur)

    plt.tight_layout()

def plot_img_table(imgs, names):
    nrows = imgs.shape[0]
    ncols = imgs.shape[1]
    fig, ax = plt.subplots(nrows, ncols, figsize=(7, 7))
    for i in range(nrows):
        for j in range(ncols):
            ax[i, j].imshow(imgs[i, j], cmap='gray')
            cv.imshow(names[i, j], imgs[i, j])
            ax[i, j].title.set_text(names[i, j])
            ax[i, j].set_xticks([])
            ax[i, j].set_yticks([])
    plt.tight_layout()

# Sobel
sobel_x = cv.Sobel(img, -1, 0, 1)
sobel_y = cv.Sobel(img, -1, 1, 0)
sobel = cv.addWeighted(sobel_x, 0.5, sobel_y, 0.5, 0)
laplacian = cv.Laplacian(img, -1)

edge_img = np.array([[sobel, laplacian], [sobel_x, sobel_y]])
edge_names = np.array([['Sobel', 'Laplacian'], ['Sobel_X', 'Sobel_Y']])
plot_img_table(edge_img, edge_names)
cv.waitKey(0)
