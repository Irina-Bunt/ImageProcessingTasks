import numpy as np
import cv2
from IPython.display import display
import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = (8,4)

def show(img: np.array):
    plt.imshow(img, cmap='gray')
    plt.axis('off')

img = cv2.imread('./sonnet.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
processed = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
show(img)
cv2.imwrite('sonnet_processed.png', processed)

show(processed)