import numpy as np
import cv2

img = cv2.imread('img_ori.jpeg', 0)
img = cv2.resize(img, (1000,1000))
kernel_size = 15
kernel = np.ones((kernel_size, kernel_size), np.uint8)
minimum = cv2.erode(img, kernel)

cv2.imshow('Original', img)
cv2.imshow('Minimum Filter', minimum)
cv2.waitKey(0)
cv2.destroyAllWindows()
