import numpy as np
import cv2

img = cv2.imread('th.jpg')
img = cv2.resize(img, (1000,1000))
kernel_size = 3
kernel = np.ones((kernel_size, kernel_size), np.uint8)
maximum = cv2.dilate(img, kernel)

cv2.imshow('Original', img)
cv2.imshow('Maximum Filter', maximum)
cv2.waitKey(0)
cv2.destroyAllWindows()
