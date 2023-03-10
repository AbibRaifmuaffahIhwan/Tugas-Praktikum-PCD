import numpy as np
import cv2

img = cv2.imread('img_ori.jpeg')
img = cv2.resize(img, (1000,1000))
kernel_size = 15
median = cv2.medianBlur(img, kernel_size)

cv2.imshow('Original', img)
cv2.imshow('Median Filter', median)
cv2.waitKey(0)
cv2.destroyAllWindows()
