import cv2
import numpy as np

img_Ori = cv2.imread("lena.jpg")
Hasil_Gray = cv2.cvtColor(img_Ori, cv2.COLOR_BGR2GRAY)

cv2.imshow("Gambar Lena Yang Original", img_Ori)
cv2.imshow("Gambar Lena GrayScale", Hasil_Gray)

cv2.waitKey(0)
cv2.destroyAllWindows()