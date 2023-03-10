import cv2
import numpy as np
from matplotlib import pyplot as plt

# Membaca gambar
img = cv2.imread('img_ori.jpeg', 0)

# Membuat filter mask Gaussian
gaussian_mask = cv2.GaussianBlur(img, (5, 5), 0)

# Mengurangi gambar yang sudah diburamkan dari gambar asli
unsharp_mask = cv2.subtract(img, gaussian_mask)

# Menambahkan hasil pengurangan pada gambar asli
sharp_img = cv2.addWeighted(img, 1.5, unsharp_mask, -0.5, 0)

# Menampilkan gambar asli dan gambar yang sudah difilter
plt.subplot(131), plt.imshow(img, cmap='gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(132), plt.imshow(unsharp_mask, cmap='gray')
plt.title('Unsharp Masking'), plt.xticks([]), plt.yticks([])
plt.subplot(133), plt.imshow(sharp_img, cmap='gray')
plt.title('Sharpened Image'), plt.xticks([]), plt.yticks([])
plt.show()
