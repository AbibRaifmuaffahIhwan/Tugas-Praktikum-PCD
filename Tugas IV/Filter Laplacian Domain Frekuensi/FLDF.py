import cv2
import numpy as np
from matplotlib import pyplot as plt

# Membaca gambar
img = cv2.imread('img_ori.jpeg', 0)

# Transformasi Fourier pada gambar
f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)

# Membuat filter Laplacian
rows, cols = img.shape
crow, ccol = rows//2, cols//2
mask = np.zeros((rows, cols), np.uint8)
mask[crow-10:crow+10, ccol-10:ccol+10] = 1
laplacian = np.zeros((rows, cols), np.float32)
laplacian[crow-3:crow+3, ccol-3:ccol+3] = -4
laplacian[crow-2:crow+2, ccol-2:ccol+2] = 8
mask_lap = laplacian * mask

# Mengalikan gambar dengan filter Laplacian
f_laplacian = fshift * mask_lap

# Transformasi Fourier yang terbalik pada gambar
f_ishift = np.fft.ifftshift(f_laplacian)
img_back = np.fft.ifft2(f_ishift)
img_back = np.abs(img_back)

# Menampilkan gambar asli dan gambar yang sudah difilter
plt.subplot(121), plt.imshow(img, cmap='gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(img_back, cmap='gray')
plt.title('Laplacian Frequency Domain'), plt.xticks([]), plt.yticks([])
plt.show()
