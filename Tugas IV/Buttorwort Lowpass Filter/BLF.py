import cv2
import numpy as np
from matplotlib import pyplot as plt

# Membaca gambar
img = cv2.imread('img_ori.jpeg', 0)

# Menghitung ukuran gambar dan menghitung frekuensi
rows, cols = img.shape
crow, ccol = rows//2, cols//2

# Membuat filter mask Butterworth Lowpass
D0 = 70
n = 2
mask = np.zeros((rows, cols), np.uint8)
x, y = np.ogrid[:rows, :cols]
dist = np.sqrt((x - crow)**2 + (y - ccol)**2)
mask = 1 / (1 + (dist / D0)**(2*n))

# Melakukan filtering pada gambar dengan filter mask Butterworth Lowpass
f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)
fshift_filtered = fshift * mask
f_filtered = np.fft.ifftshift(fshift_filtered)
img_filtered = np.fft.ifft2(f_filtered)
img_filtered = np.abs(img_filtered)

# Menampilkan gambar asli dan gambar yang sudah difilter
plt.subplot(121), plt.imshow(img, cmap='gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(img_filtered, cmap='gray')
plt.title('Butterworth Lowpass Filter'), plt.xticks([]), plt.yticks([])
plt.show()
