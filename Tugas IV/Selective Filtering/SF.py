import cv2
import numpy as np
from matplotlib import pyplot as plt

# Membaca gambar
img = cv2.imread('img_ori.jpeg', 0)

# Identifikasi wilayah yang akan dihilangkan noise-nya
blur = cv2.GaussianBlur(img, (15,15), 0)
laplacian = cv2.Laplacian(blur, cv2.CV_64F)

# Menghilangkan noise pada wilayah yang sudah diidentifikasi
noise_mask = (laplacian < 0.10) * 255
noise_mask = noise_mask.astype(np.uint8)
filtered_img = cv2.medianBlur(img, 5)

# Menggabungkan wilayah yang sudah dihilangkan noise-nya dengan wilayah yang dipertahankan
output_img = cv2.bitwise_and(img, img, mask=cv2.bitwise_not(noise_mask))
output_img = cv2.add(output_img, cv2.bitwise_and(filtered_img, noise_mask))

# Menampilkan gambar asli dan gambar yang sudah diolah
plt.subplot(121), plt.imshow(img, cmap='gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(output_img, cmap='gray')
plt.title('Selective Filtered Image'), plt.xticks([]), plt.yticks([])
plt.show()
