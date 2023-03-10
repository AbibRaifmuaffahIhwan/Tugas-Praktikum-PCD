import cv2
import numpy as np
from matplotlib import pyplot as plt

# Load image
img = cv2.imread('img_ori.jpeg',0)

# Calculate Fourier Transform
f = np.fft.fft2(img)

# Shift the zero-frequency component to the center of the spectrum
fshift = np.fft.fftshift(f)

# Create a mask with the same size as the image, centered on the image
rows, cols = img.shape
crow, ccol = rows//2, cols//2
mask = np.zeros((rows, cols), np.uint8)
r = 80
mask[crow-r:crow+r, ccol-r:ccol+r] = 1

# Apply the mask to the frequency spectrum
fshift_filtered = fshift * mask

# Shift back the zero-frequency component to the original position
f_filtered = np.fft.ifftshift(fshift_filtered)

# Perform inverse Fourier Transform to get the image back
img_filtered = np.fft.ifft2(f_filtered)
img_filtered = np.abs(img_filtered)

# Show the original and filtered images
plt.subplot(121), plt.imshow(img, cmap='gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(img_filtered, cmap='gray')
plt.title('Filtered Image'), plt.xticks([]), plt.yticks([])
plt.show()
