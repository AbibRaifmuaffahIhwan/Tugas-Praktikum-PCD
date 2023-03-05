from PIL import Image
import matplotlib.pyplot as plt
import  numpy as np
import cv2

img = cv2.imread('img1.jpeg')

# BGR
cv2.imshow("Gambar",img)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
(n, m) = (gray.shape)
i = 0
j = 0
k = 0
H = np.zeros((256), dtype = int)
print(H)
while k < 256:
    H[k] = np.count_nonzero(gray==k)
    k = k+1
Intensity = np.arange(0, 256, 1)
print(Intensity)
print(H)
plt.bar(Intensity, H, color = 'maroon', width = 0.5)
plt.xlabel('Intensity')
plt.ylabel('Frequency')
plt.title('Histogram')
plt.show()
