# Import packages
import cv2
import imutils
import numpy as np

# Membaca 2 Gambar
img1 = cv2.imread('city1.jpg')
img2 = cv2.imread("city2.jpg")

# Mengubah Ukuran Foto (Resize)
img1 = cv2.resize(img1, (634,645))
img2 = cv2.resize(img2, (634,645))

img_height = img1.shape[0]

# Grayscale
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# Menemukan Perbedaan Antara 2 Gambar
# Hitung Perbedaan ABsolut antara 2 Array
diff = cv2.absdiff(gray1, gray2)
cv2.imshow("diff(img1, img2)", diff)

# Terapkan ambang batas (Threshold). Terapkan THRESH_BINARY dan THRESH_OTSU
thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
cv2.imshow("Threshold", thresh)

# Dilation (Pelebaran)
kernel = np.ones((5,5), np.uint8)
dilate = cv2.dilate(thresh, kernel, iterations=2)
cv2.imshow("Dilate", dilate)

# Menghitung Kontur
contours = cv2.findContours(dilate.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours = imutils.grab_contours(contours)

for contour in contours:
    if cv2.contourArea(contour) > 100:
        # Calculate bounding box around contour
        x, y, w, h = cv2.boundingRect(contour)
        # Draw rectangle - bounding box on both images
        cv2.rectangle(img1, (x, y), (x+w, y+h), (0,0,255), 2)
        cv2.rectangle(img2, (x, y), (x+w, y+h), (0,0,255), 2)

# Tampilkan gambar dengan perbedaan
x = np.zeros((img_height,10,3), np.uint8)
result = np.hstack((img1, x, img2))
cv2.imshow("Differences", result)

cv2.waitKey(0)
cv2.destroyAllWindows()