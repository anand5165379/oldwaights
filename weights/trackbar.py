import cv2
import numpy as np


def nothing(x):
	pass


cv2.namedWindow('value')
cv2.createTrackbar('low', 'value', 0, 255, nothing)
cv2.createTrackbar('high', 'value', 0, 255, nothing)
# cv2.createTrackbar('dilate', 'value', )
kernel = np.ones((5, 5), np.uint8)


while True:
	img = cv2.imread('./img_1633677506062.jpg')
	img = cv2.resize(img, None, fx=0.2, fy=0.2)
	hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
	h, s, v = cv2.split(hsv)

	low = int(cv2.getTrackbarPos('low', 'value'))
	high = int(cv2.getTrackbarPos('high', 'value'))
	#h = cv2.erode(h, kernel, iterations=1)
	ret, threshold = cv2.threshold(h, low, high, cv2.THRESH_BINARY_INV)
	contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
	contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 10000]
	out = cv2.drawContours(img, contours, -1, (255, 0, 0), 2)

	cv2.imshow('Image', out)
	cv2.imshow('Threshold', threshold)

	if cv2.waitKey(1) & 0xFF == 27:
		break

cv2.destroyAllWindows()
