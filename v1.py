# -*- coding: utf-8 -*-

from imutils.perspective import four_point_transform
import imutils
import cv2
import numpy as np
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r'C:\Users\Joona\AppData\Local\Tesseract-OCR\tesseract.exe'

def is_grid_count(n):
    for i in range(3, 8):
        if n == n*n:
            return True
    return False

image = cv2.imread("bad.jpg")
image = imutils.resize(image, height=700)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#blurred = cv2.bilateralFilter(gray, 5, 50, 50)
blurred = cv2.boxFilter(gray, -1, (5,5))

ret, thresholdMask = cv2.threshold(blurred, 175, 255, cv2.THRESH_BINARY)
#anded = cv2.bitwise_and(gray, gray, mask=thresholdMask)

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
#dilated = cv2.dilate(anded, kernel, iterations=3)
dilated = cv2.dilate(thresholdMask, kernel, iterations=9)

cnts = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
cnts = imutils.grab_contours(cnts)

# filter out contours that probably aren't part of our grid
areas = [cv2.contourArea(cnt) for cnt in cnts]

areasMean = np.mean(areas)
areasStdDev = np.std(areas)
areasMin = areasMean - 3*areasStdDev
areasMax = areasMean + 3*areasStdDev

#print(areasMin, areasMax)
# hist, edges = np.histogram(areas)
# print(hist)

# maxBinIndex = np.argmax(hist, axis=0)
# if not is_grid_count(maxBinIndex):
#     print('not grid count max bin', hist[maxBinIndex])
    
# binStart = edges[maxBinIndex]
# binEndExc = edges[maxBinIndex + 1]

gridCnts = []
for i,area in enumerate(areas):
    if area >= areasMin and area < areasMax:
        gridCnts.append(cnts[i])

convexhull = cv2.convexHull(np.concatenate(gridCnts))
peri = cv2.arcLength(convexhull, True)
approx = cv2.approxPolyDP(convexhull, 0.02*peri, True)

cv2.drawContours(image, gridCnts, -1, (0, 0, 255), 2)
cv2.imwrite('outt.jpg', image)

print(approx)
print(approx.reshape(4, 2))

if len(approx) != 4:
    print("approxPolyDP len != 4")

warped = four_point_transform(cv2.bitwise_not(thresholdMask), approx.reshape(4, 2))
warped = cv2.copyMakeBorder(warped, 30, 30, 30, 30, cv2.BORDER_REPLICATE)

erodeKernel = np.ones((2,2), np.uint8)
erosion = cv2.erode(warped, erodeKernel, iterations=1)

ocrrgb = cv2.cvtColor(erosion, cv2.COLOR_GRAY2RGB)
cv2.imwrite("out.jpg", ocrrgb)

tessconfig = r'--psm 6 --oem 0 -c tessedit_char_whitelist=ABCDE1579'
result = pytesseract.image_to_string(ocrrgb, 'eng', config=tessconfig, timeout=10)

import re
result = re.sub('[^ABCDE1579 \\n]', '', result)
print(result)

# import pandas as pd
# from io import StringIO
# text = pd.read_csv(StringIO(result),delimiter='\t')
# text = text[text.conf != -1]
# for name, group in text.groupby('line_num'):
#     print(name, group)

#cv2.drawContours(output, [approx], 0, (0, 0, 255), 3)

#print(convexhull)
#cv2.drawContours(output, [approx], 0, (0, 0, 255), 3)
#rrect = cv2.minAreaRect(convexhull)
#rbox = cv2.boxPoints(rrect)
#rbox = np.int0(rbox)
#cv2.drawContours(output, [rbox], 0, (0, 0, 255), 2)
#print()
#[x,y,w,h] = cv2.boundingRect(np.concatenate(gridCnts))
#cv2.rectangle(output, (x,y), (x+w, y+h), (0, 0, 255), 3)


# boxCnt = None
# boxCntArea = 0
# for contour in cnts:
#     [x, y, w, h] = cv2.boundingRect(contour)
    
#     # check aspect ratio matches~ the ARC of the hackbox
#     if abs(1.4 - w/h) > 0.2:
#         continue;
    
#     area = cv2.contourArea(contour)
#     if area > boxCntArea:
#         boxCnt = contour
#         boxCntArea = area
    
#     #peri = cv2.arcLength(contour, True)
#     #approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
    
#     #if len(approx) != 3:
#     #    continue
    
#     #cv2.rectangle(image, (x,y), (x+w, y+h), (0, 255, 0), 3)

# #cv2.drawContours(image, [boxCnt], 0, (0, 255, 0),10)
# hackboxMask = np.zeros((gray.shape), np.uint8)
# cv2.drawContours(hackboxMask, [boxCnt], 0, 255, -1)
# cv2.drawContours(hackboxMask, [boxCnt], 0, 0, 2)

# hackboxImg = np.zeros_like(gray)
# hackboxImg[hackboxMask == 255] = gray[hackboxMask == 255]

# #warped = four_point_transform(gray, boxCnt.reshape(4, 2))
