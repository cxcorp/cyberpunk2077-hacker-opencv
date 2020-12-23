# -*- coding: utf-8 -*-

from imutils.perspective import four_point_transform
from four_point_transform_matrix import four_point_transform_matrix
import imutils
import cv2
import numpy as np
import math
import matplotlib.pyplot as plt

ref = cv2.imread("ref.png")
#image = cv2.imread("bad.jpg")
ref = imutils.resize(ref, height=500)
ref = cv2.GaussianBlur(ref, (5,5), 0)

image = cv2.imread("bad.jpg")
#image = imutils.resize(image, height=500)
#image = cv2.fastNlMeansDenoising(image, 3, 9)
#image = cv2.GaussianBlur(image, (5,5), 0)

height,width, _ = image.shape


gridsize = 6
arr = np.array([(71,52), (508,46), (507,473), (82,483)], np.int32)

M, _, dst, (width, height) = four_point_transform_matrix(image, arr)

gridX = width / (gridsize - 1)
gridY = height / (gridsize - 1)
padX = gridX / 2
padY = gridY / 2


paddedDst = np.array([
    [-padX, -padY],
    dst[1] + [padX, -padY],
    dst[2] + [padX, padY],
    dst[3] + [-padX, padY]
    ], dtype="float32")


inv_M = np.linalg.pinv(M)

padded_arr = cv2.perspectiveTransform(paddedDst.reshape(4, 1, -1), inv_M).reshape(4,2).astype(np.int32)

cv2.polylines(image, [arr.reshape((-1,1,2))], True, (0, 255, 0), 3)
cv2.polylines(image, [padded_arr.reshape((-1,1,2))], True, (0, 0, 255), 3)

warped = four_point_transform(image, padded_arr)

warpHeight, warpWidth, _ = warped.shape
#cv2.line(warped, (0, math.ceil(gridY)), (warpWidth, math.ceil(gridY)), (0, 0, 255), 2)

splitter = math.ceil(gridY) + 5
header = cv2.cvtColor(warped[:splitter,:], cv2.COLOR_BGR2GRAY)
body = cv2.cvtColor(warped[splitter:, :], cv2.COLOR_BGR2GRAY)


blur = cv2.bilateralFilter(header, 5, 50, 50)
blur = cv2.GaussianBlur(blur, (3, 3), 0)
_, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
out1 = cv2.bitwise_not(th)
#blur = cv2.GaussianBlur(body, (5, 5), 0)
#	dst	=	cv.bilateralFilter(	src, d, sigmaColor, sigmaSpace[, dst[, borderType]]	)

blur = cv2.bilateralFilter(body, 5, 100, 100)
blur = cv2.GaussianBlur(blur, (3, 3), 0)
_, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

erodeKernel = np.ones((2,2), np.uint8)
th = cv2.erode(th, erodeKernel, iterations=1)

out2 = cv2.bitwise_not(th)
out2 = cv2.cvtColor(out2, cv2.COLOR_GRAY2BGR)

for x in range(gridsize + 1):
    cv2.line(out2, (math.floor(x * gridX), 0), (math.floor(x * gridX), warpHeight), (0, 0, 255), 2)
for y in range(gridsize + 1):
    cv2.line(out2, (0, math.floor(y * gridY)), (warpWidth, math.floor(y * gridY)), (0, 0, 255), 2)

cv2.imwrite('out.jpg', out2)
cv2.imwrite('out1.jpg', out1)
cv2.imwrite('out2.jpg', out2)

# print("M", M)
# print("rect", rect)
# print("dst", dst)
# print("width", width)
# print("height", height)

#cv2.imwrite('out.jpg', warped)

# lt: 71, 72
# rt: 508, 46
# lb: 507, 473
# lb: 82, 483

# cv2.imwrite('out.jpg', blur)

# import matplotlib.pyplot as plt
# import matplotlib.ticker as ticker

# fig, ax = plt.subplots(3,1)
# fig.set_size_inches(15, 10)

# hist = cv2.calcHist([image], [2], None, [255], [0,255])
# ax[0].xaxis.set_major_locator(ticker.MultipleLocator(5))
# ax[0].plot([x for x in range(255)], hist)
# ax[0].set_xlabel('R')

# hist = cv2.calcHist([image], [1], None, [255], [0,255])
# ax[1].xaxis.set_major_locator(ticker.MultipleLocator(10))
# ax[1].plot([x for x in range(255)], hist)
# ax[1].set_xlabel('G')

# hist = cv2.calcHist([image], [0], None, [255], [0,255])
# ax[2].xaxis.set_major_locator(ticker.MultipleLocator(10))
# ax[2].plot([x for x in range(255)], hist)
# ax[2].set_xlabel('B')
# plt.show()

def hist_match(source, template):
    """
    Adjust the pixel values of a grayscale image such that its histogram
    matches that of a target image

    Arguments:
    -----------
        source: np.ndarray
            Image to transform; the histogram is computed over the flattened
            array
        template: np.ndarray
            Template image; can have different dimensions to source
    Returns:
    -----------
        matched: np.ndarray
            The transformed output image
    """

    oldshape = source.shape
    source = source.ravel()
    template = template.ravel()

    # get the set of unique pixel values and their corresponding indices and
    # counts
    s_values, bin_idx, s_counts = np.unique(source, return_inverse=True,
                                            return_counts=True)
    t_values, t_counts = np.unique(template, return_counts=True)

    # take the cumsum of the counts and normalize by the number of pixels to
    # get the empirical cumulative distribution functions for the source and
    # template images (maps pixel value --> quantile)
    s_quantiles = np.cumsum(s_counts).astype(np.float64)
    s_quantiles /= s_quantiles[-1]
    t_quantiles = np.cumsum(t_counts).astype(np.float64)
    t_quantiles /= t_quantiles[-1]

    # interpolate linearly to find the pixel values in the template image
    # that correspond most closely to the quantiles in the source image
    interp_t_values = np.interp(s_quantiles, t_quantiles, t_values)

    return interp_t_values[bin_idx].reshape(oldshape)

#matched = np.uint8(hist_match(image, ref))

#t = 20
#hsv = cv2.cvtColor(matched, cv2.COLOR_BGR2HSV)
#thresh = cv2.inRange(hsv, (113-t, 74-t, 58-t), (113+t, 74+t,58+t))

#cv2.imwrite('out.jpg', thresh)


#blurred2 = cv2.bilateralFilter(gray, 5, 50, 50)
# blurred2 = cv2.boxFilter(gray, -1, (5,5))

# ret, thresholdMask = cv2.threshold(blurred2, 175, 255, cv2.THRESH_BINARY)
# #anded = cv2.bitwise_and(gray, gray, mask=thresholdMask)

# kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
# #dilated = cv2.dilate(anded, kernel, iterations=3)
# dilated = cv2.dilate(thresholdMask, kernel, iterations=9)

# cnts = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
# cnts = imutils.grab_contours(cnts)

# # filter out contours that probably aren't part of our grid
# areas = [cv2.contourArea(cnt) for cnt in cnts]

# cv2.imwrite("out.jpg", ocrrgb)