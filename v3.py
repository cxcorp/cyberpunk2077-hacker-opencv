# -*- coding: utf-8 -*-

from imutils.perspective import four_point_transform
from four_point_transform_matrix import four_point_transform_matrix
import imutils
import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
import os

def getGraySample(lab):
    height, width, _ = lab.shape
    
    squareSize = height / 16.85
    grayMid = height / 11.6
    
    top = math.floor(grayMid - squareSize)
    left = math.floor(squareSize)
    right = math.floor(squareSize + squareSize * 2)
    bot = math.floor(grayMid + squareSize)
    
    sample = lab[top:bot, left:right]
    #print(lab)
    l, a, b = cv2.split(sample)
    return np.trunc([np.mean(l), np.mean(a), np.mean(b)]).astype(np.uint8)

def getBlueSample(lab):
    height, width, _ = lab.shape
    
    squareSize = height / 16.85
    blueMid = height / 2.0
    
    top = math.floor(blueMid - squareSize)
    left = math.floor(squareSize)
    right = math.floor(squareSize + squareSize * 2)
    bot = math.floor(blueMid + squareSize)
    
    sample = lab[top:bot, left:right]
    #print(lab)
    l, a, b = cv2.split(sample)
    return np.trunc([np.mean(l), np.mean(a), np.mean(b)]).astype(np.uint8)

def heightWise(height, heightThresh, amounts):
    interpolated = np.interp(height, heightThresh, [0, len(amounts) - 1])
    index = int(np.round(interpolated))
    return amounts[index]

for filename in os.listdir('.\\uis'):
    pathname = os.path.join('.\\uis', filename)

    image = cv2.imread(pathname)
    
    # 1. CROP CODE MATRIX FROM SCREENSHOT - according to width/height relative
    height,width, _ = image.shape
    
    uiHeight = math.floor((9/16.0) * width)
    uiTop = math.floor((height - uiHeight) / 2)
    
    left = math.floor(width / 10.0)
    top = math.floor(uiTop + uiHeight / 3.15)
    bot = math.floor(top + uiHeight / 2.8)
    right = math.floor(left + width / 3.2)
    
    cropped = image[top:bot, left:right]
    
    # 2. NORMALIZE THE GRAY HORIZONTAL STRIP
    lab = cv2.cvtColor(cropped, cv2.COLOR_BGR2Lab)
    graySample = getGraySample(lab)
    blueSample = getBlueSample(lab)
    
    blurred = cv2.cvtColor(cv2.GaussianBlur(cropped, (5, 5), 0), cv2.COLOR_BGR2Lab)
    grayLower = np.clip([x - 12 for x in graySample], 0, 255) # (255 * 5%) = ~12
    grayUpper = np.clip([x + 12 for x in graySample], 0, 255)
    grayMask = cv2.inRange(lab, grayLower, grayUpper)
    
    blueLower = np.clip([x - 12 for x in blueSample], 0, 255)
    blueUpper = np.clip([x + 12 for x in blueSample], 0, 255)
    blueMask = cv2.inRange(blurred, blueLower, blueUpper)
    
    normalized_lab = lab.copy()
    normalized_lab[grayMask>0] = blueSample
    normalized_lab[blueMask>0] = blueSample
    
    normalized = cv2.cvtColor(normalized_lab, cv2.COLOR_Lab2BGR)
    gray = cv2.cvtColor(normalized, cv2.COLOR_BGR2GRAY)
    
    # 3. FIND GRID SIZE
    #   a. blur, threshold & dilate
    #   b. find contours & get bounding rects
    #   c. 
    croppedHeight, _, _ = cropped.shape
    filterAmount = heightWise(croppedHeight, [257, 514], [4, 5])
    blurred = cv2.boxFilter(gray, -1, (filterAmount,filterAmount))
    
    _, threshold = cv2.threshold(blurred, 80, 255, cv2.THRESH_BINARY)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilateIterations = heightWise(croppedHeight, [257, 514], [3, 4, 5])
    dilated = cv2.dilate(threshold, kernel, iterations = dilateIterations)
    
    cnts = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnts = imutils.grab_contours(cnts)
    boundingRects = [cv2.boundingRect(cnt) for cnt in cnts]
    midpoints = [[x+(w/2), y+(h/2)] for [x,y,w,h] in boundingRects]
    
    out = cropped.copy()
    # for rect in boundingRects:
    #     cv2.rectangle(out, rect, (0, 255, 0), 2)
    for [x,y] in midpoints:
        cv2.drawMarker(out, (math.floor(x),math.floor(y)), (0, 0, 255), cv2.MARKER_CROSS, 30, 2)
    
   # bw = cv2.bitwise_not(dilated)
    
    cv2.imwrite(os.path.join('.\\out', os.path.splitext(filename)[0] + '.png'), out)

# cv2.imwrite("out.jpg", ocrrgb)