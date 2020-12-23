# -*- coding: utf-8 -*-

import imutils
import cv2
import numpy as np
import math
import os

def getColorSample(lab, point, rectSize):
    height, width, _ = lab.shape
    
    top = math.floor(point[1] - rectSize[1]/2)
    left = math.floor(point[0] - rectSize[0]/2)
    right = math.floor(point[0] + rectSize[0]/2)
    bot = math.floor(point[1] + rectSize[1]/2)
    
    sample = lab[top:bot, left:right]
    #print(lab)
    l, a, b = cv2.split(sample)
    return np.trunc([np.mean(l), np.mean(a), np.mean(b)]).astype(np.uint8)

def heightwise(height, heightThresh, amounts):
    interpolated = np.interp(height, heightThresh, [0, len(amounts) - 1])
    index = int(np.round(interpolated))
    return amounts[index]

def resizeAndPad(img, size, padColor=0):
    h, w = img.shape[:2]
    sh, sw = size

    # interpolation method
    if h > sh or w > sw: # shrinking image
        interp = cv2.INTER_AREA
    else: # stretching image
        interp = cv2.INTER_CUBIC

    # aspect ratio of image
    aspect = w/h  # if on Python 2, you might need to cast as a float: float(w)/h

    # compute scaling and pad sizing
    if aspect > 1: # horizontal image
        new_w = sw
        new_h = np.round(new_w/aspect).astype(int)
        pad_vert = (sh-new_h)/2
        pad_top, pad_bot = np.floor(pad_vert).astype(int), np.ceil(pad_vert).astype(int)
        pad_left, pad_right = 0, 0
    elif aspect < 1: # vertical image
        new_h = sh
        new_w = np.round(new_h*aspect).astype(int)
        pad_horz = (sw-new_w)/2
        pad_left, pad_right = np.floor(pad_horz).astype(int), np.ceil(pad_horz).astype(int)
        pad_top, pad_bot = 0, 0
    else: # square image
        new_h, new_w = sh, sw
        pad_left, pad_right, pad_top, pad_bot = 0, 0, 0, 0

    # set pad color
    if len(img.shape) == 3 and not isinstance(padColor, (list, tuple, np.ndarray)): # color image but only one color provided
        padColor = [padColor]*3

    # scale and pad
    scaled_img = cv2.resize(img, (new_w, new_h), interpolation=interp)
    scaled_img = cv2.copyMakeBorder(scaled_img, pad_top, pad_bot, pad_left, pad_right, borderType=cv2.BORDER_CONSTANT, value=padColor)

    return scaled_img

for filename in os.listdir('.\\dropzone\\in'):
    pathname = os.path.join('.\\dropzone\\in', filename)

    image = cv2.imread(pathname)
    height,width, _ = image.shape
    
    # 1. CROP FULLSCREEN SCREENIE TO THE FIXED LIMITS WE KNOW:
    #   a. top of the code matrix interior
    #   b. what we think is the max right of the code matrix, on the safe side: 60% of width
    
    uiHeight = math.floor((9/16.0) * width)
    uiTop = math.floor((height - uiHeight) / 2)
    
    top = math.floor(uiTop + uiHeight/3.15)
    bot = uiTop + uiHeight # crop to 16:9 bot
    # let's assume the matrix box can't grow over 60%
    right = math.floor(width * 0.6)
    left = 0
    cropped = image[top:bot, left:right]
    
    # approx distance between code matrix grid elements
    approxGridOffset = uiHeight / (1440/85.0)
    
    
    codeMatrixCenterOrigX = width / 3.91
    graySamplePointOrigY = uiTop + (uiHeight / 2.0) - uiHeight / 7.63251
    codeMatrixCenterCropX = codeMatrixCenterOrigX - left
    graySamplePointCroppedY = graySamplePointOrigY - top

    # 2. CROP FURTHER - GET WIDTH OF THE FIRST ROW'S GREY BG
    lab = cv2.cvtColor(cropped, cv2.COLOR_BGR2Lab)    
    graySample = getColorSample(lab, [codeMatrixCenterCropX, graySamplePointCroppedY], [30, 10])
    blueSample = getColorSample(lab, [codeMatrixCenterCropX, math.floor(graySamplePointCroppedY + approxGridOffset)], [30, 10])
    
    grayLower = np.clip([x - 12 for x in graySample], 0, 255) # (255 * 5%) = ~12
    grayUpper = np.clip([x + 12 for x in graySample], 0, 255)
    grayMask = cv2.inRange(lab, grayLower, grayUpper)
    
    cnts = cv2.findContours(grayMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnts = imutils.grab_contours(cnts)
    boundingRects = [cv2.boundingRect(cnt) for cnt in cnts]
    
    minGrayStripHeight = uiHeight / 18.947
    maxGrayStripHeight = uiHeight / 13.846
    # really low lower bound, assume width at least as wide as 4x4 code matrix
    minGrayStripWidth = width / 8.767
    
    # 2.a. FIND GRAY STRIPE'S CONTOUR
    
    grayStripRect = None
    for rect in boundingRects:
        [_, _, w, h] = rect
        if h > minGrayStripHeight and h < maxGrayStripHeight and w > minGrayStripWidth:
            grayStripRect = rect
            break
        
    if grayStripRect == None:
        print(f"NO GRAY STRIP FOUND FOR {filename}")
        print(f'  minH: {minGrayStripHeight}')
        print(f'  maxH: {maxGrayStripHeight}')
        print(f'  minW: {minGrayStripWidth}')
        for rect in boundingRects:
            print(rect)
            cv2.rectangle(cropped, rect, (0, 0, 255), 2)
        cv2.imwrite(os.path.join('.\\out', os.path.splitext(filename)[0] + '.png'), cropped)
        continue
    
    # approx dist from gray rect sides to code matrix: width/14
    # crop to approx width*(2/3)
    approxDistToCode = width / 14
    newCropPadX = math.floor(approxDistToCode - approxGridOffset/2)
    [grX, grY, grW, grH] = grayStripRect
    
    newHeight = math.floor(((grX+grW-newCropPadX) - (grX+newCropPadX)) - approxGridOffset/2)
    cropped = cropped[0:newHeight, grX+newCropPadX:grX+grW-newCropPadX]
    
    # 3. CHANGE GRAY'S COLOR TO THE BG BLUE SO IT DOESN'T MESS WITH OUR THRESHOLD
    
    cropped_lab = cv2.cvtColor(cropped, cv2.COLOR_BGR2Lab)
    blurred = cv2.GaussianBlur(cropped_lab, (5, 5), 0)

    grayMask = cv2.inRange(cropped_lab, grayLower, grayUpper)
    
    blueLower = np.clip([x - 24 for x in blueSample], 0, 255) # (255 * 10%) = ~12
    blueUpper = np.clip([x + 24 for x in blueSample], 0, 255)
    blueMask = cv2.inRange(cropped_lab, grayLower, grayUpper)
    
    cropped_lab[grayMask>0] = blueSample
    cropped_lab[blueMask>0] = blueSample
    
    # 4. FIND GRID MIDPOINTS AND CROP OUT DIGITS INTO TILES
    
    cropped_corrected = cv2.cvtColor(cropped_lab, cv2.COLOR_Lab2BGR)
    cropped_height, cropped_width, _ = cropped_corrected.shape
    
    filterAmount = heightwise(approxGridOffset, [42, 85], [3, 5, 7])
    gray = cv2.cvtColor(cropped_corrected, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (filterAmount,filterAmount), 0)
    
    _, threshold = cv2.threshold(blurred, 80, 255, cv2.THRESH_BINARY)
        

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilateIterations = heightwise(approxGridOffset, [42.5, 85], [3, 4, 5])
    dilated = cv2.dilate(threshold, kernel, iterations = dilateIterations)
   
    cnts = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnts = imutils.grab_contours(cnts)
    boundingRects = [cv2.boundingRect(cnt) for cnt in cnts]
    midpoints = [[x+(w/2), y+(h/2)] for [x,y,w,h] in boundingRects]
    
    gridSize = round((cropped_width - approxGridOffset/2) / (approxGridOffset))
    print(f'{len(boundingRects)} - {gridSize*gridSize}')
   
    threshold = cv2.bitwise_not(threshold)
    out = cv2.cvtColor(threshold, cv2.COLOR_GRAY2BGR)
    # for rect in boundingRects:
    #     cv2.rectangle(out, rect, (0, 255, 0), 2)
    
    gridTopLeftX = None
    if gridSize % 2 == 0:
        gridTopLeftX = cropped_width / 2 - approxGridOffset * (gridSize / 2)
    else:
        gridTopLeftX = cropped_width / 2 - approxGridOffset * ((gridSize-1) / 2) - approxGridOffset / 2
    
    import glob
    files = glob.glob('.\\dropzone\\tiles\\*')
    for f in files:
        os.remove(f)
    
    for i, [x,y] in enumerate(midpoints):
        #cv2.drawMarker(out, (math.floor(x),math.floor(y)), (0, 0, 255), cv2.MARKER_CROSS, 30, 2)
        
        # cv2.rectangle(out,
        #               (left, top, round(approxGridOffset/2), round(approxGridOffset/2)),
        #               (0, 255, 0),
        #               1)
        
        gridposX = round((x - approxGridOffset/2 - gridTopLeftX) / approxGridOffset)
        gridposY = round((y - approxGridOffset/2) / approxGridOffset)
        tsize = round(approxGridOffset/2)
        left = round(x - tsize/2)
        top = round(y - tsize/2)
        tile = threshold[top:top+tsize, left:left+tsize]
        tile = resizeAndPad(tile, (20,20), 255)
        cv2.imwrite(os.path.join('.\\dropzone\\tiles', f'{gridposY}-{gridposX}.png'), tile)
        
    
    cv2.imwrite(os.path.join('.\\dropzone', os.path.splitext(filename)[0] + '-out.png'), cropped_corrected)