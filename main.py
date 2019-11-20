import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from skimage.segmentation import active_contour
from  skimage.filters import gaussian
from skimage.filters import threshold_otsu
from scipy.spatial import distance

def InRangeByHelano(image, lower, upper):
    x,y,z = image.shape
    for i in range(x):
        for j in range(y):
            if (not((image[i][j][0]>= lower[0] and image[i][j][0] <= upper[0])  and (image[i][j][1]>= lower[1] and image[i][j][1] <= upper[1])  and (image[i][j][2]>= lower[2] and image[i][j][2] <= upper[2]) )):
                    image[i][j]=0
    return image

def fillHolesBinaryImage(image):


    h, w = image.shape[:2]
    imgFilled = image.copy()
    mask = np.zeros((h+2, w+2), np.uint8)
    cv.floodFill(imgFilled, mask, (0,0), 255);
    result = cv.bitwise_not(imgFilled)
    teste = image | result

    return teste

def getAccuracy(image):
    diffSize = 0
    x,y = image.shape
    for i in range(x):
        for j in range(y):
            if (image[i][j]==255):
                diffSize+=1
    return 100-((diffSize/(x*y))*100)


def getLocalaccuracy(imageDiff, groundTruth):
    totalSize = 0
    diffSize = 0
    x,y = imageDiff.shape
    for i in range(x):
        for j in range(y):
            if (groundTruth[i][j] == 255):
                totalSize+=1
            if (imageDiff[i][j] == 255):
                diffSize+=1
    return 100-(diffSize/totalSize)*100

def getDiceOverlapIndex(image1, image2):
    andOp = cv.bitwise_and(image1, image2)
    orOp = cv.bitwise_or(image1, image2)
    andSize = 0
    orSize = 0
    x, y = andOp.shape
    for i in range(x):
        for j in range(y):
            if (andOp[i][j] ==255):
                andSize+=1
            if (orOp[i][j]==255):
                orSize+=1
    return (2*andSize)/orSize

def  getDice(image1, image2):
    dice = np.sum(image2[image1==255])*2.0 / (np.sum(image2) + np.sum(image1))
    return dice


source = cv.imread('blobs.tif')
source_labeled = cv.imread('blobs_labeled.tif')
gray2 = cv.cvtColor(source_labeled.copy(), cv.COLOR_BGR2GRAY)



gray = cv.cvtColor(source.copy(), cv.COLOR_BGR2GRAY)
teste = gaussian(gray, sigma=3)

dst = cv.GaussianBlur(gray,(3,3),10,10,cv.BORDER_ISOLATED)
gray = teste - gray


thresh = threshold_otsu(teste)

ret, binary = cv.threshold(dst,150,255,cv.THRESH_BINARY)
ret2, binary2 = cv.threshold(gray2,50,255,cv.THRESH_BINARY)

filled = fillHolesBinaryImage(binary.copy())
filled2 = fillHolesBinaryImage(binary2.copy())

element = cv.getStructuringElement(cv.MORPH_OPEN,(2,2))

eroded = cv.erode(filled.copy(),element)
dilated = cv.dilate(eroded.copy(),element)
#dilated = cv.dilate(dilated,element)
#eroded = cv.erode(dilated,element)


frameDelta = cv.absdiff(dilated , filled2)

print("Global Accuracy: ", getAccuracy(frameDelta))
print("Local Accuracy: ", getLocalaccuracy(frameDelta, filled2))
print ("DICE Overlap Index :", getDice(filled2,dilated))

cv.imshow("source", source)
cv.imshow("blured", teste)
cv.imshow("source_labeled", source_labeled)
cv.imshow("delta", frameDelta)
cv.imshow("eroded", eroded)
cv.imshow("dilated", dilated)
cv.imshow("binary", binary)
cv.imshow("labeled", filled2)
cv.imshow("filled", filled)

cv.waitKey()
