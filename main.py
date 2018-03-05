import cv2
import pickle
import numpy as np
from scipy.cluster.vq import vq, kmeans


extension = ".webm"
nameVideos=["arbol", "casa,", "zagal,", "pez,", "test,"]


def GetAllDescriptors(nameVideos,extension):
    for path in nameVideos:
        capture = cv2.VideoCapture(path+extension)
        while True:
            # Capturamos
            ret, frame = capture.read()
            if ret == False:
                break
            # Escalamos
            resized = cv2.resize(frame, (640, 480))
            # Convertimos a gris
            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
            # Sacamos keypoints y descriptores
            keypoints, descriptors = orb.detectAndCompute(gray, None)
            if allDescriptors is None:
                allDescriptors = descriptors
            elif descriptors is not None:
                allDescriptors = np.vstack((allDescriptors, descriptors))
    return allDescriptors

def whiten(v, dev=None):
    v2 = v.astype(np.float64)
    if type(dev) == type(None):
        dev = np.std(a=v2, axis=0)
    ret = np.array(v2).astype(np.float64)
    row = 0
    while row < len(ret):
        column = 0
        while column < len(ret[row]):
            if dev[column] > 0.0000001:
                ret[row][column] /= dev[column]
            else:
                dev[column] = 0.
            column += 1
        row += 1
    return ret, dev

def GenerateCodebook(descriptores):
    whitened, dev = whiten(descriptores)
    codebook, distortion = kmeans(whitened, 20)
    return codebook, dev


def cropToRequirements(image):
    return image[100:-100][100:-170]

def scale(image,col=1120):
    return cv2.resize(image,None,fx=col/image.shape[1],fy=col/image.shape[1])

def morphFilters(binFrame):
    kernel = np.ones((3,3),np.uint8)
    dilateImage = cv2.dilate(binFrame,kernel,iterations=1)
    erodeImg = cv2.erode(dilateImage,kernel,iterations=2)
    dilateImage = cv2.dilate(erodeImg, kernel, iterations=4)
    return dilateImage

def extractIntReg(binFrame, padding=10, minCont=50):
    aux = np.copy(binFrame)
    regions = []
    image,contours,hierarchy = cv2.findContours(binFrame,cv2.RETR_LIST,cv2.CHAIN_APROX_SIMPLE)

    for cont in contours:
        if len(cont) > minCont:
            x,y,w,h = cv2.boundingRect(cont)
            regions.append((x-padding,y-padding,w+2*padding,h+2*padding))
    return regions

