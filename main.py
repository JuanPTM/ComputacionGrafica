import cv2
import pickle
import numpy as np
from scipy.cluster.vq import vq, kmeans

pathVideos = "movies/"
extension = ".webm"
nameVideos=["arbol", "casa", "zagal", "pez"]


def GetAllDescriptors():
    orb = cv2.ORB_create()
    allDescriptors = None
    for path in nameVideos:
        print pathVideos+path+extension
        capture = cv2.VideoCapture(pathVideos+path+extension)
        while True:
            # Capturamos
            ret, frame = capture.read()
            if ret == False:
                break

            # Convertimos a gris
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Recortamos
            graycrop = cropToRequirements(gray)

            # Escalamos
            grayScale = scale(graycrop)

            # Sacamos keypoints y descriptores
            keypoints, descriptors = orb.detectAndCompute(grayScale, None)
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

def GenerateCodebook(descriptors):
    whitened, dev = whiten(descriptors)
    codebook, distortion = kmeans(whitened, 20)
    return codebook, dev

def GenerateAllHistograms():
    orb = cv2.ORB_create()
    for path in nameVideos:
        histograms = None
        print "inicio el video", pathVideos+path+extension
        capture = cv2.VideoCapture(pathVideos+path+extension)
        while True:
            # Capturamos
            ret, frame = capture.read()
            if ret == False:
                break
            # Escalamos
            # Convertimos a gris
            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
            # Sacamos keypoints y descriptores
            keypoints, descriptors = orb.detectAndCompute(gray, None)

            if descriptors is not None:
                histogram, limites = np.histogram(vq(descriptors/dev, codebook)[:][1],bins=range(21))
                if histograms is None:
                    histograms = histogram
                else:
                    histograms = np.vstack((histograms, histogram))


def SegmentFrame(frameGray, prevframeGray):
    m = np.subtract(prevframeGray.astype(np.int16),frameGray.astype(np.int16))
    return np.absolute(m).astype(np.int8)

def FilterByRange(frameGray,minValue=90,maxValue=250):
     return 255-cv2.inRange(frameGray, minValue, maxValue)

def cropToRequirements(image):
    return image[100:-100,100:-170]

def scale(image,col=1120):
    col = float(col)
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

if __name__ == "__main__":
    try:
        codebook = pickle.load(open('codebook.pickle', 'r'))
        dev = pickle.load(open('dev.pickle', 'r'))
    except:
        print "comienzo"
        descriptors = GetAllDescriptors()
        codebook, dev = GenerateCodebook(descriptors)
        pickle.dump(dev, open('dev.pickle', 'w'))
        pickle.dump(codebook, open('codebook.pickle', 'w'))
        print "He terminado"
    # try:
    #
    # except Exception as e:
    #
    #     histograms = GenerateAllHistograms()
    #     pickle.dump(histograms, open(path[:-5]+'histograms.pickle', 'w'))
    #     raise
    capture = cv2.VideoCapture("movies/test.webm")
    while True:
        # Capturamos
        ret, frame = capture.read()
        if ret == False:
            break

        # Convertimos a gris
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Recortamos
        graycrop = cropToRequirements(gray)

        # Escalamos
        grayScale = scale(graycrop)

        binImg = FilterByRange(grayScale)

        procesedImage = morphFilters(binImg)

        cv2.imshow("d", procesedImage)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break



        #
        # resized = cv2.resize(frame, (640, 480))
        # gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        # # Sacamos keypoints y descriptores
        #
        # keypoints, descriptors = orb.detectAndCompute(gray, None)
        # if descriptors is not None:
        #     hist, limites = np.histogram(vq(descriptors/dev, codebook)[:][1],bins=range(21))
        #     dist = []
        #     for i in range(len(histograms)):
        #         dist.append(min(distance.cdist(histograms[i], np.array([hist]), "euclidean")))
        # #cv2.putText(resized,paths[dist.index(min(dist))],(10,500), cv2.FONT_HERSHEY_SIMPLEX, 4,(255,255,255),2,cv2.LINE_AA)
        # cv2.imshow("d", resized)
