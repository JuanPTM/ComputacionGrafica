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
    allHistograms =[]
    orb = cv2.ORB_create()
    allDescriptors = None
    for path in nameVideos:
        histograms = None
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

            if descriptors is not None:
                histogram, limites = np.histogram(vq(descriptors/dev, codebook)[:][1],bins=range(21))
                if histograms is None:
                    histograms = histogram
                else:
                    histograms = np.vstack((histograms, histogram))
        allHistograms.append(histograms)
    return allHistograms

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
    regions = []
    image,contours,hierarchy = cv2.findContours(binFrame,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    for cont in contours:
        if len(cont) > minCont and cv2.contourArea(cont)>2500:
            x,y,w,h = cv2.boundingRect(cont)
            paddingx = paddingy = paddingw = paddingh = 10
            if x < 10:
                paddingx = x
            if y < 10:
                paddingy = y
            if x+w+padding > image.shape[1]:
                paddingw = image.shape[1]-x-w
            if y+h+padding > image.shape[0]:
                paddingh = image.shape[0]-y-h
            regions.append((x-paddingx,y-paddingy, w+paddingx+paddingw, h+paddingy+paddingh))
    return regions,contours

def union(a,b):
  x = min(a[0], b[0])
  y = min(a[1], b[1])
  w = max(a[0]+a[2], b[0]+b[2]) - x
  h = max(a[1]+a[3], b[1]+b[3]) - y
  return (x, y, w, h)

def intersection(a,b):
  x = max(a[0], b[0])
  y = max(a[1], b[1])
  w = min(a[0]+a[2], b[0]+b[2]) - x
  h = min(a[1]+a[3], b[1]+b[3]) - y
  if w<0 or h<0: return None # or (0,0,0,0) ?
  return (x, y, w, h)

if __name__ == "__main__":
    # try:
    #     codebook = pickle.load(open('codebook.pickle', 'r'))
    #     dev = pickle.load(open('dev.pickle', 'r'))
    # except:
    #     print "comienzo"
    #     descriptors = GetAllDescriptors()
    #     codebook, dev = GenerateCodebook(descriptors)
    #     pickle.dump(dev, open('dev.pickle', 'w'))
    #     pickle.dump(codebook, open('codebook.pickle', 'w'))
    #     print "He terminado"
    # try:
    #     histograms = pickle.load(open('allHistograms.pickle', 'r'))
    # except:
    #     allHistograms = GenerateAllHistograms()
    #     pickle.dump(allHistograms, 'allHistograms.pickle', 'w'))
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

        regions, contornos = extractIntReg(procesedImage)
        cv2.drawContours(grayScale, contornos, -1, (0,255,0), 3)

        show_image = np.copy(procesedImage)

        for r in regions:
            cv2.rectangle(grayScale,(r[0],r[1]),(r[0]+r[2],r[1]+r[3]),(255,0,0))

        cv2.imshow("d", grayScale)

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
