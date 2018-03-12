import cv2
import pickle
import numpy as np
from scipy.cluster.vq import vq, kmeans
from scipy.spatial import distance
import time
pathVideos = "movies/"
extension = ".webm"
nameVideos=["arbol", "casa", "zagal", "pez"]

MAXAREA = 500

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
            crop = cropToRequirements(frame)

            # Convertimos a gris
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)


            # Escalamos
            grayScale = scale(gray)

            binImg = FilterByRange(grayScale)

            procesedImage = morphFilters(binImg)

            regions, contornos = extractIntReg(procesedImage)

            for r in regions:
                imgReg = grayScale[r[1]:r[1]+r[3], r[0]:r[0]+r[2]]
                # Sacamos keypoints y descriptores
                keypoints, descriptors = orb.detectAndCompute(imgReg, None)
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

def GenerateAllHistograms(codebook, dev):
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
            crop = cropToRequirements(frame)

            # Convertimos a gris
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)


            # Escalamos
            grayScale = scale(gray)

            binImg = FilterByRange(grayScale)

            procesedImage = morphFilters(binImg)

            regions, contornos = extractIntReg(procesedImage)

            for r in regions:
                imgReg = grayScale[r[1]:r[1]+r[3], r[0]:r[0]+r[2]]
                # Sacamos keypoints y descriptores
                keypoints, descriptors = orb.detectAndCompute(imgReg, None)
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
            if x+w+padding >= image.shape[1]:
                paddingw = image.shape[1] - x - w - 1
            if y+h+padding >= image.shape[0]:
                paddingh = image.shape[0] - y - h - 1
            regions.append((x-paddingx,y-paddingy, w+paddingx+paddingw, h+paddingy+paddingh))
            # regions.append((x,y, w, h))
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
  if w<0 or h<0: return (0,0,0,0)
  return (x, y, w, h)

def area(a):
    return a[2] * a[3]

def areaIntersec(a,b):
    AInter = area(intersection(a, b))
    AUnion = area(union(a, b))
    Porcent = 0.
    if AInter is not 0:
        Porcent = float(AInter)/AUnion
    return AInter, AUnion, Porcent
    # if AInter is not 0:
    #     PercentInterR1 = area(a)/AInter
    #     PercentInterR2 = area(b)/AInter
    # else:
    #     PercentInterR1 = 0
    #     PercentInterR2 = 0
    # return AInter, PercentInterR1,PercentInterR2

def Init_Matcher():
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
    try:
        allHistograms = pickle.load(open('allHistograms.pickle', 'r'))
    except:
        allHistograms = GenerateAllHistograms(codebook, dev)
        pickle.dump(allHistograms, open('allHistograms.pickle', 'w'))
    return allHistograms, codebook, dev

if __name__ == "__main__":

    allHistograms, codebook, dev = Init_Matcher()

    orb = cv2.ORB_create()
    capture = cv2.VideoCapture("movies/test.webm")
    while True:
        # Capturamos
        ret, frame = capture.read()
        if ret == False:
            break

        # Recortamos
        crop = cropToRequirements(frame)

        # Convertimos a gris
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)


        # Escalamos
        grayScale = scale(gray)

        binImg = FilterByRange(grayScale)

        procesedImage = morphFilters(binImg)

        regions, contornos = extractIntReg(procesedImage)

        outImage = np.copy(scale(crop))

        cv2.drawContours(outImage, contornos, -1, (0,255,0), 3)

        #
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break
        cv2.waitKey(1000)
        guessReg = []
        for r in regions:
            imgReg = grayScale[r[1]:r[1]+r[3], r[0]:r[0]+r[2]]
            # cv2.imshow("asdads", imgReg)
            keypoints, descriptors = orb.detectAndCompute(imgReg, None)
            #
            if descriptors is not None:
                hist, limites = np.histogram(vq(descriptors/dev, codebook)[:][1],bins=range(21))
                dist = []
                for i in range(len(allHistograms)):
                    distancias = distance.cdist(allHistograms[i], np.array([hist]), "euclidean")
                    distancias.sort()
                    map(lambda x: dist.append( ( x[0], nameVideos[i] )), distancias[0:20])
                dist.sort()
                # print dist
                guess = "Desconocido"
                minValue = 0
                for i in range(0,len(dist)):
                    if dist[0][1] != dist[i][1]:
                        # print dist[0], dist[i]
                        if dist[0][0]/dist[i][0] < 0.75:
                            guess = dist[0][1]
                        else:
                            guess = "Desconocido"
                        minValue = dist[0][0]/dist[i][0]
                        break
                guessReg.append((r,guess,minValue))


        newGuessReg = []
        flagsRemove = [0]*len(guessReg)

        for i in range(len(guessReg)):
            for j in range(i+1,len(guessReg)-1):
                if flagsRemove[i] is 1 or flagsRemove[j] is 1:
                    continue
                r1 = guessReg[i]
                r2 = guessReg[j]

                AInter,AUnion,Porcent = areaIntersec(r1[0], r2[0])
                if AInter is 0:
                    continue
                print r1[1],r2[1], Porcent
                if r1[1] == r2[1]:
                    if Porcent>0.25:
                        newGuessReg.append((union(r1[0], r2[0]),r1[1],r1[2]))
                        flagsRemove[i] = 1
                        flagsRemove[j] = 1
                        # break
                else:
                    if r1 is "Desconocido":
                        flagsRemove[i] = 1
                    elif r2 is "Desconocido":
                        flagsRemove[j] = 1
                    elif Porcent>0.25:
                        if r1[2]<r2[2]:
                            flagsRemove[j] = 1
                        else:
                            flagsRemove[i] = 1

        guessReg = [guessReg[i] for i in range(len(guessReg)) if flagsRemove[i] is not 1]


        newGuessReg.extend(guessReg)


        for tupla in newGuessReg:
            r = tupla[0]
            guess = tupla[1]
            cv2.rectangle(outImage,(r[0],r[1]),(r[0]+r[2],r[1]+r[3]),(255,0,0))
            cv2.putText(outImage, guess, (r[0],r[1]), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),2,cv2.LINE_AA)
            cv2.imshow("d", outImage)
