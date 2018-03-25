import cv2
import pickle
import numpy as np
from scipy.cluster.vq import vq, kmeans
from sklearn.neighbors import KNeighborsClassifier

pathVideos = "movies/"
extension = ".webm"
nameVideos = ["arbol", "casa", "zagal", "pez"]

MAXAREA = 500


def GetAllDescriptors():
    orb = cv2.ORB_create()
    allDescriptors = None
    for path in nameVideos:
        counter = 0
        segmented = None
        prevFrame = None
        print pathVideos + path + extension
        capture = cv2.VideoCapture(pathVideos + path + extension)
        while True:
            # Capturamos
            ret, frame = capture.read()
            if ret == False:
                break
            counter += 1
            if counter % 10 is not 0:
                continue

            # Convertimos a gris
            crop = cropToRequirements(frame)

            # Convertimos a gris
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            if prevFrame is not None:
                segmented = SegmentFrame(gray, prevFrame)
            prevFrame = np.copy(gray)

            # Escalamos
            grayScale = scale(gray)

            binImg = FilterByRange(grayScale)

            procesedImage = morphFilters(binImg)

            regions, contornos = extractIntReg(procesedImage)

            if len(regions) > 4:
                continue

            for r in regions:
                imgReg = grayScale[r[1]:r[1] + r[3], r[0]:r[0] + r[2]]
                # Sacamos keypoints y descriptores
                keypoints, descriptors = orb.detectAndCompute(imgReg, None)
                if descriptors is not None and descriptors.shape[0] > 20:
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
    allHistograms = []
    orb = cv2.ORB_create()
    allDescriptors = None
    for path in nameVideos:
        counter = 0
        histograms = None
        print pathVideos + path + extension
        capture = cv2.VideoCapture(pathVideos + path + extension)
        while True:
            # Capturamos
            ret, frame = capture.read()
            if ret == False:
                break
            counter += 1
            if counter % 10 is not 0:
                continue
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
                imgReg = grayScale[r[1]:r[1] + r[3], r[0]:r[0] + r[2]]
                # Sacamos keypoints y descriptores
                keypoints, descriptors = orb.detectAndCompute(imgReg, None)
                if descriptors is not None and descriptors.shape[0] > 20:
                    histogram, limites = np.histogram(vq(descriptors / dev, codebook)[0], bins=range(21))
                    #                    histogram, limites = np.histogram(vq(descriptors/dev, codebook)[:][1],bins=range(21))
                    if histograms is None:
                        histograms = histogram
                    else:
                        histograms = np.vstack((histograms, histogram))
        allHistograms.append(histograms)
    return allHistograms


def SegmentFrame(frameGray, prevframeGray):
    m = np.subtract(prevframeGray.astype(np.int16), frameGray.astype(np.int16))
    return np.absolute(m).astype(np.int8)


def FilterByRange(frameGray, minValue=90, maxValue=250):
    return 255 - cv2.inRange(frameGray, minValue, maxValue)


def cropToRequirements(image):
    return image[100:-100, 100:-170]


def scale(image, col=1120):
    col = float(col)
    return cv2.resize(image, None, fx=col / image.shape[1], fy=col / image.shape[1])


def morphFilters(binFrame):
    kernel = np.ones((3, 3), np.uint8)
    dilateImage = cv2.dilate(binFrame, kernel, iterations=1)
    erodeImg = cv2.erode(dilateImage, kernel, iterations=2)
    dilateImage = cv2.dilate(erodeImg, kernel, iterations=4)
    return dilateImage


def extractIntReg(binFrame, padding=10, minCont=50):
    regions = []
    image, contours, hierarchy = cv2.findContours(binFrame, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for cont in contours:
        if cv2.contourArea(cont) > 300:
            x, y, w, h = cv2.boundingRect(cont)
            paddingx = paddingy = paddingw = paddingh = 10
            if x < 10:
                paddingx = x
            if y < 10:
                paddingy = y
            if x + w + padding >= image.shape[1]:
                paddingw = image.shape[1] - x - w - 1
            if y + h + padding >= image.shape[0]:
                paddingh = image.shape[0] - y - h - 1
            regions.append((x - paddingx, y - paddingy, w + paddingx + paddingw, h + paddingy + paddingh))
            # regions.append((x,y, w, h))
    return regions, contours


def union(a, b):
    x = min(a[0], b[0])
    y = min(a[1], b[1])
    w = max(a[0] + a[2], b[0] + b[2]) - x
    h = max(a[1] + a[3], b[1] + b[3]) - y
    return (x, y, w, h)


def intersection(a, b):
    x = max(a[0], b[0])
    y = max(a[1], b[1])
    w = min(a[0] + a[2], b[0] + b[2]) - x
    h = min(a[1] + a[3], b[1] + b[3]) - y
    if w < 0 or h < 0: return (0, 0, 0, 0)
    return (x, y, w, h)


def area(a):
    return a[2] * a[3]


def areaIntersec(a, b):
    AInter = area(intersection(a, b))
    AUnion = area(union(a, b))
    Porcent = 0.
    if AInter is not 0:
        Porcent = float(AInter) / AUnion
    return AInter, AUnion, Porcent


def Init_Matcher():
    try:
        codebook = pickle.load(open('codebook_s1.pickle', 'r'))
        dev = pickle.load(open('dev_s1.pickle', 'r'))
    except:
        print "Comienzo a generar codebook"
        descriptors = GetAllDescriptors()
        codebook, dev = GenerateCodebook(descriptors)
        pickle.dump(dev, open('dev_s1.pickle', 'w'))
        pickle.dump(codebook, open('codebook_s1.pickle', 'w'))
        print "He terminado de generar el codebook"
    try:
        allHistograms = pickle.load(open('allHistograms_s1.pickle', 'r'))
    except:
        allHistograms = GenerateAllHistograms(codebook, dev)
        pickle.dump(allHistograms, open('allHistograms_s1.pickle', 'w'))
    return allHistograms, codebook, dev


def classifyRegion(grayScaleFrame, regions, orb, neigh):
    guessReg = []
    for r in regions:
        imgReg = grayScaleFrame[r[1]:r[1] + r[3], r[0]:r[0] + r[2]]

        keypoints, descriptors = orb.detectAndCompute(imgReg, None)

        if descriptors is not None and descriptors.shape[0] > 20:

            hist, limites = np.histogram(vq(descriptors / dev, codebook)[0], bins=range(21))
            guess = neigh.predict([hist])[0]
            prob = np.amax(neigh.predict_proba([hist])[0])
            # print prob
            # print guess
            # print neigh.predict_proba([hist])
            if prob <= 0.5:
                guess = "Desconocido"
            guessReg.append((r, guess, prob))

    return guessReg


def joinRegions(guessReg):
    newGuessReg = []
    flagsRemove = [0] * len(guessReg)

    for i in range(len(guessReg)):
        for j in range(i + 1, len(guessReg)):

            r1 = guessReg[i]
            r2 = guessReg[j]

            AInter, AUnion, Porcent = areaIntersec(r1[0], r2[0])

            if AInter is 0:
                continue
            # print r1[1],r2[1], Porcent, r1[2],r2[2]
            if r1[1] == r2[1]:
                if Porcent > 0.25:
                    newGuessReg.append((union(r1[0], r2[0]), r1[1], r1[2]))
                    flagsRemove[i] = 1
                    flagsRemove[j] = 1
            else:
                if r1[1] is "Desconocido":
                    flagsRemove[i] = 1
                elif r2[1] is "Desconocido":
                    flagsRemove[j] = 1
                elif Porcent > 0.20:
                    if r1[2] > r2[2]:
                        flagsRemove[j] = 1
                        # print r1[1], r2[1], Porcent, r1[2], r2[2], "ELEJIDO "+r1[1]
                    else:
                        flagsRemove[i] = 1
                        # print r1[1], r2[1], Porcent, r1[2], r2[2], "ELEJIDO " + r2[1]

    guessReg = [guessReg[i] for i in range(len(guessReg)) if flagsRemove[i] is not 1]

    newGuessReg.extend(guessReg)

    return newGuessReg


def printImage(image, regions):
    for reg in regions:
        r = reg[0]
        guess = reg[1]
        cv2.rectangle(image, (r[0], r[1]), (r[0] + r[2], r[1] + r[3]), (255, 0, 0), 3)
        cv2.putText(image, guess + " " + str(reg[2] * 100), (r[0], r[1]), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow("d", image)


if __name__ == "__main__":

    allHistograms, codebook, dev = Init_Matcher()

    orb = cv2.ORB_create()
    capture = cv2.VideoCapture("movies/test.webm")

    X = [x for h in allHistograms for x in h]
    Y = [nameVideos[c] for c in range(len(nameVideos)) for x in range(len(allHistograms[c]))]

    neigh = KNeighborsClassifier(n_neighbors=5)
    neigh.fit(X, Y)

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

        cv2.drawContours(outImage, contornos, -1, (0, 255, 0), 3)

        guessReg = classifyRegion(grayScale, regions, orb, neigh)

        finalReg = joinRegions(guessReg)

        printImage(outImage, finalReg)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
