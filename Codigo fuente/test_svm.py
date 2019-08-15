import glob, os, cv2
import numpy as np
from scipy import ndimage, signal
from PIL import Image
import math
import pytesseract
from sklearn.externals import joblib
from statistics import mode, StatisticsError


def sort_contours(cnts, method="left-to-right"):
	# initialize the reverse flag and sort index
	reverse = False
	i = 0
 
	# handle if we need to sort in reverse
	if method == "right-to-left" or method == "bottom-to-top":
		reverse = True
 
	# handle if we are sorting against the y-coordinate rather than
	# the x-coordinate of the bounding box
	if method == "top-to-bottom" or method == "bottom-to-top":
		i = 1
 
	# construct the list of bounding boxes and sort them from top to
	# bottom
	boundingBoxes = [cv2.boundingRect(c) for c in cnts]
	(cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
		key=lambda b:b[1][i], reverse=reverse))
 
	# return the list of sorted contours and bounding boxes
	return (cnts, boundingBoxes)


pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files (x86)/Tesseract-OCR/tesseract.exe'
directory = "./images_test_algorithm"
cont = 0
file = open('plates.txt','r')
manual_results = file.read().split("\n")
clf = joblib.load('model_svm.pkl')

for index in sorted(glob.glob(os.path.join(directory,'*.jpg'))):
	img = cv2.imread(index, 0)
	img = cv2.resize(img, (500, 250), interpolation=cv2.INTER_CUBIC)
	clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(3, 3))
	img = clahe.apply(img)
	minuendo = img.copy()
	img = signal.medfilt(img, 3)
	img = cv2.GaussianBlur(img, (7, 7), 1.82)
	img = ndimage.median_filter(img, size=(55, 55))
	img = minuendo - img
	min = np.min(img)
	img = img - min
	max = np.max(img)
	scale = float(255)/ (max)
	for row in range(0, img.shape[0]):
		for col in range(0, img.shape[1]):
			img[row, col] = int(img[row, col]*scale)

	img = cv2.adaptiveThreshold(img.astype(np.uint8), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 89, 2)
	hist = cv2.calcHist([img], [0], None, [256], [0, 256])
	indexmax = np.where(hist == np.max(hist))[0][0]
	if(indexmax < 150):
		img = abs(255 - img)
	
	aux = img.copy()
	aux[0:30, :] = 255
	aux[:, 0:10] = 255
	aux[250-30:250, :] = 255
	aux[:, 500-10:500] = 255

	aux = cv2.morphologyEx(aux, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_CROSS, (7, 7)))
	aux = abs(255-aux) 

	connected_components = cv2.connectedComponentsWithStats(aux, 8, cv2.CV_32S)
	n = connected_components[0] + 1
	for x in range(1, n):
		mask3 = connected_components[1] == x
		sum = int(np.sum(connected_components[1][mask3])) / x
		if(sum < 1000):
			aux[mask3] = 0
		if(sum > 9000):
			aux[mask3] = 0
	
	print(index)

	contours, npaContours, npaHierarchy = cv2.findContours(aux, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	mode = []
	con = 0
	npaContours = sorted(npaContours, key=cv2.contourArea)

	for npaContour in npaContours: 
		(x1, y1), radius = cv2.minEnclosingCircle(npaContour)
		if(radius > 25 and radius < 120):
			mode.append(radius)
			con += 1

	l = len(mode)
	for i in range(0, l):
		for j in range(i, l):
			if(mode[j] < mode[i]):
				naux = mode[j]
				mode[j] = mode[i]
				mode[i] = naux
	nmode = mode[math.ceil(con/2)]

	for npaContour in npaContours: 
		(x1, y1), radius = cv2.minEnclosingCircle(npaContour)
		center = (int (x1), int (y1))
		radius = int (radius)
		if(radius < (nmode-10) or radius > (nmode+10)):
			cv2.drawContours(aux, [npaContour], -1, (0,0,0), -1)

	aux = abs(255-aux) 
	aux = cv2.morphologyEx(aux, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))

	contours, npaContours, npaHierarchy = cv2.findContours(aux, cv2.RETR_TREE,  cv2.CHAIN_APPROX_SIMPLE)
	npaContours, boundingBoxes = sort_contours(npaContours, method="left-to-right")
	platePrediction  = ""
	aux2 = aux.copy()
	for npaContour in npaContours: 
		(x1, y1), radius = cv2.minEnclosingCircle(npaContour)
		radius = int (radius)
		if(radius > (nmode-10) and radius < (nmode+10)):
			x,y,w,h = cv2.boundingRect(npaContour)
			cv2.rectangle(aux2, (x, y), (x+w, y+h), (0, 0, 0), 2)
			separeted_char = np.zeros((h+50, (w)+100))
			separeted_char[:, :] = 255
			imgaux = cv2.morphologyEx(aux[y:y+h, x:x+w], cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7)))
			separeted_char[25:h+25, 50:w+50] = imgaux
			separeted_char2 = cv2.resize(separeted_char, (128, 128), interpolation=cv2.INTER_CUBIC)
			msk = separeted_char2 > 1
			separeted_char2[msk] = 1
			separeted_char2 = separeted_char2.astype(np.bool_)
			predict = clf.predict(separeted_char2.reshape(1, 16384))
			cPredict = ""
			if(predict < 10):
				cPredict = str(predict[0])
			else:
				cPredict = chr(predict+55)
			platePrediction += cPredict
			cont += 1
		
	print("Prediction: ", platePrediction)
	print("------------------------------------")
	cv2.imshow("Car Plate", aux2)
	cv2.waitKey(0)
	cv2.destroyAllWindows()


