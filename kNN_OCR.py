import numpy as np
import cv2
from matplotlib import pyplot as plt


def findAccuracy(cells):

	# Make it into a Numpy array. It size will be (50,100,20,20)
	npArrayInput = np.array(cells)

	# Now we prepare train_data and test_data.
	train = npArrayInput[:,:50].reshape(-1,400).astype(np.float32) # Size = (2500,400)
	test = npArrayInput[:,50:100].reshape(-1,400).astype(np.float32) # Size = (2500,400)

	# Create labels for train and test data
	k = np.arange(10)
	train_labels = np.repeat(k,250)[:,np.newaxis]
	test_labels = train_labels.copy()

	# Initiate kNN, train the data, then test it with test data for k=1
	knn = cv2.ml.KNearest_create()
	knn.train(train, cv2.ml.ROW_SAMPLE, train_labels)
	ret,result,neighbours,dist = knn.find_nearest(test, k=5)

	# Now we check the accuracy of classification
	# For that, compare the result with test_labels and check which are wrong
	matches = result==test_labels
	correct = np.count_nonzero(matches)
	accuracy = correct*100.0/result.size
	print accuracy


img1 = cv2.imread('digits.png')
gray1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)

# Now we split the image to 5000 cells, each 20x20 size
cells1 = [np.hsplit(row,100) for row in np.vsplit(gray1,50)]


findAccuracy(cells1)


