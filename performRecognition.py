# Import the modules
import cv2
from sklearn.externals import joblib
from skimage.feature import hog
import numpy as np

# Load the classifier created in generateClassifier.py
clf = joblib.load("digits_cls.pkl")

# Read the input image 
im = cv2.imread("photo_3.jpg")
im = cv2.resize(im, (1080, 1080), interpolation=cv2.INTER_AREA)
cv2.imshow("Original", im)
cv2.waitKey(0)
cv2.destroyAllWindows()


# Convert to grayscale and apply Gaussian filtering to remove noise
im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
im_gray = cv2.GaussianBlur(im_gray, (5, 5), 0)


# Threshold the image(aka select pixels that exceed a certain value) then convert it to a binary image
ret, im_th = cv2.threshold(im_gray, 60, 255, cv2.THRESH_BINARY_INV)
cv2.imshow("Threshold Image", im_th)

# Find contours (a curve joining all continuous points along a boundary of the same color and intensity) in the image
ctrs, hier = cv2.findContours(im_th.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Create a list of bounding rectangles for every contour found
rects = [cv2.boundingRect(ctr) for ctr in ctrs]

# For each rectangular region, calculate HOG features and predict
# the digit using Linear SVM.
for rect in rects:
    if ((rect[2] > 14) and (rect[3] > 14)):
    # Draw the rectangles
        cv2.rectangle(im, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0), 3)
    
        # Make the rectangular region around the digit
        leng = int(rect[3] * 1.6)
        topBorder = (leng - rect[3]) // 2
        leftBorder = (leng - rect[2]) // 2
        #print('Length: {}'.format(leng))
        
        #roi = im_th[pt1:pt1+leng, pt2:pt2+leng]
        temp = im_th[rect[1]:rect[1]+rect[3], rect[0]:rect[0]+rect[2]]
        roi = np.zeros((leng,leng),temp.dtype)
        roi[topBorder:(temp.shape[0]+topBorder), leftBorder:(temp.shape[1]+leftBorder)] = temp

        # Resize the image
    
        roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
        roi = cv2.dilate(roi, (3, 3))

        # Calculate the HOG features
        roi_hog_fd = hog(roi, orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1), visualise=False)
        nbr = clf.predict(np.array([roi_hog_fd], 'float64'))
        print('Prediction = {}'.format(nbr[0]))

        cv2.putText(im, str(int(nbr[0])), (rect[0], rect[1]),cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 255), 3)

cv2.imshow("Resulting Image with Rectangular ROIs", im)
cv2.waitKey(0)
cv2.destroyAllWindows()

