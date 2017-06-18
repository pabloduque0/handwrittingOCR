import cv2
import numpy as np

# In this test we will use SVM instead of kNN

size=20
bin_n = 16 # Number of bins

svm_params = dict( kernel_type = cv2.SVM_LINEAR,
                    svm_type = cv2.SVM_C_SVC,
                    C=2.67, gamma=5.383 )

thisFlags = cv2.WARP_INVERSE_MAP|cv2.INTER_LINEAR

# First deskew image
def deskew(image):
    myMoments = cv2.moments(image)
    if abs(myMoments['mu02']) < 1e-2:
        return image.copy()
    skew = myMoments['mu11']/myMoments['mu02']
    M = np.float32([[1, skew, -0.5*size*skew], [0, 1, 0]])
    image = cv2.warpAffine(image,M,(size, size),flags=thisFlags)
    return image

# HOG function
def hog(image):
    gx = cv2.Sobel(image, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(image, cv2.CV_32F, 0, 1)
    mag, ang = cv2.cartToPolar(gx, gy)
    bins = np.int32(bin_n*ang/(2*np.pi))    # quantizing binvalues in (0...16)
    bin_cells = bins[:10,:10], bins[10:,:10], bins[:10,10:], bins[10:,10:]
    mag_cells = mag[:10,:10], mag[10:,:10], mag[:10,10:], mag[10:,10:]
    hists = [np.bincount(b.ravel(), m.ravel(), bin_n) for b, m in zip(bin_cells, mag_cells)]
    hist = np.hstack(hists)     # hist is a 64 bit vector
    return hist

def GaussianFilter(sigma):
    halfSize = 3 * sigma
    maskSize = 2 * halfSize + 1 
    mat = np.ones((maskSize,maskSize)) / (float)( 2 * np.pi * (sigma**2))
    xyRange = np.arange(-halfSize, halfSize+1)
    xx, yy = np.meshgrid(xyRange, xyRange)    
    x2y2 = (xx**2 + yy**2)    
    exp_part = np.exp(-(x2y2/(2.0*(sigma**2))))
    mat = mat * exp_part

    return mat



image = cv2.imread('digits.png',0)

# If desired, image can be previously smoothed
gaussianFilter = GaussianFilter(1)
gaussianGray1 = cv2.filter2D(image, -1, gaussianFilter)

cells = [np.hsplit(row,100) for row in np.vsplit(image,50)]

# First half is trainData, remaining is testData
train_cells = [ i[:50] for i in cells ]
test_cells = [ i[50:] for i in cells]

# Training data

deskewedImage = [map(deskew,row) for row in train_cells]
hogData = [map(hog,row) for row in deskewedImage]
trainData = np.float32(hogData).reshape(-1,64)
dataResponses = np.float32(np.repeat(np.arange(10),250)[:,np.newaxis])

svm = cv2.SVM()
svm.train(trainData,dataResponses, params=svm_params)
svm.save('svm_data.dat')

# Testing data

deskewedImage = [map(deskew,row) for row in test_cells]
hogData = [map(hog,row) for row in deskewedImage]
testData = np.float32(hogData).reshape(-1,bin_n*4)
result = svm.predict_all(testData)

# Checking accuracy
mask = result==dataResponses
correct = np.count_nonzero(mask)
print correct*100.0/result.size