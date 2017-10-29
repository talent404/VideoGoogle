import cv2 
import numpy as np
import os

images = []
descriptors = np.array([])

for i in os.listdir('output'):
    img = cv2.imread('output/'+i)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sift = cv2.xfeatures2d.SIFT_create()
    kp, des = sift.detectAndCompute(gray, None)
    print(i)
    if des is None:
        continue
    images.append(img)
    if descriptors.shape != (0,): 
        descriptors = np.vstack((descriptors, des))
    else:
        descriptors = des
print(descriptors.shape)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
#descriptors = descriptors.reshape(int(len(descriptors)/128),128)
#print(type(len(descriptors)/128),len(descriptors)/128,len(descriptors))
#desc = np.reshape(descriptors, (len(descriptors)/128, 128))
desc = np.float32(descriptors)
ret, label, center = cv2.kmeans(desc,200,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
print(center)
