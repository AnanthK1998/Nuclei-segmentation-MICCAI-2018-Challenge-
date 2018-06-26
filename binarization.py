import cv2
import glob
import os
import numpy as np
import sklearn
from skimage.morphology import skeletonize
img="result/9_predict.png"
kernel = np.ones((3,3),np.uint8)

imag=cv2.imread(img)
imagename=os.path.basename(img)
edges = cv2.Canny(imag,90, 10)
closing = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
ske = (skeletonize(closing//255) * 255).astype(np.uint8)
#erosion = cv2.erode(closing, kernel, iterations=1)
cv2.imwrite("result1/"+imagename,ske)
cv2.imshow('img',ske)
cv2.waitKey()

    #print(imag)


#    imag2=[]
#    for i in imag1:
#       for j in i:
#           if j>0:
#               j=255
#               imag2.append(j)
#           else:
#               j=255
#               imag2.append(j)

