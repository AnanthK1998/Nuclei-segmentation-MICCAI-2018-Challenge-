import glob
import cv2
import os
import numpy as np
imgdir= glob.glob("test1/*.png")
newname=0
for img in imgdir:
    image= cv2.imread(img)
    img1=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    cv2.imwrite("testg/"+str(newname)+".png",img1)
    newname=newname+1
#imag=np.array(img)
#imag1=np.reshape(imag,(3*imag.shape[0]*imag.shape[1],1))
#print(max(imag1),min(imag1))
#np.t


