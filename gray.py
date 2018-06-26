import cv2
import glob
import os
#os.mkdir("testg")
imgdir=glob.glob("tests/*.png")
for img in imgdir:
    imag=cv2.imread(img)
    image_name=os.path.basename(img)
    imag1=cv2.cvtColor(imag, cv2.COLOR_BGR2GRAY)
    cv2.imwrite("testg/"+image_name,imag1)