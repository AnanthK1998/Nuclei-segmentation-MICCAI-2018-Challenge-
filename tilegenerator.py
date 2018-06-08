import glob
import cv2
import math
import numpy

import geojson
import os

from PIL import Image, ImageDraw

#from matplotlib import pyplot as plt

class TileGenerator:
    def __init__(self,w=224,h=224):
        self.w = w
        self.h = h

    def __str__(self):
        return str({"w":self.w, "h":self.h})

    def setImageSize(self,rows,cols):
        self.rows = rows
        self.cols = cols

        g1 = math.gcd(rows, self.h)
        g2 = math.gcd(cols, self.w)

        self.g1 = g1
        self.g2 = g2

        top = [-g1]
        left = [-g2]
        nex = left[-1] + w
        ne = top[-1] + h
        while nex < cols:
            left.append(nex)
            nex = left[-1] + w
        while ne < rows:
            top.append(ne)
            ne = top[-1] + h

        self.top = top
        self.left = left

        self.cur_ii = None

    def shape(self):
        return (len(self.top),len(self.left))

    def getRect(self,x,y):
        assert(type(x)==int and type(y)==int)
        return (self.left[x],self.top[y],self.w+self.g2,self.h+self.g1)

    def getRect(self,ii):
        sh = self.shape()
        assert(type(ii)==int and ii < sh[0]*sh[1])
        x = ii % sh[1]
        y = math.floor(ii / sh[1])
        return (self.left[x],self.top[y], self.w+self.g2, self.h+self.g1)

    def __iter__(self):
        return self

    def __next__(self):
        if self.cur_ii is None:
            self.cur_ii = 0
        else:
            self.cur_ii = self.cur_ii + 1

        sh = self.shape()
        if self.cur_ii >= sh[0]*sh[1]:
            raise StopIteration()

        return self.getRect(self.cur_ii)

def rectToExtent(rect):
    x,y,w,h=rect
    #return x,y,x+w,y+h
    return y,y+h, x,x+w

def extentToRect(ext):
    y1,y2,x1,x2=ext
    return x1,y1,x2-x1,y2-y1


def getSubimage(img, ext, padded = True):
    y1,y2,x1,x2 = ext

    sz = img.shape

    if len(sz)==2:
        sz = list(sz)+[1]

    if y1 > 0 and x1 > 0 and y2<sz[0] and x2<sz[1]:
        return img[y1:y2, x1:x2, :]

    if not padded:

        if y1< 0:
            y1=0
        if x1 < 0:
            x1 = 0

        if y2>=sz[0]:
            y2 = sz[0]-1

        if x2>=sz[1]:
            x2 = sz[1]-1

        return img[y1:y2, x1:x2, :]

    else:
        outim = numpy.zeros((y2-y1,x2-x1,sz[2]), img.dtype)

        starty = 0
        startx = 0
        endy = y2-y1
        endx = x2-x1

        if y1 < 0:
            starty = -y1
            y1 = 0

        if x1 < 0:
            startx = -x1
            x1 = 0

        if y2 >= sz[0]:
            endy = endy - (y2 - (sz[0] -1))
            y2 = sz[0]-1


        if x2 >= sz[1]:
            endx = endx - (x2 - (sz[1] -1))
            x2 = sz[1]-1

        outim[starty:endy,startx:endx,:] = img[y1:y2, x1:x2, :]

        return outim


def poly2mask(gjcoord, maskimg):
    cc = numpy.array(gjcoord).ravel().tolist()
    #print(cc)
    ImageDraw.Draw(maskimg).polygon(cc,outline=255,fill=1)
    return maskimg

if __name__=="__main__":

    imgdir = glob.glob('renamed/*png')

    cv2.namedWindow("img")

    w = 224
    h = 224

    tg = TileGenerator(w, h)

    #import pickle
    #pickle.dump(tg,open('tgconf.pkl','wb'))

    #tg = pickle.load(open('tgconf.pkl','rb'))
    #print(tg)

    for image in imgdir:
        print(image)
        jsonfilename = (os.path.basename(image).split('_')[-1]).replace('png','json')

        annot = geojson.load(open(jsonfilename))

        print(jsonfilename)
        img = cv2.imread(image)
        #im = Image.open(image)
        rows = img.shape[0]
        cols = img.shape[1]

        #mask = Image.fromarray(numpy.zeros((rows,cols),numpy.uint8))
        mask = Image.new("L",(cols,rows),0)
        for cc in annot:
            mask = poly2mask(cc.coordinates, mask)

        mask = numpy.array(mask)
        mask = numpy.dstack((mask,mask,mask))

        #cv2.imshow("img",mask)
        #cv2.waitKey()

        print(img.shape)

        tg.setImageSize(rows,cols)

        shp = tg.shape()

        print(shp)

        for r12 in tg:
            print(r12)
            e12 = rectToExtent(r12)
            #print(e12)
            #
            sub = getSubimage(img,e12)

            submask = getSubimage(mask,e12)

            cv2.imshow("img", numpy.hstack((sub.astype(submask.dtype),submask)))
            cv2.waitKey()
        #plt.hold(True)
        #plt.plot(r12[0], r12[1], 'bx')
        #plt.plot(r12[0]+r12[2], r12[1]+r12[3], 'bx')
        #plt.hold(False)

        #break
    #plt.show()
