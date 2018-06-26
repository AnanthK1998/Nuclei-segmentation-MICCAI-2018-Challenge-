import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
#from pprint import pprint
import geojson 
from shapely.geometry import Polygon
from PIL import Image, ImageDraw
import cv2
import glob
import tilegenerator as tg

def adjacency(annot_json):
    """
    create adjacency matrix of contours (adjacent=> touching/overlapping)
    """
    topRight=[]
    bottomLeft=[]
    topLeft=[]
    bottomRight=[]
    count=0
    countG=0
    for i in annot_json:
        x=[]
        y=[]
        
        for j in i["coordinates"]:
            x.append(j[0])
            y.append(j[1])
        x_max=max(x)+1
        y_max=max(y)+1
        x_min=min(x)-1
        y_min=min(y)-1
        topRight.append((x_max,y_min))
        bottomLeft.append((x_min,y_max))
        topLeft.append((x_min,y_min))
        bottomRight.append((x_max,y_max))
        count=count+1 
    print(count)    
    
    #mat=np.zeros((count,count))
    G = nx.Graph()
    #color_map=[]
    for i in range(count):
        poly1 = Polygon([topLeft[i],topRight[i],bottomRight[i],bottomLeft[i]])
        #print(poly1)
        for j in range(count):
            if j>i:
                poly2 = Polygon([topLeft[j],topRight[j],bottomRight[j],bottomLeft[j]])
                poly_inter = poly1.intersects(poly2)
                #print(poly_inter)
                if poly_inter==True:
                    #mat[i][j]=1
                    G.add_edge(i, j)
                    countG=countG+1
                else:
                    #mat[i][j]=0
                    pass
                    
        #print(i,np.where(mat[i,:])[0])
    
    return G

def mask2poly(mask):
    
    tmp = mask.copy()
    _, contours, hierarchy = cv2.findContours(tmp, cv2.RETR_EXTERNAL , cv2.CHAIN_APPROX_SIMPLE)
        
    #print(hierarchy)
    features = []        
    for ci in contours:
        coords = []
        for pt in ci:
            pt=pt.flatten().tolist()
            coords.append(pt)
              
        coords.append(coords[0])
            
        gj = geojson.Polygon(coordinates=coords)
          
        features.append(gj)
    return features

def polygoncentroid(gjcoord):
    pts = np.array(gjcoord)
    return pts.mean(axis=0)





if __name__=="__main__":
    datadir="C:/Users/Ashwin/Desktop/htic/Nuclei/stage 1/train/train (0)"
    imgname = glob.glob(datadir+'/images/*.png')
    print(imgname)
    img = cv2.imread(imgname[0]) # A image
    
    rows,cols,_ = img.shape
    
    gtimgs = glob.glob(datadir+'/masks/*.png')
    
    annot_json=[]
    for gi in gtimgs:
        mask = cv2.imread(gi)
        if len(mask.shape)>2:
            mask = mask[:,:,0]
            
        feats = mask2poly(mask)
        annot_json = annot_json + feats
    
    
        
#number =0
#for number in range(670):
 #   f=open("json1/train_"+str(number)+".json","wt")
 #   annot = open("json/train_"+str(number)+".json")
 #   annot_json = geojson.load(annot)
    G = adjacency(annot_json)
    #print(countG)    
    d=nx.greedy_color(G,strategy='connected_sequential_bfs', interchange=True)
    #print(d)
    
    mask = Image.new("L",(cols,rows),0)
    
    #print(colour)    
    maxcolor = max(d.values())
    centres = []
    for i in range(len(annot_json)):
        color = maxcolor + 1 
        if i in d.keys():
            color = d[i]
        cc = annot_json[i]
        cen = polygoncentroid(cc.coordinates)
        centres.append(cen)
        mask = tg.poly2mask(cc.coordinates, mask, color+2)
   
    
     # simulated a network output, also the B image
    mask = np.array(mask)
    plt.imshow(mask,cmap='gray')
    plt.hold(True)
    ptnum = 1
    for ci in centres:
        plt.plot(ci[0],ci[1],'bx')
        plt.text(ci[0],ci[1],str(ptnum))
        ptnum = ptnum+1

    
    
    maxmask = mask.max()
    print(maxmask)
    
    analyzedpolygons = []
    combined = None
    analyzedcenters = []
    for ii in range(2,maxmask+1):
        
        poly = mask2poly((mask==ii).astype(np.uint8))
        print(len(poly))
        analyzedpolygons = analyzedpolygons + poly
        for pp in poly:
            analyzedcenters.append(polygoncentroid(pp.coordinates))
        #plt.imshow(mask==ii)
        cv2.imwrite('debug/'+str(ii)+'.png',255*(mask==ii).astype(np.uint8))
        #plt.show()
        if combined is None:
            combined = ii*(mask==ii).astype(np.uint8)
        else:
            combined = combined + ii*(mask==ii).astype(np.uint8)
        
    plt.figure()
    plt.imshow(combined,cmap='gray')
    plt.hold(True)
    ptnum = 1
    for ci in analyzedcenters:
        plt.plot(ci[0],ci[1],'rx')
        plt.text(ci[0],ci[1],str(ptnum))
        ptnum = ptnum+1
    print(len(analyzedpolygons))
    plt.show()
    
"""
    img=cv2.imread('renamed/train_'+str(number)+'.png')
    annot=open("json/train_"+str(number)+".json")
    b= geojson.load(annot)
    #plt.imshow(img)
    #plt.hold(True)
    features=[]
    count=0
    for i in b:
        cc = np.array(i.coordinates)
        #plt.plot(cc[:,0],cc[:,1])
        #plt.hold(True)
    
        
        
        img=cv2.imread('renamed/train_'+str(number)+'.png')
        size=img.shape
        cc = np.array(i.coordinates).ravel().tolist()
        width=size[1]
        height=size[0]
        
        # polygon = [(x1,y1),(x2,y2),...] or [x1,y1,x2,y2,...]
        # width = ?
        # height = ?
        
        img = Image.new('L', (width, height), 0)
        ImageDraw.Draw(img).polygon(cc, outline=1, fill=1)
        mask = np.array(img)
        
        tmp = mask[:,:].copy()
        _, contours, hierarchy = cv2.findContours(tmp, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        
        
        
        for ci in contours:
            
            coords = []
            for pt in ci:
                pt=pt.flatten().tolist()
                coords.append(pt)
                
            coords.append(coords[0])
            
            gj = geojson.Polygon(coordinates=coords,color=d[count],maskno=count,shape=img.shape)
            
            features.append(gj)
        count=count+1    
    f.write(str(features))
    f.close()
        
        
#nx.draw_networkx_nodes(G,node_color=colour, pos=nx.spring_layout(G))
#nx.draw(G,node_color=colour, with_labels=True, font_weight='bold')

"""