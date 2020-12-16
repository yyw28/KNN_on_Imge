import numpy as np
import pandas as pd
import skimage.io
from skimage.transform import resize
import matplotlib.pyplot as plt
import numpy.matlib
img = skimage.io.imread('/Users/yuwenyu/Desktop/Data_Mining/HW3/Yuwen_Yu_CS5790_HW3/image.png')
skimage.io.imshow(img)
plt.show()
#print(img[2,1,0],img[2,1,1],img[2,1,2])
#height x width x 3; height 244, width 198
img.size
img.shape
#print(img)

df=np.empty((244*198,3))

for col in range(198):
    for row in range(244):
        df[row+col*244,:]=img[row,col,:]
#print(df[2,:])
df = df/ 255
points=df

ks=[2,3,6,10]
def centroid_func(k):
    if k == 2:
        centroids=np.array([[0, 0, 0],[0.1, 0.1, 0.1]])
    elif k ==3:
        centroids=np.array([[0, 0, 0],[0.1, 0.1, 0.1],[0.2, 0.2, 0.2]])
    elif k ==6:
        centroids=np.array([[0, 0, 0],[0.1, 0.1, 0.1],[0.2, 0.2, 0.2],[0.3, 0.3, 0.3],[0.4, 0.4, 0.4], [0.5, 0.5,0.5]])
    else:
        centroids=np.array([[0, 0, 0], [0.1, 0.1, 0.1], [0.2, 0.2, 0.2], [0.3, 0.3, 0.3], [0.4, 0.4, 0.4], [0.5, 0.5,0.5], [0.6, 0.6, 0.6], [0.7, 0.7, 0.7], [0.8, 0.8, 0.8], [0.9, 0.9, 0.9]])
    return centroids

def getAssign(kmat,points):
    nK = kmat.shape[0]
    nP = points.shape[0]
    distances = np.empty((nP,nK))
    for i in range(nK):
        k = kmat[i]
        squares = (k-points)**2
        distances[:,i] = squares.sum(1)
    assignment = distances.argmin(1)
    return(assignment,distances)

def getSSE(kmat,assignments,distances):
    nK = kmat.shape[0]
    SSE = 0
    for i in range(nK):
        SSE += distances[assignments==i,i].sum()
    return(round(SSE))
    
def getCentroid(kmat,points,assignments):
    nK = kmat.shape[0]
    for i in range(nK):
        assignedPoints = points[assignments==i,:]
        if(assignedPoints.size != 0):
            kmat[i] = assignedPoints.mean(0)
    return(kmat)

mycolors = np.array([[60, 179, 113],[0, 191, 255],[255, 255, 0],[255, 0, 0],[0, 0, 0],[169, 169, 169],[255, 140, 0], [128, 0, 128], [255, 192, 203], [255, 255, 255]])
mycolors=mycolors/255

for j in range(len(ks)):
    print('for '+ str(ks[j]) +' classes cluster')
    kmat = centroid_func(ks[j])
    oldKmat = kmat.copy()  
    for i in range(51):
        assignments,distances = getAssign(kmat,points)
        kmat = getCentroid(kmat,points,assignments)
        if (oldKmat == kmat).all():
            break
        oldKmat = kmat.copy()
    print( str(i) +'th iterations' + ', SSE is', getSSE(kmat,assignments,distances))
    
    nK = kmat.shape[0]
    points2 = np.empty(points.shape)
    for i in range(nK):
        points2[assignments==i,:] = mycolors[i]
    
    df2 = points2
    img2 = np.zeros(img.shape)
    for col in range(198):
        for row in range(244):
            img2[row,col,:] = df2[row+col*244,:]
    
    skimage.io.imshow(img2)
    plt.title('classes %s'% (ks[j]))
    plt.savefig('Q1.class%s.png' %(ks[j]))
    plt.show()




