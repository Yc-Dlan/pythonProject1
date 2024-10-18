from platform import version

import cv2
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np


print(cv.__version__)
'''
img=cv.imread("image/exchange/1.jpg")
img=cv.cvtColor(img,cv.COLOR_BGR2RGB)
tsize,bsize,lsize,rsize=(50,50,50,50)
replicate=cv.copyMakeBorder(img,tsize,bsize,lsize,rsize,borderType=cv.BORDER_REPLICATE)
reflect=cv.copyMakeBorder(img,tsize,bsize,lsize,rsize,borderType=cv.BORDER_REFLECT)
reflect101=cv.copyMakeBorder(img,tsize,bsize,lsize,rsize,borderType=cv.BORDER_REFLECT_101)
wrap=cv.copyMakeBorder(img,tsize,bsize,lsize,rsize,borderType=cv.BORDER_WRAP)
constant=cv.copyMakeBorder(img,tsize,bsize,lsize,rsize,borderType=cv.BORDER_CONSTANT,value=0)

plt.subplot(231),plt.imshow(img,"gray"),plt.title("O")
plt.subplot(232),plt.imshow(replicate,"gray"),plt.title("replicate")
plt.subplot(233),plt.imshow(reflect,"gray"),plt.title("reflect")
plt.subplot(234),plt.imshow(reflect101,"gray"),plt.title("reflect101")
plt.subplot(235),plt.imshow(wrap,"gray"),plt.title("wrap")
plt.subplot(236),plt.imshow(constant,"gray"),plt.title("constant")

plt.show()'''