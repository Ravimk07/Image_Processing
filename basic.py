# -*- coding: utf-8 -*-
"""
Created on Sat Sep 15 12:07:05 2018

@author: Ravi KAmble
""" 

import numpy as np 
import cv2
 
#img = cv2.imread('image1.jpg')
#cv2.imshow('original',img)
#cv2.waitKey(0)
#cv2.destryAllWindows() 

#%%%%%%%%%%%%%%%%%%%%% Laplacian Derivatives %%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#import cv2
#import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('image1.jpg',0)

laplacian = cv2.Laplacian(img,cv2.CV_64F)
sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)
sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5)

plt.subplot(2,2,1),plt.imshow(img,cmap = 'gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,2),plt.imshow(laplacian,cmap = 'gray')
plt.title('Laplacian'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,3),plt.imshow(sobelx,cmap = 'gray')
plt.title('Sobel X'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,4),plt.imshow(sobely,cmap = 'gray')
plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])

plt.show() 