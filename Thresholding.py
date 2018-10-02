# -*- coding: utf-8 -*-
"""
Created on Tue Oct  2 12:02:11 2018

@author: Ravi Kamble
"""

import cv2 
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('original.jpg',0)
img = cv2.medianBlur(img,5)

ret,th1 = cv2.threshold(img,120,255,cv2.THRESH_BINARY)
 
contours,hierarchy = cv2.findContours(th1, 1, 2)

cnt = contours[0]
area = cv2.contourArea(cnt)

plt.imshow(th1,'gray')
print('Area',area)