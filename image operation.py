
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 15 12:07:05 2018

@author: Ravi KAmble
"""
import cv2
 
def main():
    imagepath = 'C:\\Users\\Administrator\\Python projects\\python image processing\\Python-OpenCV3-master\\Dataset\\4.1.04.tiff'
    img = cv2.imread(imagepath,0)
    
    cv2.imshow('Image',img) 
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
main()
