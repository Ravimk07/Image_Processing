# -*- coding: utf-8 -*-
"""
Created on Sat Jul 28 12:33:40 2018

@author: admin
"""

from matplotlib import pyplot as plt
from skimage import data
from skimage.feature import blob_dog, blob_log, blob_doh
from math import sqrt
from skimage.color import rgb2gray
import glob
from skimage.io import imread



example_file = glob.glob(r"C:\Users\admin\Desktop\wint_sky.gif")[0]
im = imread(example_file, as_grey=True)
#plt.imshow(im)
#plt.show()

gray = rgb2gray(im)    
plt.imshow(gray, cmap = plt.get_cmap('gray'))
plt.show()

blobs_log = blob_log(gray, max_sigma=30, num_sigma=10, threshold=.1)
# Compute radii in the 3rd column.
blobs_log[:, 2] = blobs_log[:, 2] * sqrt(2)
numrows = len(blobs_log)
print("Number of stars counted : " ,numrows)

fig, ax = plt.subplots(1, 1)
plt.imshow(gray, cmap = plt.get_cmap('gray'))
for blob in blobs_log:
    y, x, r = blob
    c = plt.Circle((x, y), r+5, color='lime', linewidth=2, fill=False)
    ax.add_patch(c)
    
