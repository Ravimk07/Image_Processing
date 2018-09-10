# -*- coding: utf-8 -*-
"""
Created on Wed Aug 22 23:01:17 2018

@author: admin
"""

from skimage import data, io, filters
image = data.coins() # or any NumPy array!
edges = filters.sobel(image)
io.imshow(edges)

import numpy as np
import matplotlib.pyplot as plt
# Load a small section of the image.
image = data.coins()[0:95, 70:370]
fig, axes = plt.subplots(ncols=2, nrows=3,
figsize=(8, 4))
ax0, ax1, ax2, ax3, ax4, ax5 = axes.flat
ax0.imshow(image, cmap=plt.cm.gray)
ax0.set_title('Original', fontsize=24)
ax0.axis('off')
