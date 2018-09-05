
import h5py
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage import img_as_float

# Simple read
"""
import h5py
path='C:/Users/User/PycharmProjects/OCT_Project/model_TW.h5'
hdf5_file = h5py.File(path, "r")

[1]
list(hdf5_file[list(hdf5_file.keys())[9]])


[2]
import pandas as pd
pd.read_hdf(hdf5_file,'conv2d_12')

[3]
data=list(hdf5_file.keys())

"""

hdf5_path = 'C:/Users/User/PycharmProjects/OCT_Project/dataset/Cropped_BM3D/Cropped_BM3D.hdf5'
subtract_mean = True

# open the hdf5 file
hdf5_file = h5py.File(hdf5_path, "r")

# subtract the training mean
if subtract_mean:
    mm = hdf5_file["train_mean"][0, ...]
    mm = mm[np.newaxis, ...]

# Total number of samples
data_num = hdf5_file["train_img"].shape[0]

i_s = 2
i_e = 5

# read batch images and remove training mean
images = hdf5_file["train_img"][i_s:i_e, ...]
# images -= mm

"""
im = Image.open(images[0])
im.show()
"""

plt.imshow(img_as_float(images[2]))
plt.show()


