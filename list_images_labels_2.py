from random import shuffle
import glob
import cv2

# Not segregated into train, val, and test

"""
Source: http://machinelearninguru.com/deep_learning/data_preparation/hdf5/hdf5.html
"""


"""
List images and their labels
"""

# shuffle the addresses before saving
shuffle_data = False

# address to where you want to save the hdf5 file
hdf5_path = 'C:/Users/User/PycharmProjects/OCT_Project/dataset/Cropped_BM3D/Cropped_BM3D_2.hdf5'

# read addresses and labels from the folder
dme_path = 'C:/Users/User/PycharmProjects/OCT_Project/dataset/Cropped_BM3D/dme/*.png'
normal_path = 'C:/Users/User/PycharmProjects/OCT_Project/dataset/Cropped_BM3D/normal/*.png'

addrs_dme = glob.glob(dme_path)
labels_dme = [1 for addr in addrs_dme]  # 0 = Normal, 1 = DME

addrs_normal = glob.glob(normal_path)
labels_normal = [0 for addr in addrs_normal]  # 0 = Normal, 1 = DME

addrs = addrs_dme+addrs_normal
labels = labels_dme+labels_normal

# shuffle data
if shuffle_data:
    c = list(zip(addrs, labels))
    shuffle(c)
    addrs, labels = zip(*c)


"""
Create a HDF5 file
"""

import numpy as np
import h5py

data_order = 'th'  # 'th' for Theano, 'tf' for Tensorflow

# check the order of data and chose proper data shape to save images
if data_order == 'th':
    shape = (len(addrs), 3, 224, 224)
elif data_order == 'tf':
    shape = (len(addrs), 224, 224, 3)


# open a hdf5 file and create arrays
hdf5_file = h5py.File(hdf5_path, mode='w')
hdf5_file.create_dataset("img_dme", shape, np.int8)
hdf5_file.create_dataset("img_normal", shape, np.int8)


"""
Load images and save them
"""

# loop over addresses
for i in range(len(addrs_dme)):
    # print how many images are saved every 1000 images

    if i % 1000 == 0 and i > 1:
        print('Data: {}/{}'.format(i, len(addrs)))


    # read an image and resize to (224, 224)
    # cv2 load images as BGR, convert it to RGB
    addr_dme = addrs_dme[i]
    img_dme = cv2.imread(addr_dme)
    img_dme = cv2.resize(img_dme, (224, 224), interpolation=cv2.INTER_CUBIC)
    img_dme = cv2.cvtColor(img_dme, cv2.COLOR_BGR2RGB)

    addr_normal = addrs_normal[i]
    img_normal = cv2.imread(addr_normal)
    img_normal = cv2.resize(img_normal, (224, 224), interpolation=cv2.INTER_CUBIC)
    img_normal = cv2.cvtColor(img_normal, cv2.COLOR_BGR2RGB)

    # add any image pre-processing here

    # if the data order is Theano, axis orders should change
    if data_order == 'th':
        img_dme = np.rollaxis(img_dme, 2)
        img_normal = np.rollaxis(img_normal, 2)

    # save the image and calculate the mean so far
    hdf5_file["img_dme"][i, ...] = img_dme[None]
    hdf5_file["img_normal"][i, ...] = img_normal[None]


# save the mean and close the hdf5 file
hdf5_file.close()

print("Complete...")


"""
# Read the HDF5 file


import h5py
import numpy as np
from random import shuffle
from math import ceil
import matplotlib.pyplot as plt

hdf5_path = 'C:/Users/User/PycharmProjects/OCT_Project/dataset/Cropped_BM3D/Cropped_BM3D_2.hdf5'
subtract_mean = False

# open the hdf5 file
hdf5_file = h5py.File(hdf5_path, "r")

# subtract the training mean
if subtract_mean:
    mm = hdf5_file["mean"][0, ...]
    mm = mm[np.newaxis, ...]

# Total number of samples
data_num = hdf5_file["img"].shape[0]

# create list of batches to shuffle the data
nb_class = 2
batch_size = 5
batches_list = list(range(int(ceil(float(data_num) / batch_size))))
shuffle(batches_list)

# loop over batches
for n, i in enumerate(batches_list):
    i_s = i * batch_size  # index of the first image in this batch
    i_e = min([(i + 1) * batch_size, data_num])  # index of the last image in this batch

    # read batch images and remove training mean
    images = hdf5_file["img"][i_s:i_e, ...]
    if subtract_mean:
        images -= mm

    # read labels and convert to one hot encoding
    labels = hdf5_file["labels"][i_s:i_e]
    labels_one_hot = np.zeros((batch_size, nb_class))
    labels_one_hot[np.arange(batch_size), labels] = 1

    print(n+1, '/', len(batches_list))
    print(labels[0], labels_one_hot[0, :])

    plt.imshow(images[0])
    plt.show()
    if n == 5:  # break after 5 batches
        break
hdf5_file.close()

"""