from numpy import array
from numpy import argmax
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import h5py

#Load the data
hdf5_path = './dataset/Cropped_BM3D/Cropped_BM3D_2.hdf5'

h5f = h5py.File(hdf5_path,'r')
list(h5f.keys())

all_input_3D_labels = h5f['labels'][:]


# integer encode
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(all_input_3D_labels)


# binary encode
onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
onehot_encoded = onehot_encoder.fit_transform(integer_encoded)

h5f.close()

"""
# invert first example
inverted = label_encoder.inverse_transform([argmax(onehot_encoded[0, :])])
print(inverted)
"""

h5f = h5py.File(hdf5_path, mode='w')

h5f.create_dataset("labels_OHE",  (4096, 2), np.int8)
h5f["labels_OHE"][...] = onehot_encoded

h5f.close()
