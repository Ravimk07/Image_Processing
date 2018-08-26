import matplotlib.pyplot as plt
import numpy as np
import sys
from skimage import data,io,img_as_float,exposure
from skimage.transform import rescale, resize, downscale_local_mean
from skimage.color import rgb2gray
from skimage.exposure import equalize_hist

CLASS1=0
CLASS2=1
Label_kelas = ["cat","dog"]
count_datas = 10 #per class
KNN=5

ic = io.ImageCollection('class1/*jpg')
ic2 = io.ImageCollection('class2/*jpg')

TOTAL_ACCURACY=0



def acquire_image(k,index):
	result = np.empty((0))
	result2 = np.empty((0))
	
	data_per_slice = (count_datas/k)
	begin_datas_slice = index*data_per_slice

	for i in range(begin_datas_slice,begin_datas_slice+data_per_slice):
		result = np.append(result,i)

	for j in range(0,count_datas):
		if j not in result:
			result2 = np.append(result2,j)
	
	# return testing index and training index for k-fold
	return result,result2


def classification(index_testing,index_training):
	print("=========================================================================")
	distances = []
	global TOTAL_ACCURACY

	# calculate all distance
	for i in index_testing:
		distance_i=[]
		distance_i2=[]
		for j in index_training:
			#test data against train data
			# Test class1
			image_test=extract_feature(ic[int(i)])
			image_train=extract_feature(ic[int(j)])
			distance_i.append(distance_formula(image_test,image_train))
			
			# Test class2
			image_test=extract_feature(ic[int(i)])
			image_train=extract_feature(ic2[int(j)])
			distance_i.append(distance_formula(image_test,image_train))


			# Test class1
			image_test=extract_feature(ic2[int(i)])
			image_train=extract_feature(ic[int(j)])
			distance_i2.append(distance_formula(image_test,image_train))
			
			# Test class2
			image_test=extract_feature(ic2[int(i)])
			image_train=extract_feature(ic2[int(j)])
			distance_i2.append(distance_formula(image_test,image_train))


		distances.append(distance_i)
		distances.append(distance_i2)

	# print distances
	# knn process
	
	predict_true = 0.0

	for idx1,item in enumerate(distances):
		# untuk masing-masing daftar list distance
		# for each list distances
		print("DATA TESTING KE "+str(idx1))
		print("label : "+str(Label_kelas[int(idx1%2)]))
		print("Daftar distance : ")
		print(item)
		
		index_choosen = np.array([])
		for i in range(KNN):
			minimum = sys.maxint
			minimum_index = 0

			for idx,val in enumerate(item):
				if val < minimum and idx not in index_choosen:
					minimum=val
					minimum_index=idx
			
			index_choosen = np.append(index_choosen,minimum_index)

		classified = classified_as(index_choosen)
		print("k terdekat : "+ print_k_nearest(index_choosen,item))
		print("diklasifikasikan sebagai : "+classified
		
		if Label_kelas[int(idx1%2)]==classified:
			print("klasifikasi benar")
			predict_true+=1 
		else:
			print("klasifikasi salah")
		print("")

	acc = (predict_true/float(len(distances)))*100.0
	TOTAL_ACCURACY+=acc

	print "ACCURACY "+str(acc)+"%"
	print "========================================================================="



def print_k_nearest(index_choosen,dis):
	string = "\n"
	for i in index_choosen:
		string += Label_kelas[int(i%2)]
		string+="="+str(dis[int(i)])
		
		string+="\n"
	return string

def classified_as(index_choosen):
	kelas1=0
	kelas2=0
	for i in index_choosen:
		if(i%2==0):
			kelas1=kelas1+1
		else:
			kelas2=kelas2+1

	if kelas1>kelas2:
		return Label_kelas[0]
	elif kelas2>kelas1:
		return Label_kelas[1]
	else:
		return "tidak tahu"


def extract_feature(img):
	image_resized = resize(img, (100, 100), mode='reflect')
	gray_image = rgb2gray(image_resized)
	equalized_image = equalize_hist(gray_image)
	gray_image[gray_image>0.75]=1.0
	
	return gray_image


def extract_feature2(img):
	image_resized = resize(img, (100, 100), mode='reflect')
	gray_image = rgb2gray(image_resized)
	equalized_image = equalize_hist(gray_image)
	
	fig, axes = plt.subplots(nrows=2, ncols=2,
                         sharex=True, sharey=True)
	
	ax = axes.ravel()

	ax[0].imshow(img, cmap='gray')
	ax[0].set_title("Original image")


	ax[1].imshow(image_resized, cmap='gray')
	ax[1].set_title("Resized image")

	ax[2].imshow(gray_image, cmap='gray')
	ax[2].set_title("Image gray")

	gray_image[gray_image>0.75]=1.0

	ax[3].imshow(gray_image, cmap='gray')
	ax[3].set_title("Threshold")


	ax[0].set_xlim(0, 512)
	ax[0].set_ylim(512, 0)

	ax[1].set_xlim(0, 100)
	ax[1].set_ylim(100, 0)

	ax[2].set_xlim(0, 100)
	ax[2].set_ylim(100, 0)

	ax[2].set_xlim(0, 100)
	ax[2].set_ylim(100, 0)

	plt.tight_layout()
	plt.show()
	# print gray_image

	return gray_image


def distance_formula(img1,img2):
	distance = np.abs(img1-img2)
	return np.sum(distance)


# K=1
testing,training=acquire_image(5,0)
classification(testing,training)

# K=2
testing,training=acquire_image(5,1)
classification(testing,training)

# K=3
testing,training=acquire_image(5,2)
classification(testing,training)

# K=4
testing,training=acquire_image(5,3)
classification(testing,training)

# K=5
testing,training=acquire_image(5,4)
classification(testing,training)

print "TOTAL ACCURACY = "+str(TOTAL_ACCURACY/5)

# extract_feature2(ic[3])




