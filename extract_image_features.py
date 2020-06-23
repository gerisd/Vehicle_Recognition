'''
Script extracts the bounding boxes from csv file and gets vehicle's ROI.
The csv file contains (x, y) bounding box coords, class #, and the filename  
The ROI is passed into a pre-trained ResNet network.
Using transfer learning feature extraction, the ResNet model will output a feature vector
that will be stored for training a linear machine learning model.

'''
from HDF5DatasetWriter import HDF5DatasetWriter
from tensorflow.keras.applications import ResNet50
from keras.applications import imagenet_utils
from imutils import paths
import argparse
import csv
import os
import cv2
import numpy as np
import pickle


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--dataset", required=True, help="path to input directory of images")
ap.add_argument("-c", "--csv", required=True, help="path to csv file containing labels and bounding boxes")
args = vars(ap.parse_args())

#Path to images 
imagePath = list(paths.list_images(args["dataset"]))

#Path to store the HDF5 File
HDF5file = f"results/ResNetOutput.hdf5"

#Load ResNet model (without the FC layer head)
resnet = ResNet50(weights="imagenet", include_top=False)

#initalize dictionary containing each row in csv file 
csv_dict = {}
#Transfer each row of labels and bounding boxes from dict into list
input_detections = []
#List containing all the vehicle's ROI
finalImages = [] 

#Path to csv file
train_Path = os.path.join(args['csv'], "train_labels_and_boxes.csv")

#read training labels csv file
with open(train_Path, mode='r') as csv_file:
	csv_reader = csv.DictReader(csv_file)
	data = list(csv_reader)

#Iterate through each row in OrderedDict
for row in data:
	for k, v in row.items():
		#If key is fname, that means we are on the last value in the row 
		if k == 'fname':
			csv_dict[k] = v 
			#should append all the values to the input list 
			#they are the (x,y) bounding box coords, the class and the filename for each image 
			(box_x1, box_y1, box_x2, box_y2, clss, fname) = csv_dict.values()
			input_detections.append((int(box_x1), int(box_y1), int(box_x2), int(box_y2), int(clss), fname))
		csv_dict[k] = v

#Path to the class names csv file
class_Path = os.path.join(args['csv'], "class_names.csv")

#list stores class names
class_names = []
class_index = []
#read class names csv file
with open(class_Path, mode='r') as csv_file:
	csv_reader = csv.reader(csv_file)
	line_counter = 0

	for row in csv_reader:
		for (i, val) in enumerate(row): 
			class_names.append(val)
			class_index.append(i)

#List containing each vehicle's class
labels = []

#loop through the images
for (i, image) in enumerate(imagePath):
	img = cv2.imread(image)
	
	#Get images input detections (labels and bounding boxes)
	(x1, y1, x2, y2, clss, fname) = input_detections[i]

	#append class to list
	labels.append(clss) 

	#get vehicle's ROI from image
	imageROI = img[y1:y2, x1:x2]

	#convert color to rgb
	imageROI = cv2.cvtColor(imageROI, cv2.COLOR_BGR2RGB)

	#resize image to 224x224 since ResNet accepts images of that shape
	imageROI = cv2.resize(imageROI, (224, 224))

	#convert dims from (H, W, C) to (1, H, W, C) to pass through ResNet, which is height, width, and channels respectively
	imageROI = np.expand_dims(imageROI, axis=0)

	#preprocess using mean subtraction
	imageROI = imagenet_utils.preprocess_input(imageROI)

	#append the image to list to be given to the resnet model
	finalImages.append(imageROI)

finalImages = np.vstack(finalImages)

#Pass the ROI images through the ResNet network 
output = resnet.predict(finalImages, batch_size=len(finalImages))

#Flatten the output results into a 1-D vector	
output = output.reshape((output.shape[0]), 100352)

#Write to disk in HDF5 format
hdf5 = HDF5DatasetWriter((len(input_detections), 100352), HDF5file, dataKey="features")

#store class labels 
hdf5.storeClassLabels(class_index)

#store results 
hdf5.add(output, labels)
'''
#store class names and their index to pickle 
names = open("./results/class_names.pickle", 'wb')
pickle.dump(class_names, names)

index = open("./results/class_index.pickle", "wb")
pickle.dump(class_index, index)
'''
#close file objects
hdf5.close()
#names.close()
#index.close()
