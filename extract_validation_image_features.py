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
HDF5file = f"results/ResNetValOutput.hdf5"

#Load ResNet model (without the FC layer head)
resnet = ResNet50(weights="imagenet", include_top=False)

#validation path
val_Path = os.path.join(args['csv'], "Validation_Labels_and_Boxes.csv")

#dict and list to store validation results
val_dict = {}
val_detections = []

#Read validation labels to store for testing
with open(val_Path, mode='r') as file:
	csv_reader = csv.DictReader(file)
	data = list(csv_reader)

for row in data:
	for k, v in row.items():
		#if key is fname, that means we are on the last value in the row
		if k == 'fname':
			val_dict[k] = v
			#should append all the values to the input list 
			#they are the (x,y) bounding box coords, the class and the filename for each image 
			(box_x1, box_y1, box_x2, box_y2, clss, fname) = val_dict.values()
			val_detections.append((int(box_x1), int(box_y1), int(box_x2), int(box_y2), int(clss), fname))
		val_dict[k] = v			

#Path to the class names csv file
class_Path = os.path.join(args['csv'], "class_names.csv")

#list stores class names
class_names = []
class_index = []

labels = []

#list to store preprocessed images for ResNet
finalImages = []

#read class names csv file
with open(class_Path, mode='r') as csv_file:
	csv_reader = csv.reader(csv_file)
	line_counter = 0

	for row in csv_reader:
		for (i, val) in enumerate(row): 
			class_names.append(val)
			class_index.append(i)

#loop through the images
for (i, image) in enumerate(imagePath):
	img = cv2.imread(image)
	
	#Get images input detections (labels and bounding boxes)
	(x1, y1, x2, y2, clss, fname) = val_detections[i]

	#Get ROI
	val_imageROI = img[y1:y2, x1:x2] 

	#append class to list
	labels.append(clss) 

	#convert color to rgb
	val_imageROI = cv2.cvtColor(val_imageROI, cv2.COLOR_BGR2RGB)

	#resize image to 224x224 since ResNet accepts images of that shape
	val_imageROI = cv2.resize(val_imageROI, (224, 224))

	#convert dims from (H, W, C) to (1, H, W, C) to pass through ResNet, which is height, width, and channels respectively
	val_imageROI = np.expand_dims(val_imageROI, axis=0)

	#preprocess using mean subtraction
	val_imageROI = imagenet_utils.preprocess_input(val_imageROI)

	finalImages.append(val_imageROI)

#Pass the ROI images through the ResNet network 
output = resnet.predict(finalImages, batch_size=len(finalImages))

#Flatten the output results into a 1-D vector	
output = output.reshape((output.shape[0]), 100352)

#Write to disk in HDF5 format
hdf5 = HDF5DatasetWriter((len(val_detections), 100352), HDF5file, dataKey="features")

#store class labels 
hdf5.storeClassLabels(class_index)

#store results 
hdf5.add(output, labels)

#close file objects
hdf5.close()