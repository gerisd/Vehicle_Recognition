# Vehicle Recognition

Performs vehicle recognition by using a trained linear model

## Overview
* Convert dataset from mat file to a usable file 
* Find ROI from the image's bounding boxes and use transfer learning on ResNet model to get image features
* Train multiple models and select the model with the best score and predictability
* Test classifier by using a piCamera module or a test dataset
___

## Dataset 
The dataset used for this project was retrieved from [Cars Dataset](https://ai.stanford.edu/~jkrause/cars/car_dataset.html#:~:text=Overview,or%202012%20BMW%20M3%20coupe.)


## Project Components
* extract_image_features.py: Script stores image features into hdf5 file, recieved from passing image ROI into a pretrained ResNet model
* extract_validation_image_features.py: Script stores the validation image features into an hdf5 file, used for training the linear model
* train model.py: Script trains multipel models on input vector features, selecting the best model and storing its weights into a pickle file
