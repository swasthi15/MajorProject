from __future__ import print_function

# keras imports
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.preprocessing import image
from keras.models import Model
from keras.models import model_from_json
from keras.layers import Input

# other imports
from sklearn.linear_model import LogisticRegression
import numpy as np
import os
import json
import pickle
# import cv2
import cv2


# config variables
# model_name 		= config["model"]
weights 		= "imagenet"
include_top 	= False
train_path 		= "../dataset/train"
test_path 		= "../dataset/test"
features_path 	= "output/features.h5"
labels_path 	= "output/labels.h5"
test_size 		= 0.10
results 		= "output/results.txt"
model_path 		= "output/model"
seed 			= 9
classifier_path = "output/classifier_naive_bayes.pickle"
# classifier_path = "output/classifier_svm.pickle"
# classifier_path = "output/classifier_logistic.pickle"
# classifier_path = "output/classifier_decision_tree.pickle"


# load the trained logistic regression classifier
print (" loading the classifier...")
classifier = pickle.load(open(classifier_path, 'rb'))

# pretrained models needed to perform feature extraction on test data too!
base_model = ResNet50(weights=weights)
base_model.layers.pop()
model = Model(inputs=base_model.inputs, outputs=base_model.layers[-1].output)
image_size = (224, 224)

# get all the train labels
train_labels = os.listdir(train_path)

# get all the test images paths
test_images = os.listdir(test_path)

# loop through each image in the test data
for image_path in test_images:
	path 		= test_path + "/" + image_path
	img 		= image.load_img(path, target_size=image_size)
	x 			= image.img_to_array(img)
	x 			= np.expand_dims(x, axis=0)
	x 			= preprocess_input(x)
	feature 	= model.predict(x)
	flat 		= feature.flatten()
	flat 		= np.expand_dims(flat, axis=0)
	preds 		= classifier.predict(flat)
	prediction 	= train_labels[preds[0]]
	
	# perform prediction on test image
	print (train_labels[preds[0]])
	img_color = cv2.imread(path, 1)
	width = 350
	height = 350
	dim = (width, height) 
	img_color = cv2.resize(img_color,dim, interpolation = cv2.INTER_AREA)
	# cv2.putText(img_color, "I think it is a " + prediction, (140,445), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
	# cv2.imshow("test image", img_color)

	# # key tracker
	# key = cv2.waitKey(0) & 0xFF
	# if (key == ord('q')):
    #         cv2.destroyAllWindows()