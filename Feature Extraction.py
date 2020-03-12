#first step is to extract the image features using resnet
# filter warnings
import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)

# keras imports
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.preprocessing import image
from keras.models import Model
# from keras.models import model_from_json
from keras.layers import Input

# other imports
from sklearn.preprocessing import LabelEncoder
import numpy as np
import glob
# import cv2
import h5py
import os
import json
import datetime
import time


# model_name    = config["model"]
weights     = "imagenet"
include_top   = False
train_path    = "../dataset/train"
test_path  = "../dataset/test"
features_path   = "output/features.h5"
labels_path   = "output/labels.h5"
test_size     = 0.10
results     = "output/results.txt"
model_path    = "output/model"

# start time
print ("start time - {}".format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M")))
start = time.time()

base_model = ResNet50(weights=weights)
base_model.layers.pop()
model = Model(inputs=base_model.inputs, outputs=base_model.layers[-1].output)
image_size = (224, 224)

print ("successfully loaded base model and model...")

# path to training dataset
train_labels = os.listdir(train_path)

# encode the labels
print ("encoding labels...")
le = LabelEncoder()
le.fit([tl for tl in train_labels])
print(list(le.classes_))

# variables to hold features and labels
features = []
labels   = []

# loop over all the labels in the folder
count = 1
for i, label in enumerate(train_labels):
  cur_path = train_path + "/" + label
  count = 1
  for image_path in glob.glob(cur_path + "/*"):
    img = image.load_img(image_path, target_size=image_size)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    feature = model.predict(x)
    flat = feature.flatten()
    features.append(flat)
    labels.append(label)
    print (" processed - " + str(count))
    count += 1
    # print(count)
  print (" completed label - " + label)

# encode the labels using LabelEncoder
le = LabelEncoder()
le_labels = le.fit_transform(labels)

# get the shape of training labels
print (" training labels: {}".format(le_labels))
print (" training labels shape: {}".format(le_labels.shape))

# save features and labels
h5f_data = h5py.File(features_path, 'w')
h5f_data.create_dataset('dataset_1', data=np.array(features))

h5f_label = h5py.File(labels_path, 'w')
h5f_label.create_dataset('dataset_1', data=np.array(le_labels))

h5f_data.close()
h5f_label.close()

# save model and weights
model_json = model.to_json()
with open(model_path + str(test_size) + ".json", "w") as json_file:
  json_file.write(model_json)

# save weights
model.save_weights(model_path + str(test_size) + ".h5")
print(" saved model and weights to disk..")

print ("features and labels saved..")

# end time
end = time.time()
print ("end time - {}".format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M")))