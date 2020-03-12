#training using decision tree
from __future__ import print_function

from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.metrics import confusion_matrix
import numpy as np
import h5py
import os
import json
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, accuracy_score
from sklearn.metrics import precision_recall_curve,average_precision_score,roc_curve,auc

# config variables
test_size     = 0.10
seed      = 9
features_path   = "output/features.h5"
labels_path   = "output/labels.h5"
results     = "output/results_decision_tree.txt"
classifier_path = "output/classifier_decision_tree.pickle"
train_path    = "../dataset/train"
num_classes   = 2
# classifier_path = config["classifier_path"]

# import features and labels
h5f_data  = h5py.File(features_path, 'r')
h5f_label = h5py.File(labels_path, 'r')

features_string = h5f_data['dataset_1']
print(type(features_string))
labels_string   = h5f_label['dataset_1']

features = np.array(features_string)
labels   = np.array(labels_string)

print(features)
h5f_data.close()
h5f_label.close()

# verify the shape of features and labels
print ("features shape: {}".format(features.shape))
print ("labels shape: {}".format(labels.shape))

print ("training started...")
# split the training and testing data
(trainData, testData, trainLabels, testLabels) = train_test_split(np.array(features),
                                                                  np.array(labels),
                                                                  test_size=test_size,
                                                                  random_state=seed)

print (" splitted train and test data...")
print (" train data  : {}".format(trainData.shape))
print (" test data   : {}".format(testData.shape))
print (" train labels: {}".format(trainLabels.shape))
print (" test labels : {}".format(testLabels.shape))

print(trainData)
# use logistic regression as the model
print (" creating model...")
model = tree.DecisionTreeClassifier()
model.fit(trainData, trainLabels)

print (" evaluating model...")
f = open(results, "w")


# evaluate the model of test data
preds = model.predict(testData)

# write the classification report to file
f.write("{}\n".format(classification_report(testLabels, preds)))
f.close()

# dump classifier to file
print (" saving model...")
pickle.dump(model, open(classifier_path, 'wb'))

# display the confusion matrix
print (" confusion matrix")

# get the list of training lables
labels = sorted(list(os.listdir(train_path)))

a = precision_score(testLabels,preds)
accuracy = accuracy_score(testLabels,preds)
print("precision is: ",a)
print("accuracy is:",accuracy)

# precision, recall, thresholds = precision_recall_curve(testLabels,preds)

# print("precision is: ",precision)
# print("recall is: ",recall)

print("average_precision_score is: ",average_precision_score(testLabels,preds))

fpr, tpr, thresholds = roc_curve(testLabels,preds)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=1, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy' , linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()


# plot the confusion matrix
cm = confusion_matrix(testLabels, preds)
sns.heatmap(cm,
            annot=True,
            cmap="Set2")
plt.show()

