# -*- coding: utf-8 -*-
"""
@author: 
    
"""

import numpy as np
import matplotlib.pyplot as plt 
from tkinter.filedialog import askopenfilename
from tensorflow.keras.models import Sequential
import cv2
from skimage.io import imshow
from skimage.feature import hog

import os
import argparse
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Conv2D,MaxPooling2D,Flatten,Dense, Dropout
from scipy.stats import skew


# === GETTING INPUT SIGNAL

filename = askopenfilename()



import matplotlib.pyplot as plt
import matplotlib.image as mpimg
img = mpimg.imread(filename)

plt.imshow(img)
plt.title('ORIGINAL IMAGE')
plt.show()


# PRE-PROCESSING

h1=224
w1=224

dimension = (w1, h1) 
resized_image = cv2.resize(img,(h1,w1))

fig = plt.figure()
plt.title('RESIZED IMAGE')
plt.imshow(resized_image)

SP = np.shape(resized_image)
try:
    
    Red = resized_image[:,:,0]
    Green = resized_image[:,:,1]
    Blue = resized_image[:,:,2]

    

    plt.imshow(Red)
    plt.title('RED IMAGE')
    plt.show()


    plt.imshow(Green)
    plt.title('GREEN IMAGE')
    plt.show()

    plt.imshow(Blue)
    plt.title('BLUE IMAGE')
    plt.show()
    GRAY = resized_image[:,:,0]

except:
    GRAY = resized_image

    
    
plt.imshow(GRAY)
plt.title('GRAY IMAGE')
plt.show()

from skimage.transform import rotate
from skimage.feature import local_binary_pattern
from skimage import data
from skimage.color import label2rgb


# settings for LBP
radius = 3
n_points = 8 * radius

image = GRAY

def overlay_labels(image, lbp, labels):
    mask = np.logical_or.reduce([lbp == each for each in labels])
    return label2rgb(mask, image=image, bg_label=0, alpha=0.5)


def highlight_bars(bars, indexes):
    for i in indexes:
        bars[i].set_facecolor('r')

METHOD = 'uniform'
image = GRAY

lbp = local_binary_pattern(image, n_points, radius, METHOD)


def hist(ax, lbp):
    n_bins = int(lbp.max() + 1)
    return ax.hist(lbp.ravel(), density=True, bins=n_bins, range=(0, n_bins),
                   facecolor='0.5')


# plot histograms of LBP of textures
fig, (ax_img, ax_hist) = plt.subplots(nrows=2, ncols=3, figsize=(9, 6))
plt.gray()

titles = ('edge', 'flat', 'corner')
w = width = radius - 1
edge_labels = range(n_points // 2 - w, n_points // 2 + w + 1)
flat_labels = list(range(0, w + 1)) + list(range(n_points - w, n_points + 2))
i_14 = n_points // 4            # 1/4th of the histogram
i_34 = 3 * (n_points // 4)      # 3/4th of the histogram
corner_labels = (list(range(i_14 - w, i_14 + w + 1)) +
                 list(range(i_34 - w, i_34 + w + 1)))

label_sets = (edge_labels, flat_labels, corner_labels)

for ax, labels in zip(ax_img, label_sets):
    ax.imshow(overlay_labels(image, lbp, labels))

for ax, labels, name in zip(ax_hist, label_sets, titles):
    counts, _, bars = hist(ax, lbp)
    highlight_bars(bars, labels)
    ax.set_ylim(top=np.max(counts[:-1]))
    ax.set_xlim(right=n_points + 2)
    ax.set_title(name)

ax_hist[0].set_ylabel('Percentage')
for ax in ax_img:
    ax.axis('off')
    
plt.show()

LBP_fea = counts

Testfea = LBP_fea
    

import os
import numpy as np
import cv2
from matplotlib import pyplot as plt


test_data1 = os.listdir('Dataset/benign/')
test_data2 = os.listdir('Dataset/malignant/')
test_data3 = os.listdir('Dataset/normal/')

dot= []
labels_target = []

for img in test_data1:
    
    try:
        img_1 = plt.imread('Dataset/benign' + "/" + img)
        img_resize = cv2.resize(img_1,((224, 224)))
        dot.append(np.array(img_resize))
        labels_target.append(0)
        
    except:
        None
        
for img in test_data2:
    
    try:
        img_2 = plt.imread('Dataset/malignant/'+ "/" + img)
        img_resize = cv2.resize(img_2,(224, 224))
        
        dot.append(np.array(img_resize))
        labels_target.append(1)
        
    except:
        None

for img in test_data3:
    
    try:
        img_3 = plt.imread('Dataset/normal/'+ "/" + img)
        img_resize = cv2.resize(img_3,(224, 224))
        
        dot.append(np.array(img_resize))
        labels_target.append(2)
        
    except:
        None
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(dot,labels_target,test_size = 0.2, random_state = 101)

x_train1=np.zeros((len(x_train),224,224,3))

try:
    
    for i in range(0,len(x_train)):
            x_train1[i,:,:,:]=x_train[i]
except:
        
            x_train1[i,:,:]=x_train[i]

x_test1=np.zeros((len(x_test),224,224,3))

try:
        
    for i in range(0,len(x_test)):
            x_test1[i,:,:,:]=x_test[i]     
except:

            x_test1[i,:,:]=x_test[i]


# temp_m = np.mean(resized_image)
# temp_st = np.std(resized_image)
# temp_vt = np.var(resized_image)
# temp_sk = np.mean(skew(resized_image))
# temp_md = np.median(resized_image)

# Testfea = [temp_m,temp_st,temp_vt,temp_sk,temp_md]

# Trainfea = []
# for ii in range(0,len(dot)):
#     temp_m = np.mean(dot[ii])
#     temp_st = np.std(dot[ii])
#     temp_vt = np.var(dot[ii])
#     feas = [temp_m,temp_st,temp_vt,temp_sk,temp_md]
#     Trainfea.append(feas)


Trainfea = []
for ijk in range(0,len(labels_target)):
    
    
    image = GRAY
    
    def overlay_labels(image, lbp, labels):
        mask = np.logical_or.reduce([lbp == each for each in labels])
        return label2rgb(mask, image=image, bg_label=0, alpha=0.5)
    
    
    def highlight_bars(bars, indexes):
        for i in indexes:
            bars[i].set_facecolor('r')
    
    METHOD = 'uniform'
    image = GRAY
    
    lbp = local_binary_pattern(image, n_points, radius, METHOD)
    
    
    def hist(ax, lbp):
        n_bins = int(lbp.max() + 1)
        return ax.hist(lbp.ravel(), density=True, bins=n_bins, range=(0, n_bins),
                       facecolor='0.5')
    
    
    # plot histograms of LBP of textures
    # fig, (ax_img, ax_hist) = plt.subplots(nrows=2, ncols=3, figsize=(9, 6))
    # plt.gray()
    
    # titles = ('edge', 'flat', 'corner')
    w = width = radius - 1
    edge_labels = range(n_points // 2 - w, n_points // 2 + w + 1)
    flat_labels = list(range(0, w + 1)) + list(range(n_points - w, n_points + 2))
    i_14 = n_points // 4            # 1/4th of the histogram
    i_34 = 3 * (n_points // 4)      # 3/4th of the histogram
    corner_labels = (list(range(i_14 - w, i_14 + w + 1)) +
                     list(range(i_34 - w, i_34 + w + 1)))
    
    label_sets = (edge_labels, flat_labels, corner_labels)
    
    # for ax, labels in zip(ax_img, label_sets):
        # ax.imshow(overlay_labels(image, lbp, labels))
    
    for ax, labels, name in zip(ax_hist, label_sets, titles):
        counts, _, bars = hist(ax, lbp)
        # highlight_bars(bars, labels)
        # ax.set_ylim(top=np.max(counts[:-1]))
        # ax.set_xlim(right=n_points + 2)
        # ax.set_title(name)
    
    # ax_hist[0].set_ylabel('Percentage')
    # for ax in ax_img:
    #     ax.axis('off')
        
    # plt.show()
    
    Trainfea.append(counts)


from keras.utils import to_categorical


y_train1=np.array(y_train)
y_test1=np.array(y_test)

train_Y_one_hot = to_categorical(y_train1)
test_Y_one_hot = to_categorical(y_test)




x_train2=np.zeros((len(x_train),50,50,3))
for i in range(0,len(x_train)):
        x_train2[i,:,:,:]=x_train2[i]

x_test2=np.zeros((len(x_test),50,50,3))
for i in range(0,len(x_test)):
        x_test2[i,:,:,:]=x_test2[i]


    
from keras.layers import Dense, Conv2D
from keras.layers import Flatten
from keras.layers import MaxPooling2D
from keras.layers import Dropout
from keras.models import Sequential

print("-----------------------------------------------")
print("------> Convolutional Neural Network  ")
print("------------------------------------------------")
print()

# initialize the model
model=Sequential()


#CNN layes 
model.add(Conv2D(filters=16,kernel_size=2,padding="same",activation="relu",input_shape=(50,50,3)))
model.add(MaxPooling2D(pool_size=2))

model.add(Conv2D(filters=32,kernel_size=2,padding="same",activation="relu"))
model.add(MaxPooling2D(pool_size=2))

model.add(Conv2D(filters=64,kernel_size=2,padding="same",activation="relu"))
model.add(MaxPooling2D(pool_size=2))

model.add(Dropout(0.2))
model.add(Flatten())

model.add(Dense(500,activation="relu"))

model.add(Dropout(0.2))

model.add(Dense(3,activation="softmax"))

#summary the model 
model.summary()

#compile the model 
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
y_train1=np.array(y_train)

train_Y_one_hot = to_categorical(y_train1)
test_Y_one_hot = to_categorical(y_test)

#fit the model 
history=model.fit(x_train2,train_Y_one_hot,batch_size=2,epochs=2,verbose=1)
accuracy = model.evaluate(x_test2, test_Y_one_hot, verbose=1)

print("-----------------------------------------------")
print("  --------------> RNN  ")
print("------------------------------------------------")
print()
# ==============================
# RNN
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Define the model
modelr = keras.Sequential()

# CNN layers
modelr.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(50, 50, 3)))
modelr.add(layers.MaxPooling2D((2, 2)))
modelr.add(layers.Conv2D(64, (3, 3), activation='relu'))
modelr.add(layers.MaxPooling2D((2, 2)))
modelr.add(layers.Conv2D(64, (3, 3), activation='relu'))

# LSTM layer
modelr.add(layers.TimeDistributed(layers.Flatten()))
modelr.add(layers.LSTM(64, return_sequences=False))  # You can adjust the LSTM units

# Dense layers for classification
modelr.add(layers.Dense(64, activation='relu'))
modelr.add(layers.Dense(3, activation='softmax'))  # Adjust num_classes for your dataset

# Compile the model
modelr.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Print a summary of the model architecture
modelr.summary()

history=modelr.fit(x_train2,train_Y_one_hot,batch_size=2,epochs=10,verbose=1)
acc_rnn = modelr.evaluate(x_test2, test_Y_one_hot, verbose=1)

# Random Forest 

from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
clf = RandomForestClassifier(max_depth=2, random_state=0)
clf.fit(Trainfea, labels_target)
print(clf.predict([Testfea]))
Class = clf.predict([Testfea])


# Hybrid

# create the sub-models
estimators = []
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier

model11 = KNeighborsClassifier(n_neighbors=3)
estimators.append(('KNN', model11))

model12 = RandomForestClassifier(max_depth=2, random_state=0)
estimators.append(('rendomforest', model12))


from sklearn import svm
model13 = svm.SVC(kernel='precomputed')


ensemble = VotingClassifier(estimators)
ensemble.fit(Trainfea, labels_target)
y_pred = ensemble.predict([Testfea])   

Class = y_pred
print('----------- Classification Result --------------')  
print('Class Value =',str(int(Class)))
 
print('----------- Hybrid (Random Forest - KNN - SVM )Algorithm Class --------------')  
print('Class =',y_pred)

if int(Class) == 0:
    print('-------------------')
    print('Identified ---')
    print('Benign')
    print('-------------------')
    
elif int(Class) == 1:
    print('-------------------')    
    print('Identified ---')    
    print('Malignant')
    print('-------------------')
    
elif int(Class) == 2:
    print('-------------------')    
    print('Identified ---')    
    print('Normal')
    print('-------------------')
    
y_pred = ensemble.predict(Trainfea)   
Class = clf.predict(Trainfea)

from sklearn.metrics import accuracy_score

Accuracy_Hybrid = 100 - accuracy_score(labels_target,y_pred)

print("Performance Analysis")
print()
error = accuracy[0]
acc_cnn=100-error
print("1. CNN Accuracy =",acc_cnn,'%')
print()
error=100-acc_cnn
print("2. CNN Error Rate =",error)
print()

print('3. Accuracy of Hybrid Algorithm = ',Accuracy_Hybrid,' %')
print()
print('4. Error Rate of Hybrid Algorithm = ',100-Accuracy_Hybrid,' %')
print()

error=acc_rnn[1]
acc_rnn=100-error

print("1. RNN Accuracy =",acc_rnn,'%')
print()
error=100-acc_rnn
print("2. RNN Error Rate =",error)
print()
