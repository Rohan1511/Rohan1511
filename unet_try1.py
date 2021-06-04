# -*- coding: utf-8 -*-
"""
Created on Wed Jun  2 17:25:19 2021

@author: SHAGUN
"""

#### Now lets import the training and validation data ready.


# pip install matplotlib
# pip install segmentation_models
# pip install opencv-python

%env SM_FRAMEWORK=tf.keras
# This is very important otherwise you will see lots of errors


import os 
import tensorflow as tf
import segmentation_models as sm
import glob
import cv2
import os
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import keras 


os.chdir(r'E:\FLood\Nasa_floodchallenge\Pre_processing')
# Here we wont we touching validation data because we dont have labels. we shall only be
#working with train_1,and train_2 : Lets assume one of them as training and other one as validation


dset_root = r'E:\\FLood\\Nasa_floodchallenge\Pre_processing'
train_dir = os.path.join(dset_root, 'train_1') # saving full path for training
train_dir_labels = os.path.join(dset_root, 'train_1_labels') # saving full path for training
valid_dir = os.path.join(dset_root, 'train_2') # Path for validation data
valid_dir_labels = os.path.join(dset_root, 'train_2_labels') # saving full path for training

#Checking the directory is nor empty and have files. 
print('Train1: {}'.format(len(os.listdir(train_dir))))
print('Train1_lab: {}'.format(len(os.listdir(train_dir_labels))))
print('Validation: {}'.format(len(os.listdir(valid_dir))))
print('Validation_labels: {}'.format(len(os.listdir(valid_dir_labels))))


#Importing Training and Testing data

# A function which will help to check if the training data and validation data 
# is imported correctly.
def training_data(traindata,labeldir,fromimg):
    train_images=[]
    for img_path in os.listdir(traindata)[fromimg:fromimg+3]: #for testing : just with 5 files
        img = cv2.imread(os.path.join(traindata,img_path))
        train_images.append(img)
    print('Size: {}'.format(np.shape(train_images)))
    plt.subplot(2,3,1)
    plt.imshow(train_images[0])
    plt.title('1st')

    # plot flood label mask
    plt.subplot(2,3,2)
    plt.imshow(train_images[1])
    plt.title('2nd')

    # plot water body mask
    plt.subplot(2,3,3)
    plt.imshow(train_images[2])
    plt.title('3rd')
    
    train_labels=[]
    for img_path in os.listdir(labeldir)[fromimg:fromimg+3]: #for testing : just with 5 files
        img = cv2.imread(os.path.join(labeldir,img_path),0)
        train_labels.append(img)
    
    plt.subplot(2,3,4)
    plt.imshow(train_labels[0])
    plt.title('1st')

    # plot flood label mask
    plt.subplot(2,3,5)
    plt.imshow(train_labels[1])
    plt.title('2nd')

    # plot water body mask
    plt.subplot(2,3,6)
    plt.imshow(train_labels[2])
    plt.title('3rd')
    
    train_labels = np.expand_dims(train_labels,3)
    print('Size: {}'.format(np.shape(train_labels)))
    
    return train_images, train_labels
    
# Visualizing.
a, b = training_data(train_dir, train_dir_labels, 100)






# Importing Dataset 

def importing_dataset(traindata,labeldir,fromimg,no_images):
    train_images=[]
    for img_path in os.listdir(traindata)[fromimg:fromimg+no_images-1]: #for testing : just with 5 files
        img = cv2.imread(os.path.join(traindata,img_path))
        train_images.append(img)
    
    train_labels=[]
    for img_path in os.listdir(labeldir)[fromimg:fromimg+no_images-1]: #for testing : just with 5 files
        img = cv2.imread(os.path.join(labeldir,img_path),0) /255
        train_labels.append(img)
    
    train_labels = np.expand_dims(train_labels,3)
    # Convering both into numpy arrays 
    train_images = np.array(train_images)
    train_labels = np.array(train_labels)
    
    print('Size of X_train: {}'.format(np.shape(train_images)))
    print('Size of X_label: {}'.format(np.shape(train_labels)))
    
    return train_images, train_labels
    

# Visualizing.
X_train, X_label = importing_dataset(train_dir, train_dir_labels, 100,1000)
print('Size of X_train: {}'.format(np.shape(X_train)))
print('Size of X_label: {}'.format(np.shape(X_label)))

# Check the unique values and we want our values to be 0 and 1; and if it is a 
# multiclass segmentation problem then we want it to be 0,1,2 and so on.. 

np.unique(X_label)

## Since our values are 0 and 1, we donot need this chunk of code. 

#from sklearn.preprocessing import LabelEncoder
#labelencoder = LabelEncoder()
#n, h, w = train_masks.shape
#train_masks_reshaped = train_masks.reshape(-1,1)
#train_masks_reshaped_encoded = labelencoder.fit_transform(train_masks_reshaped)
#train_masks_encoded_original_shape = train_masks_reshaped_encoded.reshape(n, h, w)




################### Training and Testing Data #############################

##In our case we have already decided to choose 2 areas and training and testing the 
# Third one as validations. but we need some data for testing. note that testing is not validation. 


from sklearn.model_selection import train_test_split
x1, x_test, y1, y_test = train_test_split(X_train, X_label, test_size = 0.10, random_state = 0)
print('Size of X_train: {}'.format(np.shape(x1)))
print('Size of X_test: {}'.format(np.shape(x_test)))
print('Size of Y_train: {}'.format(np.shape(y1)))
print('Size of Y_label: {}'.format(np.shape(y_test)))


## One hot encoding 

#one layer for one lables. so each label pixel is a vecor .. cant explain bhot lamba ho jayeha 
# see one hot encoding

n_classes = 2
y_train = y1
from tensorflow.keras.utils import to_categorical
train_masks_cat = to_categorical(y_train, num_classes=n_classes)
y_train_cat = train_masks_cat.reshape((y_train.shape[0], y_train.shape[1], y_train.shape[2], n_classes))


test_masks_cat = to_categorical(y_test, num_classes=n_classes)
y_test_cat = test_masks_cat.reshape((y_test.shape[0], y_test.shape[1], y_test.shape[2], n_classes))
print('Size of X_train: {}'.format(np.shape(x1)))
print('Size of X_test: {}'.format(np.shape(x_test)))
print('Size of Y_train: {}'.format(np.shape(y_train_cat)))
print('Size of Y_label: {}'.format(np.shape(y_test_cat)))


######################################################
#Reused parameters in all models

activation='softmax'

LR = 0.0001
optim = tf.optimizers.Adam(LR)

# Segmentation models losses can be combined together by '+' and scaled by integer or float factor
# set class weights for dice_loss (car: 1.; pedestrian: 2.; background: 0.5;)
dice_loss = sm.losses.DiceLoss(class_weights=np.array([0.5, 0.5])) 
focal_loss = sm.losses.CategoricalFocalLoss()
total_loss = dice_loss + (1 * focal_loss)

# actulally total_loss can be imported directly from library, above example just show you how to manipulate with losses
# total_loss = sm.losses.binary_focal_dice_loss # or sm.losses.categorical_focal_dice_loss 

metrics = [sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5)]


########################################################################
###Model 1

from tensorflow.keras.datasets import mnist
from tensorflow.keras import utils
from tensorflow.keras import Sequential
from tensorflow.keras import models
from tensorflow.keras.layers import Input, Dense, Dropout, Activation, Flatten
from tensorflow.keras.optimizers import Adam, RMSprop

activation='sigmoid'
BACKBONE1 = 'resnet34'
preprocess_input1 = sm.get_preprocessing(BACKBONE1)

# preprocess input
X_train1 = preprocess_input1(x1)
X_test1 = preprocess_input1(x_test)


# define model
model1 = sm.Unet(BACKBONE1, encoder_weights='imagenet', classes=n_classes, activation=activation)

# compile keras model with defined optimozer, loss and metrics
model1.compile(optim, total_loss, metrics=metrics)

#model1.compile(optimizer='adam', loss='categorical_crossentropy', metrics=metrics)

print(model1.summary())


history1=model1.fit(X_train1, 
          y_train_cat,
          batch_size=4, 
          epochs=50,
          verbose=1,
          validation_data=(X_test1, y_test_cat))


model1.save('res34_backbone_50epochs.hdf5')
############################################################
###Model 2

BACKBONE2 = 'inceptionv3'
preprocess_input2 = sm.get_preprocessing(BACKBONE2)

# preprocess input
X_train2 = preprocess_input2(X_train)
X_test2 = preprocess_input2(X_test)

# define model
model2 = sm.Unet(BACKBONE2, encoder_weights='imagenet', classes=n_classes, activation=activation)


# compile keras model with defined optimozer, loss and metrics
model2.compile(optim, total_loss, metrics)
#model2.compile(optimizer='adam', loss='categorical_crossentropy', metrics=metrics)


print(model2.summary())


history2=model2.fit(X_train2, 
          y_train_cat,
          batch_size=8, 
          epochs=50,
          verbose=1,
          validation_data=(X_test2, y_test_cat))


model2.save('inceptionv3_backbone_50epochs.hdf5')

#####################################################
###Model 3

BACKBONE3 = 'vgg16'
preprocess_input3 = sm.get_preprocessing(BACKBONE3)

# preprocess input
X_train3 = preprocess_input3(X_train)
X_test3 = preprocess_input3(X_test)


# define model
model3 = sm.Unet(BACKBONE3, encoder_weights='imagenet', classes=n_classes, activation=activation)

# compile keras model with defined optimozer, loss and metrics
model3.compile(optim, total_loss, metrics)
#model3.compile(optimizer='adam', loss='categorical_crossentropy', metrics=metrics)


print(model3.summary())

history3=model3.fit(X_train3, 
          y_train_cat,
          batch_size=8, 
          epochs=50,
          verbose=1,
          validation_data=(X_test3, y_test_cat))


model3.save('vgg19_backbone_50epochs.hdf5')


##########################################################

###
#plot the training and validation accuracy and loss at each epoch
loss = history1.history['loss']
val_loss = history1.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'y', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

acc = history1.history['iou_score']
val_acc = history1.history['val_iou_score']

plt.plot(epochs, acc, 'y', label='Training IOU')
plt.plot(epochs, val_acc, 'r', label='Validation IOU')
plt.title('Training and validation IOU')
plt.xlabel('Epochs')
plt.ylabel('IOU')
plt.legend()
plt.show()

#####################################################

from keras.models import load_model

### FOR NOW LET US FOCUS ON A SINGLE MODEL

#Set compile=False as we are not loading it for training, only for prediction.
model1 = load_model('saved_models/res34_backbone_50epochs.hdf5', compile=False)
model2 = load_model('saved_models/inceptionv3_backbone_50epochs.hdf5', compile=False)
model3 = load_model('saved_models/vgg19_backbone_50epochs.hdf5', compile=False)

#IOU
y_pred1=model2.predict(X_test2)
y_pred1_argmax=np.argmax(y_pred1, axis=3)


#Using built in keras function
#from keras.metrics import MeanIoU
n_classes = 4
IOU_keras = MeanIoU(num_classes=n_classes)  
IOU_keras.update_state(y_test[:,:,:,0], y_pred1_argmax)
print("Mean IoU =", IOU_keras.result().numpy())


#To calculate I0U for each class...
values = np.array(IOU_keras.get_weights()).reshape(n_classes, n_classes)
print(values)
class1_IoU = values[0,0]/(values[0,0] + values[0,1] + values[0,2] + values[0,3] + values[1,0]+ values[2,0]+ values[3,0])
class2_IoU = values[1,1]/(values[1,1] + values[1,0] + values[1,2] + values[1,3] + values[0,1]+ values[2,1]+ values[3,1])
class3_IoU = values[2,2]/(values[2,2] + values[2,0] + values[2,1] + values[2,3] + values[0,2]+ values[1,2]+ values[3,2])
class4_IoU = values[3,3]/(values[3,3] + values[3,0] + values[3,1] + values[3,2] + values[0,3]+ values[1,3]+ values[2,3])

print("IoU for class1 is: ", class1_IoU)
print("IoU for class2 is: ", class2_IoU)
print("IoU for class3 is: ", class3_IoU)
print("IoU for class4 is: ", class4_IoU)

#Vaerify the prediction on first image
plt.imshow(train_images[0, :,:,0], cmap='gray')
plt.imshow(train_masks[0], cmap='gray')
##############################################################

#Test some random images
# import random
# test_img_number = random.randint(0, len(X_test2))
# test_img = X_test2[test_img_number]
# ground_truth=y_test[test_img_number]
# test_img_input=np.expand_dims(test_img, 0)

# test_img_input1 = preprocess_input2(test_img_input)

# test_pred1 = model2.predict(test_img_input1)
# test_prediction1 = np.argmax(test_pred1, axis=3)[0,:,:]


# plt.figure(figsize=(12, 8))
# plt.subplot(231)
# plt.title('Testing Image')
# plt.imshow(test_img[:,:,0], cmap='gray')
# plt.subplot(232)
# plt.title('Testing Label')
# plt.imshow(ground_truth[:,:,0], cmap='gray')
# plt.subplot(233)
# plt.title('Prediction on test image')
# plt.imshow(test_prediction1, cmap='gray')
# plt.show()