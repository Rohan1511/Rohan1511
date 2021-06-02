# -*- coding: utf-8 -*-
"""
Created on Tue Jun  1 03:28:23 2021

@author: SHAGUN
"""
# Code for preprocessing of the data.This code is written to preprocess the data
# For NASA flood challenge. The dataset can be downloaded from here
# https://nasa-impact.github.io/etci2021/ and then excracted somewhere in your loacal computer.

#* In this case we have our data stored in E Drive . Have a look at the dataset. Our goal 
#is to arrange it so that we can have the training, testing and validation data ready and we can 
# perform computations that we want.. 


# Importing Libraries.

import os
from glob import glob
import numpy as np
import pandas as pd


# Setting Directory path 
dset_root = 'E:\\FLood\\Nasa_floodchallenge'
train_dir = os.path.join(dset_root, 'train') # saving full path for training
valid_dir = os.path.join(dset_root, 'val') # Path for validation data

# Changing Directory
os.chdir('E:\\FLood\\Nasa_floodchallenge')

# Quik look at the training and validation data. 
n_train_regions = len(glob(train_dir+'/*/')) #Counting directories inside 
n_valid_regions = len(glob(valid_dir+'/*/')) 

print('Number of training temporal-regions: {}'.format(n_train_regions))
print('Number of validation temporal-regions: {}'.format(n_valid_regions))
# NOTE: make sure number of regions is NOT 0


## Defining useful functions: 

# Enter the path of image and get the name
def get_filename(filepath): 
    return os.path.split(filepath)[1]


# Visualization once dataframe is ready
def visualize(df_row, figsize=[25, 15]):
    # get image paths
    vv_image_path = df_row['vv_image_path']
    vh_image_path = df_row['vh_image_path']
    flood_label_path = df_row['flood_label_path']
    water_body_label_path = df_row['water_body_label_path']

    # create RGB image from S1 images
    rgb_name = get_filename(vv_image_path)
    vv_image = cv2.imread(vv_image_path, 0) / 255.0
    vh_image = cv2.imread(vh_image_path, 0) / 255.0
    rgb_image = s1_to_rgb(vv_image, vh_image)

    # get water body label mask
    water_body_label_image = cv2.imread(water_body_label_path, 0) / 255.0

    # plot images
    plt.figure(figsize=tuple(figsize))
    if df_row.isnull().sum() > 0:
        # plot RGB S1 image
        plt.subplot(1,2,1)
        plt.imshow(rgb_image)
        plt.title(rgb_name)

        # plot water body mask
        plt.subplot(1,2,2)
        plt.imshow(water_body_label_image)
        plt.title('Water body mask')
    else:
        flood_label_image = cv2.imread(flood_label_path, 0) / 255.0

        # plot RGB S1 image
        plt.subplot(1,3,1)
        plt.imshow(rgb_image)
        plt.title(rgb_name)

        # plot flood label mask
        plt.subplot(1,3,2)
        plt.imshow(flood_label_image)
        plt.title('Flood mask')

        # plot water body mask
        plt.subplot(1,3,3)
        plt.imshow(water_body_label_image)
        plt.title('Water body mask')


# Input vv and vh and output rgb
def s1_to_rgb(vv_image, vh_image):
    ratio_image = np.clip(np.nan_to_num(vh_image/vv_image, 0), 0, 1)
    rgb_image = np.stack((vv_image, vh_image, 1-ratio_image), axis=2)
    return rgb_image

# Visualizate -2 
def visualize_result(df_row, prediction, figsize=[25, 15]):
    vv_image = cv2.imread(df_row['vv_image_path'], 0) / 255.0
    vh_image = cv2.imread(df_row['vh_image_path'], 0) / 255.0
    rgb_input = s1_to_rgb(vv_image, vh_image)

    plt.figure(figsize=tuple(figsize))
    plt.subplot(1,2,1)
    plt.imshow(rgb_input)
    plt.title('RGB w/ result')
    plt.subplot(1,2,2)
    plt.imshow(prediction)
    plt.title('Result')
    
    
# Navigate through directories and get all the vv paths, vvnames and region names
vv_image_paths = sorted(glob(train_dir+'/**/vv/*.png', recursive=True)) 
vv_image_names = [get_filename(pth) for pth in vv_image_paths] 
region_name_dates = ['_'.join(n.split('_')[:2]) for n in vv_image_names]

#
vh_image_paths, flood_label_paths, water_body_label_paths, region_names = [], [], [], []
for i in range(len(vv_image_paths)):
    # get vh image path
    vh_image_name = vv_image_names[i].replace('vv', 'vh')
    vh_image_path = os.path.join(train_dir, region_name_dates[i], 'tiles', 'vh', vh_image_name)
    vh_image_paths.append(vh_image_path)

    # get flood mask path
    flood_image_name = vv_image_names[i].replace('_vv', '')
    flood_label_path = os.path.join(train_dir, region_name_dates[i], 'tiles', 'flood_label', flood_image_name)
    flood_label_paths.append(flood_label_path)

    # get water body mask path
    water_body_label_name = vv_image_names[i].replace('_vv', '')
    water_body_label_path = os.path.join(train_dir, region_name_dates[i], 'tiles', 'water_body_label', water_body_label_name)
    water_body_label_paths.append(water_body_label_path)

    # get region name
    region_name = region_name_dates[i].split('_')[0]
    region_names.append(region_name)


train_paths = {'vv_image_path': vv_image_paths,
        'vh_image_path': vh_image_paths,
        'flood_label_path': flood_label_paths,
        'water_body_label_path': water_body_label_paths,
        'region': region_names
}


# Making dataFrame 
train_df = pd.DataFrame(train_paths)
print(train_df.shape)
train_df.head()
    


## Same for validation data 
vv_image_paths = sorted(glob(valid_dir+'/**/vv/*.png', recursive=True))
vv_image_names = [get_filename(pth) for pth in vv_image_paths]
region_name_dates = ['_'.join(n.split('_')[:2]) for n in vv_image_names]


vh_image_paths, flood_label_paths, water_body_label_paths, region_names = [], [], [], []
for i in range(len(vv_image_paths)):
    # get vh image path
    vh_image_name = vv_image_names[i].replace('vv', 'vh')
    vh_image_path = os.path.join(valid_dir, region_name_dates[i], 'tiles', 'vh', vh_image_name)
    vh_image_paths.append(vh_image_path)

    # get flood mask path ()
    flood_label_paths.append(np.NaN)

    # get water body mask path
    water_body_label_name = vv_image_names[i].replace('_vv', '')
    water_body_label_path = os.path.join(valid_dir, region_name_dates[i], 'tiles', 'water_body_label', water_body_label_name)
    water_body_label_paths.append(water_body_label_path)

    # get region name
    region_name = region_name_dates[i].split('_')[0]
    region_names.append(region_name)

valid_paths = {'vv_image_path': vv_image_paths,
        'vh_image_path': vh_image_paths,
        'flood_label_path': flood_label_paths,
        'water_body_label_path': water_body_label_paths,
        'region': region_names
}


valid_df = pd.DataFrame(valid_paths)
valid_df.sort_values(by=['vv_image_path'])  # important line for submitting results

print(valid_df.shape)
valid_df.head()


# Visualize 
import cv2
import numpy as np
import matplotlib.pyplot as plt
visualize(train_df.iloc[16501])


# Since we do not have flood labels for the validation set of this dataset, we will not be able to fairly evaluate our trained model for the flood segmentation task. Instead we can split our training dataset (that contains flood masks) into a smaller training and development set. We will leave the true validation set for inference in another section.
print(np.unique(train_df.region))

# Set training and validation. 
#1) Lets take 'nebraska', 'bangladesh' as training data 
regions = ['nebraska', 'bangladesh','northal']
development_region =['northal'][0]
regions.remove(development_region)
train_regions = regions


# filter the dataframe to only get images from specified regions
sub_train_df = train_df[train_df['region'] != development_region]
development_df = train_df[train_df['region'] == development_region]

#Note : If this cell is not working then check the size of variables in cells above and if not matching run the above cells again 
# Training Data (X): 
try_stack2=[]
try_stack=[]
for i in range (sub_train_df.index.min(),sub_train_df.index.max()):
  vv_image = cv2.imread(sub_train_df['vv_image_path'][i],0) /255.0
  vh_image = cv2.imread(sub_train_df['vh_image_path'][i],0) /255.0
  water_mask = cv2.imread(sub_train_df['water_body_label_path'][i],0) /255.0
  try_stack = np.stack([vv_image, vh_image, water_mask], axis=2)
  try_stack2.append(try_stack)

np.shape(try_stack2)

############ Not needed; This was to debug during organising training images############
def Sh(i):
    
    try_stack2=[]
    try_stack=[]
    
    
    vv_image = cv2.imread(sub_train_df['vv_image_path'][i],0) /255.0
    vh_image = cv2.imread(sub_train_df['vh_image_path'][i],0) /255.0
    water_mask = cv2.imread(sub_train_df['water_body_label_path'][i],0) /255.0
      
    try_stack = np.stack([vv_image, vh_image, water_mask], axis=2)
    #try_stack2.append(try_stack)
    
    plt.subplot(1,4,1)
     
    plt.imshow(vv_image)
    plt.title('rgb_name')
    
    # plot flood label mask
    plt.subplot(1,4,2)
    plt.imshow(vh_image)
    plt.title('Flood mask')
    
    # plot water body mask
    plt.subplot(1,4,3)
    plt.imshow(water_mask)
    plt.title('Water body mask')
    
    plt.subplot(1,4,4)
    plt.imshow(try_stack)
    plt.title('Water boklkjdy mask')


# Rearranging_trainingdata.py
    

path_to_save = '/FLood/Nasa_floodchallenge/Pre_processing/train_1/'

# This is when the training data is divided 
# =============================================================================
# for i in range (sub_train_df.index.min(),sub_train_df.index.max()):
#   vv_image = cv2.imread(sub_train_df['vv_image_path'][i],0) /255.0
#   vh_image = cv2.imread(sub_train_df['vh_image_path'][i],0) /255.0
#   water_mask = cv2.imread(sub_train_df['water_body_label_path'][i],0) /255.0
#   try_stack = np.stack([vv_image, vh_image, water_mask], axis=2)
#   #try_stack2.append(try_stack)
#   # write in file  
#   name_img = sub_train_df['vv_image_path'][i].split('\\')[-1]
#   path_save = os.path.join(path_to_save,name_img)
#   cv2.imwrite(path_save, try_stack[:,:,0:3]*255)
# 
# 
# path_to_save = '/FLood/Nasa_floodchallenge/Pre_processing/train_2/'
# ## This is when the training data is divided 
# for i in range (development_df.index.min(),development_df.index.max()):
#   vv_image = cv2.imread(development_df['vv_image_path'][i],0) /255.0
#   vh_image = cv2.imread(development_df['vh_image_path'][i],0) /255.0
#   water_mask = cv2.imread(development_df['water_body_label_path'][i],0) /255.0
#   try_stack = np.stack([vv_image, vh_image, water_mask], axis=2)
#   #try_stack2.append(try_stack)
#   # write in file  
#   name_img = development_df['vv_image_path'][i].split('\\')[-1]
#   path_save = os.path.join(path_to_save,name_img)
#   cv2.imwrite(path_save, try_stack[:,:,0:3]*255)
# =============================================================================
  
os.mkdir('train_2_labels')
path_to_save = '/Flood/Nasa_floodchallenge/Pre_processing/train_2_labels/'

for i in range (development_df.index.min(),development_df.index.max()):
    flood_mask = cv2.imread(development_df['flood_label_path'][i],0)
    name_img = development_df['vv_image_path'][i].split('\\')[-1]
    path_save = os.path.join(path_to_save,name_img)
    cv2.imwrite(path_save, flood_mask)
    

  
  