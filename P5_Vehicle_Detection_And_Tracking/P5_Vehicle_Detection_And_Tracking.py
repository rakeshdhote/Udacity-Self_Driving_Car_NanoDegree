#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  3 21:35:39 2017

@author: rakesh
"""
#%%   ########################################################################

import cv2
import glob
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pickle
import time
import os

from moviepy.editor import *
from moviepy.editor import VideoFileClip
from skimage.feature import hog
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.svm import LinearSVC
from P5_utility_functions import *
from sklearn.model_selection import train_test_split # if sklearn version >= 0.18
#from sklearn.cross_validation import train_test_split # if sklearn version <= 0.17

#%%   ########################################################################
# Define a function to extract features from a single image window
# This function is very similar to extract_features()
# just for a single image rather than list of images

def single_img_features(img,
                        color_space='RGB',
                        spatial_size=(32, 32),
                        hist_bins=32,
                        orient=9,
                        pix_per_cell=8,
                        cell_per_block=2,
                        hog_channel=0,
                        spatial_feat=True,
                        hist_feat=True,
                        hog_feat=True):

    # Define an empty list to receive features
    img_features = []

    #2) Apply color conversion if other than 'RGB'
    if color_space != 'RGB':
        if color_space == 'HSV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        elif color_space == 'LUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
        elif color_space == 'HLS':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        elif color_space == 'YUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        elif color_space == 'YCrCb':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    else: feature_image = np.copy(img)

    minmax_scale = MinMaxScaler(feature_range=(0, 1))

    # Compute spatial features if flag is set
    if spatial_feat == True:
        spatial_features = bin_spatial(feature_image, size=spatial_size)

        # Append features to list
        img_features.append(spatial_features)
#        img_features.append(sp_feature)

    # Compute histogram features if flag is set
    if hist_feat == True:
        hist_features = color_hist(feature_image, nbins=hist_bins)
#        ht_feature = minmax_scale.fit_transform(hist_features)

        # Append features to list
        img_features.append(hist_features)
#        img_features.append(ht_feature)

    # Compute HOG features if flag is set
    if hog_feat == True:
        if hog_channel == 'ALL':
            hog_features = []
            for channel in range(feature_image.shape[2]):
                hog_features.extend(get_hog_features(feature_image[:,:,channel],
                                    orient, pix_per_cell, cell_per_block,
                                    vis=False, feature_vec=True))
        else:
            hog_features = get_hog_features(feature_image[:,:,hog_channel], orient,
                        pix_per_cell, cell_per_block, vis=False, feature_vec=True)

#        hg_feature = minmax_scale.fit_transform(hog_features)

        # Append features to list
        img_features.append(hog_features)
#        img_features.append(hg_feature)

    # Return concatenated array of features
    return np.concatenate(img_features)

#%%   ########################################################################
# Define a function you will pass an image
# and the list of windows to be searched (output of slide_windows())

def search_windows(img,
                    windows,
                    clf,
                    scaler,
                    color_space='RGB',
                    spatial_size=(32, 32),
                    hist_bins=32,
                    hist_range=(0, 256),
                    orient=9,
                    pix_per_cell=8,
                    cell_per_block=2,
                    hog_channel=0, #0
                    spatial_feat=True,
                    hist_feat=True,
                    hog_feat=True):

    # Create an empty list to receive positive detection windows
    on_windows = []

    # Iterate over all windows in the list
    for window in windows:

        # Extract the test window from original image
        test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))

        # Extract features for that window using single_img_features()
        features = single_img_features(test_img,
                                       color_space=color_space,
                                       spatial_size=spatial_size,
                                       hist_bins=hist_bins,
                                       orient=orient,
                                       pix_per_cell=pix_per_cell,
                                       cell_per_block=cell_per_block,
                                       hog_channel=hog_channel,
                                       spatial_feat=spatial_feat,
                                       hist_feat=hist_feat,
                                       hog_feat=hog_feat)

        # Scale extracted features to be fed to classifier
        test_features = scaler.transform(np.array(features).reshape(1, -1))

        # Predict using your classifier
        dec = clf.decision_function(test_features)
        prediction = int(dec > dec_threshold)

        # If positive (prediction == 1) then save the window
        if prediction == 1:
            on_windows.append(window)

    # Return windows for positive detections
    return on_windows


#%%   ########################################################################
# Feature Extraction
def feature_extraction(cars, notcars):

    car_features = extract_features(cars,
                            color_space=color_space,
                            spatial_size=spatial_size,
                            hist_bins=hist_bins,
                            orient=orient,
                            pix_per_cell=pix_per_cell,
                            cell_per_block=cell_per_block,
                            hog_channel=hog_channel,
                            spatial_feat=spatial_feat,
                            hist_feat=hist_feat,
                            hog_feat=hog_feat)

    notcar_features = extract_features(notcars,
                            color_space=color_space,
                            spatial_size=spatial_size,
                            hist_bins=hist_bins,
                            orient=orient,
                            pix_per_cell=pix_per_cell,
                            cell_per_block=cell_per_block,
                            hog_channel=hog_channel,
                            spatial_feat=spatial_feat,
                            hist_feat=hist_feat,
                            hog_feat=hog_feat)

    return car_features, notcar_features

#%%   ########################################################################
# Data split - train/test dataset

def data_train_test_split(car_features, notcar_features, split_ratio, rand_state):

    features = np.vstack((car_features, notcar_features)).astype(np.float64)

    # Fit a per-column scaler
    features_scaler = StandardScaler().fit(features)
#    features_scaler = MinMaxScaler(feature_range=(0, 1)).fit(features)

    # Apply the scaler to X
#    features_scaled = features_scaler.transform(features)
    features_scaled = features_scaler.transform(features)

    # Define the labels vector with cars labeled as 1 and non-car images as 0
    labels = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

    # Split up data into randomized training and test sets
    features_train, features_test, labels_train, labels_test = train_test_split(features_scaled, labels, test_size=split_ratio, random_state=rand_state)

    return features_train, features_test, labels_train, labels_test, features_scaler

#%%   ########################################################################
## Model training/test and pickling the model

def model_train_test(features_train, features_test, labels_train, labels_test, fname_pickle):

    # Use a linear SVC
    clf = LinearSVC(C=0.01)

    # Check the training time for the SVC
    t0=time.time()

    # Model training
    clf.fit(features_train, labels_train)

    # end time
    t1 = time.time()

    test_accuracy = round(clf.score(features_test, labels_test), 4)

    print('Time to train classifier (Linear SVC) : ', round(t1-t0, 2), ' sec')
    print('Test Accuracy of the classifier = ', test_accuracy)

    # Save classifier
    with open(fname_pickle, 'wb') as f:
        pickle.dump(clf, f)

    print('Trained Model pickled to : ', fname_pickle)

    return clf

#%%

def heatmap_windows(image, hot_windows, threshold):

    # Create heatmap
    heatmap = np.zeros_like(image[:,:,0]).astype(np.float)
    heatmap = add_heat(heatmap, hot_windows)
#    heatmap = apply_threshold(heatmap, threshold)
#    plt.imshow(heatmap, cmap='gist_heat')

    # Create labels
    labels = label(heatmap)
    retimg, boxesv = draw_labeled_boxes(image, labels)

    return heatmap, labels, retimg, boxesv
#%%

def process_pipeline(image):
#    image = mpimg.imread('aaaa.jpg')
#    draw_image = np.copy(image)

    # Uncomment the following line if you extracted training
    # data from .png images (scaled 0 to 1 by mpimg) and the
    # image you are searching is a .jpg (scaled 0 to 255)
    #image = image.astype(np.float32)/255

    windows = []
    # xy_window = (64,64)
    windows64 = slide_window(image,
                           x_start_stop=x_start_stop,
                           y_start_stop=y_start_stop,
                           xy_window=xy_window,
                           xy_overlap=xy_overlap)
    windows.extend(windows64)

    # xy_window = (96,96)
    windows96 = slide_window(image,
                           x_start_stop=x_start_stop,
                           y_start_stop=y_start_stop,
                           xy_window=(96,96),
                           xy_overlap=xy_overlap)
    windows.extend(windows96)

        # xy_window = (96,96)
    windows128 = slide_window(image,
                           x_start_stop=x_start_stop,
                           y_start_stop=y_start_stop,
                           xy_window=(128,128),
                           xy_overlap=xy_overlap)
    windows.extend(windows128)

    hot_windows = search_windows(image,
                                 windows,
                                 clf,
                                 features_scaler,
                                 color_space=color_space,
                                 spatial_size=spatial_size,
                                 hist_bins=hist_bins,
                                 orient=orient,
                                 pix_per_cell=pix_per_cell,
                                 cell_per_block=cell_per_block,
                                 hog_channel=hog_channel,
                                 spatial_feat=spatial_feat,
                                 hist_feat=hist_feat,
                                 hog_feat=hog_feat)

#    window_img = draw_boxes(image.copy(), hot_windows, color=(0, 0, 255), thick=6)
#    plt.imshow(window_img)
    heatmap, labels, retimg, boxesv = heatmap_windows(image, hot_windows, threshold=1)
#    plt.imshow(heatmap,cmap='gist_heat')
#    plt.imshow(retimg)

        # Find blobs using cv2.findContours
    hm = (heatmap*255).astype('uint8')
    _, contours, _ = cv2.findContours(hm.copy(), mode = cv2.RETR_LIST,method = cv2.CHAIN_APPROX_SIMPLE)

    # Compute bounding boxes
    vehicles_out = []
    for contour in contours:
        (x0, y0, size_x, size_y) = cv2.boundingRect(contour)
        vehicles_out.append((x0, y0, x0+size_x, y0+size_y))

    window_img = draw_boxes(image.copy(), vehicles_out, color=(0, 0, 255), thick=6)
#    plt.imshow(window_img)
    # Add text
#    result = cv2.putText(retimg, 'Number of Cars = '+str(labels[1]),(100,100), font, 2,(255,255,255) ,2)
#    result = retimg
#    return result

    if hmap == True:
        return retimg, heatmap
    else:
        return retimg
#%%

##############################################################
#%% Main function

if __name__ == '__main__':

    #%%  Read Data

    dir_vehicles = '../DataSet/vehicles/**/*.png'
    dir_nonvehicles = '../DataSet/non-vehicles/**/*.png'

    # List of cars/not cars images
    cars = glob.glob(dir_vehicles, recursive=True)
    notcars = glob.glob(dir_nonvehicles, recursive=True)

    #%% Parameter definations

    color_space = 'YCrCb' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb  'HLS'
    orient = 9  # HOG orientations
    pix_per_cell = 8 # HOG pixels per cell
    cell_per_block = 2 # HOG cells per block
    hog_channel = "ALL" #0 # Can be 0, 1, 2, or "ALL"
    spatial_size = (16, 16) # Spatial binning dimensions
    hist_bins = 16    # Number of histogram bins
    spatial_feat = True # Spatial features on or off
    hist_feat = True # Histogram features on or off
    hog_feat = True # HOG features on or off
    x_start_stop = [None, None]
    y_start_stop = [400, 640]
    xy_window=(64, 64)
    xy_overlap=(0.75, 0.75)

    split_ratio = 0.25 # train/test split ratio
    fname_pickle = 'lsvc_classifier.pkl'

    threshold = 3 #3
    dec_threshold = 0.5
    font = cv2.FONT_HERSHEY_SIMPLEX

    #%% Extract Features and test train data split

    # Read images and extract features
    car_features, notcar_features = feature_extraction(cars, notcars)

    # Random state
    rand_state = np.random.randint(0, 1000000) # 37629184 #

    # train/test data split
    features_train, features_test, labels_train, labels_test, features_scaler = data_train_test_split(car_features, notcar_features, split_ratio, rand_state)

    #%% Model training/test and pickling the model
    os.remove(fname_pickle)

    if os.path.isfile(fname_pickle):
        # Load classifier
        with open(fname_pickle, 'rb') as f:
            clf = pickle.load(f)
        print(">>>> Loaded trained model")

    else:
        clf = model_train_test(features_train, features_test, labels_train, labels_test, fname_pickle)
        print(">>>>>> Model Trained and saved to a pickle file")

#%% image pipeline

    images = glob.glob('test_images/*.jpg')

#    fname = images[4]
    for fname in images:

        if '.png' in fname:
            # Read in each one by one
            image = (mpimg.imread(fname)*255).astype('uint8')
        else: # jpg files
            image = mpimg.imread(fname)

        hmap=True
        retimg, heatmap = process_pipeline(image)
        mpimg.imsave(fname.strip('.jpg')+'_processed.jpg',retimg)
        mpimg.imsave(fname.strip('.jpg')+'_heatmap.jpg',heatmap, cmap='gist_heat')

#%%  Video processing

    #original videos
    ip_clip = "project_video.mp4"
    output_clip = 'project_video_output.mp4'

    #test videos
#    ip_clip = "test_video.mp4"
#    output_clip = 'test_video_output.mp4'

    input_clip = VideoFileClip(ip_clip) # Input video

    hmap=False

    # Run the video processing code
    process_clip = input_clip.fl_image(process_pipeline)
    process_clip.write_videofile(output_clip, audio=False)