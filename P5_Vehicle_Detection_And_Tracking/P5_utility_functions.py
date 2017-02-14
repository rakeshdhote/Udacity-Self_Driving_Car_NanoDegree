#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  3 21:36:07 2017

@author: rakesh
"""

#%%   ########################################################################
# Import libraries

import matplotlib.image as mpimg
import numpy as np
import cv2
from skimage.feature import hog
from scipy.ndimage.measurements import label

#%%   ########################################################################
# Define a function to return HOG features in an image

def get_hog_features(img,
                    orient,
                    pix_per_cell,
                    cell_per_block,
                    vis=False,
                    feature_vec=True):
    """
	Define a function to return HOG features in an image
	:param img: image
    :param orient: HOG orientation parameter
    :param pix_per_cell: HOG pixels per cell
    :param cell_per_block: HOG cell per block parameter
    :param vis: boolean to return image with HOG features
    :param feature_vec: boolean to return HOG feature vectors
	:return:
	    numpy array of HOG features
	    image with HOG features
	"""

    # Call with two outputs if vis==True
    if vis == True:
        features, hog_image = hog(img,
                                orientations=orient,
                                pixels_per_cell=(pix_per_cell, pix_per_cell),
                                cells_per_block=(cell_per_block, cell_per_block),
                                transform_sqrt=False, # True
                                visualise=vis,
                                feature_vector=feature_vec)
        return features, hog_image

    # Otherwise call with one output
    else:
        features = hog(img,
                        orientations=orient,
                        pixels_per_cell=(pix_per_cell, pix_per_cell),
                        cells_per_block=(cell_per_block, cell_per_block),
                        transform_sqrt=False, # True
                        visualise=vis,
                        feature_vector=feature_vec)
        return features

#%%   ########################################################################
# Define a function to compute binned color features

def bin_spatial(img,
                size=(32, 32)):
    """
	Feature Extraction from vehicle and non-vehicle dataset
	:param img: image
	:param size: tuple of spatial binning
	:return:
	    numpy array of spatial binned features
	"""

    # Use cv2.resize().ravel() to create the feature vector
    features = cv2.resize(img, size).ravel()

    # Return the feature vector
    return features

#%%   ########################################################################
# Define a function to compute color histogram features

def color_hist(img, nbins=32, bins_range=(0, 256)):
    """
	Define a function to compute color histogram features
	:param img: img
	:param nbins: # of color histogram bins
	:param bins_range: tuple of histogram binning
	:return:
	    numpy array of color histogram binned features
	"""

    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:,:,0], bins=nbins, range=bins_range)
    channel2_hist = np.histogram(img[:,:,1], bins=nbins, range=bins_range)
    channel3_hist = np.histogram(img[:,:,2], bins=nbins, range=bins_range)

    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))

    # Return the individual histograms, bin_centers and feature vector
    return hist_features

#%%   ########################################################################
# Function to extract features from a list of images

def extract_features(imgs,
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
    """
	Function to extract features from a list of images
	:param imgs: List of images
	:param color_space: Color space
    :param spatial_size: Tuple of spatial binning size
    :param hist_bins: Tuple of histogram binning size
    :param orient: HOG orientation parameter
    :param pix_per_cell: HOG pixels per cell
    :param hog_channel: HOG channel
    :param spatial_feat: Boolean to define if spatial feature is used
    :param hist_feat: Boolean to define if histogram feature is used
    :param hog_feat: Boolean to define if HOG feature is used
	:return:
	    feature vector
	"""

    # Create a list to append feature vectors to
    features = []

    # Iterate through the list of images
    for file in imgs:
        file_features = []

        if '.png' in file:
            # Read in each one by one
            image = (mpimg.imread(file)*255).astype('uint8')
        else: # jpg files
            image = mpimg.imread(file)

        # apply color conversion if other than 'RGB'
        if color_space != 'RGB':
            if color_space == 'HSV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            elif color_space == 'LUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
            elif color_space == 'HLS':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
            elif color_space == 'YUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
            elif color_space == 'YCrCb':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        else: feature_image = np.copy(image)

        if spatial_feat == True:
            spatial_features = bin_spatial(feature_image, size=spatial_size)
            file_features.append(spatial_features)

        if hist_feat == True:
            # Apply color_hist()
            hist_features = color_hist(feature_image, nbins=hist_bins)
            file_features.append(hist_features)

        if hog_feat == True:
        # Call get_hog_features() with vis=False, feature_vec=True
            if hog_channel == 'ALL':
                hog_features = []
                for channel in range(feature_image.shape[2]):
                    hog_features.append(get_hog_features(feature_image[:,:,channel],
                                        orient, pix_per_cell, cell_per_block,
                                        vis=False, feature_vec=True))
                hog_features = np.ravel(hog_features)
            else:
                hog_features = get_hog_features(feature_image[:,:,hog_channel], orient,
                            pix_per_cell, cell_per_block, vis=False, feature_vec=True)
            # Append the new feature vector to the features list
            file_features.append(hog_features)

        features.append(np.concatenate(file_features))

    # Return list of feature vectors
    return features

#%%   ########################################################################
# Extract List of sliding windows for an image

def slide_window(img,
                x_start_stop=[None, None],
                y_start_stop=[None, None],
                xy_window=(64, 64),
                xy_overlap=(0.5, 0.5)):
    """
	Function to extract features from a list of images
	:param img: Images
	:param x_start_stop: Region of interest (pixel coordinates) in X-direction 
    :param y_start_stop: Region of interest (pixel coordinates) in Y-direction 
    :param xy_window: Tuple of sliding window size in pixels
    :param xy_overlap: Tuple for overlapping windows
	:return:
	    List of sliding windows
	"""

    # If x and/or y start/stop positions not defined, set to image size
    if x_start_stop[0] == None:
        x_start_stop[0] = 0
    if x_start_stop[1] == None:
        x_start_stop[1] = img.shape[1]
    if y_start_stop[0] == None:
        y_start_stop[0] = 0
    if y_start_stop[1] == None:
        y_start_stop[1] = img.shape[0]

    # Compute the span of the region to be searched
    xspan = x_start_stop[1] - x_start_stop[0]
    yspan = y_start_stop[1] - y_start_stop[0]

    # Compute the number of pixels per step in x/y
    nx_pix_per_step = np.int(xy_window[0]*(1 - xy_overlap[0]))
    ny_pix_per_step = np.int(xy_window[1]*(1 - xy_overlap[1]))

    # Compute the number of windows in x/y
    nx_windows = np.int(xspan/nx_pix_per_step) - 1
    ny_windows = np.int(yspan/ny_pix_per_step) - 1

    # Initialize a list to append window positions to
    window_list = []

    # Loop through finding x and y window positions
    # Note: you could vectorize this step, but in practice
    # you'll be considering windows one by one with your
    # classifier, so looping makes sense
    for ys in range(ny_windows):
        for xs in range(nx_windows):

            # Calculate window position
            startx = xs*nx_pix_per_step + x_start_stop[0]
            endx = startx + xy_window[0]
            starty = ys*ny_pix_per_step + y_start_stop[0]
            endy = starty + xy_window[1]

            # Append window position to list
            window_list.append(((startx, starty), (endx, endy)))

    # Return the list of windows
    return window_list

#%%   ########################################################################
# Draw bounding boxes in an image

def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    """
	Draw bounding boxes in an image
	:param img: Images
	:param bboxes: List of boxes
    :param color: Color of bounding box
    :param thick: Thickness of bounding box
	:return:
	    Image with bounding boxes drawn
	"""

    # Make a copy of the image
    imcopy = np.copy(img)

    # Iterate through the bounding boxes
    for bbox in bboxes:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(imcopy, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, thick)

    # Return the image copy with boxes drawn
    return imcopy

#%% #######################################################################################
# Add heatmap to detect vehicles with high confidence

def add_heat(heatmap, hot_windows):
    """
	Add heatmap to detect vehicles with high confidence
	:param heatmap: heatmap
	:param hot_windows: List of true positive windows
	:return:
	    Heat map image
	"""

    # Iterate through list of bboxes
    for box in hot_windows:
        # Add += 1 for all pixels inside each bbox
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1
    # Return updated heatmap
    return heatmap

#%%  #######################################################################################
# Implement threshold on overlapping boxes in heatmap to detect vehicles with high confidence

def apply_threshold(heatmap, threshold):
	"""
	Implement threshold on overlapping boxes in heatmap to detect vehicles with high confidence
	:param heatmap: heatmap
	:param threshold: threshold value
	:return:
	    Heat map image with threshold implemented
	"""

    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0

    # Return thresholded map
    return heatmap
#%%  #######################################################################################
# Draw labeled boxes

def draw_labeled_boxes(img, labels):
	"""
	Draw labeled boxes
	:param img: Image
	:param labels: List of detected labels
	:return:
	    image with detected labels
	"""

    carboxes = []

    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):

        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()

        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))

        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)
        carboxes.append(bbox)

    # Return the image
    return img, carboxes
