#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  2 14:20:52 2017

@author: rakesh
"""
##############################################################
#%% Import Packages

import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
from moviepy.editor import VideoFileClip
from moviepy.editor import *

##############################################################
#%% Undistort image
def undistortImage(img, mtx, dist):
    """
	 Undistort image using camera matrix and distortion coefficients
	:param img: Image
	:param mtx: Camera matrix
    :param dist: camera distortion coefficients
	:return:
	    undistorted image
	"""

    return cv2.undistort(img, mtx, dist, None, mtx)

##############################################################
#%% Calibrate Car Camera

def calibrateCarCamera(img, nx, ny):
    """
	 Calibrate camera
	:param img: Image
	:param nx: Checkerboard corners in X-direction
    :param ny: Checkerboard corners in Y-direction
	:return:
	    None
        The camera matrix and distortion coefficients are pickled in a dictionary to a file.
	"""

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((nx*ny,3), np.float32)
    objp[:,:2] = np.mgrid[0:nx, 0:ny].T.reshape(-1,2)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d points in real world space
    imgpoints = [] # 2d points in image plane.

#    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

    # If found, add object points, image points
    if ret == True:
        objpoints.append(objp)
        imgpoints.append(corners)

        # Draw and display the corners
        cv2.drawChessboardCorners(img, (nx,ny), corners, ret)

    # Test undistortion on an image
    img_size = (img.shape[1], img.shape[0])

    # Do camera calibration given object points and image points
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)

#    dst = cv2.undistort(img, mtx, dist, None, mtx)
    dst = undistortImage(img, mtx, dist)

    # Save the camera calibration result for later use (we won't worry about rvecs / tvecs)
    dist_pickle = {}
    dist_pickle["mtx"] = mtx
    dist_pickle["dist"] = dist
    pickle.dump( dist_pickle, open("camera_calibration_pickle.p", "wb" ) )
    return None
##############################################################
#%% Perspective transformation of an image

def warpedImage(img, pfile):
    """
	 Undistort image using camera matrix and distortion coefficients
	:param img: Image
	:param pfile: Dictionary of camera matrix and distortion coefficients
	:return:
	    Perspective transformation of an image
	"""

    undist = undistortImage(img, pfile['mtx'], pfile['dist'])
    return cv2.warpPerspective(undist, pfile['M'], (img.shape[1], img.shape[0]))

##############################################################
#%% Calculate camera matrix and distortion coefficients

def corners_unwarp(pfile, img, checkerboard = 0, hood_pixels = 0, offset = 0):
    """
	 Determine camera matrix and distortion coefficients
	:param pfile: Dictionary of camera matrix and distortion coefficients
	:param img: Image
    :param checkboard: binary value for checkerboard/non-checkerboard house
    :param hood_pixels: Number of pixels in vertical directions spanning vehicle hood
    :param offset: pixel offset values
	:return:
	   Pickle dictionary consisting of
            camera matrix,
            distortion coefficients,
            perspective tranformation matrix and
            inverse perspective transformation matrix.
	"""

    # Use the OpenCV undistort() function to remove distortion
    undist = undistortImage(img, pfile['mtx'], pfile['dist'])

    # Geometric transformations for checkerboard image
    if checkerboard == 0:
        ht_window = np.uint(img.shape[0]/1.5)
        hb_window = np.uint(img.shape[0]) - hood_pixels
        c_window = np.uint(img.shape[1]/2)
        ctl_window = c_window - .2*np.uint(img.shape[1]/2)
        ctr_window = c_window + .2*np.uint(img.shape[1]/2)
        cbl_window = c_window - 1*np.uint(img.shape[1]/2)
        cbr_window = c_window + 1*np.uint(img.shape[1]/2)

        # Source and destination points
        src = np.float32([[cbl_window,hb_window],[cbr_window,hb_window],[ctr_window,ht_window],[ctl_window,ht_window]])
        dst = np.float32([[0,img.shape[0]],[img.shape[1],img.shape[0]],[img.shape[1],0],[0,0]])

    # Geometric transformations for non-checkerboard image
    elif checkerboard == 1:
        # Convert undistorted image to grayscale
        gray = cv2.cvtColor(undist, cv2.COLOR_RGB2GRAY)

        # Search for corners in the grayscaled image
        ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
        img_size = (gray.shape[1], gray.shape[0])

        # Source and destination points
        src = np.float32([corners[0], corners[nx-1], corners[-1], corners[-nx]])
        dst = np.float32([[offset, offset], [img_size[0]-offset, offset],
                                     [img_size[0]-offset, img_size[1]-offset],
                                     [offset, img_size[1]-offset]])

    # Given src and dst points, calculate the perspective transform matrix
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)

    # Warp the image using OpenCV warpPerspective()
    # Save the camera calibration result
    dist_pickle = {}
    dist_pickle["M"] = M
    dist_pickle["Minv"] = Minv
    dist_pickle["mtx"] = pfile['mtx']
    dist_pickle["dist"] = pfile['dist']
    pickle.dump(dist_pickle, open("perspective_matrix_pickle.p", "wb" ) )

    return dist_pickle

##############################################################
#%%

def roadPerspectiveTransFormation(pfile, img, hood_pixels=0):
    """
	 Determine camera matrix and distortion coefficients
	:param pfile: Dictionary of camera/perspective matrix and distortion coefficients
	:param img: Image
    :param hood_pixels: Number of pixels in vertical directions spanning vehicle hood
	:return:
	   warped image
	"""

    # undistort image
    undist = undistortImage(img, pfile['mtx'], pfile['dist'])

    # source/destination points for perspective transformation
    ht_window = np.uint(img.shape[0]/1.5)
    hb_window = np.uint(img.shape[0]) - hood_pixels
    c_window = np.uint(img.shape[1]/2)
    ctl_window = c_window - .2*np.uint(img.shape[1]/2)
    ctr_window = c_window + .2*np.uint(img.shape[1]/2)
    cbl_window = c_window - 1*np.uint(img.shape[1]/2)
    cbr_window = c_window + 1*np.uint(img.shape[1]/2)

    src = np.float32([[cbl_window,hb_window],[cbr_window,hb_window],[ctr_window,ht_window],[ctl_window,ht_window]])
    dst = np.float32([[0,img.shape[0]],[img.shape[1],img.shape[0]],[img.shape[1],0],[0,0]])

    # Calculate
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(undist, M, ((img.shape[1], img.shape[0])))
    return warped

##############################################################
#%%  Smoothen image using Gaussian blur

def gaussian_blur(img, kernel=5):
    """
	 Determine camera matrix and distortion coefficients
	:param img: Image
    :param kernel: kernel size (odd number)
	:return:
	   blurred image
	"""
    blur = cv2.GaussianBlur(img,(kernel,kernel),0)
    return blur
##############################################################
#%% Absolute Sobel filter

def abs_sobel_thresh1(img, orient='x', sobel_kernel=3, thresh=(0, 255)):
    """
	 Apply x or y gradient with the OpenCV Sobel() function and take the absolute value
	:param img: Image
    :param orient: X/Y direction gradient using Sobel function
    :param sobel_kernel: kernel size (odd number)
    :param thresh: low/high threshold value
	:return:
	   thresholded image
	"""

    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(img, cv2.CV_64F, 1, 0))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(img, cv2.CV_64F, 0, 1))

    # Rescale back to 8 bit integer
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))

    # Create a copy and apply the threshold
    binary_output = np.zeros_like(scaled_sobel)

    # implement thresholds
    binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1

    # Return the result
    return binary_output
##############################################################
#%% Magnitude Sobel filter

def mag_thresh1(img, sobel_kernel=3, mag_thresh=(0, 255)):
    """
	 Magnitude of the Gradient
	:param img: Image
    :param sobel_kernel: kernel size (odd number)
    :param thresh: low/high threshold value
	:return:
	   thresholded image
	"""

    # Take both Sobel x and y gradients
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

    # Calculate the gradient magnitude
    gradmag = np.sqrt(sobelx**2 + sobely**2)

    # Rescale to 8 bit
    scale_factor = np.max(gradmag)/255
    gradmag = (gradmag/scale_factor).astype(np.uint8)

    # Create a binary image of ones where threshold is met, zeros otherwise
    binary_output = np.zeros_like(gradmag)
    binary_output[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1

    # Return the binary image
    return binary_output
##############################################################
#%% Sobel filter thresold

def dir_threshold1(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    """
	 Computes the direction of the gradient
	:param img: Image
    :param sobel_kernel: kernel size (odd number)
    :param thresh: low/high threshold value
	:return:
	   thresholded image
	"""

    # Calculate the x and y gradients
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

    # Take the absolute value of the gradient direction,
    # apply a threshold, and create a binary image result
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    binary_output =  np.zeros_like(absgraddir)
    binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1

    # Return the binary image
    return binary_output
##############################################################
#%% Implement thresholding pipeline to detect lanes using gradient

def thresholding_pipeline1(img, kernels= 5, s_thresh=(50,225), l_thresh=(50,225)):
    """
	 Implement thresholding pipeline to detect lanes using gradient
	:param img: Image
    :param kernel: kernel size (odd number)
    :param s_thresh: low/high threshold value for `saturation` channel in HSV color space
    :param l_thresh: low/high threshold value for `light` channel in HSL color space
	:return:
	   thresholded image
	"""
    img = np.copy(img)

    # Convert to HSV color space and separate the V channel
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
    l_channel = hls[:,:,1]
    s_channel = hls[:,:,2]

    ####################
    # Sobel xy - channel l
    sobx = abs_sobel_thresh1(l_channel,'x',kernels,s_thresh)
    soby = abs_sobel_thresh1(l_channel,'y',kernels,s_thresh)
    l_sobelxy = np.copy(cv2.bitwise_or(sobx,soby))

    ####################
    # Sobel xy - channel s
    sobx = abs_sobel_thresh1(s_channel,'x',kernels,s_thresh)
    soby = abs_sobel_thresh1(s_channel,'y',kernels,s_thresh)
    s_sobelxy = np.copy(cv2.bitwise_or(sobx,soby))

    ####################
    # Threshold color channel
    image = cv2.bitwise_or(l_sobelxy,s_sobelxy)
    image = gaussian_blur(image,kernels)
    return image

##############################################################
#%% Select white/yellow pixels lanes

def color_pixels_hsv(img, white_color_range, yellow_color_range):
    """
	 Select white/yellow pixels lanes in HSV color space
	:param img: Image
    :param white_color_range: low/high threshold value to select white color
    :param yellow_color_range: low/high threshold value to select yellow color
	:return:
	   white/yellow pixels in image
	"""

    img = np.copy(img)

    # Convert to HSV color space and separate the V channel
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float)

    # Create masks for white and yellow colors
    white_mask = cv2.inRange(hsv, white_color_range[0], white_color_range[1])
    yellow_mask = cv2.inRange(hsv, yellow_color_range[0], yellow_color_range[1])

    # Select white and yellow pixels
    white_pixels = cv2.bitwise_and(hsv, hsv, mask= white_mask)
    yellow_pixels = cv2.bitwise_and(hsv, hsv, mask= yellow_mask)

    # Select white and yellow pixels in image
    pixels = cv2.bitwise_or(white_pixels,yellow_pixels)
    return pixels
##############################################################
#%% Impelement lane detection pipeline

def lane_detection_pipeline(img, pfile_cb, kernels = 5, hood_pixels=0):
    """
	 Determine camera matrix and distortion coefficients
	:param img: Image
	:param pfile_cb: Dictionary of camera/perspective matrix and distortion coefficients
    :param kernel: kernel size (odd number)
    :param hood_pixels: Number of pixels in vertical directions spanning vehicle hood
	:return:
	   detected left/right lanes in the image
	"""
    # Lane detection using gradients in the HSL space
    hslwrp = thresholding_pipeline1(img, kernels=kernels, s_thresh=s_thresh, l_thresh=l_thresh)
    # lane detection using color thresholding in HSV space
    hsv = color_pixels_hsv(img, white_color_rangehsv, yellow_color_rangehsv)

    # combined hsv + hsl
    hsvgray = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
    hsvgray = cv2.cvtColor(hsvgray, cv2.COLOR_RGB2GRAY)

    # lane detection using combined color and gradient
    lrlanes = np.copy(cv2.bitwise_or(hslwrp,hsvgray))

    # Thresholding image
    _, lrlanes = cv2.threshold(lrlanes,1,255,cv2.THRESH_BINARY)
    return lrlanes
##############################################################
#%% Reverse perspecitive transformation

def roadTransFormation(pfile, img):
    """
	 Reverse perspecitive transformation of image
	:param img: Image
	:param pfile: Dictionary of camera/perspective matrix and distortion coefficients
	:return:
	   detected left/right lanes in the image
	"""
    warped = cv2.warpPerspective(img, pfile['Minv'], ((img.shape[1], img.shape[0])))
    return warped
##############################################################
#%% Extract x/y pixels of left and right lanes


def detect_lanes(lrimg, slabs):
    """
	 Reverse perspecitive transformation of image
	:param lrimg: Image
	:param slabs: # slabs for sliding windows
	:return:
	   detected left/right lane x/y pixel arrays
	"""

    thresh = 0.5

    # Line Finding Method: Peaks in a Histogram
    hist = np.sum(lrimg[lrimg.shape[0]/2:,:], axis=0)
    hist_norm = hist / hist.max(axis=0)

    llane = hist_norm[0:lrimg.shape[1]/2]
    rlane = hist_norm[lrimg.shape[1]/2:]

    llanemax = llane.argmax(axis=0)
    rlanemax = rlane.argmax(axis=0) + lrimg.shape[1]/2

    # Determine mid point of the left and right lane
    midlane= ((llanemax+rlanemax)/2).astype(int)

    # array to save x/y pixels of left/right lanes
    lxlane = np.array([])
    lylane = np.array([])
    rxlane = np.array([])
    rylane = np.array([])

    for i in range(slabs):
        # Calculate y1, y2 for sliding window
        y1 = lrimg.shape[0]-lrimg.shape[0]*i/slabs
        y2 = lrimg.shape[0]-lrimg.shape[0]*(i+1)/slabs

        # determine ROI for current slab
        roi = lrimg[y2:y1,:]

        # Extract left lane pixels
        lroi = np.copy(roi)
        lroi[:,midlane:] = 0
        lpxls = np.argwhere(lroi>thresh)
        lx = lpxls.T[1]
        ly = lpxls.T[0] +y2
        lxlane=np.append(lxlane, lx) # x pixel values for left lane
        lylane=np.append(lylane, ly) # y pixel values for left lane

        # Extract right lane pixels
        rroi = np.copy(roi)
        rroi[:,:midlane] = 0
        rpxls = np.argwhere(rroi>thresh)
        rx = rpxls.T[1]
        ry = rpxls.T[0] +y2
        rxlane=np.append(rxlane, rx) # x pixel values for right lane
        rylane=np.append(rylane, ry) # y pixel values for right lane

    return lxlane, lylane, rxlane, rylane
##############################################################
#%% Fit

def fitlane(img, x, y, poly = 2, num_pts = 10):
    """
	 Reverse perspecitive transformation of image
	:param img: Image
	:param x: x pixel values
	:param y: y pixel values
	:param poly: degree of polynomial
	:param num_pts: # points for fitting line
	:return:
	   x/y values of a fitted line
	"""
   # fit line
    line_fit = np.polyfit(x, y, poly)

   # y points
    ypts = np.arange(num_pts+1)*img.shape[0]/num_pts
   # calcluate fitted x value
    fitlane = line_fit[0]*ypts**2 + line_fit[1]*ypts + line_fit[2]
    return fitlane, ypts

##############################################################
#%% Determine lane curvature

def lanecurvature1(img, y, x, poly = 2):
    """
	 Reverse perspecitive transformation of image
	:param img: Image
	:param x: x pixel values
	:param y: y pixel values
	:param poly: degree of polynomial
	:return:
	   lane curvature
	"""
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension

    y_eval = np.max(np.arange(img.shape[0]))
#   y_eval = np.max(y)

    linefit_cr = np.polyfit(y*ym_per_pix, x*xm_per_pix, poly)
   # curvature radius
    curverad = ((1 + (2*linefit_cr[0]*y_eval*ym_per_pix + linefit_cr[1])**2)**1.5) / np.absolute(2*linefit_cr[0])

    return curverad
##############################################################
#%% Detect lanes and draw lanes back to the road

def plot_road(imgorig, image, pfile_cb, lfitx, lfity, rfitx, rfity, avgcurvature,vposition):
    """
	 Reverse perspecitive transformation of image
	:param imorig: original image
	:param image: detected left/right lanes
	:param pfile_cb: Dictionary of camera/perspective matrix and distortion coefficients
	:param lfitx: x values of fitted left line
	:param lfity: y values of fitted left line
	:param rfitx: x values of fitted right line
	:param rfity: y values of fitted right line
	:param avgcurvature: average curvature of lanes
	:param vposition: vehicle position wrt center
	:return:
	   detected road lanes highlighted with color wrap with radius of curvature and vehicle position value
	"""

    # define color warp for detected lanes
    warp_zero = np.zeros_like(image).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([lfitx, lfity]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([rfitx, rfity])))])
    pts = np.hstack((pts_left, pts_right))
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

    # Reverse perespective transformation
    newwarp = cv2.warpPerspective(color_warp, pfile_cb['Minv'], (image.shape[1], image.shape[0]))
    undist = undistortImage(imgorig, pfile_cb['mtx'], pfile_cb['dist'])
    result = cv2.addWeighted(imgorig, 1, newwarp, 0.5, 0)

    # Add text
    cv2.putText(result, 'Radius of Curvature = '+str(round(avgcurvature,1))+' m',(100,100), font, 2,(255,255,255),2)
    cv2.putText(result, 'Vehicle Position = '+str(round(vposition,2))+' m',(100,160), font, 2,(255,255,255),2)

    return result
##############################################################
#%% Remove noise from the detected left/right lane image

def remove_noise(lrlanes, threshold = 0.08):
    """
	 Remove noise from the detected left/right lane image
	:param lrlanes: left/right lane image
	:param threshold: threshold values
	:return:
	   image with noise removed
	"""
    # Peaks in a Histogram
    hist = np.sum(lrlanes, axis=0)
    hist_norm = hist / hist.max(axis=0)
    histn = hist_norm > threshold

    # create mask
    mask = np.tile(histn.transpose(), (lrlanes.shape[0], 1)).astype(np.uint8)
    result = cv2.bitwise_and(lrlanes,mask)
    return result

##############################################################
#%% Image processing pipeline to process files in a folder

def img_processing(images):
    """
	 Image processing pipeline to process files in a folder
	:param images: list of images
	:return:
	   None
       Processed images are saved in a folder
	"""

    ##Step through the list and search for chessboard corners
    for idx, fname in enumerate(images):

        # Read image
        img = mpimg.imread(fname)

        # warp image
        warped = roadPerspectiveTransFormation(pfile_cb, img, hood_pixels=0)

        # Detect lanes and remove noise
        lrlanes = lane_detection_pipeline(warped, pfile_cb, kernels = 5,hood_pixels=0)
        lrlanes = remove_noise(lrlanes, threshold = 0.08)
        lxlane, lylane, rxlane, rylane = detect_lanes(lrlanes, slabs)

        # Fit lines and determine curvature
        lfitx, lfity= fitlane(lrlanes, lylane, lxlane, poly, num_pts)
        rfitx, rfity= fitlane(lrlanes, rylane, rxlane, poly, num_pts)

        lcurvature = lanecurvature1(lrlanes, lfity, lfitx, poly)
        rcurvature = lanecurvature1(lrlanes, rfity, rfitx, poly)
        avgcurvature = (lcurvature+rcurvature)/2.0

        vposition = ( img.shape[1]/2 - (lfitx[-1]+rfitx[-1])/2)*xm_per_pix
        # Obtain the resulting image
        result = plot_road(img, lrlanes, pfile_cb, lfitx, lfity, rfitx, rfity, avgcurvature,vposition)

        # Save image to the folder
        fname1 = fname.strip('.jpg')+'_processed.jpg'
        mpimg.imsave(fname1, result)

    return None

##############################################################
#%% Running average

def runningavg(data, window_width):
    # Adopted from http://stackoverflow.com/questions/11352047/finding-moving-average-from-data-points-in-python
    """
	 Running average of the list
	:param data: list of data
	:param window_width: width of window for running average
	:return:
       List of running average values
	"""
    cumsum_vec = np.cumsum(np.insert(data, 0, 0))
    ma_vec = (cumsum_vec[window_width:] - cumsum_vec[:-window_width]) / window_width
    return ma_vec

##############################################################
#%% Determine curvature in video frame at time t = 0

def initcurvature(img):
    """
	 Determine curvature in video frame at time t = 0
	:param img: image at time t = 0 in video stream
	:return:
       List of running average values
	"""

    # Define global variables for smoothining purpose
    global runningcur
    global smoothcurvature

    # Warp image
    warped = roadPerspectiveTransFormation(pfile_cb, img, hood_pixels=0)

    # Detect lanes
    lrlanes = lane_detection_pipeline(warped, pfile_cb, kernels = 5,hood_pixels=0)
    lrlanes = remove_noise(lrlanes, threshold = 0.08)
    lxlane, lylane, rxlane, rylane = detect_lanes(lrlanes, slabs)

    # Fit lines and determine curvature
    lfitx, lfity = fitlane(lrlanes, lylane, lxlane, poly, num_pts)
    rfitx, rfity = fitlane(lrlanes, rylane, rxlane, poly, num_pts)
    lcurvature = lanecurvature1(lrlanes, lfity, lfitx, poly)
    rcurvature = lanecurvature1(lrlanes, rfity, rfitx, poly)
    avgcurvature = (lcurvature+rcurvature)/2.0

    runningcur = np.ones(window_width)*avgcurvature
    smoothcurvature = np.ones(2)*avgcurvature
    vposition = ( img.shape[1]/2 - (lfitx[-1]+rfitx[-1])/2)*xm_per_pix

    return runningcur, smoothcurvature

##############################################################
#%% Video image processing

def video_processing_pipeline(img):
    """
	 Video processing pipeline
	:param img: image
	:return:
       detected road lanes highlighted with color wrap with radius of curvature and vehicle position value
	"""

    # Global variables for smoothining data
    global runningcur
    global smoothcurvature

    # warp image
    warped = roadPerspectiveTransFormation(pfile_cb, img, hood_pixels=0)

    # Detect lanes
    lrlanes = lane_detection_pipeline(warped, pfile_cb, kernels = 5,hood_pixels=0)
    lrlanes = remove_noise(lrlanes, threshold = 0.08)
    lxlane, lylane, rxlane, rylane = detect_lanes(lrlanes, slabs)

    # Fit lines and determine curvature
    lfitx, lfity = fitlane(lrlanes, lylane, lxlane, poly, num_pts)
    rfitx, rfity = fitlane(lrlanes, rylane, rxlane, poly, num_pts)
    lcurvature = lanecurvature1(lrlanes, lfity, lfitx, poly)
    rcurvature = lanecurvature1(lrlanes, rfity, rfitx, poly)
    avgcurvature = (lcurvature+rcurvature)/2.0

    # Running average
    runningcur = np.append(runningcur,avgcurvature)
    sc = runningavg(runningcur, window_width)
    smoothcurvature = np.append(smoothcurvature,sc)

    # Smoothen the line via first-order filter
    curvature = alpha* smoothcurvature[-1] + (1-alpha)* smoothcurvature[-2]
    vposition = ( img.shape[1]/2 - (lfitx[-1]+rfitx[-1])/2)*xm_per_pix

    # Process the image
    result = plot_road(img, lrlanes, pfile_cb, lfitx, lfity, rfitx, rfity,curvature,vposition)

    return result

##############################################################
#%% Main function

if __name__ == '__main__':

    # define constants
    # checkerboard corners in X and Y directions
    nx, ny = 9, 6

    ksize = 5 # Kernel size - Choose a larger odd number to smooth gradient measurements

    # hsv threshold
    white_color_rangehsv = ((0,0,200),(255,30,255))
    yellow_color_rangehsv = ((0,80,100),(80,255,255))

    # hsl thresholds
    s_thresh=(10,100)
    l_thresh=(20,100)

    rows, cols = 2, 2
    poly = 2
    num_pts = 10
    slabs = 10

    xm_per_pix = 3.7/700

    # Write some Text
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Load checkerboard image in RGB
    fname = "camera_cal/calibration3.jpg"
    img = mpimg.imread(fname)

    # calibrate camera using checkerboard image
    _ = calibrateCarCamera(img, nx, ny)

    # Load pickle file
    pfile_cb = pickle.load( open( "camera_calibration_pickle.p", "rb" ) )

    # Determine camera matrix, distortion coefficients, perspective and inververse perspective transformations
    pfile_cb = corners_unwarp(pfile_cb, img, checkerboard = 0, hood_pixels=0, offset = 75)

    # Load if not
#    pfile_cb = pickle.load( open( "perspective_matrix_pickle.p", "rb" ) )

    ######################################
    #%% Loop through images in a list and save processed image to the folder

    images = glob.glob('test_images/test*.jpg')
    _ = img_processing(images)

    ######################################
    #%% Video analytics
    window_width = 25
    alpha = 0.6

    # Smoothing utilities
    runningcur = np.array([])
    smoothcurvature  = np.array([])

    input_clip = VideoFileClip("project_video.mp4") # Input video
    output_clip = 'project_video_processed.mp4' # Output video
    #clip1 = input_clip.subclip(10,12)

    # Calculate curvature at time t = 0
    frame0 = input_clip.get_frame(t=0)
    runningcur, smoothcurvature = initcurvature(frame0)

    # Run the video processing code
    process_clip = input_clip.fl_image(video_processing_pipeline)
    process_clip.write_videofile(output_clip, audio=False)
