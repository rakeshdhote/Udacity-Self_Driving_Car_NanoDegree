# Self-Driving Car Engineer Nanodegree 
# Advanced Lane Finding 
- - - 
[TOC] 
## 1. Project Overview 
The objective of this project is to create a image/video processing pipeline to detect road lanes under different environmental conditions using image processing techniques. 
 
## 2. Camera Calibration and Distortion Correction 
<table> 
<tr> 
<td style="text-align: center;"> 
**Original Image** 
</td> 
<td style="text-align: center;"> 
**Distortion Correction** 
</td> 
<td style="text-align: center;"> 
**Distortion Correction and Warped Image** 
</td> 
</tr> 
<tr> 
<td style="text-align: center;"> 
<img src='camera_cal/calibration3.jpg' style="width: 300px;"> 
</td> 
<td style="text-align: center;"> 
<img src='camera_cal/calibration3_undist.jpg' style="width: 300px;"> 
</td> 
<td style="text-align: center;"> 
<img src='camera_cal/calibration3_undist_warped.jpg' style="width: 300px;"> 
</td> 
</tr> 
</table> 
 
The video image obtained from the car camera is not a true image due to distortions and inherent lens structural properties. These inaccuracies in the image may lead to incorrect decision making for a self-driving car. Hence, it is essential to minimize these inaccuracies by correcting the distortions and camera calibration. 
Typically, a camera is calibrated by taking a checker-board picture, finding square corners and then using the image-processing algorithm to correct the distortion. The `OpenCV` function `cv2.findChessboardCorners` is used to detect square corners by mapping `3D` real world `object points` to `2D` `image points`. Later,  `object points` and `image points` are used to calibrate camera using the `cv2.calibrateCamera` function. 
The code snippet to calibrate camera and undistort (distortion correction) an image is provided below.   
 
 

```python
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

    dst = undistortImage(img, mtx, dist)

    # Save the camera calibration result for later use (we won't worry about rvecs / tvecs)
    dist_pickle = {}
    dist_pickle["mtx"] = mtx
    dist_pickle["dist"] = dist
    pickle.dump( dist_pickle, open("camera_calibration_pickle.p", "wb" ) )
    return None
```

```python
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
```

The following figure presents the camera calibration along within distortion correction implemented on a real-life road image. 

<table>
<tr>
    <td style="text-align: center;">
        **Original Image**
    </td>
    <td style="text-align: center;">
        **Distortion Correction**
    </td>
    <td style="text-align: center;">
        **Distortion Correction and Warped Image**
    </td>
</tr>
<tr>
    <td style="text-align: center;">
        <img src='images/test2.jpg' style="width: 200px;">
    </td>
    <td style="text-align: center;">
        <img src='images/test2_undist.jpg' style="width: 200px;">
    </td>
    <td style="text-align: center;">
        <img src='images/test2_undist_warped.jpg' style="width: 200px;">
    </td>
</tr>
</table>

- - -
 
## 3. Lane Detection Pipeline 
After the camera image is corrected; the next step is to identify lane lines. Typically, the lane lines are `white` and `yellow` in color. `Color` and `edges` are two common attributes, which can be used to detect lanes. 
 
### 3.1 Color Transformation 
Though it is easy to detect `white` and `yellow` lane colors in the day light, they become difficult to identify under night/twilight/dawn light conditions or different environmental conditions including rain, snow. The lane detection algorithm should be robust to detect road lines under various light/environmental conditions. Red-Green-Blue (RGB) is not a robust color space to identify road lanes. Hue-Saturation-Vibrance (HSV) can prove to a working solution to discriminate colors under various light/environmental conditions. Initial experimentation suggested that implementing masks of thresholded `white` and `yellow` color using the `cv2.inRange` function to extract lane pixels. 
The following snippet extracts `white` and `yellow` pixels from image using `white_color_range` and `yellow_color_range` threshold values.  

```python
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
```


<table>
<tr>
    <td style="text-align: center;">
        **Original Image**
    </td>
    <td style="text-align: center;">
        **White Pixels**
    </td>
</tr>
<tr>
    <td style="text-align: center;">
        <img src='images/test21.jpg' style="width: 300px;">
    </td>
    <td style="text-align: center;">
        <img src='images/test2_whitepixels.jpg' width="300px">
    </td>
</tr>
<tr>
    <td style="text-align: center;">
        **Yellow pixels**
    </td>
    <td style="text-align: center;">
        **White and Yellow Pixels**
    </td>
</tr>
<tr>
    <td style="text-align: center;">
        <img src='images/test2_yellowpixels.jpg' width="300px">
    </td>
    <td style="text-align: center;">
        <img src='images/test2_whiteyellowpixels.jpg' width="300px">
    </td>
</tr>
</table>

### 3.2 Gradient Transformation
 
The edges can be a good indicator of road lanes. The edges/lines can be determined by the edge detection algorithm. Initial experiments reveal that the Hue-Saturation-Lightness (HSL) color space can be a good choice to detect a gradient under different light/environmental conditions. L- and S- channels are used to extract edges [[Reference]](https://medium.com/@vivek.yadav/robust-lane-finding-using-advanced-computer-vision-techniques-mid-project-update-540387e95ed3#.qc1y9h6y0). 

Sobel filters is one of the popular filters used to extract lines/edges in an image via convolution operations. The horizontal and vertical edges can be extracted using Sobel filters in X/Y directions and later combined to obtain lane lines. 

The following snippet extracts lines using Sobel filters.  
```python
#%% Implement thresholding pipeline to detect lanes using gradient

def thresholding_pipeline(img, kernels= 5, s_thresh=(50,225), l_thresh=(50,225)):
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
```

<table>
<tr>
    <td style="text-align: center;">
        **Original Image**
    </td>
    <td style="text-align: center;">
        **S-channel Threshold**
    </td>
</tr>
<tr>
    <td style="text-align: center;">
        <img src='images/test21.jpg' style="width: 300px;">
    </td>
    <td style="text-align: center;">
        <img src='images/test2_s_sobelxy.jpg' style="width: 300px;">
    </td>
</tr>
<tr>
    <td style="text-align: center;">
        **L-channel Threshold**
    </td>
    <td style="text-align: center;">
        **S- and L- channels Threshold**
    </td>
</tr>
<tr>
    <td style="text-align: center;">
        <img src='images/test2_l_sobelxy.jpg' style="width: 300px;">
    </td>
    <td style="text-align: center;">
        <img src='images/test2_sl_sobelxy.jpg' style="width: 300px;">
    </td>
</tr>
</table>


### 3.3 Combined Color and Gradient Transformations

The lane detection algorithm can be made robust by combining the pixels obtained using color and gradient (edge detection) transformations. This can be achieved by using the function described below:

```python
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
```language
```

<table>
<tr>
    <td style="text-align: center;">
        **Original Image**
    </td>
    <td style="text-align: center;">
        **Combined Color and Gradient Transformations**
    </td>
</tr>
<tr>
    <td style="text-align: center;">
        <img src='images/test21.jpg' style="width: 300px;">
    </td>
    <td style="text-align: center;">
        <img src='images/test2_lrlanes.jpg' style="width: 300px;">
    </td>
</tr>
</table>

### 3.4 Apply a perspective transform to rectify binary image ("birds-eye view")

Next, the detected lanes need to be transformed by perspective transformation to birds-eye view for line fitting and curvature calculations. The perspective transformation is conducted by mapping the source points (envelope of detected lanes) and destination points. The mapping of source points (intersection of red lines) and destination points (intersection of blue lines) is presented in the following figure.  
 

|Source  | Destination |
| ------------- |:-------------:|
| (0, 690) | (0 , 720)|
| (1280, 690) | (1280, 720)|
| (768, 480) | (1280, 0)|
| (512, 480) | (0 , 0)|

<table>
<tr>
    <td style="text-align: center;">
        <img src='images/solidWhiteRight.jpg' style="width: 300px;">
    </td>
</tr>
</table>

<table>
<tr>
    <td style="text-align: center;">
        **Original Image**
    </td>
    <td style="text-align: center;">
        **Perspective Transformation**
    </td>
</tr>
<tr>
    <td style="text-align: center;">
        <img src='images/test21.jpg' style="width: 200px;">
    </td>
    <td style="text-align: center;">
        <img src='images/test2_undist_warped.jpg' style="width: 300px;">
    </td>
</tr>
</table>

### 3.5 Detect lane pixels and fit to find lane boundary
In the bird's eye view, the lanes are detected. The next step is to apply color and gradient threshold filters to obtain a binarized image of lanes. The lanes are then determined using the following procedure: 
1. Identify peaks in a bottom half of the image 
2. Split the image in 10 equal horizontal strips 
3. Starting from the very bottom strip, identify histogram peaks and select non-zero pixel values. The approximate mid-point of identified peaks is used to determine left and right lane pixels. 
4. Repeat step 3 for all the strips 
5. Save the pixel coordinates in a numpy array. 
6. 
Once the left and right lane pixels are identified and saved in separate arrays, the lines are fitted separately using `np.polyfit` function. The following snippet summarizes the lane detection and line fitting procedures.  

```python
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
        lcurvature = lanecurvature(lrlanes, lylane, lxlane, poly)
        rcurvature = lanecurvature(lrlanes, rylane, rxlane, poly)
        avgcurvature = (lcurvature+rcurvature)/2.0

        # Vehicle position
        vposition = ( img.shape[1]/2 - (lfitx[-1]+rfitx[-1])/2)*xm_per_pix
        # Obtain the resulting image
        result = plot_road(img, lrlanes, pfile_cb, lfitx, lfity, rfitx, rfity, avgcurvature,vposition)

        # Save image to the folder
        fname1 = fname.strip('.jpg')+'_processed.jpg'
        mpimg.imsave(fname1, result)

    return None
```

```python
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

```

<table>
<tr>
    <td style="text-align: center;">
        **Original Image**
    </td>
    <td style="text-align: center;">
        **Lane Detection**
    </td>
    <td style="text-align: center;">
        **Lane Fit**
    </td>    
</tr>
<tr>
    <td style="text-align: center;">
        <img src='images/test21.jpg' style="width: 200px">
    </td>
    <td style="text-align: center;">
        <img src='images/test2_lrlanesp.jpg' style="width: 200px">
    </td>
    <td style="text-align: center;">
        <img src='images/test2_lanefit.jpg' style="width: 200px">
    </td>    
</tr>
</table>

- - -

## 4. Calculating Curvature of Road and Vehicle Position
The radius of curvature of the road is estimated by the formula provided in the [Reference](http://www.intmath.com/applications-differentiation/8-radius-curvature.php). it is to be noted that the correction is conducted to while calculating curvature value by mapping `pixel space` to the `world space`. The following snippet summarizes the steps. 

The vehicle position is determined as follows: 
1. Under the assumption that the camera is mounted at the centre of the car hood, we can consider the car position as centre of the image. 
2. The intersection of fitted lane lines and X-axis is calculated, and the road position is estimated. 
3. The difference between center of the camera (i.e. horizontal centre of the image) and mid-point of the intersection of fitted lane lines provides an estimate of the vehicle position with respect to center. If the vehicle position is left of the camera center (negative value), the car is offsetted to the left with respect to the center of the road lane. Conversely, if the offset is positive, the car is on the right side of the camera center.  

```python
#%% Determine lane curvature 

def lanecurvature(img, y, x, poly = 2):
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

    linefit_cr = np.polyfit(y*ym_per_pix, x*xm_per_pix, poly)
    curverad = ((1 + (2*linefit_cr[0]*y_eval*ym_per_pix + linefit_cr[1])**2)**1.5) / np.absolute(2*linefit_cr[0])

    return curverad
```

The following figures present the advanced lane detection procedure for the test images. 

<table>
<tr>
    <td style="text-align: center;">
        **Original Image**
    </td>
    <td style="text-align: center;">
        **Processed Image**
    </td>
</tr>
<tr>
    <td style="text-align: center;">
        <img src='test_images/test1.jpg' style="width: 400px;">
    </td>
    <td style="text-align: center;">
        <img src='test_images/test1_processed.jpg' style="width: 400px;">
    </td>
</tr>
<tr>
    <td style="text-align: center;">
        <img src='test_images/test2.jpg' style="width: 400px;">
    </td>
    <td style="text-align: center;">
        <img src='test_images/test2_processed.jpg' style="width: 400px;">
    </td>
</tr>
<tr>
    <td style="text-align: center;">
        <img src='test_images/test3.jpg' style="width: 400px;">
    </td>
    <td style="text-align: center;">
        <img src='test_images/test3_processed.jpg' style="width: 400px;">
    </td>
</tr>
<tr>
    <td style="text-align: center;">
        <img src='test_images/test4.jpg' style="width: 400px;">
    </td>
    <td style="text-align: center;">
        <img src='test_images/test4_processed.jpg' style="width: 400px;">
    </td>
</tr>
<tr>
    <td style="text-align: center;">
        <img src='test_images/test5.jpg' style="width: 400px;">
    </td>
    <td style="text-align: center;">
        <img src='test_images/test5_processed.jpg' style="width: 400px;">
    </td>
</tr>
<tr>
    <td style="text-align: center;">
        <img src='test_images/test6.jpg' style="width: 400px;">
    </td>
    <td style="text-align: center;">
        <img src='test_images/test6_processed.jpg' style="width: 400px;">
    </td>
</tr>
</table>

- - -

## 5. Video Processing Pipeline

In order to process a video, the image processing pipeline developed in the earlier section is utilized. The following snippet wraps the necessary steps in a `video_processing_pipeline` function. A smoothing is conducted to present the curvature and vehicle position via the first-order filter.


```
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
```

Click on the image to run the video.
[![Track-2](images/videoimg.png)](https://youtu.be/aJRmShMjjak)


- - -

## 6. Conclusions 

An image/video processing pipeline is built to detect road lanes under different environmental conditions using image processing techniques. The pipeline predicts the road lanes along with the radius of curvature and relative vehicle position with respect to the lane centre. The pipeline does remarkably well under different test image and video scenarios. 

Following are the opportunities to build a more robust pipeline for videos: 
* Keep track of lane detection in previous frame 
* Check if the curvatures over different frames are in close range 
* Check if the lanes are roughly parallel 
* Incoporate other light/enviornmental/road conditions such as faded lines, exiting a highway, potholes, bumps, etc. 


- - - 
## 7. Reflections

This was a fun project to design a robust pipeline for road lane detection than Project 1. The lessons gave me the opportunity to experiment with various image processing techniques and manipulating color spaces. The developed pipeline along with deep learning will be useful in building a robust self-driving car project.  

- - -

## References
* [Radius of Curvature](http://www.intmath.com/applications-differentiation/8-radius-curvature.php)
