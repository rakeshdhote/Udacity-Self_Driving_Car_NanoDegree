# Self-Driving Car Engineer Nanodegree

# Advanced Lane Finding
- - -
[TOC]

## 1. Project Overview

The objective of this project is to create a image/video processing pipeline to detect road lanes under different enviornmental conditions using image processing techniques. 


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

### 2.1 Camera Calibration
Compute the camera calibration matrix and distortion coefficients given a set of chessboard images

```
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
```

### 2.2 Distortion Correction
Apply the distortion correction to the raw image

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
        <img src='images/test2.jpg' style="width: 300px;">
    </td>
    <td style="text-align: center;">
        <img src='images/test2_undist.jpg' style="width: 300px;">
    </td>
    <td style="text-align: center;">
        <img src='images/test2_undist_warped.jpg' style="width: 300px;">
    </td>
</tr>
</table>

## 3. Lane Detection Pipeline
### 3.1 Use color transforms, gradients, etc., to create a thresholded binary image
### 3.2 Apply a perspective transform to rectify binary image ("birds-eye view")
### 3.3 Detect lane pixels and fit to find lane boundary

## 4. Calculating Curvature of Road and Vehicle Position
### 4.1 Determine curvature of the lane and vehicle position with respect to center
### 4.2 Warp the detected lane boundaries back onto the original image
### 4.3 Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position

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



## 5. Video Processing Pipeline
[![Track-2](images/videoimg.png)](https://youtu.be/aJRmShMjjak)



## 6. Conclusions

## 7. Refelections

## References

