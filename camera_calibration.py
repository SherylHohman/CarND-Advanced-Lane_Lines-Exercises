
# coding: utf-8

# In[1]:

get_ipython().run_cell_magic('HTML', '', '<style> code {background-color : pink !important;} </style>')


# Camera Calibration with OpenCV
# ===
# 
# ### Run the code in the cell below to extract object points and image points for camera calibration.  

# In[2]:

import numpy as np
import cv2
import glob
import pickle
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib qt')

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6*8,3), np.float32)
objp[:,:2] = np.mgrid[0:8, 0:6].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d points in real world space
imgpoints = [] # 2d points in image plane.

# Make a list of calibration images
images = glob.glob('calibration_wide/GO*.jpg')

# Step through the list and search for chessboard corners
for idx, fname in enumerate(images):
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (8,6), None)

    # If found, add object points, image points
    if ret == True:
        objpoints.append(objp)
        imgpoints.append(corners)

        # Draw and display the corners
        cv2.drawChessboardCorners(img, (8,6), corners, ret)
        #write_name = 'corners_found'+str(idx)+'.jpg'
        #cv2.imwrite(write_name, img)
        cv2.imshow('img', img)
        cv2.waitKey(500)
        
# save objpoints and imgpoints to pickle file
objpoints_pickle = {}
objpoints_pickle["objpoints"] = objpoints
objpoints_pickle["imgpoints"] = imgpoints
pickle.dump( objpoints_pickle, open( "calibration_wide/wide_objpoints_pickle.p", "wb" ) )


cv2.destroyAllWindows()


# ### If the above cell ran sucessfully, you should now have `objpoints` and `imgpoints` needed for camera calibration.  Run the cell below to calibrate, calculate distortion coefficients, and test undistortion on an image!

# In[5]:

import pickle
get_ipython().magic('matplotlib inline')

# Test undistortion on an image
img = cv2.imread('calibration_wide/test_image.jpg')
img_size = (img.shape[1], img.shape[0])

# Do camera calibration given object points and image points
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size,None,None)


dst = cv2.undistort(img, mtx, dist, None, mtx)
cv2.imwrite('calibration_wide/test_undist.jpg',dst)

# Save the camera calibration result for later use (we won't worry about rvecs / tvecs)
dist_pickle = {}
dist_pickle["mtx"]  = mtx
dist_pickle["dist"] = dist
pickle.dump( dist_pickle, open( "calibration_wide/wide_dist_pickle.p", "wb" ) )
#dst = cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)

# Visualize undistortion
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
ax1.imshow(img)
ax1.set_title('Original Image', fontsize=30)
ax2.imshow(dst)
ax2.set_title('Undistorted Image', fontsize=30)


# In[6]:

import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Read in the saved objpoints and imgpoints
#   because browser quiz cannot perform the extraction of these points
#     ret, corners = cv2.findChessboardCorners(gray, (8,6), None)

# This file was written/saved in the second cell.
objpoints_pickle = pickle.load( open( "./calibration_wide/wide_objpoints_pickle.p", "rb" ) )
objpoints = objpoints_pickle["objpoints"]
imgpoints = objpoints_pickle["imgpoints"]

# Read in an image
img = cv2.imread('./calibration_wide/test_image.jpg')

# TODO: Write a function that takes an image, object points, and image points
# performs the camera calibration, image distortion correction and 
# returns the undistorted image
def cal_undistort(img, objpoints, imgpoints):
    # Use cv2.calibrateCamera() and cv2.undistort()
    
    #shape = gray_img.shape[::-1] # equiv to img.shape[0:2]
    shape = img.shape[0:2]
    gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, shape, None, None)
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    return undist

undistorted = cal_undistort(img, objpoints, imgpoints)

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
f.tight_layout()
ax1.imshow(img)
ax1.set_title('Original Image', fontsize=50)
ax2.imshow(undistorted)
ax2.set_title('Undistorted Image', fontsize=50)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)


# ## Start Here: 

# ### Part 1: objpoints & imgpoints
# - Library Imports
# 
# 
# - Prepare arrays to store imgpoints (for img), and gridpattern (objpoints) (of chessboard grid pattern)  
# - Use **mgrid** to generate numpy array representing **objpoints**, each corner of the chessboard  
# .
# 
# - Read in Image,  
# - Turn in into grayscale  
# .
# 
# - Use **findChessBoardCorners** to obtain **imgpoints** (measurements) for the (objpoint locations) in the image  
#     - (takes in grayscale image)  
#     - (returns imgpoints)  
#     - Store imgpoints and corresponding objpoints  

#  ### Pickle the objpoints and imgpoints for each image

# In[4]:

# Here are the steps to pickle the objpoints and imgpoints for each image

import numpy as np
import cv2
import glob
import pickle
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib qt')

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6*8,3), np.float32)
objp[:,:2] = np.mgrid[0:8, 0:6].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d points in real world space
imgpoints = [] # 2d points in image plane.

# Make a list of calibration images
images = glob.glob('calibration_wide/GO*.jpg')

# Get and Store chessboard corners for each image as it's Read in

for idx, fname in enumerate(images):
    
    # read in image
    img = cv2.imread(fname)
    # turn it into grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chessboard corners for the image
    ret, corners = cv2.findChessboardCorners(gray, (8,6), None)

    # If (corners) found, add object points, image points to array of all obj ang img points
    if ret == True:
    
        # same   set of (grid) values for each image -- it's the chessboard pattern
        objpoints.append(objp)
        
        # unique set of values for each image processed (..from findChessboardCorners)
        imgpoints.append(corners)

        # Draw and display the corners (not required)
        cv2.drawChessboardCorners(img, (8,6), corners, ret)
        cv2.imshow('img', img)
        cv2.waitKey(500)
        
# save objpoints and imgpoints to pickle file
objpoints_pickle = {}
objpoints_pickle["objpoints"] = objpoints
objpoints_pickle["imgpoints"] = imgpoints

pickle.dump( objpoints_pickle, open( "calibration_wide/wide_objpoints_pickle.p", "wb" ) )
print("Imgpoints and Objpoints have been saved to 'calibration_wide/wide_objpoints_pickle.p'")


# #### Read objpoints and imgpoints from pickle file

# In[6]:

import pickle

# Read objpoints and imgpoints from pickle file
objpoints_pickle = pickle.load( open( "calibration_wide/wide_objpoints_pickle.p", "rb" ) )
objpoints = objpoints_pickle["objpoints"]
imgpoints = objpoints_pickle["imgpoints"]


# ### Part2: Callibrate and Undistort Image
# Calibrate the image (from objpoints and imgpoints - found from findChessboardCorners)  
#    - (cv2.calibrateCamera)  
#    
# Undistort the image  
#    - (cv2.undistort)  
# 

# In[7]:

import cv2

# takes in: an image, chessboard grid object points, image's image points (as meassured by findChessboardCorners)
#    - performs the camera calibration (mathematically correlate imgpoints to objpoints), 
#    - undistorts the image (image distortion correction)
# returns: the undistorted image

def cal_undistort(img, objpoints, imgpoints):
    # Uses cv2.calibrateCamera() and cv2.undistort()
    
    # throw away (color depth dim==1) portion of image shape (shape = gray_img.shape[::-1] == gray_img.shape[0:2])
    shape = img.shape[0:2]
    
    # Calibrate imgpoints to the objpoints
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, shape, None, None)
    
    # Undistort the image
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    
    return undist


# ### Part 3: Display Images: Before (distorted)  and  After (Undistorted)
# 
# Read an image (whose imgpoints are stored in imgpoints)  
# Calibrate the image (uses imgpoints and objpoints prev found for that image)  
# Undistort the image  
# 
# .  
# Display Distorted and Undistorted image  
# 
# .  
# This *could* be re-written as a loop, to see before and after for all images that have been processed  
# 
# .  
# Currently, it displays a before and after only for a single image (==first image that was stored in imgpoints array)  

# In[13]:

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
get_ipython().magic('matplotlib inline')

#   Could loop through all images to show before and after.
#   Here, we only display before and after for a single image..

# Read in an Example image == First Image in Pickled objpoints, imgpoints
img = cv2.imread('calibration_wide/test_image.jpg')

# calibrate and undistort the image
undistorted = cal_undistort(img, objpoints, imgpoints)

# display an Example undistorted image
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
f.tight_layout()

# Before
ax1.imshow(img)
ax1.set_title('Original Image', fontsize=50)

# After
ax2.imshow(undistorted)
ax2.set_title('Undistorted Image', fontsize=50)

# Show the images
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)


# In[ ]:

Summary:
    
0:
Read in image
convert to grayscale
store 2d shape

initialize imgpoints and corresponding objpoints arrays
  - objpoints is a property of the chessboard pattern - it's a grid
  - imgpoints will be measured, litterally, where is space do these grid intersections land in this image (think inches, or whatever)

1:
FindChessBoardCorners: returns the measurement in the image for each "objpoint" grid corner  

2:
CalibrateCamera: correlate imgpoints to objpoints: find the math equation that would undistort the image

3:
Undistort: apply that math equation to the image, returning an undistorted image


