## Ivan Blanco writeup 

### Udacity self driving car nanondegree

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/camera_calibration.jpg "Undistorted"
[image2]: ./output_images/undistort_image.jpg "Undistorsioned image"
[image3]: ./output_images/threshold_binary.jpg "Binary Example"
[image4]: ./output_images/warped_straight_lines_example.jpg "Warp Example"
[image41]: ./output_images/warped_straight_lines.jpg "warp straight"
[image5]: ./output_images/fit_polynomial.jpg "Fit polynomial"
[image51]: ./output_images/fit_polynomial_coefficients.jpg "Fit polynomial coefficients"
[image6]: ./output_images/test1.jpg "Output"
[image61]: ./output_images/image_grid.jpg "Output"
[video1]: ./output_images/project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in a regular python file in "./src/camera_calibration.py".
  
I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]

Notice that the function `findchessboardcorners` does not find corners in a few images, which 
might lower the accuracy of the coefficients.

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the 
test images. I just stored the `mtx` and `dist` coefficients from `camera_calibration.py` and 
applied them to a test image. Notice that I had to use the `mpimg.imread` to get an image with 
the right color scheme. The results are specially noticeable on the left white car. The results 
are: 

![alt text][image2]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

The main function is `pipeline()` in `threshold_binary_image.py`. I used a combination of color and gradient thresholds to 
generate a binary image. I followed the steps described during the lessons and I tried to find 
the parameters that best work for my case. This was by fare the most time consuming task, it 
takes a lot of time to adjust all the parameters in multiple images. After a lot of trial and 
error, here's an example of my output for this step.

![alt text][image3]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `perspective_transform(img)`, 
which appears in lines 112 through 148 in the file `threshold_binary_image.py`. The `perspective_transform()` 
function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  
Initially I used a similar values from the previous project, those worked well, but I preffer to 
use the suggested ones on this writeup. Just for documentation, that was my initial choice:

```python
table_offset = img.shape[1] / 8 - 40
src = np.float32(
    [[(img_size[0] / 2) - table_offset + 10, img_size[1] / 2 + img.shape[0] / 6],
    [((img_size[0] / 6) - 5), img_size[1]],
    [(img_size[0] * 5 / 6) + table_offset, img_size[1]],
    [(img_size[0] / 2 + table_offset + 10), img_size[1] / 2 + img.shape[0] / 6]])
dst = np.float32(
    [[(img_size[0] / 4), 0],
    [(img_size[0] / 4), img_size[1]],
    [(img_size[0] * 3 / 4), img_size[1]],
    [(img_size[0] * 3 / 4), 0]])
```

Then I chose the hardcoded the source and destination points in the following manner:

```python
src = np.float32(
    [[(img_size[0] / 2) - 55, img_size[1] / 2 + 100],
    [((img_size[0] / 6) - 10), img_size[1]],
    [(img_size[0] * 5 / 6) + 60, img_size[1]],
    [(img_size[0] / 2 + 55), img_size[1] / 2 + 100]])
dst = np.float32(
    [[(img_size[0] / 4), 0],
    [(img_size[0] / 4), img_size[1]],
    [(img_size[0] * 3 / 4), img_size[1]],
    [(img_size[0] * 3 / 4), 0]])
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 585, 460      | 320, 0        | 
| 203, 720      | 320, 720      |
| 1127, 720     | 960, 720      |
| 695, 460      | 960, 0        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]

![alt text][image41]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

To identify the lane lines I created the function `fit_polynomial(warped_img)`.
This function find the lane pixels points with `find_lane_pixels` using the same techniques as 
described in the lessons and then uses the function `np.polyfit(points_y, points_x, 2)` to 
find the coefficients of a second polynomial that best fits those points. 
Following the advice from the lessons and to speed up the pipeline I also implemented the 
targeted search for the coefficients in `search_around_poly()`. This function creates a search 
area or windows along of the detected lane markings. The following is an example of how that 
function works:

![alt text][image5]

And this is the output of the function highlighting the search areas in green.

![alt text][image51]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in function `measure_curvature_real()`. Using the formula described on the lessons and
 the previously calculated polynoms it easy to obtain the curvature in pixels. To have a 
 meaningful value in meters, the curvature has to be re-calculated. That was a bit difficult 
 since the previous polynoms where in "pixel" coordinates and now was necessary to have them in 
 meters. That forced to modify the function `search_around_poly()` to also provide the 
 coefficients in meters.
 After that, it was easy to calculate the offset of the car via:
 ```python
 lane_center_position_px = (r_fit_x_int + l_fit_x_int) / 2
 offset = img.shape[1] / 2 * xm_per_pix - lane_center_position_px
```

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in in the function `draw_result`. I found again some issues here because 
I needed the inverse matrix transformation `inv_matrix` used to warp the image. After some 
gymnastics on previous functions and experimentation with `cv2.putText`,
here is an example of my result on a test image:

![alt text][image6]

Tested on all test images:
![alt text][image61]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./output_images/project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

- It takes a lot of time to find the right hyper-parameters for the different threshold functions.
- Unfortunately my current implementation has some wobbling in a few shaded areas and does not 
perform well in the `harder_challenge.mp4` video. 
- The sudden changes on light conditions have a huge impact on the detection algorithms.

###### Future improvements:

- Add "buffering" to the pipeline as suggested in the tips to smooth the lanes.
- Consider other channels for the color transformation, specially the "l" channel.
- Consider other color spaces
- Investigate the combination of different functions (substraction) or weights when generating the 
binary image

