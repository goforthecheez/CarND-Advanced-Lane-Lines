## Advanced Lane Finding Project Writeup

---

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.


## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

All code pointers below refer to code in the [Advanced-Lane-Lines.ipynb](./Advanced-Lane-Lines.ipynb) IPython notebook.

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for computing the camera matrix `mtx` and the distortion coefficients `dist` is contained in Section 1 of the notebook.

I start by preparing "object points" `objp`, which will be the (x, y, z) coordinates of the chessboard corners in the world. I assume the chess board is located in the (x, y) plane at z = 0, with the object points the same for each calibration image; i.e. I assume it's the same chess board in each image and the chess board is flat. Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image. `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration matrix `mtx` and distortion coefficients `dist` using the `cv2.calibrateCamera()` function. 

In Section 2 of the notebook, I applied this distortion correction using the `cv2.undistort()` function.

* I applied distortion correction to calibration images `./camera_cal/calibration1.jpg` thru `./camera_cal/calibration5.jpg`, which are close-ups of the chess board and have the most pronounced distortion.
* The results can be found in the [./output_images/camera_cal](./output_images/camera_cal) folder. They show that the straight lines in the chess board have been restored!

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To apply distortion correction to raw input images, I use the same method -- including the same camera matrix and distortion coefficients -- as for undistorting a calibration image. This can be found in Section 2 of the notebook, after applying distortion correction to the calibration images.

* I applied distortion correction to all images in the `./test_images` folder.
* Output images are located in the [./output_images/undistort](./output_images/undistort) folder.

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

In Section 3 of the notebook, I first use gradient thresholds, then an HLS transform and color thresholds, and finally combine both methods to generate a thresholded binary image that emphasizes lane lines and minimizes visual noise in the middle of the roadway.

To perform gradient thresholding, I use the `cv2.Sobel()` function, applied to grayscaled input images. I compute both the absolute value of the gradient in the x and y directions (`abs_sobel_thresh()`, where the direction can be set using the `orient` parameter), the magnitude of the gradient (`mag_thresh()`) using the Pythagorean theorem, and the direction of the gradient (`dir_thresh()`) by using the arctangent of gradients in the x and y directions.

I only include a pixel if both its gradients in the x and y directions are large (to eliminate the majority of noise), or if the magnitude of the gradient is large and in a more-or-less vertical direction (to retain pixels whose gradients are in the right direction to be lane lines). This is done in the `apply_gradient_thresholds()` function.

* I applied gradient thresholding to all images in the `./test_images` folder.
* Output images are located in the [./output_images/grad_thresh](./output_images/grad_thresh) folder.

The results showed that the solid yellow left lane line in some of the images was not being detected. I solved this problem by transforming images into HLS color space, and applying thresholding to the hue and saturation channels (`hue_threshold()` and `saturation_threshold()`, respectively). I use HLS color space, ignoring the lightness channel, since we want to be able to detect the lane under different lighting conditions. Lane lines are both high-saturation and in a fixed range of colors, so I included a pixel if it met both the hue and saturation thresholds. This is done in the `apply_color_thresholds()` function.

* I applied color thresholding to all images in the `./test_images` folder.
* Output images are located in the [./output_images/color_thresh](./output_images/color_thresh) folder.

Finally, I combine the gradient and color thresholds in the `apply_combined_thresholds()` function. A pixel is included if it meets either the gradient or color thresholding conditions.

* I applied combined threshold to all images in the `./test_images` folder.
* Output images are located in the [./output_images/combined_thresh](./output_images/combined_thresh) folder.

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

I apply a perspective transform in the `warp()` function in Section 4 of the notebook.

This function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points. To reverse the perspective transform (which will have to be done after curve-fitting to generate the final output), the `src` and `dst` arguments can be flipped.

To verify that `warp()` works as expected, I first apply it to an undistorted chessboard calibration image. Here, the `src` points are found using the `findChessboardCorners()` function and selecting the outermost corners. The `dst` points were set to 100 pixels offset in both the x- and y-directions from the corners of the original image.

* The output of this process can be found at[./output_images/warped/calibration2.jpg](./output_images/warped/calibration2.jpg).

Finally, I define a perspective transform for roads. This can be found in the cell containing the `warp_road()` function.

I define the `src` points like this:

```python
height = 720
width = 1280

top_left = [int(0.47 * width), int(0.62 * height)]
top_right = [int(0.53 * width), int(0.62 * height)]
bottom_right = [int(0.85 * width), height]
bottom_left = [int(0.17 * width), height]
vertices = [top_left, top_right, bottom_right, bottom_left]
src = np.float32(vertices)
```

I define the `dst` points like this:

```python
left_margin = 200
right_margin = 400
dst = np.float32([[left_margin, 0],
                  [width - right_margin, 0],
                  [width - right_margin, height],
                  [left_margin, height]])
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 601, 446      | 200, 0        | 
| 678, 446      | 880, 0        |
| 1088, 720     | 880, 720      |
| 217, 720      | 200, 720      |

I use a larger `right_margin` than `left_margin` to order to help the right lane line stay inside the image boundaries. The car tends to drift to the left in all the images and the project video.

* I corrected for image distortion, applied thresholding, and applied the perspective transform to all images in the `./test_images` folder.
* Output images are located in the [./output_images/warped](./output_images/warped) folder.

I verified that `./test_images/straight_lines1.jpg` and `./test_images/straight_lines2.jpg` look parallel, and the curves do a good job staying within the image boundaries.

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

This code is in Section 5 of the notebook.

I identify the left and right lanes by using a sliding window search and fit curves to them in the function `fit_curves_from_scratch()`, which takes a thresholded, perspective-transformed binary image as input. The function works as follows:

* Find the bottom end of each lane line by taking a histogram of the lower half of the road image and finding the left and right peaks. These (`leftx_base` and `rightx_base`) will be the x-axis centers of the first windows. Due to artificially shifting the image to the left in the perspective transform step, I define a `dead_space` region at the right of the histogram to ignore when finding peaks.
* Define a window height and width (via a `margin` from the x-axis centers `leftx_current` and `rightx_current`, which are initialized to `leftx_base` and `rightx_base`).
* Then, looping over the height of the image by window-height:
    * Define windows for the left and right lanes (via `win_{y, xleft, xright}_{low, high}` variables). For example, the left window has opposing corners `(win_xleft_low, win_y_low), (win_xleft_high, win_y_high)`. These may be optionally drawn on the input image in green.
    * Add the "good" (nonzero) pixels within the windows to lists `left_lane_inds` and `right_lane_indx` of left and right lane pixels (identified by their indexes).
    * If more than a threshold number `minpix` of "good" pixels appear within a window, reset its x-axis centers `leftx_current` and `rightx_current` to the "good" pixels' mean x-value for the next iteration.
* Finally, use the left and right lane pixel lists color the left lane red and the right lane yellow. Also extract from these pixel arrays of x-values and y-values for the lane lines, and fit 2nd-order polynomials to them using `np.polyfit()`.

To visualize the sliding windows and fit curves, I defined a `visualize_fit_from_scratch()` function that takes as inputs the input binary warped image and return values of `fit_curves_from_scratch()`. It uses `np.linspace` to compute an array of y-values `ploty` to plot, then uses the coefficients of the fit curves `left_fit` and `right_fit` to compute x-values. These values are then used to plot the left and right lane lines in yellow.

* I fit curves using a sliding window search and visualized the results for all images in the `./test_images` folder.
* Output images are located in the [./output_images/fit](./output_images/fit) folder.

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

This code is in Section 6 of the notebook, in the `compute_curvature_and_offset()` function, which takes the fit curve coeffiecients `left_fit` and `right_fit` as arguments.

I want to compute curvature and position at the location of the car, which is at the bottom edge of the image, so I define `y_eval` to be the height of the image - 1 (to get an index). Since we want to work in real-world distances, I first transform the fit curve coeffiecients into meters. I am given scaling factors `xm_per_pix` and `ym_per_pix` for converting distances from pixels to meters, so this is done like so:

```python
scaling_factors = [xm_per_pix/ym_per_pix**2, xm_per_pix/ym_per_pix, xm_per_pix]
```

These scaling factors are then applied to `left_fit` and `right_fit` to get `left_fit_scaled` and `right_fit_scaled`. I also need to convert the input y-value into meters, which I do by scaling `y_eval` by `ym_per_pix` to get `y_scaled`.

Now that all the `*_scaled` variables hold values are in meters, I simply apply the radius of curvature definition (1 + (2*A*y + B)^2)^(3/2))/|2*A| like so:

```python
left_curverad = (((1 + (2 * left_fit_scaled[0] * y_scaled + left_fit_scaled[1])**2)**1.5)
                 / np.absolute(2 * left_fit_scaled[0]))
right_curverad = (((1 + (2 * right_fit_scaled[0] * y_scaled + right_fit_scaled[1])**2)**1.5)
                  / np.absolute(2 * right_fit_scaled[0]))
```

Offsets from the center of the lane are computed by evaluating the polynomial at the bottom edge of the image and taking the mean:

```python
leftx = (left_fit_scaled[0] * y_scaled**2 + left_fit_scaled[1] * y_scaled + left_fit_scaled[2])
rightx = (right_fit_scaled[0] * y_scaled**2 + right_fit_scaled[1] * y_scaled + right_fit_scaled[2])
lane_center = (rightx + leftx) / 2
```

Since I assume the camera is mounted at the center of the car, I find the image center by dividing the image width by 2 and scaling by `xm_per_pix`, subject to correcting for the leftward shift during the perspective transform step:

```python
image_center = ((width - (right_margin - left_margin))/ 2) * xm_per_pix
```

Finally, the offset is the difference between the image and lane centers.

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

The code for this is in Section 7 of the notebook.

* I applied the complete pipeline, unwarped, and annotated all the images in the `./test_images` folder.
* Output images are available in the [./output_images/final](./output_images/final) folder.

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./output_images/project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

* One issue/potential failure point I faced was reliably being able to detect the lane lines without much noise, during thresholding. To allow for changes in the color of the pavement, I used looser color thresholding, which can start picking up lighter patches in the pavement. To prevent visual noise from trees casting shadows on the pavement, I used stricter gradient thresholding, which can overlook lane lines in the distance that appear shorter. The effect of unreliable lane detection is jittering in the edges of the green lane area. I could reduce the jittering by averaging the curves seen in several previous frames of the video, if I an unable to fit a reasonable curve this frame.

* Another issue/potential failure point I faced was keeping both lane lines within the image boundaries during the perspective transform. Since the car tends to drift to the left, I solved this problem by mapping the lane line to a destination rectangle that is just a bit narrower than the original image size, but with a wider right margin. If the car were to also drifts to the right, I would need to use wide margins on both sides. Doing this would keep both lane lines in view, but would also start picking up objects in neighboring lanes, which can start interfering with the sliding window search. A solution, which I did employ, is to restrict the range explored for finding histogram peaks during the sliding window search.

* My method is also fairly slow, since it runs the sliding window search from scratch on every single video frame. It would be faster to apply the method discussed in the course content, where a band around the pixels in the left and right lanes from the previous frame were used to search for lane lines in the current frame.