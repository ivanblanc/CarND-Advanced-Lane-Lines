import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle
import glob
import os

from moviepy.editor import VideoFileClip

# Define conversions in x and y from pixels space to meters
ym_per_pix = 30 / 720  # meters per pixel in y dimension
xm_per_pix = 3.7 / 700  # meters per pixel in x dimension


def undistort(img, mtx, dist):
    """ Returns the undistorted image """
    return cv2.undistort(img, mtx, dist, None, mtx)


def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0, 255)):
    """ Returns a binary image after applying Sobel threshold """
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Apply x or y gradient with the OpenCV Sobel() function
    # and take the absolute value
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel))
    # Rescale back to 8 bit integer
    scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))
    # Create a copy and apply the threshold
    binary_output = np.zeros_like(scaled_sobel)
    # Here I'm using inclusive (>=, <=) thresholds, but exclusive is ok too
    binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
    # Return the result
    return binary_output


def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):
    """ Returns a binary image after applying magnitude threshold """
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Take both Sobel x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Calculate the gradient magnitude
    gradmag = np.sqrt(sobelx ** 2 + sobely ** 2)
    # Rescale to 8 bit
    scale_factor = np.max(gradmag) / 255
    gradmag = (gradmag / scale_factor).astype(np.uint8)
    # Create a binary image of ones where threshold is met, zeros otherwise
    binary_output = np.zeros_like(gradmag)
    binary_output[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1
    # Return the binary image
    return binary_output


def dir_threshold(img, sobel_kernel=3, d_thresh=(0.7, 1.3)):
    """ Define a function to threshold an image for a given range and Sobel kernel """
    # Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Calculate the x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Take the absolute value of the gradient direction,
    # apply a threshold, and create a binary image result
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    binary_output = np.zeros_like(absgraddir)
    binary_output[(absgraddir >= d_thresh[0]) & (absgraddir <= d_thresh[1])] = 1
    # Return the binary image
    return binary_output


def color_threshold(img, s_thresh=(90, 255)):
    """ Returns a binary image after applying color threshold """
    # Some other factors to consider 170 255
    # Convert to HLS color space and separate the V channel
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    s_channel = hls[:, :, 2]
    # l_channel = hls[:, :, 1] #TODO (ivan) consider this in future improvements
    # Threshold color channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
    return s_binary


def pipeline(img, ksize=3, dir_th=(0.7, 1.3), sob_th=(20, 100), mag_th=(30, 100), s_th=(170, 255)):
    """ My pipeline for the project, returns a binary image after some processing. """
    # First make a copy of the image
    pic = np.copy(img)

    # Apply each of the threshold functions
    gradx = abs_sobel_thresh(pic, orient='x', sobel_kernel=ksize, thresh=sob_th)
    grady = abs_sobel_thresh(pic, orient='y', sobel_kernel=ksize, thresh=sob_th)
    mag_binary = mag_thresh(pic, sobel_kernel=ksize, mag_thresh=mag_th)
    dir_binary = dir_threshold(pic, sobel_kernel=ksize, d_thresh=dir_th)

    # Combine the results of all different functions.
    combined = np.zeros_like(dir_binary)
    combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1

    s_binary = color_threshold(pic, s_thresh=s_th)

    # combine color image with the binary color
    color_binary = np.zeros_like(combined)
    color_binary[(combined == 1) | (s_binary == 1)] = 1
    return color_binary


def perspective_transform(img):
    """ Perform a perspective transformation, returns a birds eye view and aux parameters. """
    img_size = (img.shape[1], img.shape[0])

    # Original implementation, not performing very well
    # table_offset = img.shape[1] / 8 - 40
    # src = np.float32(
    #     [[(img_size[0] / 2) - table_offset + 10, img_size[1] / 2 + img.shape[0] / 6],
    #      [((img_size[0] / 6) - 5), img_size[1]],
    #      [(img_size[0] * 5 / 6) + table_offset, img_size[1]],
    #      [(img_size[0] / 2 + table_offset + 10), img_size[1] / 2 + img.shape[0] / 6]])
    # dst = np.float32(
    #     [[(img_size[0] / 4), 0],
    #      [(img_size[0] / 4), img_size[1]],
    #      [(img_size[0] * 3 / 4), img_size[1]],
    #      [(img_size[0] * 3 / 4), 0]])

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

    # Given src and dst points, calculate the perspective transform matrix
    trans_matrix = cv2.getPerspectiveTransform(src, dst)
    inv_trans_matrix = cv2.getPerspectiveTransform(dst, src)

    # Warp the image using OpenCV warpPerspective()
    warped = cv2.warpPerspective(img, trans_matrix, img_size, flags=cv2.INTER_LINEAR)

    return warped, src, inv_trans_matrix


def plot_one_image(img, title):
    """ Aux function to plot 1 images """
    plt.imshow(img, cmap='gray')
    plt.title(title, fontsize=40)


def plot_two_images(img1, img2, title1, title2):
    """ Aux function to plot 2 images and titles """
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()

    ax1.imshow(img1, cmap='gray')
    ax1.set_title(title1, fontsize=40)

    ax2.imshow(img2, cmap='gray')
    ax2.set_title(title2, fontsize=40)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)


def load_matrix_coefficients(file_path):
    """ Load the camera calibration coefficients calculated previously. """
    dist_pickle = pickle.load(open(file_path, "rb"))
    return dist_pickle["mtx"], dist_pickle["dist"]


def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    """ Draw lines into an image. """
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)


def search_around_poly(binary_warped, left_fit, right_fit, margin=100):
    """ Performa a targeted search around previous coefficients, returns coefficients. """
    # Grab activated pixels
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    left_lane_inds = ((nonzerox > (
            left_fit[0] * (nonzeroy ** 2) + left_fit[1] * nonzeroy + left_fit[2] - margin)) & (
                              nonzerox < (
                              left_fit[0] * (nonzeroy ** 2) + left_fit[1] * nonzeroy +
                              left_fit[2] + margin)))
    right_lane_inds = ((nonzerox > (
            right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy + right_fit[
        2] - margin)) & (nonzerox < (
            right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy + right_fit[2] + margin)))

    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit new polynomials
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # for the curvature:
    left_fit_cr = np.polyfit(lefty * ym_per_pix, leftx * xm_per_pix, 2)
    right_fit_cr = np.polyfit(righty * ym_per_pix, rightx * xm_per_pix, 2)

    result = np.zeros_like(binary_warped)  # just for keeping function signature, useless.

    # # ## Visualization ##
    # # Generate x and y values for plotting
    # ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
    # left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    # right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]
    #
    # # Create an image to draw on and an image to show the selection window
    # out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
    # window_img = np.zeros_like(out_img)
    # # Color in left and right line pixels
    # out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    # out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
    #
    # # Generate a polygon to illustrate the search window area
    # # And recast the x and y points into usable format for cv2.fillPoly()
    # left_line_window1 = np.array([np.transpose(np.vstack([left_fitx - margin, ploty]))])
    # left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx + margin,
    #                                                                 ploty])))])
    # left_line_pts = np.hstack((left_line_window1, left_line_window2))
    # right_line_window1 = np.array([np.transpose(np.vstack([right_fitx - margin, ploty]))])
    # right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx + margin,
    #                                                                  ploty])))])
    # right_line_pts = np.hstack((right_line_window1, right_line_window2))
    #
    # # Draw the lane onto the warped blank image
    # cv2.fillPoly(window_img, np.int_([left_line_pts]), (0, 255, 0))
    # cv2.fillPoly(window_img, np.int_([right_line_pts]), (0, 255, 0))
    # result = cv2.addWeighted(out_img, 1, window_img, 0.1, 0)
    #
    # # Plot the polynomial lines onto the image
    # plt.plot(left_fitx, ploty, color='orange')
    # plt.plot(right_fitx, ploty, color='orange')
    # ## End visualization steps ##

    return result, left_fit, right_fit, left_fit_cr, right_fit_cr


def find_lane_pixels(binary_warped, nwindows=9, margin=100, minpix=50):
    """ Returns the pixel positions of the lane markings
    Choose:
    - Choose the number of sliding windows
    - width of the windows +/- margin
    - minimum number of pixels found to recenter window
    """
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0] // 2:, :], axis=0)
    # Create an output image to draw on and visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0] // 2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Set height of windows - based on nwindows above and image shape
    window_height = np.int(binary_warped.shape[0] // nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated later for each window in nwindows
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window + 1) * window_height
        win_y_high = binary_warped.shape[0] - window * window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin

        # Draw the windows on the visualization image
        cv2.rectangle(out_img, (win_xleft_low, win_y_low),
                      (win_xleft_high, win_y_high), (0, 255, 0), 2)
        cv2.rectangle(out_img, (win_xright_low, win_y_low),
                      (win_xright_high, win_y_high), (0, 255, 0), 2)

        # Identify the nonzero pixels in x and y within the window #
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                          (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                           (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices (previously was a list of lists of pixels)
    try:
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
    except ValueError:
        # Avoids an error if the above is not implemented fully
        pass

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    return leftx, lefty, rightx, righty, out_img


def fit_polynomial(binary_warped):
    """ Find polynomial coefficients that best fit the image. """
    # Find our lane pixels first
    leftx, lefty, rightx, righty, out_img = find_lane_pixels(binary_warped, 9, 100, 50)

    # Fit a second order polynomial to each using `np.polyfit`
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    out_img = np.copy(binary_warped)  # This is just to keep signature, useless
    # ## Visualization ##
    # # Colors in the left and right lane regions
    # out_img[lefty, leftx] = [255, 0, 0]
    # out_img[righty, rightx] = [0, 0, 255]
    #
    # # Plots the left and right polynomials on the lane lines
    # plt.plot(left_fitx, ploty, color='yellow')
    # plt.plot(right_fitx, ploty, color='yellow')

    return out_img, left_fit, right_fit


def measure_curvature_real(img, left_fit_cr, right_fit_cr):
    """ Calculates the curvature of polynomial functions in meters. """

    # Define y-value where we want radius of curvature
    # We'll choose the maximum y-value, corresponding to the bottom of the image
    y_eval = img.shape[0]

    # Calculation of R_curve (radius of curvature)
    left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval * ym_per_pix + left_fit_cr[
        1]) ** 2) ** 1.5) / np.absolute(2 * left_fit_cr[0])
    right_curverad = ((1 + (2 * right_fit_cr[0] * y_eval * ym_per_pix + right_fit_cr[
        1]) ** 2) ** 1.5) / np.absolute(2 * right_fit_cr[0])

    # Calculation of car position
    car_position = img.shape[1] / 2
    dist = img.shape[0] * ym_per_pix

    l_fit_x_int = left_fit_cr[0] * dist ** 2 + left_fit_cr[1] * dist + left_fit_cr[2]
    r_fit_x_int = right_fit_cr[0] * dist ** 2 + right_fit_cr[1] * dist + right_fit_cr[2]
    lane_center_position_px = (r_fit_x_int + l_fit_x_int) / 2
    # img.shape[1] / 2 = 640 px for all test pictures
    offset = img.shape[1] / 2 * xm_per_pix - lane_center_position_px

    direction = "just in center"
    if offset < 0:
        direction = "right"
    elif offset > 0:
        direction = "left"

    # print("Real", dist, l_fit_x_int, r_fit_x_int, lane_center_position_px, offset, direction)
    # print("lane width real", r_fit_x_int - l_fit_x_int)
    return left_curverad, right_curverad, offset, direction


def draw_result(warped, left_fit, right_fit, image, inv_matrix, rad, pos, direction):
    """ Draws the pipeline result and returns the image. """
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    ploty = np.linspace(0, warped.shape[0] - 1, num=warped.shape[0])
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))
    cv2.polylines(color_warp, np.int_([pts_left]), False, (255, 0, 0), 10)
    cv2.polylines(color_warp, np.int_([pts_right]), False, (0, 0, 255), 10)

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, inv_matrix, (image.shape[1], image.shape[0]))
    # Combine the result with the original image
    undist = np.copy(image)

    # Add the curvature
    text = 'Radius of Curvature = ' + '{:04.2f}'.format(rad) + '(m)'
    cv2.putText(undist, text, (40, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (222, 222, 222), 2,
                cv2.LINE_AA)
    text = 'Vehicle is ' + '{:01.3f}'.format(pos) + 'm ' + direction + ' of center.'
    cv2.putText(undist, text, (40, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (222, 222, 222), 2,
                cv2.LINE_AA)

    result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)
    return result


def process_picture(img_path):
    """ Process picture via pipeline """
    original_image = mpimg.imread(img_path)
    # Read in the saved mtx and dist from the previous step
    mtx, dist = load_matrix_coefficients("../output_images/camera_coef.p")

    # Un-distort the image
    undistort_img = undistort(original_image, mtx, dist)
    # plot_two_images(original_image, undistort_img, "Original image", "Undistort image")
    # plt.savefig('../output_images/undistort_image.jpg')

    # Apply pipeline steps to get a binary image
    binary_image = pipeline(undistort_img, ksize=15)
    # plot_two_images(undistort_img, binary_image, "Undistort image", "Binary image")
    # plt.savefig('../output_images/threshold_binary.jpg')

    # Apply perspective transformation
    warped_img, src, inv_t = perspective_transform(binary_image)
    # plot_two_images(binary_image, warped_img, "Binary image", "Warped image")
    # plt.savefig('../output_images/warped_straight_lines.jpg')

    # Just for documentation, generate image with src lines drawn
    ln = [[[src[0][0], src[0][1], src[1][0], src[2][1]]],
          [[src[2][0], src[2][1], src[3][0], src[3][1]]]]
    draw_lines(undistort_img, ln, [0, 0, 255])
    # warped_img2, src, Minv = perspective_transform(undistort_img)
    # plot_two_images(undistort_img, warped_img2, "Undistort with lines", "Warped with lines")
    # plt.savefig('../output_images/warped_straight_lines_example.jpg')

    # Find initial polynomials (do this only once)
    out_img, left_fit, right_fit = fit_polynomial(warped_img)
    # plot_two_images(warped_img, out_img, "Warped image", "Polynomial fit")
    # plt.title("Polynomial fit")
    # plt.imshow(out_img)
    # plt.savefig('../output_images/fit_polynomial.jpg')

    # Polynomial fit values from the previous frame
    out_img2, left_fitx, right_fitx, left_fit_cr, right_fit_cr = search_around_poly(warped_img,
                                                                                    left_fit,
                                                                                    right_fit,
                                                                                    margin=100)
    # plt.title("Polynomial fit with coefficients")
    # plt.imshow(out_img2)
    # plt.savefig('../output_images/fit_polynomial_coefficients.jpg')

    # Calculate the radius of curvature in meters for both lane lines
    left_curverad, right_curverad, dst, dir = measure_curvature_real(warped_img, left_fit_cr,
                                                                     right_fit_cr)

    plt.clf()
    result = draw_result(warped_img, left_fitx, right_fitx, original_image, inv_t,
                         (right_curverad + left_curverad) * 0.5, abs(dst), dir)
    return result


class Pipeline:

    def __init__(self):
        # Read in the saved mtx and dist from the previous step
        self.mtx, self.dist = load_matrix_coefficients("../output_images/camera_coef.p")
        self.left_line = Line()
        self.right_line = Line()

    def process_image(self, img):
        original_image = img

        # Un-distort the image
        undistort_img = undistort(original_image, self.mtx, self.dist)

        # Apply pipeline steps to get a binary image
        binary_image = pipeline(undistort_img, ksize=15)

        # Apply perspective transformation
        warped_img, src, inv_t = perspective_transform(binary_image)

        # Find initial polynomials (do this only once)
        if not self.left_line.detected and not self.right_line.detected:
            out_img, left_fit, right_fit = fit_polynomial(warped_img)
            self.right_line.detected = True
            self.left_line.detected = True
            self.left_line.current_fit = left_fit
            self.right_line.current_fit = right_fit

        # Polynomial fit values from the previous frame
        left_fit = self.left_line.current_fit
        right_fit = self.right_line.current_fit

        out_img2, left_fitx, right_fitx, left_fit_cr, right_fit_cr = search_around_poly(warped_img,
                                                                                        left_fit,
                                                                                        right_fit,
                                                                                        margin=100)

        # Calculate the radius of curvature in meters for both lane lines
        left_curverad, right_curverad, dst, ori = measure_curvature_real(warped_img, left_fit_cr,
                                                                         right_fit_cr)
        plt.clf()
        result = draw_result(warped_img, left_fitx, right_fitx, original_image, inv_t,
                             (right_curverad + left_curverad) * 0.5, abs(dst), ori)
        return result


class Line:
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False
        # x values of the last n fits of the line
        self.recent_xfitted = []
        # average x values of the fitted line over the last n iterations
        self.bestx = None
        # polynomial coefficients averaged over the last n iterations
        self.best_fit = None
        # polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]
        # radius of curvature of the line in some units
        self.radius_of_curvature = None
        # distance in meters of vehicle center from the line
        self.line_base_pos = None
        # difference in fit coefficients between last and new fits
        self.diffs = np.array([0, 0, 0], dtype='float')
        # x values for detected line pixels
        self.allx = None
        # y values for detected line pixels
        self.ally = None


def process_video(video_name):
    """ Process a video using the pipeline. """
    clip1 = VideoFileClip("../" + video_name)

    pipe = Pipeline()
    white_clip = clip1.fl_image(pipe.process_image)  # .subclip(0, 5)
    white_clip.write_videofile("../output_images/" + video_name, audio=False)
    # clip1.save_frame("../test_images/screenshot_" + video_name + "5s.jpg", t=5)
    # clip1.save_frame("../test_images/screenshot_" + video_name + "12s.jpg", t=12)


def plot_image_grid(images, title, cols=4, rows=5, figsize=(15, 10), cmap=None):
    """ Plot images on a grid. """
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    fig.tight_layout()
    indexes = range(cols * rows)
    for ax, index in zip(axes.flat, indexes):
        if index < len(images):
            image_path, image = images[index]
            if not cmap:
                ax.imshow(image)
            else:
                ax.imshow(image, cmap=cmap)
            ax.set_title(os.path.basename(image_path))
            ax.axis('off')


if __name__ == "__main__":
    # Read in an image
    # image_path = '../test_images/test1.jpg'
    # image_path = '../test_images/straight_lines2.jpg'
    # image_path = '../test_images/extra_5s.jpg'
    # result = process_picture(image_path)
    # plt.axis('off')
    # plt.tight_layout()
    # plt.title("Pipeline result")
    # plt.imshow(result)
    # plt.savefig('../output_images/test1.jpg', bbox_inches='tight')

    # Video
    video_path = 'project_video.mp4'
    # video_path = 'challenge_video.mp4'
    # video_path = 'harder_challenge_video.mp4'
    process_video(video_path)

    # Create image grid
    # test_images = list(
    #     map(lambda img_name: (img_name, ""), glob.glob('../test_images/*.jpg')))
    # plot_image_grid(
    #     list(map(lambda img: (img[0], process_picture(img[0])), test_images)), "Pipeline result", 4,
    #     int(len(test_images) / 4), (15, 13), "gray")
    # plt.axis('off')
    # plt.tight_layout()
    # plt.savefig('../output_images/image_grid.jpg', bbox_inches='tight')
