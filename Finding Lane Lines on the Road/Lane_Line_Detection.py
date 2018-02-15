import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import os
import math
from sklearn.linear_model import LinearRegression
from moviepy.editor import VideoFileClip
from IPython.display import HTML


def read_image_print_dimension(path):
    image=mpimg.imread(path)
    print('Image: ', type(image), 'Dimension: ', image.shape)
    plt.imshow(image)
    return image


# Read and save images to array
Images = [read_image_print_dimension('IMAGES/' + i) for i in os.listdir('IMAGES/')]


'''
cv2.inRange() for color selection
cv2.fillPoly() for regions selection
cv2.line() to draw lines on an image given endpoints
cv2.addWeighted() to coadd / overlay two images cv2.cvtColor() to grayscale or change color cv2.imwrite() to output images to file
cv2.bitwise_and() to apply a mask to an image
'''

# Grayscale transform: return an image with only one color channel
def gray(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


# Canny transform
def canny(image, low_threshold, high_threshold):
    return cv2.Canny(image, low_threshold, high_threshold)


# Gaussian Noise kernel
def gaussian_blur(image, kernel_size):
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)


# Image mask: Only keeps the region of the image defined by the polygon formed from vertices
def region_mask(image, vertices):
    # defining a blank mask to start with
    mask = np.zeros_like(image)
    # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(image.shape) > 2:
        channel_count = image.shape[2]
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
    # filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    # returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image


# Input an image after Canny transform and return an image with hough lines.
def hough_lines(image, rho, theta, threshold, min_line_len, max_line_gap):
    H_lines = cv2.HoughLinesP(image, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    # print("Hough lines: ", lines)
    return H_lines


# Returns x-coordinate of intersection of two lines.
def intersection_x(coef1, intercept1, coef2, intercept2):
    return (intercept2 - intercept1) / (coef1 - coef2)


def draw_linear_regression_line(coef, intercept, intersection_x, image, image_shape=[540, 960], color=[255, 0, 0], thickness=2):
    # Get starting and ending points of regression line.
    # print("Coef: ", coef, "Intercept: ", intercept, "intersection_x: ", intersection_x)
    point_one = (int(intersection_x), int(intersection_x * coef + intercept))
    if coef > 0:
        point_two = (image_shape[1], int(image_shape[1] * coef + intercept))
    elif coef < 0:
        point_two = (0, int(0 * coef + intercept))
    # print("Point one: ", point_one, "Point two: ", point_two)
    # Draw line using cv2.line
    cv2.line(image, point_one, point_two, color, thickness)


def find_line_fit(slope_intercept):  # slope_intercept is an array [[slope, intercept], [slope, intercept]...]
    # Initialize arrays
    keep_slopes = []
    keep_intercepts = []
    # print("Slope & intercept: ", slope_intercept)
    if len(slope_intercept) == 1:
        return slope_intercept[0][0], slope_intercept[0][1]
    # Remove points with slope not within 1.5 standard deviations of the mean
    slopes = [pair[0] for pair in slope_intercept]
    mean_slope = np.mean(slopes)
    slope_std = np.std(slopes)
    for pair in slope_intercept:
        slope = pair[0]
        if slope - mean_slope < 1.5 * slope_std:
            keep_slopes.append(slope)
            keep_intercepts.append(pair[1])
    if not keep_slopes:
        keep_slopes = slopes
        keep_intercepts = [pair[1] for pair in slope_intercept]
    # Take estimate of slope, intercept to be the mean of remaining values
    slope = np.mean(keep_slopes)
    intercept = np.mean(keep_intercepts)
    # print("Slope: ", slope, "Intercept: ", intercept)
    return slope, intercept


# Linear Regression Option
def find_linear_regression_line(points):
    # Separate points into X and y to fit LinearRegression model
    points_x = [[point[0]] for point in points]
    points_y = [point[1] for point in points]
    # Fit points to LinearRegression line
    clf = LinearRegression().fit(points_x, points_y)
    # Get parameters from line
    coef = clf.coef_[0]
    intercept = clf.intercept_
    # print("Coefficients: ", coef, "Intercept: ", intercept)
    return coef, intercept


# Average/extrapolate the line segments to map out the full extent of the lane
# Separating line segments by their slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left line and the right line.
# Average the position of each of the lines and extrapolate to the top and bottom of the lane
def draw_lines(image, lines, color=[255, 0, 0], thickness=2):
    image_shape = [540, 960]
    # Initialise arrays
    positive_slope_points = []
    negative_slope_points = []
    positive_slope_intercept = []
    negative_slope_intercept = []
    for line in lines:
        for x1, y1, x2, y2 in line:
            slope = (y1 - y2) / (x1 - x2)
            # print("Points: ", [x1, y1, x2, y2])
            length = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
            # print("Length: ", length)
            if not math.isnan(slope):
                if length > 50:
                    if slope > 0:
                        positive_slope_points.append([x1, y1])
                        positive_slope_points.append([x2, y2])
                        positive_slope_intercept.append([slope, y1 - slope * x1])
                    elif slope < 0:
                        negative_slope_points.append([x1, y1])
                        negative_slope_points.append([x2, y2])
                        negative_slope_intercept.append([slope, y1 - slope * x1])
    # If either array is empty, waive length requirement
    if not positive_slope_points:
        for line in lines:
            for x1, y1, x2, y2 in line:
                slope = (y1 - y2) / (x1 - x2)
                if slope > 0:
                    positive_slope_points.append([x1, y1])
                    positive_slope_points.append([x2, y2])
                    positive_slope_intercept.append([slope, y1 - slope * x1])
    if not negative_slope_points:
        for line in lines:
            for x1, y1, x2, y2 in line:
                slope = (y1 - y2) / (x1 - x2)
                if slope < 0:
                    negative_slope_points.append([x1, y1])
                    negative_slope_points.append([x2, y2])
                    negative_slope_intercept.append([slope, y1 - slope * x1])
    if not positive_slope_points:
        print("positive_slope_points are empty")
    if not negative_slope_points:
        print("negative_slope_points are empty")
    positive_slope_points = np.array(positive_slope_points)
    negative_slope_points = np.array(negative_slope_points)
    # print("Positive slope line points: ", positive_slope_points)
    # print("Negative slope line points: ", negative_slope_points)
    # print("positive slope points dtype: ", positive_slope_points.dtype)
    # Get intercept and coefficient of fitted lines
    pos_coef, pos_intercept = find_line_fit(positive_slope_intercept)
    neg_coef, neg_intercept = find_line_fit(negative_slope_intercept)
    # Following is Linear Regression Option:
    # Get intercept and coefficient of linear regression lines
    # pos_coef, pos_intercept = find_linear_regression_line(positive_slope_points)
    # neg_coef, neg_intercept = find_linear_regression_line(negative_slope_points)
    # Get intersection point
    intersection_x_coord = intersection_x(pos_coef, pos_intercept, neg_coef, neg_intercept)
    # Plot lines
    draw_linear_regression_line(pos_coef, pos_intercept, intersection_x_coord, image)
    draw_linear_regression_line(neg_coef, neg_intercept, intersection_x_coord, image)


def draw_hough_lines_image(image, rho, theta, threshold, min_line_len, max_line_gap):
    H_lines=hough_lines(image, rho, theta, threshold, min_line_len, max_line_gap)
    # print("Hough lines: ", H_lines)
    H_lines_image = np.zeros(image.shape, dtype=np.uint8)
    draw_lines(H_lines_image, H_lines)
    return H_lines_image


# H_lines_image should be the output of the draw_hough_lines_image(), an image with lines drawn on it. A blank image (all black) with lines drawn on it.
# initial_image should be the image before any processing.
# Return should be initial_image * alpha + H_lines_image * bate + lambda
def weighted_img(H_lines_image, initial_image, alpha=0.8, beta=1., Lambda=0.):
    return cv2.addWeighted(initial_image, alpha, H_lines_image, beta, Lambda)


# Pipeline
'''
Read in and grayscale the image
Define a kernel size and apply Gaussian smoothing
Define our parameters for Canny and apply to get edges image
Mask edges image using cv2.fillPoly() (ignore everything outside region of interest)
Define Hough transform parameters and run Hough transform on masked edge-detected image
Draw line segments
Draw lines extrapolated from line segments
Combine line image with original image to see how accurate the line annotations are.
'''
def draw_lane_lines(image):
    image_shape = image.shape
    # Greyscale image
    greyscaled_image = gray(image)
    plt.subplot(2, 2, 1)
    plt.imshow(greyscaled_image, cmap="gray")
    # Gaussian Blur
    blurred_grey_image = gaussian_blur(greyscaled_image, 5)
    # Canny edge detection
    edges_image = canny(blurred_grey_image, 50, 150)
    # Mask edges image
    vertices = np.array([[(0, image_shape[0]), (465, 320), (475, 320), (image_shape[1], image_shape[0])]], dtype=np.int32)
    edges_image_with_mask = region_mask(edges_image, vertices)
    # Plot masked edges image
    bw_edges_image_with_mask = cv2.cvtColor(edges_image_with_mask, cv2.COLOR_GRAY2BGR)
    plt.subplot(2, 2, 2)
    plt.imshow(bw_edges_image_with_mask)
    # Hough lines
    rho = 2  # distance resolution in pixels of the Hough grid
    theta = np.pi / 180  # angular resolution in radians of the Hough grid
    threshold = 45  # minimum number of votes (intersections in Hough grid cell)
    min_line_len = 40  # minimum number of pixels making up a line
    max_line_gap = 100  # maximum gap in pixels between connectable line segments
    lines_image = draw_hough_lines_image(edges_image_with_mask, rho, theta, threshold, min_line_len, max_line_gap)
    # Convert Hough from single channel to RGB to prep for weighted
    hough_rgb_image = cv2.cvtColor(lines_image, cv2.COLOR_GRAY2BGR)
    # hough_rgb_image.dtype: uint8.  Shape: (540,960,3).
    # hough_rgb_image is like [[[0 0 0], [0 0 0],...] [[0 0 0], [0 0 0],...]]
    # Plot Hough lines image
    plt.subplot(2, 2, 3)
    plt.imshow(hough_rgb_image)
    # Combine lines image with original image
    final_image = weighted_img(hough_rgb_image, image)
    # Plot final image
    plt.subplot(2, 2, 4)
    plt.imshow(final_image)
    plt.show()
    return final_image


for i in range(len(Images)):
    draw_lane_lines(Images[i])


# Test on video: Puts image through pipeline and returns 3-channel image for processing video
def process_image(image):
    output_image = draw_lane_lines(image)
    # print(output_image.shape)
    return output_image


VideoFileClip("lanes.mp4").fl_image(process_image).write_videofile('detected_lanes.mp4', audio=False)


HTML("""
<video width="960" height="540" controls>
  <source src="{0}">
</video>
""".format('detected_lanes.mp4'))
