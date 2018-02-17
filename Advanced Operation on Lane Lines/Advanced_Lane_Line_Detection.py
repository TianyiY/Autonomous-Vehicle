import glob
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pickle
from scipy import signal


# Prepare object points
nx = 9 # Number of inside corners in any given row
ny = 6 # Number of inside corners in any given column
# Read in and make a list of calibration images
calibration_images = glob.glob("CAMERA_CALIBRATION/calibration*.jpg")
# Initialize image and object point arrays
object_points_list = []
image_points_list = []
# Generate object points
object_points = np.zeros((nx*ny, 3), np.float32)
object_points[:,:2] = np.mgrid[0:nx, 0:ny].T.reshape(-1,2) # x, y coordinates

def image_undistort(image, object_points_list, image_points_list):
    # Calibrate camera
    retval, cameraMatrix, distCoeffs, rotation_vectors, translation_vectors = cv2.calibrateCamera(object_points_list, image_points_list, image.shape[0:2], None, None)
    # Undistort image
    undistorted_image = cv2.undistort(image, cameraMatrix, distCoeffs, None, cameraMatrix)
    # Returns undistorted image
    return undistorted_image

for calibration_image in calibration_images:
    # Read in image
    img = cv2.imread(calibration_image)
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Find the chessboard corners
    # Parameters: (image, chessboard dims, param for any flags)
    # chessboard dims = inside corners, not squares.
    ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
    # If found, draw corners
    if ret == True:
        # Fill image point and object point arrays
        image_points_list.append(corners)
        object_points_list.append(object_points)
        # Draw and display the corners
        cv2.drawChessboardCorners(img, (nx, ny), corners, ret)
        plt.imshow(img)
        #plt.show()

### You can uncomment the following block to visualize the undistortion operation
'''
import os
from PIL import Image

imgs=os.listdir('CAMERA_CALIBRATION')
#print(imgs)
imgs_len=len(imgs)
for i in range(imgs_len):
    img=cv2.imread('CAMERA_CALIBRATION'+'/'+imgs[i])
    undistorted_img=image_undistort(img, object_points_list, image_points_list)
    # Visualize undistorted images
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    ax1.imshow(img)
    ax1.set_title('Original Image', fontsize=30)
    ax2.imshow(undistorted_img)
    ax2.set_title('Undistorted Image', fontsize=30)
    plt.show()
'''

# Test undistortion operation on one image
img = cv2.imread('CAMERA_CALIBRATION/calibration11.jpg')
img_size = (img.shape[1], img.shape[0])
# Do camera calibration given object points and image points
retval, cameraMatrix, distCoeffs, rotation_vectors, translation_vectors = cv2.calibrateCamera(object_points_list,
                                                                                              image_points_list,
                                                                                              img.shape[0:2], None,
                                                                                              None)
undistorted_img = cv2.undistort(img, cameraMatrix, distCoeffs, None, cameraMatrix)
cv2.imwrite('CAMERA_CALIBRATION/undistorted_image11.jpg', undistorted_img)
# Save the camera calibration result for later use
undistorted_pickle = {}
undistorted_pickle["cameraMatrix"] = cameraMatrix
undistorted_pickle["distCoeffs"] = distCoeffs
pickle.dump(undistorted_pickle, open( "Camera_Calibration_Result.p", "wb" ) )
undistorted_img = cv2.cvtColor(undistorted_img, cv2.COLOR_BGR2RGB)
# Visualize undistortion
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
ax1.imshow(img)
ax1.set_title('Original Image', fontsize=30)
ax2.imshow(undistorted_img)
ax2.set_title('Undistorted Image', fontsize=30)
plt.show()


# Apply the distortion correction to the raw image
with open("Camera_Calibration_Result.p", mode='rb') as f:
    camera_calibration_param = pickle.load(f)
cameraMatrix = camera_calibration_param["cameraMatrix"]
distCoeffs = camera_calibration_param["distCoeffs"]

raw_image = cv2.imread("test_images/quiz.png")
undistorted_image = cv2.undistort(raw_image, cameraMatrix, distCoeffs, None, cameraMatrix)
# Visualize undistorted image
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
ax1.imshow(raw_image)
ax1.set_title('Original Image', fontsize=30)
ax2.imshow(undistorted_image)
ax2.set_title('Undistorted Image', fontsize=30)
plt.show()


# hyper-parameters
raw_shape = raw_image.shape
raw_height = raw_image.shape[0]
offset = 50
offset_height = raw_height - offset
half_frame = raw_image.shape[1] // 2
steps = 6
pixels_per_step = offset_height / steps
window_radius = 200
median_filter_kernel_size = 51
horizontal_offset = 40
blank_canvas = np.zeros((720, 1280))
colour_canvas = cv2.cvtColor(blank_canvas.astype(np.uint8), cv2.COLOR_GRAY2RGB)

def apply_threshold_v2(image, x_grad_thresh=(20,100), s_thresh=(170,255)):
    # Grayscale image
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # Sobel x
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0) # Take the derivative in x
    abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= x_grad_thresh[0]) & (scaled_sobel <= x_grad_thresh[1])] = 1
    # Threshold colour channel
    # Convert to HLS colour space and separate the S channel
    # image is the undistorted image
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    s_channel = hls[:,:,2]
    # Threshold colour channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
    # Stack each channel to view their individual contributions in green and blue respectively
    # This returns a stack of the two binary images, whose components you can see as different colors
    color_binary = np.dstack(( np.zeros_like(sxbinary), sxbinary, s_binary))
    # Combine the two binary thresholds
    combined_binary = np.zeros_like(sxbinary)
    combined_binary[(s_binary == 1) | (sxbinary == 1)] = 1
    return combined_binary

# use color transforms, gradients, etc. to create a thresholded binary image
x_grad_thresh_temp = (40,100)
s_thresh_temp=(150,255)
combined_binary = apply_threshold_v2(undistorted_image, x_grad_thresh=x_grad_thresh_temp, s_thresh=s_thresh_temp)
plt.imshow(combined_binary, cmap="gray")
plt.show()

def region_of_interest(image, vertices):
    # Applies an image mask to only keeps the region of the image defined by the polygon
    # defining a blank mask to start with
    mask = np.zeros_like(image)
    # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(image.shape) > 2:
        channel_count = image.shape[2]  # 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
    # filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    # returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image

# only show the region of interest
vertices = np.array([[(0,raw_shape[0]), (550, 470), (700, 470), (raw_shape[1],raw_shape[0])]], dtype=np.int32)
masked_image = region_of_interest(combined_binary, vertices)
plt.imshow(masked_image, cmap="gray")
plt.show()


# Apply a perspective transform to rectify binary image (top-down-view)
source = np.float32([[120, 720], [550, 470], [700, 470], [1160, 720]])   # vertices of source image
destination = np.float32([[200,720], [200,0], [1080,0], [1080,720]])    # vertices of transformed image
Map = cv2.getPerspectiveTransform(source, destination)
Map_Inverse = cv2.getPerspectiveTransform(destination, source)
warped = cv2.warpPerspective(combined_binary, Map, (raw_shape[1], raw_shape[0]), flags=cv2.INTER_LINEAR)
plt.imshow(warped, cmap="gray")
plt.show()

def get_pixel_in_window(image, x_center, y_center, size):
    # image: binary image
    # x_center: x coordinate of the window center
    # y_center: y coordinate of the window center
    # size: size of the window in pixel
    # return: x, y of detected pixels
    half_size = size // 2
    window = image[int(y_center - half_size):int(y_center + half_size), int(x_center - half_size):int(x_center + half_size)]
    x, y = (window.T == 1).nonzero()
    x = x + x_center - half_size
    y = y + y_center - half_size
    return x, y

def collapse_into_single_arrays(leftx, lefty, rightx, righty):
    leftx = [x for array in leftx for x in array]
    lefty = [x for array in lefty for x in array]
    rightx = [x for array in rightx for x in array]
    righty = [x for array in righty for x in array]
    leftx = np.array(leftx)
    lefty = np.array(lefty)
    rightx = np.array(rightx)
    righty = np.array(righty)
    return leftx, lefty, rightx, righty

def histogram_pixels(warped_thresholded_image, offset=50, steps=6, window_radius=200, median_filter_kernel_size=51, horizontal_offset=50):
    # Initialize arrays
    left_x = []
    left_y = []
    right_x = []
    right_y = []
    # Parameters
    height = warped_thresholded_image.shape[0]
    offset_height = height - offset
    width = warped_thresholded_image.shape[1]
    half_frame = warped_thresholded_image.shape[1] // 2
    pixels_per_step = offset_height / steps
    for step in range(steps):
        left_x_window_centres = []
        right_x_window_centres = []
        y_window_centres = []
        # Define the window (horizontal slice)
        window_start_y = height - (step * pixels_per_step) + offset
        window_end_y = window_start_y - pixels_per_step + offset
        # Take a count of all the pixels at each x-value in the horizontal slice
        histogram = np.sum(warped_thresholded_image[int(window_end_y):int(window_start_y), int(horizontal_offset):int(width - horizontal_offset)], axis=0)
        # plt.plot(histogram)
        # Smoothen the histogram
        histogram_smooth = signal.medfilt(histogram, median_filter_kernel_size)
        # plt.plot(histogram_smooth)
        # Identify the left and right peaks
        left_peaks = np.array(signal.find_peaks_cwt(histogram_smooth[:half_frame], np.arange(1, 10)))
        right_peaks = np.array(signal.find_peaks_cwt(histogram_smooth[half_frame:], np.arange(1, 10)))
        if len(left_peaks) > 0:
            left_peak = max(left_peaks)
            left_x_window_centres.append(left_peak)
        if len(right_peaks) > 0:
            right_peak = max(right_peaks) + half_frame
            right_x_window_centres.append(right_peak)
        # Add coordinates to window centres
        if len(left_peaks) > 0 or len(right_peaks) > 0:
            y_window_centres.append((window_start_y + window_end_y) // 2)
        # Get pixels in the left window
        for left_x_centre, y_centre in zip(left_x_window_centres, y_window_centres):
            left_x_additional, left_y_additional = get_pixel_in_window(warped_thresholded_image, left_x_centre, y_centre, window_radius)
            # plt.scatter(left_x_additional, left_y_additional)
            # Add pixels to list
            left_x.append(left_x_additional)
            left_y.append(left_y_additional)
        # Get pixels in the right window
        for right_x_centre, y_centre in zip(right_x_window_centres, y_window_centres):
            right_x_additional, right_y_additional = get_pixel_in_window(warped_thresholded_image, right_x_centre, y_centre, window_radius)
            # plt.scatter(right_x_additional, right_y_additional)
            # Add pixels to list
            right_x.append(right_x_additional)
            right_y.append(right_y_additional)
    if len(right_x) == 0 or len(left_x) == 0:
        print("Init no peaks for left or right")
        print("left_x: ", left_x)
        print("right_x: ", right_x)
        horizontal_offset = 0
        left_x = []
        left_y = []
        right_x = []
        right_y = []
        for step in range(steps):
            left_x_window_centres = []
            right_x_window_centres = []
            y_window_centres = []
            # Define the window (horizontal slice)
            window_start_y = height - (step * pixels_per_step) + offset
            window_end_y = window_start_y - pixels_per_step + offset
            # Take a count of all the pixels at each x-value in the horizontal slice
            histogram = np.sum(warped_thresholded_image[int(window_end_y):int(window_start_y), int(horizontal_offset):int(width - horizontal_offset)], axis=0)
            # plt.plot(histogram)
            # Smoothen the histogram
            histogram_smooth = signal.medfilt(histogram, median_filter_kernel_size)
            # plt.plot(histogram_smooth)
            # Identify the left and right peaks
            left_peaks = np.array(signal.find_peaks_cwt(histogram_smooth[:half_frame], np.arange(1, 10)))
            right_peaks = np.array(signal.find_peaks_cwt(histogram_smooth[half_frame:], np.arange(1, 10)))
            if len(left_peaks) > 0:
                left_peak = max(left_peaks)
                left_x_window_centres.append(left_peak)
            if len(right_peaks) > 0:
                right_peak = max(right_peaks) + half_frame
                right_x_window_centres.append(right_peak)
            # Add coordinates to window centres
            if len(left_peaks) > 0 or len(right_peaks) > 0:
                y_window_centres.append((window_start_y + window_end_y) // 2)
            # Get pixels in the left window
            for left_x_centre, y_centre in zip(left_x_window_centres, y_window_centres):
                left_x_additional, left_y_additional = get_pixel_in_window(warped_thresholded_image, left_x_centre, y_centre, window_radius)
                # plt.scatter(left_x_additional, left_y_additional)
                # Add pixels to list
                left_x.append(left_x_additional)
                left_y.append(left_y_additional)
            # Get pixels in the right window
            for right_x_centre, y_centre in zip(right_x_window_centres, y_window_centres):
                right_x_additional, right_y_additional = get_pixel_in_window(warped_thresholded_image, right_x_centre, y_centre, window_radius)
                # plt.scatter(right_x_additional, right_y_additional)
                # Add pixels to list
                right_x.append(right_x_additional)
                right_y.append(right_y_additional)
    return collapse_into_single_arrays(left_x, left_y, right_x, right_y)

def fit_second_order_poly(indep, dep, return_coeffs=False):
    fit = np.polyfit(indep, dep, 2)
    fitdep = fit[0]*indep**2 + fit[1]*indep + fit[2]
    if return_coeffs == True:
        return fitdep, fit
    else:
        return fitdep

def draw_poly(image, poly, poly_coeffs, steps, color=[255, 0, 0], thickness=10, dashed=False):
    image_height = image.shape[0]
    pixels_per_step = image_height // steps
    for i in range(steps):
        start = i * pixels_per_step
        end = start + pixels_per_step
        start_point = (int(poly(start, poly_coeffs=poly_coeffs)), start)
        end_point = (int(poly(end, poly_coeffs=poly_coeffs)), end)
        if dashed == False or i % 2 == 1:
            image = cv2.line(image, end_point, start_point, color, thickness)
    return image

def lane_poly(yval, poly_coeffs):
    # Returns x value for poly given a y-value. (x = Ay^2 + By + C)
    return poly_coeffs[0]*yval**2 + poly_coeffs[1]*yval + poly_coeffs[2]

# Detect lane pixels and fit to find lane boundary.
# Histogram and get pixels in window
leftx, lefty, rightx, righty = histogram_pixels(warped, horizontal_offset=horizontal_offset)
# Fit a second order polynomial to each fake lane line
left_fit, left_coeffs = fit_second_order_poly(lefty, leftx, return_coeffs=True)
print("Left coeffs:", left_coeffs)
print("righty[0]: ,", righty[0], ", rightx[0]: ", rightx[0])
right_fit, right_coeffs = fit_second_order_poly(righty, rightx, return_coeffs=True)
print("Right coeffs: ", right_coeffs)

# Plot data
plt.plot(left_fit, lefty, color='green', linewidth=3)
plt.plot(right_fit, righty, color='green', linewidth=3)
plt.imshow(warped, cmap="gray")
plt.show()
print("Left coeffs: ", left_coeffs)
print("Right fit: ", right_coeffs)
polyfit_left = draw_poly(blank_canvas, lane_poly, left_coeffs, 30)
polyfit_drawn = draw_poly(polyfit_left, lane_poly, right_coeffs, 30)
plt.imshow(polyfit_drawn, cmap="gray")
plt.show()

def evaluate_poly(indep, poly_coeffs):
    return poly_coeffs[0]*indep**2 + poly_coeffs[1]*indep + poly_coeffs[2]

def highlight_lane_line_area(mask_template, left_poly, right_poly, start_y=0, end_y =720):
    area_mask = mask_template
    for y in range(start_y, end_y):
        left = evaluate_poly(y, left_poly)
        right = evaluate_poly(y, right_poly)
        area_mask[y][int(left):int(right)] = 1
    return area_mask

trace = colour_canvas
trace[polyfit_drawn > 1] = [0,0,255]
area = highlight_lane_line_area(blank_canvas, left_coeffs, right_coeffs)
trace[area == 1] = [0,255,0]
plt.imshow(trace)
plt.show()


def center(y, left_poly, right_poly):
    return (1.5 * evaluate_poly(y, left_poly) - evaluate_poly(y, right_poly)) / 2

# Determine curvature of the lane and vehicle position with respect to center
# Determine curvature of the lane
# Define y-value where we want radius of curvature
# choose the maximum y-value, corresponding to the bottom of the image
y_eval = 500
left_curverad = np.absolute(((1 + (2 * left_coeffs[0] * y_eval + left_coeffs[1])**2) ** 1.5)/(2 * left_coeffs[0]))
right_curverad = np.absolute(((1 + (2 * right_coeffs[0] * y_eval + right_coeffs[1]) ** 2) ** 1.5)/(2 * right_coeffs[0]))
print("Left lane curve radius: ", left_curverad, "pixels")
print("Right lane curve radius: ", right_curverad, "pixels")
curvature = (left_curverad + right_curverad) / 2
centre = center(719, left_coeffs, right_coeffs)
min_curvature = min(left_curverad, right_curverad)

def add_figures_to_image(image, curvature, vehicle_position, min_curvature, left_coeffs=(0,0,0), right_coeffs=(0,0,0)):
    # Convert from pixels to meters
    vehicle_position = vehicle_position / 12800 * 3.7
    curvature = curvature / 128 * 3.7
    min_curvature = min_curvature / 128 * 3.7
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(image, 'Radius of Curvature = %d(m)' % curvature, (50, 50), font, 1, (255, 255, 255), 2)
    left_or_right = "Left" if vehicle_position < 0 else "Right"
    cv2.putText(image, 'Vehicle is %.2fm %s of Center' % (np.abs(vehicle_position), left_or_right), (50, 100), font, 1, (255, 255, 255), 2)
    cv2.putText(image, 'Min Radius of Curvature = %d(m)' % min_curvature, (50, 150), font, 1, (255, 255, 255), 2)
    cv2.putText(image, 'Left Poly Coefficients = %.3f %.3f %.3f' % (left_coeffs[0], left_coeffs[1], left_coeffs[2]), (50, 200), font, 1, (255, 255, 255), 2)
    cv2.putText(image, 'Right Poly Coefficients = %.3f %.3f %.3f' % (right_coeffs[0], right_coeffs[1], right_coeffs[2]), (50, 250), font, 1, (255, 255, 255), 2)

# Warp the detected lane boundaries back onto the original image
# Warp lane boundaries back onto original image
lane_lines = cv2.warpPerspective(trace, Map_Inverse, (raw_shape[1], raw_shape[0]), flags=cv2.INTER_LINEAR)
# Convert to colour
combined_image = cv2.add(lane_lines, undistorted_image)
add_figures_to_image(combined_image, curvature=curvature, vehicle_position=centre, min_curvature=min_curvature,
                     left_coeffs=left_coeffs, right_coeffs=right_coeffs)
plt.imshow(combined_image)
plt.show()