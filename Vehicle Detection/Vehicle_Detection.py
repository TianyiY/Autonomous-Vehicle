import cv2
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from skimage.feature import hog
import glob
import time
from random import choice
from skimage.measure import label, regionprops
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV
from scipy.ndimage.measurements import label
from moviepy.editor import VideoFileClip


# Function to return HOG features and visualization
def get_hog_features(img, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True):
    # Call with two outputs if vis==True
    if vis == True:
        features, hog_image = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=True,
                                  visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    # Otherwise call with one output
    else:
        features = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=True,
                       visualise=vis, feature_vector=feature_vec)
        return features

# Function to compute color histogram features
# bins_range=(0, 256) is commanded out, for png the range is (0, 1), very confusing
def color_hist(img, nbins=32): #, bins_range=(0, 256)):
    # Compute the histogram of the RGB channels separately
    rhist = np.histogram(img[:,:,0], bins=nbins) #, range=bins_range)
    ghist = np.histogram(img[:,:,1], bins=nbins) #, range=bins_range)
    bhist = np.histogram(img[:,:,2], bins=nbins) #, range=bins_range)
    # Generating bin centers
    bin_edges = rhist[1]
    bin_centers = (bin_edges[1:]  + bin_edges[0:len(bin_edges)-1])/2
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((rhist[0], ghist[0], bhist[0]))
    # Return the individual histograms, bin_centers and feature vector
    # return rhist, ghist, bhist, bin_centers, hist_features
    return hist_features

# Function to compute color histogram features
def bin_spatial(img, color_space='RGB', size=(32, 32)):
    # Convert image to new color space (if specified)
    if color_space != 'RGB':
        if color_space == 'HSV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        elif color_space == 'LUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
        elif color_space == 'HLS':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        elif color_space == 'YUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        elif color_space == 'YCrCb':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    else: feature_image = np.copy(img)
    # Use cv2.resize().ravel() to create the feature vector
    features = cv2.resize(feature_image, size).ravel()
    # Return the feature vector
    return features

# Function to extract features from a list of images
def extract_features(imgs, color_space='RGB', spatial_size=(32, 32), hist_bins=32, orient=9, pix_per_cell=8,
                     cell_per_block=2, hog_channel=0, spatial_feat=True, hist_feat=True, hog_feat=True):
    # Create a list to append feature vectors to
    features = []
    cnt = imgs.shape[0]
    # Iterate through the list of images
    for i in range(0, cnt):
        file_features = []
        # Read in each one by one
        image = imgs[i,:,:,:]
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
                    hog_features.append(get_hog_features(feature_image[:,:,channel], orient, pix_per_cell, cell_per_block,
                                        vis=False, feature_vec=True))
                hog_features = np.ravel(hog_features)
            else:
                hog_features = get_hog_features(feature_image[:,:,hog_channel], orient, pix_per_cell, cell_per_block,
                                                vis=False, feature_vec=True)
            # Append the new feature vector to the features list
            file_features.append(hog_features)
        features.append(np.concatenate(file_features))
    # Return list of feature vectors
    return features

# Function to draw boxes on top of an image
def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    # Make a copy of the image
    imcopy = np.copy(img)
    # Iterate through the bounding boxes
    for bbox in bboxes:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
    # Return the image copy with boxes drawn
    return imcopy

# Function to find all the slide windows on a given image shape
def slide_window(img_shape, x_start_stop=[None, None], y_start_stop=[None, None], xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
    # If x and/or y start/stop positions not defined, set to image size
    if x_start_stop[0] == None:
        x_start_stop[0] = 0
    if x_start_stop[1] == None:
        x_start_stop[1] = img_shape[1]
    if y_start_stop[0] == None:
        y_start_stop[0] = 0
    if y_start_stop[1] == None:
        y_start_stop[1] = img_shape[0]
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

# Function to extract features from a single image window
def single_img_features(img, color_space='RGB', spatial_size=(32, 32), hist_bins=32, orient=9, pix_per_cell=8,
                        cell_per_block=2, hog_channel=0, spatial_feat=True, hist_feat=True, hog_feat=True):
    # Define an empty list to receive features
    img_features = []
    # Apply color conversion if other than 'RGB'
    if color_space != 'RGB':
        if color_space == 'HSV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        elif color_space == 'LUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
        elif color_space == 'HLS':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        elif color_space == 'YUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        elif color_space == 'YCrCb':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    else: feature_image = np.copy(img)
    # Compute spatial features if flag is set
    if spatial_feat == True:
        spatial_features = bin_spatial(feature_image, size=spatial_size)
        # Append features to list
        img_features.append(spatial_features)
    # Compute histogram features if flag is set
    if hist_feat == True:
        hist_features = color_hist(feature_image, nbins=hist_bins)
        # Append features to list
        img_features.append(hist_features)
    # Compute HOG features if flag is set
    if hog_feat == True:
        if hog_channel == 'ALL':
            hog_features = []
            for channel in range(feature_image.shape[2]):
                hog_features.extend(get_hog_features(feature_image[:,:,channel], orient, pix_per_cell, cell_per_block,
                                    vis=False, feature_vec=True))
        else:
            hog_features = get_hog_features(feature_image[:,:,hog_channel], orient, pix_per_cell, cell_per_block,
                                            vis=False, feature_vec=True)
        # Append features to list
        img_features.append(hog_features)
    # Return concatenated array of features
    return np.concatenate(img_features)

# Function to search the car in all the given windows in a given image
def search_windows(img, windows, clf, scaler, color_space='RGB', orient=9, pix_per_cell=8, cell_per_block=2, hog_channel=0):
    # Create an empty list to receive positive detection windows
    on_windows = []
    # Iterate over all windows in the list
    for window in windows:
        # Extract the test window from original image
        window_img = img[window[0][1]:window[1][1], window[0][0]:window[1][0], :]
        test_img = cv2.resize(window_img, (64, 64))
        # Extract features for that window using single_img_features()
        features = single_img_features(test_img, color_space=color_space, orient=orient, pix_per_cell=pix_per_cell,
                                        cell_per_block=cell_per_block, hog_channel=hog_channel, spatial_feat=True,
                                       hist_feat=True, hog_feat=True)
        # Scale extracted features to be fed to classifier
        test_features = scaler.transform(np.array(features).reshape(1, -1))
        # Predict using your classifier
        prediction = clf.predict_proba(test_features)[0][1]
        # If positive (prediction == 1) then save the window
        if prediction > 0.85:
            on_windows.append(window)
    # Return windows for positive detections
    return on_windows

# Function to add boxed regions to a heatmap
def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1
    # Return updated heatmap
    return heatmap

# Function to apply a threshold on a heatmap
def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap

# Function to infer labels from a list of boxes
def bbox_from_labels(labels):
    bbox_list = []
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Append the box to the list
        bbox_list.append(bbox)
    # Return the list
    return bbox_list


## Explore the Data
# Grab all the training images
vehicles = glob.glob('TRAINING_DATA/vehicles/*/*.png')
non_vehicles = glob.glob('TRAINING_DATA/non-vehicles/*/*.png')
# Plot some sample vehicle images
img_cnt = 8
f, ax = plt.subplots(1, img_cnt, figsize = (18, 16))
for i in range(0, img_cnt):
    sample = mpimg.imread(vehicles[i])
    ax[i].imshow(sample)
    ax[i].set_xticks([])
    ax[i].set_yticks([])
plt.show()
print('Total vehicle: %d' %len(vehicles))
img_size = sample.shape
print('Image size: {}'.format(img_size))
# Plot some sample non-vehicle images
f, ax = plt.subplots(1, img_cnt, figsize = (18, 16))
for i in range(0, img_cnt):
    sample = mpimg.imread(non_vehicles[i])
    ax[i].imshow(sample)
    ax[i].set_xticks([])
    ax[i].set_yticks([])
plt.show()
print('Total non-vehicle: %d' %len(non_vehicles))
img_size = sample.shape
print('Image size: {}'.format(img_size))

# Get the X training tdata
X = np.zeros((len(vehicles) + len(non_vehicles), *img_size), dtype=np.float32)
i = 0
for img in vehicles:
    X[i, :, :, :] = mpimg.imread(img)
    i += 1
for img in non_vehicles:
    X[i, :, :, :] = mpimg.imread(img)
    i += 1
print(X.shape)

# Get the y training data
y = np.hstack((np.ones(len(vehicles)), np.zeros(len(non_vehicles)))).astype(np.float32)
print(y.shape)

'''
# Save the raw data as a pickle
import pickle

raw_pickle = {}
raw_pickle['X'] = X
raw_pickle['y'] = y
with open('TRAINING_DATA/raw_pickle.p','wb') as output_file:
    pickle.dump(raw_pickle, output_file)
# Load the raw data pickle
with open('training_data/raw_pickle.p', 'rb') as input_file:
    p = pickle.load(input_file)
X, y = p['X'], p['y']
'''


## Feature Selection
ORIENT = 9
PIX_PER_CELL = 8
CELL_PER_BLOCK = 2

# Function to plot example images on HOG
def plot_hog_examples(img, orient=9, pix_per_cell=8, cell_per_block=2):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    _, ax = plt.subplots(1, 6, figsize=(18, 6))
    ax[0].imshow(img)
    ax[0].set_title('original')
    _, gray_hog = get_hog_features(gray, orient, pix_per_cell, cell_per_block, vis=True)
    ax[1].imshow(gray_hog, cmap='gray')
    ax[1].set_title('gray hog')
    _, h_hog = get_hog_features(hls[:, :, 0], orient, pix_per_cell, cell_per_block, vis=True)
    ax[2].imshow(h_hog, cmap='gray')
    ax[2].set_title('h hog')
    _, l_hog = get_hog_features(hls[:, :, 1], orient, pix_per_cell, cell_per_block, vis=True)
    ax[3].imshow(l_hog, cmap='gray')
    ax[3].set_title('l hog')
    _, s_hog = get_hog_features(hls[:, :, 2], orient, pix_per_cell, cell_per_block, vis=True)
    ax[4].imshow(s_hog, cmap='gray')
    ax[4].set_title('s hog')
    _, v_hog = get_hog_features(hsv[:, :, 2], orient, pix_per_cell, cell_per_block, vis=True)
    ax[5].imshow(v_hog, cmap='gray')
    ax[5].set_title('v hog')
    plt.show()

# Plot different HOGs from random images
for _ in range(0, 3):
    plot_hog_examples(mpimg.imread(choice(vehicles)), ORIENT, PIX_PER_CELL, CELL_PER_BLOCK)
    plot_hog_examples(mpimg.imread(choice(non_vehicles)), ORIENT, PIX_PER_CELL, CELL_PER_BLOCK)

color_space = 'HLS' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
hog_channel = 'ALL' # Can be 0, 1, 2, or "ALL"

t1 = time.time()
features = extract_features(X, color_space = color_space, orient = ORIENT, pix_per_cell = PIX_PER_CELL,
                            cell_per_block = CELL_PER_BLOCK, hog_channel = hog_channel, spatial_feat = True, hist_feat = True)
t2 = time.time()
print(round(t2 - t1, 2), 'Seconds to extract features...' )

X = np.array(features).astype(np.float32)
print('Before scaler, mean:{}, std:{}'.format(np.mean(X),np.std(X)) )
X_scaler = StandardScaler().fit(features)
scaled_X = X_scaler.transform(features)
print('After scaler, mean:{}, std:{}'.format(np.mean(scaled_X),np.std(scaled_X)) )
print(scaled_X.shape, y.shape)


## Train Classifier
X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size=0.2, random_state=12345)
print('Training data size: {}'.format(X_train.shape[0]))
print('Test data size: {}'.format(X_test.shape[0]))
svc = LinearSVC()
clf = CalibratedClassifierCV(svc)
t1 = time.time()
clf.fit(X_train, y_train)
t2 = time.time()
print(round(t2 - t1, 2), 'Seconds to train SVC...' )
print('Train Accuracy of SVC = ', round(clf.score(X_train, y_train), 4))
print('Test Accuracy of SVC = ', round(clf.score(X_test, y_test), 4))


## Sliding Window
y_start_stops = [[400, 500], [400, 600], [400, 600]]
x_start_stops = [[600, 800], [600, 1280], [600, 1280]]
xy_windows = [(64, 64), (96, 96), (128, 128)]
test_img = mpimg.imread('TEST_IMAGES/test1.jpg')
windows = []
for i in range(len(xy_windows)):
    windows+= slide_window(test_img.shape, x_start_stop=x_start_stops[i], y_start_stop=y_start_stops[i],
                           xy_window=xy_windows[i],xy_overlap=(0.5, 0.5))
window_img = draw_boxes(test_img, windows, color=(0,0,255), thick = 6)
plt.figure()
plt.imshow(window_img)
plt.show()

test_imgs = glob.glob('TEST_IMAGES/*.jpg')
for img in test_imgs:
    img = mpimg.imread(img)
    proc_img = img.astype(np.float32)/255.0
    windows = []
    for i in range(len(xy_windows)):
        windows+= slide_window(img.shape, x_start_stop=x_start_stops[i], y_start_stop=y_start_stops[i],
                               xy_window=xy_windows[i],xy_overlap=(0.75, 0.5))
    on_windows = search_windows(proc_img, windows, clf, X_scaler, 'HLS', ORIENT, PIX_PER_CELL, CELL_PER_BLOCK, 'ALL')
    window_img = draw_boxes(img, on_windows, color=(0,0,255), thick = 6)
    plt.figure()
    plt.imshow(window_img)
    plt.show()


## Heat Map
test_imgs = glob.glob('TEST_IMAGES/*.jpg')
for img in test_imgs:
    img = mpimg.imread(img)
    proc_img = img.astype(np.float32)/255.0
    windows = []
    for i in range(len(xy_windows)):
        windows+= slide_window(img.shape, x_start_stop=x_start_stops[i], y_start_stop=y_start_stops[i],
                               xy_window=xy_windows[i],xy_overlap=(0.75, 0.5))
    on_windows = search_windows(proc_img, windows, clf, X_scaler, 'HLS', ORIENT, PIX_PER_CELL, CELL_PER_BLOCK, 'ALL')
    heatmap = np.zeros_like(img[:,:,0]).astype(np.float)
    heatmap = add_heat(heatmap, on_windows)
    heatmap =apply_threshold(heatmap, 0)
    labels = label(heatmap)
    bbox_list = bbox_from_labels(labels)
    window_img = draw_boxes(img, bbox_list, color=(0,0,255), thick = 6)
    _, ax = plt.subplots(1, 3, figsize = (12, 8))
    ax[0].imshow(heatmap, cmap = 'hot')
    ax[1].imshow(labels[0], cmap = 'gray')
    ax[2].imshow(window_img)
    plt.show()


## Video
class vehicle_detector():
    def __init__(self):
        self.x_shape = 1280
        self.y_shape = 720
        self.windows = self.get_windows(self.x_shape, self.y_shape)
        self.last_on_windows = []

    def get_windows(self, x_shape, y_shape):
        windows = []
        for i in range(len(xy_windows)):
            windows+= slide_window([y_shape, x_shape], x_start_stop=x_start_stops[i], y_start_stop=y_start_stops[i],
                                   xy_window=xy_windows[i], xy_overlap=(0.75, 0.5))
        return windows

    def pipeline(self, img):
        proc_img = img.astype(np.float32) / 255.0
        on_windows = []
        for window in self.windows:
            window_img = proc_img[window[0][1]:window[1][1], window[0][0]:window[1][0], :]
            test_img = cv2.resize(window_img, (64, 64))
            features = single_img_features(test_img, color_space=color_space, orient=ORIENT, pix_per_cell=PIX_PER_CELL,
                                           cell_per_block=CELL_PER_BLOCK, hog_channel=hog_channel, spatial_feat=True,
                                           hist_feat=True, hog_feat=True)
            test_features = X_scaler.transform(np.array(features).reshape(1, -1))
            prediction = clf.predict_proba(test_features)[0][1]
            if prediction > 0.75:
                on_windows.append(window)
        if len(on_windows) > 0:
            heatmap = np.zeros_like(img[:, :, 0]).astype(np.float)
            heatmap = add_heat(heatmap, on_windows)
            for i in range(len(self.last_on_windows)):
                heatmap = add_heat(heatmap, self.last_on_windows[i])
            self.last_on_windows.append(on_windows)
            if (len(self.last_on_windows) > 5):
                self.last_on_windows.pop(0)
            heatmap = apply_threshold(heatmap, 2)
            labels = label(heatmap)
            bbox_list = bbox_from_labels(labels)
            window_img = draw_boxes(img, bbox_list, color=(0, 0, 255), thick=6)
            return window_img
        else:
            return img

VehicleDetector = vehicle_detector()
clip = VideoFileClip('project_video.mp4')
project_clip = clip.fl_image(VehicleDetector.pipeline)
project_clip.write_videofile('proc_project_video.mp4', audio = False)