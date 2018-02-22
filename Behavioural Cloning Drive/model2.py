import cv2
import json
import scipy.misc
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from scipy.stats import bernoulli
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Lambda
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

DRIVING_LOG_CSV = "./DATA/driving_log.csv"
IMAGE_PATH = "./DATA/"
STEERING = 0.22
RESIZED_IMAGE_SIZE = (128, 128)
RESIZED_IMAGE_DIM = (128, 128, 3)
NUMBER_SAMPLES_PER_EPOCH = 20032
NUMBER_VALIDATION_SAMPLES = 6400
NUMBER_EPOCH = 3

def get_all_image_files_and_angle():
    data = pd.read_csv(DRIVING_LOG_CSV)
    n = len(data)
    image_files_and_angle = []
    for i in range(n):
        image_files_and_angle.append((data.left[i].strip(), data.steering[i] + STEERING))
        image_files_and_angle.append((data.right[i].strip(), data.steering[i] - STEERING))
        image_files_and_angle.append((data.center[i].strip(), data.steering[i]))
    return image_files_and_angle

def crop(img, top_crop=0.35, bottom_crop=0.1):
    top = int(np.ceil(img.shape[0] * top_crop))
    bottom = img.shape[0] - int(np.ceil(img.shape[0] * bottom_crop))
    return img[top:bottom, :]

def random_flip(image, angle, prob=0.5):
    if bernoulli.rvs(prob):
        return np.fliplr(image), -1 * angle
    else:
        return image, angle

def random_shear(image, steering_angle, shear_range=200):
    rows, cols, ch = image.shape
    dx = np.random.randint(-shear_range, shear_range + 1)
    random_point = [cols / 2 + dx, rows / 2]
    pts1 = np.float32([[0, rows], [cols, rows], [cols / 2, rows / 2]])
    pts2 = np.float32([[0, rows], [cols, rows], random_point])
    dsteering = dx / (rows / 2) * 360 / (2 * np.pi * 25.0) / 6.0
    M = cv2.getAffineTransform(pts1, pts2)
    image = cv2.warpAffine(image, M, (cols, rows), borderMode=1)
    steering_angle += dsteering
    return image, steering_angle

def pre_process(image, angle):
    if bernoulli.rvs(0.9):
        image, angle = random_shear(image, angle)
    image = crop(image)
    image, angle = random_flip(image, angle)
    image = scipy.misc.imresize(image, RESIZED_IMAGE_SIZE)
    return image, angle

def read_images(image_files_and_angle):
    X = []
    y = []
    for image_file, angle in image_files_and_angle:
        raw_image = plt.imread(IMAGE_PATH + image_file)
        new_image, new_angle = pre_process(raw_image, angle)
        X.append(new_image)
        y.append(new_angle)
    return X, y

def generate_next_batch(batch_size=64):
    data = get_all_image_files_and_angle()
    while True:
        data = shuffle(data)
        X_batch, y_batch = read_images(data[:batch_size])
        yield np.array(X_batch), np.array(y_batch)

# The model is based on:
# https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf
model = Sequential()
model.add(Lambda(lambda x: x / 127.5 - 1.0, input_shape=RESIZED_IMAGE_DIM))
# Five Convolutional Layer
model.add(Convolution2D(24, 5, 5, border_mode='same'))
model.add(MaxPooling2D())
model.add(Activation('relu'))
model.add(Convolution2D(36, 5, 5, border_mode='same'))
model.add(MaxPooling2D())
model.add(Activation('relu'))
model.add(Convolution2D(48, 5, 5, border_mode='same'))
model.add(MaxPooling2D())
model.add(Activation('relu'))
model.add(Convolution2D(64, 3, 3, border_mode='same'))
model.add(MaxPooling2D())
model.add(Activation('relu'))
model.add(Convolution2D(64, 3, 3, border_mode='same'))
model.add(MaxPooling2D())
model.add(Activation('relu'))
# Five Fully-Connected Layer
model.add(Flatten())
model.add(Dense(1164))
model.add(Activation('relu'))
model.add(Dense(100))
model.add(Activation('relu'))
model.add(Dense(50))
model.add(Activation('relu'))
model.add(Dense(10))
model.add(Activation('relu'))
model.add(Dense(1))
model.compile('adam', loss='mse')
model.fit_generator(generator=generate_next_batch(), samples_per_epoch=NUMBER_SAMPLES_PER_EPOCH, nb_epoch=NUMBER_EPOCH,
                    validation_data=generate_next_batch(), nb_val_samples=NUMBER_VALIDATION_SAMPLES)

json_string = model.to_json()
with open("model.json", 'w') as outfile:
    json.dump(json_string, outfile)
model.save_weights('model.h5')