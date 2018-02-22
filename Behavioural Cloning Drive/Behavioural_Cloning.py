import numpy as np
import matplotlib.image as mpimg
from keras.models import Sequential
from keras.layers import Dense, Flatten, ELU
from keras.layers import Convolution2D, Cropping2D
from keras.layers.core import Lambda
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import csv


REDUCED = False
if REDUCED == False:
    csv_filepath = 'DATA/driving_log.csv'
else:
    csv_filepath = 'DATA/driving_log_reduced.csv'
samples = []


def add_to_samples(csv_filepath, samples):
    with open(csv_filepath) as f:
        reader = csv.reader(f)
        for line in reader:
            samples.append(line)
    return samples


samples = add_to_samples(csv_filepath, samples)
samples = samples[1:]
print("Samples number: ", len(samples))

train_samples, validation_samples = train_test_split(samples, test_size=0.05)


def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1:  # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset : offset + batch_size]
            images = []
            angles = []
            for batch_sample in batch_samples:
                name = './DATA/' + batch_sample[0]
                center_image = mpimg.imread(name)
                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)
            X_train = np.array(images)
            y_train = np.array(angles)
            yield shuffle(X_train, y_train)


# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)


def resize_comma(image):
    import tensorflow as tf  # This import is required here otherwise the model cannot be loaded in drive1.py
    return tf.image.resize_images(image, 40, 160)


# build model
model = Sequential()
# Crop 70 pixels from the top of the image and 25 from the bottom
model.add(Cropping2D(cropping=((70, 25), (0, 0)), dim_ordering='tf', input_shape=(160, 320, 3)))
# Resize the data
model.add(Lambda(resize_comma))
model.add(Lambda(lambda x: (x/255.0) - 0.5))
model.add(Convolution2D(16, 8, 8, subsample=(4, 4), border_mode="same"))
model.add(ELU())
model.add(Convolution2D(32, 5, 5, subsample=(2, 2), border_mode="same"))
model.add(ELU())
model.add(Convolution2D(64, 5, 5, subsample=(2, 2), border_mode="same"))
model.add(Flatten())
# model.add(Dropout(.2))
model.add(ELU())
model.add(Dense(512))
# model.add(Dropout(.5))
model.add(ELU())
model.add(Dense(50))
model.add(ELU())
model.add(Dense(1))
adam = Adam(lr=0.0001)
model.compile(optimizer=adam, loss="mse", metrics=['accuracy'])
model.summary()


# Train model
batch_size = 32
nb_epoch = 20
checkpointer = ModelCheckpoint(filepath="./tmp/comma-4c.{epoch:02d}-{val_loss:.2f}.hdf5", verbose=1, save_best_only=False)
model.fit_generator(train_generator, samples_per_epoch=len(train_samples), validation_data=validation_generator,
                    nb_val_samples=len(validation_samples), nb_epoch=nb_epoch, callbacks=[checkpointer])


# save model
model_json = model.to_json()
with open("model1.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights("model1.h5")
