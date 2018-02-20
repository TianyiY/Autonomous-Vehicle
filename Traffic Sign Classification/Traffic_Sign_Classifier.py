import pickle
import cv2
import matplotlib.pyplot as plt
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from tensorflow.contrib.layers import flatten
from sklearn.model_selection import train_test_split


training_file_path = 'TRAFFIC_SIGN_IMAGES/TRAINING_TEST_DATASET/train.p'
testing_file_path = 'TRAFFIC_SIGN_IMAGES/TRAINING_TEST_DATASET/test.p'
with open(training_file_path, mode='rb') as f:
    train_data = pickle.load(f)
with open(testing_file_path, mode='rb') as f:
    test_data = pickle.load(f)
X_train, y_train = train_data['features'], train_data['labels']
X_test, y_test = test_data['features'], test_data['labels']
X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=0.2, random_state=0)

# Number of training examples
n_train = X_train.shape[0]
# Number of testing examples.
n_test = X_test.shape[0]
# Shape of an traffic sign image?
image_shape = X_train[0].shape
# Number of unique classes/labels in the dataset
n_classes = len(set(y_train))
print("Number of training examples: ", n_train)
print("Number of testing examples: ", n_test)
print("Image data shape: ", image_shape)
print("Number of classes: ", n_classes)

# Show a random sample from each class of the traffic sign dataset
rows, cols = 4, 12
fig, ax_array = plt.subplots(rows, cols) # ax_array is a array object consistint of plt object
plt.suptitle('Random Sample from Training Sst (One for Each Class)')
for class_idx, ax in enumerate(ax_array.ravel()):
    if class_idx < n_classes:
        cur_X = X_train[y_train == class_idx]
        cur_img = cur_X[np.random.randint(len(cur_X))]
        ax.imshow(cur_img)
        ax.set_title('{:02d}'.format(class_idx))
    else:
        ax.axis('off')
# Hide both x and y ticks
plt.setp([a.get_xticklabels() for a in ax_array.ravel()], visible=False)
plt.setp([a.get_yticklabels() for a in ax_array.ravel()], visible=False)
plt.draw()

# Bar chart of classes distribution (Each class distributes unevenly, which brings bias to CNN)
train_distribution, test_distribution = np.zeros(n_classes), np.zeros(n_classes)
for c in range(n_classes):
    train_distribution[c] = np.sum(y_train == c) / n_train
    test_distribution[c] = np.sum(y_test == c) / n_test
fig, ax = plt.subplots()
col_width = 0.5
bar_train = ax.bar(np.arange(n_classes), train_distribution, width=col_width, color='r')
bar_test = ax.bar(np.arange(n_classes)+col_width, test_distribution, width=col_width, color='b')
ax.set_ylabel('Percentage of Presence')
ax.set_xlabel('Class Label')
ax.set_title('Classes Distribution in Traffic-sign Dataset')
ax.set_xticks(np.arange(0, n_classes, 5) )
ax.set_xticklabels(['{:02d}'.format(c) for c in range(0, n_classes, 5)])
ax.legend((bar_train[0], bar_test[0]), ('train set', 'test set'))
plt.show()


# Convert RGB to YUV, and choose only Y channel (reduce computing workload at the minimum presion loss)
# Then convert the images to the format that mean value is 0 and std is 1
def preprocess_features(X, equalize_hist=True):
    # Convert from RGB to YUV
    X = np.array([np.expand_dims(cv2.cvtColor(rgb_img, cv2.COLOR_RGB2YUV)[:, :, 0], 2) for rgb_img in X])
    # Adjust image contrast
    if equalize_hist:
        X = np.array([np.expand_dims(cv2.equalizeHist(img), 2) for img in X])
    X = np.float32(X)
    # Standardize features
    X -= np.mean(X, axis=0)
    X /= (np.std(X, axis=0) + np.finfo('float32').eps)
    return X

X_train_norm = preprocess_features(X_train)
X_test_norm = preprocess_features(X_test)

# Creat the generator to perform online data augmentation
# rotation_range: rotation angles
# zoom_range: zoom in and zoom out multiples
# width_shift_range: shift left and right scales
# height_shift_range: shift up and down scales
image_datagen = ImageDataGenerator(rotation_range=15., zoom_range=0.2, width_shift_range=0.1, height_shift_range=0.1)

# Take a random image from the training set
img_rgb = X_train[10]
# Plot the original image
plt.figure(figsize=(1,1))
plt.imshow(img_rgb)
plt.title('Example of RGB image (class = {})'.format(y_train[10]))
plt.axis('off')
plt.show()
# Plot some randomly augmented images
rows, cols = 4, 10
fig, ax_array = plt.subplots(rows, cols)
for ax in ax_array.ravel():
    augmented_img, _ = image_datagen.flow(np.expand_dims(img_rgb, 0), y_train[10:11]).next()
    ax.imshow(np.uint8(np.squeeze(augmented_img)))
plt.setp([a.get_xticklabels() for a in ax_array.ravel()], visible=False)
plt.setp([a.get_yticklabels() for a in ax_array.ravel()], visible=False)
plt.suptitle('Random Examples of Data Augment (Starting from the Previous Image)')
plt.show()


def weight_variable(shape, mu=0, sigma=0.1):
    initialization = tf.truncated_normal(shape=shape, mean=mu, stddev=sigma)
    return tf.Variable(initialization)

def bias_variable(shape, start_val=0.1):
    initialization = tf.constant(start_val, shape=shape)
    return tf.Variable(initialization)

def conv2d(x, W, strides=[1,1,1,1], padding='SAME'):
    return tf.nn.conv2d(input=x, filter=W, strides=strides, padding=padding)

def max_pool2x2(x):
    return tf.nn.max_pool(value=x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

n_classes = 43
tf.reset_default_graph()
# Placeholders
x = tf.placeholder(dtype=tf.float32, shape=(None, 32, 32, 1))
y = tf.placeholder(dtype=tf.int64, shape=None)
keep_prob = tf.placeholder(tf.float32)


# Network architecture
def my_net(x, n_classes):
    c1_out = 64
    conv1_W = weight_variable(shape=(3, 3, 1, c1_out))
    conv1_b = bias_variable(shape=(c1_out,))
    conv1 = tf.nn.relu(conv2d(x, conv1_W) + conv1_b)
    pool1 = max_pool2x2(conv1)
    drop1 = tf.nn.dropout(pool1, keep_prob=keep_prob)
    c2_out = 128
    conv2_W = weight_variable(shape=(3, 3, c1_out, c2_out))
    conv2_b = bias_variable(shape=(c2_out,))
    conv2 = tf.nn.relu(conv2d(drop1, conv2_W) + conv2_b)
    pool2 = max_pool2x2(conv2)
    drop2 = tf.nn.dropout(pool2, keep_prob=keep_prob)
    fc0 = tf.concat([flatten(drop1), flatten(drop2)], 1)
    fc1_out = 64
    fc1_W = weight_variable(shape=(fc0._shape[1].value, fc1_out))
    fc1_b = bias_variable(shape=(fc1_out,))
    fc1 = tf.matmul(fc0, fc1_W) + fc1_b
    drop_fc1 = tf.nn.dropout(fc1, keep_prob=keep_prob)
    fc2_out = n_classes
    fc2_W = weight_variable(shape=(drop_fc1._shape[1].value, fc2_out))
    fc2_b = bias_variable(shape=(fc2_out,))
    logits = tf.matmul(drop_fc1, fc2_W) + fc2_b
    return logits


# Training pipeline
learning_rate = 0.001
logits = my_net(x, n_classes=n_classes)
cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y)
loss_function = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
training_operation = optimizer.minimize(loss=loss_function)
correct_prediction = tf.equal(tf.argmax(logits, 1), y)
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


Epoch = 2
Batch_per_Epoch = 3
Batch_size = 128

def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, Batch_size):
        batch_x, batch_y = X_data[offset:offset+Batch_size], y_data[offset:offset+Batch_size]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y, keep_prob:1.0})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples


saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print("Training Start")
    print()
    for epoch in range(Epoch):
        batch_counter = 0
        for batch_x, batch_y in image_datagen.flow(X_train, y_train, batch_size=Batch_size):
            batch_counter += 1
            sess.run(training_operation, feed_dict={x: batch_x, y: batch_y, keep_prob:0.5})
            if batch_counter == Batch_per_Epoch:
                break
        # at the end of each epoch, evaluate accuracy on both training and validation set
        train_accuracy = evaluate(X_train, y_train)
        validation_accuracy = evaluate(X_validation, y_validation)
        print("Epoch {} ...".format(epoch+1))
        print("Training Accuracy = {:.3f}, Validation Accuracy = {:.3f}".format(train_accuracy, validation_accuracy))
        print()
        saver.save(sess, save_path='model.ckpt', global_step=epoch)

with tf.Session() as sess:
    # restore saved session with highest validation accuracy
    saver.restore(sess, 'model.ckpt')
    test_accuracy = evaluate(X_test_norm, y_test)
    print('Performance on test set: {:.3f}'.format(test_accuracy))