import numpy as np
import tensorflow as tf
import time
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle
from sklearn.utils import shuffle
import pandas as pd
from sklearn.model_selection import train_test_split


# Load data
training_file_path = 'TRAFFIC_SIGN_IMAGES/TRAINING_TEST_DATASET/train.p'
testing_file_path = 'TRAFFIC_SIGN_IMAGES/TRAINING_TEST_DATASET/test.p'
with open(training_file_path, mode='rb') as f:
    train = pickle.load(f)
with open(testing_file_path, mode='rb') as f:
    test = pickle.load(f)
X_train, y_train = train['features'], train['labels']
X_test, y_test = test['features'], test['labels']
# Basic data summary.
n_train = len(X_train)
n_test = len(X_test)
image_shape = X_train[0].shape
n_classes = len(set(y_train))
print("Number of training examples =", n_train)
print("Number of testing examples =", n_test)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)
# Plot sample images
print('Sample images')
for i in range(12):
    plt.subplot(3,4,i+1)
    plt.imshow(X_train[i*1500+1])

# Plot a histogram of the count of the number of examples of each sign in the test set
plt.hist(y_train, bins=n_classes)
plt.title('Number of examples of each sign in the training set')
plt.xlabel('Sign')
plt.ylabel('Count')
plt.plot()

# Shuffle training examples
X_train, y_train = shuffle(X_train, y_train, random_state=0)
# Normalization
X_train_orig = X_train
X_test_orig = X_test
# Normalise input (images still in colour)
X_train = (X_train - X_train.mean()) / (np.max(X_train) - np.min(X_train))
X_test = (X_test - X_test.mean()) / (np.max(X_test) - np.min(X_test))

def plot_norm_image(image_index):
    # Plots original image on the left and normalised image on the right
    plt.subplot(2,2,1)
    plt.imshow(X_train_orig[image_index])
    plt.subplot(2,2,2)
    plt.imshow(X_train[image_index])

plot_norm_image(0)

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=0)

# Network parameters
n_input = 32 * 32 * 3
nb_filters = 32
kernel_size = (3, 3)
input_shape = (32, 32, 3)
n_fc1 = 512
n_fc2 = 128
in_channels = 3
pool_size = 2 # (2,2)
dropout_conv = 0.9
dropout_fc = 0.9
weights_stddev = 0.1
weights_mean = 0.0
biases_mean = 0.0

padding = 'VALID'
if padding == 'SAME':
    conv_output_length = 6
elif padding == 'VALID':
    conv_output_length = 5
else:
    raise Exception("Unknown padding.")

# tf Graph input
x_unflattened = tf.placeholder("float", [None, 32, 32, 3])
x = x_unflattened
y_rawlabels = tf.placeholder("int32", [None])
y = tf.one_hot(y_rawlabels, depth=43, on_value=1., off_value=0., axis=-1)

## Create model
def conv2d(x, W, b, strides=3):
    # strides = [batch, in_height, in_width, channels]
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)

def maxpool2d(x, k=2, padding_setting='SAME'):
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding=padding_setting)

def conv_net(model_x, model_weights, model_biases, model_pool_size, model_dropout_conv, model_dropout_fc, padding='SAME'):
    # Convolution Layer 1
    conv1 = conv2d(model_x, model_weights['conv1'], model_biases['conv1'])
    # Max Pooling (down-sampling)
    conv1 = maxpool2d(conv1, k=model_pool_size, padding_setting=padding)
    conv1 = tf.nn.dropout(conv1, model_dropout_conv)
    # Fully connected layer 1
    # Reshape conv1 output to fit fully connected layer input
    conv1_shape = conv1.get_shape().as_list()
    fc1 = tf.reshape(conv1, [-1, conv1_shape[1]*conv1_shape[2]*conv1_shape[3]])
    fc1 = tf.add(tf.matmul(fc1, model_weights['fc1']), model_biases['fc1'])
    fc1 = tf.nn.relu(fc1)
    fc1 = tf.nn.dropout(fc1, model_dropout_fc)
    # Fully connected layer 2
    fc2 = tf.add(tf.matmul(fc1, model_weights['fc2']), model_biases['fc2'])
    fc2 = tf.nn.relu(fc2)
    fc2 = tf.nn.dropout(fc2, model_dropout_fc)
    # Output layer
    output = tf.add(tf.matmul(fc2, model_weights['out']), model_biases['out'])
    # Softmax is outside the model
    return output

# Initialise neurons with slightly positive initial bias to avoid dead neurons.
def weight_variable(shape, weight_mean, weight_stddev):
    initial = tf.truncated_normal(shape, stddev=weight_stddev, mean=weight_mean)
    return tf.Variable(initial)

def bias_variable(shape, bias_mean):
    initial = tf.constant(bias_mean, shape=shape)
    return tf.Variable(initial)

weights = {'conv1': weight_variable([kernel_size[0], kernel_size[1], in_channels, nb_filters], weights_mean, weights_stddev),
           'fc1': weight_variable([nb_filters * conv_output_length**2, n_fc1], weights_mean, weights_stddev),
           'fc2': weight_variable([n_fc1, n_fc2], weights_mean, weights_stddev),
           'out': weight_variable([n_fc2, n_classes], weights_mean, weights_stddev)}

biases = {'conv1': bias_variable([nb_filters], biases_mean),
          'fc1': bias_variable([n_fc1], biases_mean),
          'fc2': bias_variable([n_fc2], biases_mean),
          'out': bias_variable([n_classes], biases_mean)}

# Training parameters
learning_rate = 0.001
initial_learning_rate = learning_rate
training_epochs = 150
batch_size = 100
display_step = 1
n_train = len(X_train)
anneal_mod_frequency = 15
annealing_rate = 1 # Annealing rate = 1: learning rate remains constant.
print_accuracy_mod_frequency = 1

# Construct model
pred = conv_net(x, weights, biases, pool_size, dropout_conv, dropout_fc, padding=padding)
pred_probs = tf.nn.softmax(pred)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Function to initialize the variables
init = tf.initialize_all_variables()

# Launch the graph
sess = tf.Session()

# Initialise variables
sess.run(init)

# Initialise time logs
init_time = time.time()
epoch_time = init_time

five_epoch_moving_average = 0.
epoch_accuracies = []

# Training cycle
for epoch in range(training_epochs):
    if five_epoch_moving_average > 0.96:
        break

    avg_cost = 0.
    total_batch = int(n_train / batch_size)

    # Loop over all batches
    for i in range(total_batch):
        batch_x, batch_y = np.array(X_train[i * batch_size : (i + 1) * batch_size]), np.array(y_train[i * batch_size : (i + 1) * batch_size])
        # tf.train.batch([X_train, y_train], batch_size=100, enqueue_many=True)
        # Run optimization op (backprop) and cost op (to get loss value)
        _, c = sess.run([optimizer, cost], feed_dict={x_unflattened: batch_x, y_rawlabels: batch_y})
        # Compute average loss
        avg_cost += c / total_batch
        # print(avg_cost)
    # Display logs per epoch step
    if epoch % display_step == 0:
        print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(avg_cost))
        last_epoch_time = epoch_time
        epoch_time = time.time()
        # print("Time since last epoch: ", epoch_time - last_epoch_time)
    # Anneal learning rate
    if (epoch + 1) % anneal_mod_frequency == 0:
        learning_rate *= annealing_rate
        print("New learning rate: ", learning_rate)

    if (epoch + 1) % print_accuracy_mod_frequency == 0:
        correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        # Line below needed only when not using `with tf.Session() as sess`
        with sess.as_default():
            epoch_accuracy = accuracy.eval({x_unflattened: X_val, y_rawlabels: y_val})
            epoch_accuracies.append(epoch_accuracy)
            if epoch >= 4:
                five_epoch_moving_average = np.sum(epoch_accuracies[epoch - 5:epoch]) / 5
                print("Five epoch moving average: ", five_epoch_moving_average)
            print("Accuracy (validation):", epoch_accuracy)
print("Optimization Finished!")

# Test model
correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
# Calculate accuracy
accuracy_train = tf.reduce_mean(tf.cast(correct_prediction, "float"))
print("Accuracy (train):", accuracy_train.eval({x_unflattened: X_train, y_rawlabels: y_train}))
train_predict_time = time.time()
print("Time to calculate accuracy on training set: ", train_predict_time - epoch_time)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
# Line below needed only when not using `with tf.Session() as sess`
with sess.as_default():
    print("Accuracy (test):", accuracy.eval({x_unflattened: X_test, y_rawlabels: y_test}))
test_predict_time = time.time()
print("Time to calculate accuracy on test set: ", test_predict_time - train_predict_time)

# Print parameters for reference
print("\nParameters:")
print("Learning rate (initial): ", initial_learning_rate)
print("Anneal learning rate every ", anneal_mod_frequency, " epochs by ", 1 - annealing_rate)
print("Learning rate (final): ", learning_rate)
print("Training epochs: ", training_epochs)
print("Batch size: ", batch_size)
print("Dropout (conv): ", dropout_conv)
print("Dropout (fc): ", dropout_fc)
print("Padding: ", padding)
print("weights_mean: ", weights_mean)
print("weights_stddev: ", weights_stddev)
print("biases_mean: ", biases_mean)

# Helper function to read image copied from lane lines project
def read_image_and_print_dims(image_path):
    # reading in an image
    image = mpimg.imread(image_path)
    # printing out some stats and plotting
    print('This image is:', type(image), 'with dimensions:', image.shape)
    plt.imshow(image)  #call as plt.imshow(gray, cmap='gray') to show a grayscaled image
    return image

japanese_sign = read_image_and_print_dims('TRAFFIC_SIGN_IMAGES/DEMO_IMAGES/japanese-sign.jpg')
german_sign = read_image_and_print_dims('TRAFFIC_SIGN_IMAGES/DEMO_IMAGES/german-sign.jpg')
two_way_sign = read_image_and_print_dims('TRAFFIC_SIGN_IMAGES/DEMO_IMAGES/two-way-sign.jpg')
speed_limit_stop = read_image_and_print_dims('TRAFFIC_SIGN_IMAGES/DEMO_IMAGES/speed-limit-stop.jpg')
shark_sign = read_image_and_print_dims('TRAFFIC_SIGN_IMAGES/DEMO_IMAGES/shark-sign.jpg')

# The Model's Predictions on the New Images
def predict(img):
    classification = sess.run(tf.argmax(pred, 1), feed_dict={x_unflattened: [img]})
    print(classification)
    print('NN predicted', classification[0])

def show_and_pred_X_train(index):
    plt.imshow(X_train[index])
    predict(X_train[index])

def show_and_pred_image(image):
    plt.imshow(image)
    predict(image)

def read_show_and_pred_image(image_path):
    image = mpimg.imread(image_path)
    plt.imshow(image)
    predict(image)
    return image

show_and_pred_X_train(40)

def read_show_and_pred_image_tsdata(image_name):
    # Read image from dir TRAFFIC_SIGN_IMAGES/DEMO_IMAGES/, show image and print model's prediction
    return read_show_and_pred_image('TRAFFIC_SIGN_IMAGES/DEMO_IMAGES/' + image_name)

japanese_sign = read_show_and_pred_image_tsdata("japanese_sign_resized.png")
german_sign = read_show_and_pred_image_tsdata("german_sign_resized.png")
two_way_sign = read_show_and_pred_image_tsdata("two_way_sign_resized.png")
speed_limit_stop = read_show_and_pred_image_tsdata("speed_limit_stop_resized.png")
shark_sign = read_show_and_pred_image_tsdata("shark_sign_resized.png")

# Visualize the softmax probabilities
def certainty_of_predictions(img):
    top_five = sess.run(tf.nn.top_k(tf.nn.softmax(pred), k=5), feed_dict={x_unflattened: [img]})
    print("Top five: ", top_five)
    return top_five

def show_and_pred_certainty_image(image):
    plt.imshow(image)
    return certainty_of_predictions(image)

def show_and_pred_certainty_X_train(index):
    plt.imshow(X_train[index])
    return certainty_of_predictions(X_train[index])

sign_names = pd.read_csv("signnames.csv")
sign_names.head()

# Plot model's probabilities (y) and traffic sign labels (x)
def plot_certainty_arrays(probabilities, labels):
    y_pos = np.arange(len(labels))
    plt.bar(y_pos, probabilities, align='center', alpha=0.5)
    plt.xticks(y_pos, labels)
    plt.ylabel('Probability')
    plt.xlabel('Traffic sign')
    plt.title('Model\'s certainty of its predictions')
    plt.show()
    print("Traffic Sign Key")
    for label in labels:
        print(label, ": ", sign_names.loc[label]['SignName'])

show_and_pred_certainty_X_train(40)
plot_certainty_arrays([ 1.,  0.,  0.,  0.,  0.], [0, 1, 2, 3, 4])
japanese_sign_certainties = show_and_pred_certainty_image(japanese_sign)
japanese_sign_certainties[1][0]
plot_certainty_arrays(japanese_sign_certainties[0][0], japanese_sign_certainties[1][0])
german_sign_certainties = show_and_pred_certainty_image(german_sign)
plot_certainty_arrays(german_sign_certainties[0][0], german_sign_certainties[1][0])
two_way_sign_certainties = show_and_pred_certainty_image(two_way_sign)
plot_certainty_arrays(two_way_sign_certainties[0][0], two_way_sign_certainties[1][0])
speed_limit_stop_certainties = show_and_pred_certainty_image(speed_limit_stop)
plot_certainty_arrays(speed_limit_stop_certainties[0][0], speed_limit_stop_certainties[1][0])
shark_sign_certainties = show_and_pred_certainty_image(shark_sign)
plot_certainty_arrays(shark_sign_certainties[0][0], shark_sign_certainties[1][0])

# Close the current session.
sess.close()

