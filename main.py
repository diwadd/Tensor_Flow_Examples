import csv

import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split

RANDOM_STATE = 1
IMAGE_SIZE = 28
TEST_SIZE = 0.1

tf.set_random_seed(RANDOM_STATE)

def read_data(file_name):

    f = open(file_name, "r")
    file_contents = csv.reader(f, delimiter=",")
    file_contents = list(file_contents)

    file_contents = np.asarray(file_contents[1:], dtype=np.int32)
    return file_contents


def show_image(row_image):

    label = row_image[0]
    row_image = row_image[1:]

    print("Printing: " + str(label))

    square_image = np.resize(row_image, (IMAGE_SIZE, IMAGE_SIZE))
    for i in range(IMAGE_SIZE):
        for j in range(IMAGE_SIZE):
            if square_image[i][j] == 0:
                print("0", end="")
            else:
                print("1", end="")
        print()


def labels_to_vectors(row_of_labels):

    N = len(row_of_labels)
    vector_labels = np.zeros((N, 10))
    for i in range(N):
        vector_labels[i][row_of_labels[i]] = 1

    return vector_labels


class Network:

    def weight_variable(self, shape):
        initial = tf.truncated_normal(shape, stddev=1.0, seed=RANDOM_STATE)
        return tf.Variable(initial)

    def bias_variable(self, shape):
        initial = tf.truncated_normal(shape, stddev=1.0, seed=RANDOM_STATE)
        return tf.Variable(initial)

    def __init__(self, n_layers = 30):
        self.input = tf.placeholder(tf.float32, shape=[None, IMAGE_SIZE*IMAGE_SIZE], name="input_image")

        self.w_one = self.weight_variable([IMAGE_SIZE*IMAGE_SIZE, n_layers])
        self.b_one = self.bias_variable([n_layers])
        self.layer_one = tf.sigmoid(tf.matmul(self.input, self.w_one) + self.b_one)

        self.w_two = self.weight_variable([n_layers, 10])
        self.b_two = self.bias_variable([10])
        self.output = tf.sigmoid(tf.matmul(self.layer_one, self.w_two) + self.b_two)

        print("input: " + str((self.input).get_shape()))
        print("w_one: " + str(self.w_one.get_shape()))
        print("b_one: " + str(self.b_one.get_shape()))
        print("layer_one: " + str((self.layer_one).get_shape()))
        print("w_two: " + str(self.w_two.get_shape()))
        print("b_two: " + str(self.b_two.get_shape()))
        print("output: " + str((self.output).get_shape()))

        #print("C: " + str(C.get_shape()))

    def setup_loss(self, mini_batch_size):

        self.expected_output = tf.placeholder(tf.float32, shape=[None, 10], name="expected_output")

        s = tf.subtract(self.output, self.expected_output)
        self.C = tf.reduce_sum(tf.multiply(s, s))
        m = tf.constant( 2.0*mini_batch_size, dtype=tf.float32 )
        self.C = tf.divide(self.C, m)


    def setup_minimize(self, learning_rate):
        self.train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(self.C)

    def evaluate_performance(self,sess, x_test, y_test):

        n_train_samples = len(x_test)
        n_pos = 0
        for i in range(n_train_samples):
            parameter_dict = {self.input: np.resize(x_test[i], (1, IMAGE_SIZE*IMAGE_SIZE) )}

            output = (self.output).eval(session=sess, feed_dict=parameter_dict)
            output = np.argmax(output)
            expected_output = np.argmax(y_test[i])

            if output == expected_output:
                n_pos = n_pos + 1

        return n_pos / n_train_samples

    def train(self, x_train, x_test, y_train, y_test, n_epochs, mini_batch_size, learning_rate):

        sess = tf.InteractiveSession()
        init = tf.initialize_all_variables()
        sess = tf.Session()
        sess.run(init)

        self.setup_loss(mini_batch_size)
        self.setup_minimize(learning_rate)

        n_batches_per_epoch = int(len(x_train) / mini_batch_size)

        for epoch in range(n_epochs):
            ptr = 0
            print("Epoch number: %10s" % (str(epoch)))
            for batch in range(n_batches_per_epoch):
                network_input, expected_output = x_train[ptr:ptr + mini_batch_size], y_train[ptr:ptr + mini_batch_size]

                ptr = ptr + mini_batch_size
                parameter_dict = {self.input: network_input, self.expected_output: expected_output}

                (self.train_step).run(session=sess, feed_dict=parameter_dict)

                c_val = (self.C).eval(session=sess, feed_dict=parameter_dict)
            print("chi^2 value: " + str(c_val))
            acc = self.evaluate_performance(sess, x_test, y_test)
            print("Accuracy: " + str(acc))
            #input("Press Enter to continue...")



print("Recognize Digits")

file_name = "/home/tadek/Coding_Competitions/Kaggle/DigitRecognizer/train.csv"

data = read_data(file_name)

row_labels = data[:, 0]
images = data[:, 1:]/255.0
vector_labels = labels_to_vectors(row_labels)

x_train, x_test, y_train, y_test = train_test_split(images, vector_labels, test_size=TEST_SIZE, random_state=RANDOM_STATE)

print("=== Data shape ===")
print(data.shape)
print(row_labels.shape)
print(images.shape)
print("==================")



for i in range(30):
    print(str(row_labels[i]) + " ", end="")
    for j in range(len(vector_labels[i])):
        print(str(vector_labels[i][j]) + " ", end="")
    print()

#show_image(data[3336])

nn = Network(100)
nn.train(x_train, x_test, y_train, y_test, 30, 10, 0.001)
