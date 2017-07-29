from keras.datasets.mnist import load_data
import tensorflow as tf
import numpy as np
import keras
import pickle

def create_model(_W=None, _b=None):
    num_classes = 11
    img_rows = 28
    img_cols = 28
    img_channel = 1
    depth1 = 16
    depth2 = 32

    x = tf.placeholder(tf.float32, [None, img_rows, img_cols, img_channel])
    y = tf.placeholder(tf.int64, [None, img_rows/4, img_cols/4])

    W = []
    b = []
    if not _W:
        W.append(tf.Variable(tf.random_normal([3, 3, img_channel, depth1])))
        W.append(tf.Variable(tf.random_normal([3, 3, depth1, depth2])))
        W.append(tf.Variable(tf.random_normal([img_rows / 4, img_cols / 4, depth2, num_classes])))
    else:
        W.append(tf.Variable(_W[0]))
        W.append(tf.Variable(_W[1]))
        W.append(tf.Variable(_W[2]))

    if not _b:
        b.append(tf.Variable(tf.random_normal([depth1])))
        b.append(tf.Variable(tf.random_normal([depth2])))
        b.append(tf.Variable(tf.random_normal([num_classes])))
    else:
        b.append(tf.Variable(_b[0]))
        b.append(tf.Variable(_b[1]))
        b.append(tf.Variable(_b[2]))

    vals = [y, x]

    vals.append(tf.nn.conv2d(vals[-1], W[0], strides=[1,1,1,1], padding='SAME') + b[0])
    vals.append(tf.nn.elu(vals[-1]))
    vals.append(tf.nn.max_pool(vals[-1], ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME'))

    vals.append(tf.nn.conv2d(vals[-1], W[1], strides=[1,1,1,1], padding='SAME') + b[1])
    vals.append(tf.nn.elu(vals[-1]))
    vals.append(tf.nn.max_pool(vals[-1], ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME'))

    vals.append(tf.nn.conv2d(vals[-1], W[2], strides=[1,1,1,1], padding='SAME') + b[2])
    vals.append(tf.reshape(vals[-1], [-1, num_classes]))

    vals.append(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=vals[-1], labels=tf.reshape(y, [-1])))
    vals.append(tf.reduce_mean(vals[-1]))

    return vals, W, b


def train(vals, W, b):
    (x_train, y_train), (x_test, y_test) = load_data()
    x_train = np.expand_dims(x_train, axis=-1).astype(np.float32)
    x_test = np.expand_dims(x_test, axis=-1).astype(np.float32)
    x_train = x_train / 255
    x_test = x_test / 255

    training_epochs = 1
    num_train_examples = len(x_train)
    batch_size = 100
    learning_rate = .01
    display_step = 1

    gen = get_batch(x_train, y_train, batch_size)
    gen_test = get_batch(x_test, y_test, len(x_test))
    x_test, y_test = next(gen_test)

    pred = tf.argmax(vals[-4], axis=-1)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(vals[-1])

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)

        # Training cycle
        for epoch in range(training_epochs):
            avg_cost = 0.
            total_batch = int(num_train_examples / batch_size)
            # Loop over all batches
            for i in range(total_batch):
                batch_x, batch_y = next(gen)

                # Run optimization op (backprop) and cost op (to get loss value)
                f = {vals[1]: batch_x, vals[0]: batch_y}
                _, c = sess.run([optimizer, vals[-1]], feed_dict=f)
                # Compute average loss
                avg_cost += c / total_batch
            print avg_cost

            # Display logs per epoch step
            if epoch % display_step == 0:
                print("Epoch:", '%04d' % (epoch + 1), "cost=", \
                      "{:.9f}".format(avg_cost))

                # Test model
                f = {vals[1]: x_test, vals[0]: y_test}
                np_pred = pred.eval(f, sess)
                correct_prediction = np_pred == y_test

                # Calculate accuracy
                accuracy = np.sum(correct_prediction / float(len(x_test) * 7 * 7))

                print 'accuracy =', accuracy
                r = np.random.randint(len(x_test))
                print np_pred[r].reshape(-1)
                print y_test[r].reshape(-1)
                print

        print("Optimization Finished!")

        trained_weights = []
        trained_biases = []
        for weight, bias in zip(W, b):
            trained_weights.append(weight.eval())
            trained_biases.append(bias.eval())

        return trained_weights, trained_biases


def save_weights(W, b):
    # save weights
    with open('mnist_fcn_train.p', 'w') as file:
        pickle.dump([W, b], file)

def load_weights():
    with open('mnist_fcn_train.p', 'r') as file:
        result = pickle.load(file)
        return result[0], result[1]

def get_batch(x, y, batch_size):
    # generator to yield batches
    num_samples = len(x)
    indices = np.arange(num_samples)
    while True:
        np.random.shuffle(indices)
        for start in range(0, num_samples, batch_size):
            end = min(num_samples, start + batch_size)
            idx = indices[start:end]

            x_batch, temp = x[idx], y[idx]
            y_batch = np.zeros((len(idx), 7, 7), dtype=np.int64) + 10
            for i in range(len(idx)):
                y_batch[i,2:5,2:5] = temp[i]

        yield x_batch, y_batch


if __name__ == '__main__':
    W, b = load_weights()
    vals, W, b = create_model(W, b)
    W, b = train(vals, W, b)
    save_weights(W, b)
