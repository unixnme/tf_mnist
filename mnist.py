# simple tensorflow model for mnist
from keras.datasets.mnist import load_data
import tensorflow as tf
import numpy as np

def create_model():
    num_classes = 10
    img_rows = 28
    img_cols = 28
    img_channel = 1
    depth = 16

    x = tf.placeholder(tf.float32, [None, img_rows, img_cols, img_channel])
    y = tf.placeholder(tf.int32, [None])

    W = []
    b = []

    W.append(tf.Variable(tf.random_normal([img_rows, img_cols, img_channel, depth])))
    b.append(tf.Variable(tf.random_normal([depth])))

    W.append(tf.Variable(tf.random_normal([img_rows/2, img_cols/2, depth, num_classes])))
    b.append(tf.Variable(tf.random_normal([num_classes])))

    vals = [y, x]

    vals.append(tf.nn.conv2d(vals[-1], W[0], strides=[1,1,1,1], padding='SAME'))
    vals.append(tf.nn.bias_add(vals[-1], b[0]))
    vals.append(tf.nn.elu(vals[-1]))
    vals.append(tf.nn.max_pool(vals[-1], ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME'))

    vals.append(tf.nn.conv2d(vals[-1], W[1], strides=[1,1,1,1], padding='VALID'))
    vals.append(tf.nn.bias_add(vals[-1], b[1]))
    vals.append(tf.reshape(vals[-1], [-1, num_classes]))
    vals.append(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=vals[-1], labels=y))
    vals.append(tf.reduce_mean(vals[-1], axis=-1))

    return vals


def train(vals):
    (x_train, y_train), (x_test, y_test) = load_data()
    x_train = np.expand_dims(x_train, axis=-1).astype(np.float32)
    x_test = np.expand_dims(x_test, axis=-1).astype(np.float32)
    x_train = x_train / 255 - 0.5
    x_test = x_test / 255 - 0.5

    training_epochs = 1
    num_train_examples = len(x_train)
    batch_size = 100
    learning_rate = .01
    display_step = 1

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
                start = i*batch_size
                end = min((i+1)*batch_size, num_train_examples)

                batch_x, batch_y = x_train[start:end], y_train[start:end]
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
        print("Optimization Finished!")

if __name__ == '__main__':
    vals = create_model()
    train(vals)