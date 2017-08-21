import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST/", one_hot = True)

# mnist >  train, test > images, labels

#number of nodes in each hidden layer
nodes_hl1_len = 784
nodes_hl2_len = (784*2)

#classes is the number of distint outputs
classes_len = 10
batch_size = 100

x = tf.placeholder(tf.float32,[None,784])
y = tf.placeholder(tf.float32)

#defining the model of the network
def neural_network_model (data) :

    hidden_layer_1_weights = tf.Variable(tf.random_normal([784,nodes_hl1_len]))
    hidden_layer_1_biases  = tf.Variable(tf.random_normal([nodes_hl1_len]))

    hidden_layer_2_weights = tf.Variable(tf.random_normal([nodes_hl1_len, nodes_hl2_len]))
    hidden_layer_2_biases  = tf.Variable(tf.random_normal([nodes_hl2_len]))

    output_layer_weights = tf.Variable(tf.random_normal([nodes_hl2_len,classes_len]))
    output_layer_biases = tf.Variable(tf.random_normal([classes_len]))

    l1 = tf.add(tf.matmul(data, hidden_layer_1_weights), hidden_layer_1_biases)
    l1 = tf.nn.relu(l1)

    l2 = tf.add(tf.matmul(l1, hidden_layer_2_weights), hidden_layer_2_biases)
    l2 = tf.nn.relu(l2)

    output_layer = tf.add(tf.matmul(l2, output_layer_weights), output_layer_biases)

    return output_layer


def train_neural_network (x):

    prediction = neural_network_model(x)

    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y) )

    optimizer = tf.train.AdamOptimizer().minimize(cost)

    epoch_len = 10

    with tf.Session() as sess :

        sess.run(tf.global_variables_initializer())

        for epoch in range(epoch_len) :

            epoch_loss = 0;

            for _ in range(int(mnist.train.num_examples/batch_size)) :
                epoch_x, epoch_y = mnist.train.next_batch(batch_size)
                _, c = sess.run([optimizer, cost], feed_dict = {x: epoch_x, y: epoch_y})
                epoch_loss += c

            print( 'Epoch',epoch,'/',epoch_len )

        #testing the data
        correct = tf.equal ( tf.argmax(prediction,1), tf.argmax(y,1) )
        accuracy = tf.reduce_mean ( tf.cast(correct,'float') )
        print ('Accuracy :', accuracy.eval({x:mnist.test.images, y: mnist.test.labels}))

train_neural_network ( x )
