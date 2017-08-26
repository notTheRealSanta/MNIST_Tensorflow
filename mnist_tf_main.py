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

x = tf.placeholder(tf.float32,[None,784], name = "x")
y = tf.placeholder(tf.float32, name="y")

#defining the model of the network
def neural_network_model (data) :

    with tf.name_scope ("Layer_1") :
        hidden_layer_1_weights = tf.Variable(tf.random_normal([784,nodes_hl1_len]), name = "W")
        hidden_layer_1_biases  = tf.Variable(tf.random_normal([nodes_hl1_len]), name = "B")

    with tf.name_scope ("Layer_2") :
        hidden_layer_2_weights = tf.Variable(tf.random_normal([nodes_hl1_len, nodes_hl2_len]), name = "W")
        hidden_layer_2_biases  = tf.Variable(tf.random_normal([nodes_hl2_len]), name = "B")

    with tf.name_scope ("Output_Layer") :
        output_layer_weights = tf.Variable(tf.random_normal([nodes_hl2_len,classes_len]), name = "W")
        output_layer_biases = tf.Variable(tf.random_normal([classes_len]), name = "B")
        tf.summary.histogram("weights",output_layer_weights)
        tf.summary.histogram("biases", output_layer_biases )

    l1 = tf.add(tf.matmul(data, hidden_layer_1_weights), hidden_layer_1_biases)
    l1 = tf.nn.relu(l1)

    l2 = tf.add(tf.matmul(l1, hidden_layer_2_weights), hidden_layer_2_biases)
    l2 = tf.nn.relu(l2)

    output_layer = tf.add(tf.matmul(l2, output_layer_weights), output_layer_biases)

    return output_layer


def train_neural_network (x):

    prediction = neural_network_model(x)

    with tf.name_scope ("Cost"):
        cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y) )
    tf.summary.scalar('cost', cost)

    optimizer = tf.train.AdamOptimizer().minimize(cost)

    epoch_len = 50

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
        with tf.name_scope ("accuracy") :
            correct = tf.equal ( tf.argmax(prediction,1), tf.argmax(y,1) )
            accuracy = tf.reduce_mean ( tf.cast(correct,'float') )

        tf.summary.scalar('accuracy', accuracy)
        print ('Accuracy :', accuracy.eval({x:mnist.test.images, y: mnist.test.labels}))

        merged_summary = tf.summary.merge_all()
        writer =  tf.summary.FileWriter("/tmp/mnist_demo/4")
        writer.add_graph(sess.graph)


train_neural_network ( x )
