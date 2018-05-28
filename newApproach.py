#Artistic Approach to Mathmatical Machine Learning Computer Vision
from PIL import Image,ImageFilter
import matplotlib.pyplot as plt
import math
import numpy as np
import tensorflow as tf

def generate_training_image(original_path,generated_path,filter):
    original = Image.open(original_path)

    #to black and white image
    generated = original.convert('1')

    #resize
    size = min(generated.size)
    size = size-(size%filter)
    print('generated size is : {}'.format(size))
    generated = generated.resize((size,size)) #resize image dimension

    #save
    generated.save(generated_path)
    print('Generated : {}'.format(generated_path))

def train(input,iterations):
    import tensorflow as tf
    tf.set_random_seed(777)  # for reproducibility

    x_data = []
    y_data = []

    i_dim = input.shape[0]
    j_dim = input.shape[1]

    for i in range(i_dim):
        for j in range(j_dim):
            x_data.append([j,i_dim-i-1])
            y_data.append([int(input[i,j])])

    # placeholders for a tensor that will be always fed.
    X = tf.placeholder(tf.float32, shape=[None, 2])
    Y = tf.placeholder(tf.float32, shape=[None, 1])

    W = tf.Variable(tf.random_normal([2, 1]), name='weight')
    b = tf.Variable(tf.random_normal([1]), name='bias')

    # Hypothesis using sigmoid: tf.div(1., 1. + tf.exp(tf.matmul(X, W)))
    hypothesis = tf.sigmoid(tf.matmul(X, W) + b)

    # cost/loss function
    cost = -tf.reduce_mean(Y * tf.log(hypothesis+1e-4) + (1 - Y) *
                           tf.log(1 - hypothesis+1e-4))

    train = tf.train.AdamOptimizer(1e-4).minimize(cost)

    # Launch graph
    with tf.Session() as sess:
        # Initialize TensorFlow variables
        sess.run(tf.global_variables_initializer())

        for step in range(iterations):
            cost_val, cur_w, cur_b, _ = sess.run([cost, W, b, train], feed_dict={X: x_data, Y: y_data})
            if step%200 == 0:
                cur_array = []
                cur_array.append(cur_w[0])
                cur_array.append(cur_w[1])
                cur_array.append(cur_b[0])
                np.savetxt("./res/trainedArray/school_newApproach.csv",cur_array,delimiter=",")
                print(step, cost_val, cur_w[0],cur_w[1], cur_b[0])

###################  Main Code Starts here  #######################
filter = 10

# 1. Generate training images
generate_training_image('./res/school_mountain.png','./res/manipulated/school_mountain_gen.png',filter)

# 2. Change input image to numpy array
input = np.array(Image.open('./res/manipulated/school_mountain_gen.png'))
train(input,10000)
