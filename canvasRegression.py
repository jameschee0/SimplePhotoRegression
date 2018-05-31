#Artistic Approach to Mathmatical Machine Learning Computer Vision
from PIL import Image,ImageFilter
import matplotlib.pyplot as plt
import math
import numpy as np
import tensorflow as tf

def generate_training_data(arr):
    size = np.shape(arr)[0]
    depth = np.shape(arr)[2]

    result = np.zeros((size,size))
    x = []
    y = []
    for i in range(size):
        for j in range(size):
            index = size-i-1
            for k in range(depth):
                result[index][j] = result[index][j]+arr[i][j][k]
            if(result[index][j]!=0):
                result[index][j] = result[index][j]/result[index][j]
                x.append([j])
                y.append([index])

    return (result,x,y)

def train(x_data,y_data,name):
    tf.set_random_seed(777)

    ### Designing Model Graph ###
    # placeholders for a tensor that will be always fed.
    X = tf.placeholder(tf.float32, shape=[None,1])
    Y = tf.placeholder(tf.float32, shape=[None,1])
    #Weight and Bias Variable
    W = tf.Variable(tf.random_normal([1,1]), name='weight')
    b = tf.Variable(tf.random_normal([1]), name='bias')

    # Hypothesis
    hypothesis = tf.matmul(X, W) + b

    # Simplified cost function
    cost = tf.reduce_mean(tf.square(hypothesis - Y))

    # Minimize
    optimizer = tf.train.AdamOptimizer(learning_rate=1e-3)
    train = optimizer.minimize(cost)

    ### Launch the graph in a session ###
    sess = tf.Session()
    # Initializes global variables in the graph.
    sess.run(tf.global_variables_initializer())

    result = []
    step = 0

    while True:
        step = step +1
        cost_val, hy_val, cur_weight, cur_bias, _ = sess.run(
            [cost, hypothesis, W, b, train], feed_dict={X: x_data, Y: y_data})
        if step%1000 == 0:
            result = [cur_weight,cur_bias]
            print('{} Cost : {} Weight: {} Bias: {}'.format(step,cost_val,cur_weight,cur_bias))
            np.savetxt("./CanvasArray/"+name+".csv",result,delimiter=",")
        if cost_val<1e-10:
            break
    return result

###################  Main Code Starts here  #######################
## Variables
filter = 10
name = "img6"
# 2. Change input image to numpy array
input = np.array(Image.open("./canvas_image/"+name+".png"))
(input,x_data,y_data) = generate_training_data(input)
result = train(x_data,y_data,name)
