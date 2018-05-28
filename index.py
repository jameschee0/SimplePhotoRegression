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
    generated = generated.resize((3000,3000)) #resize image dimension

    #save
    generated.save(generated_path)
    print('Generated : {}'.format(generated_path))

def crop(input,filter,i,j):
    cropped = input[filter*i:filter*(i+1),filter*j:filter*(j+1)]
    return cropped

def check(input):
    count = 0
    for i in range(input.shape[0]):
        for j in range(input.shape[1]):
            if input[i][j] == 1:
                count = count + 1
    return count

def train(input,iterations):
    tf.set_random_seed(777)

    #Generate formatted training data
    x_data = []
    y_data = []

    for i in range(input.shape[0]):
        for j in range(input.shape[1]):
            if input[i][j] == 1:
                x_data.append([j])
                y_data.append([i])

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
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
    train = optimizer.minimize(cost)

    ### Launch the graph in a session ###
    sess = tf.Session()
    # Initializes global variables in the graph.
    sess.run(tf.global_variables_initializer())

    result = []

    for step in range(iterations):
        cost_val, hy_val, cur_weight, cur_bias, _ = sess.run(
            [cost, hypothesis, W, b, train], feed_dict={X: x_data, Y: y_data})
        if step ==iterations-1:
            result = [cur_weight[0][0],cur_bias[0]]
            print('Cost : {} Weight: {} Bias: {}'.format(cost_val,cur_weight,cur_bias))

    return result

def arraytoimg(input,dir):
    im = Image.fromarray(input,'1')
    im.save(dir)

###################  Main Code Starts here  #######################
## Variables
filter = 10

# 1. Generate training images
generate_training_image('./res/school.jpeg','./res/manipulated/school_gen.png',filter)

# 2. Change input image to numpy array
input = np.array(Image.open('./res/manipulated/school_gen.png'))

## Another Variables
input_dim = input.shape[0]
result_array = np.zeros((input_dim,input_dim))

# 3. generate result array
stride = input.shape[0]/filter
for i in range(stride):
    for j in range(stride):
        data = crop(input,filter,i,j)
        print('cropped i:{} j:{} '.format(i,j))

        # 1) train and get output // Done
        if check(data)>math.floor(filter/2):
            parameter = train(data,2000)
            # 2) Generate array with outputed equation
            for k in range(filter):
                value = parameter[0]*k+parameter[1] ## pure value of hypothesis ##
                value = math.floor(value)
                if value>=0 and value<filter:
                    x_axis = int(filter*i+value)
                    y_axis = int(filter*j+k)
                    result_array[x_axis,y_axis] = 1
                    print('x: {} y: {} arr:{}'.format(x_axis,y_axis,result_array[x_axis,y_axis]))
        if j%10==0:
            np.savetxt("./res/trainedArray/school.csv",result_array,delimiter=",")
            arraytoimg(result_array,'./res/regressed/school_reg.png')

# 4. generate new image
arraytoimg(result_array,'./res/regressed/school_reg.png')
