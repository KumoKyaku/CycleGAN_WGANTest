from tfImport import *
from ops import *

class Discriminator:
    def __init__(self, name, is_training, norm='instance'):
        self.name = name
        self.is_training = is_training
        self.norm = norm
        self.reuse = False

    def __call__(self, input):
        """
        Args:
        input: batch_size x image_size x image_size x 3

        """
        with tf.variable_scope(self.name,reuse=self.reuse):

            # conv layers
            with tf.variable_scope('conv1'):
                conv1 = conv2d(input, input.get_shape()[3], output_filters = 64, kernel = 4, strides = 2)
                conv1 = leaky_relu(norm(conv1,self.is_training,self.norm))
            
            with tf.variable_scope('conv2'):
                conv2 = conv2d(conv1, 64, output_filters = 128, kernel = 4, strides = 2)
                conv2 = leaky_relu(norm(conv2,self.is_training,self.norm))
        
            with tf.variable_scope('conv3'):
                conv3 = conv2d(conv2, 128, output_filters = 256, kernel = 4, strides = 2)
                conv3 = leaky_relu(norm(conv3,self.is_training,self.norm))

            with tf.variable_scope('conv4'):
                conv4 = conv2d(conv3, 256, output_filters = 512, kernel = 4, strides = 2)
                conv4 = leaky_relu(norm(conv4,self.is_training,self.norm))

            with tf.variable_scope('conv5'):
                conv5 = conv2d(conv4, 512, output_filters = 1, kernel = 5, strides = 1)
                conv5 = leaky_relu(norm(conv5,self.is_training,self.norm))

            with tf.variable_scope('output'):
                output = tf.reshape(conv5,shape=[conv5.shape[0].value,-1])
                output = dense(output,1)

        self.reuse = True
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

        return output
