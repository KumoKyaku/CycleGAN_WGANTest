from tfImport import *
from ops import *


class Generator:
    def __init__(self, name, is_training, norm='instance', image_size=(270, 480), mean=0.0):
        self.name = name
        self.reuse = False
        self.norm = norm
        self.is_training = is_training
        self.image_size = image_size
        self.mean = mean

    def __call__(self, input):
        """
        Args:
        input: batch_size x width x height x 3
        Returns:
        output: same size as input
        """

        input = tf.pad(input, [[0, 0], [10, 10], [
                       10, 10], [0, 0]], mode='REFLECT')

        batch_size = input.get_shape()[0].value
        oriHeight = input.get_shape()[1].value
        oriWidth = input.get_shape()[2].value
        channel = input.get_shape()[3].value

        with tf.variable_scope(self.name, reuse=self.reuse):
            # conv layers
            with tf.variable_scope('conv1'):
                conv1 = conv2d_padhalfKernel(input, input.get_shape()[3],
                                             output_filters=32, kernel=9, strides=1, mean=self.mean)
                conv1 = relu(norm(conv1, self.is_training, self.norm))
            with tf.variable_scope('conv2'):
                conv2 = conv2d_padhalfKernel(
                    conv1, 32, output_filters=64, kernel=3, strides=2, mean=self.mean)
                conv2 = relu(norm(conv2, self.is_training, self.norm))

            with tf.variable_scope('conv3'):
                conv3 = conv2d_padhalfKernel(
                    conv2, 64, output_filters=128, kernel=3, strides=2, mean=self.mean)
                conv3 = relu(norm(conv3, self.is_training, self.norm))

            with tf.variable_scope('res'):
                # 9 blocks for higher resolution
                res_output = n_res_blocks(conv3, reuse=self.reuse, norm=self.norm,
                                          is_training=self.is_training, n=9, mean=self.mean)

            with tf.variable_scope('deconv1'):
                deconv1 = resize_conv2d(
                    res_output, 128, 64, 3, 2, self.is_training, mean=self.mean)
                deconv1 = relu(norm(deconv1, self.is_training, self.norm))

            with tf.variable_scope('deconv2'):
                deconv2 = resize_conv2d(
                    deconv1, 64, 32, 3, 2, self.is_training, mean=self.mean)
                deconv2 = relu(norm(deconv2, self.is_training, self.norm))

            with tf.variable_scope('deconv3'):
                deconv3 = resize_conv2d(deconv2, 32, 3, 9, 1, self.is_training, mean=self.mean)
                deconv3 = tf.nn.tanh(
                    norm(deconv3, self.is_training, self.norm))

        # set reuse=True for next call
        self.reuse = True
        self.variables = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

        output = tf.image.resize_images(deconv3, (oriHeight, oriWidth),
                                        method=tf.image.ResizeMethod.BILINEAR)

        output = tf.slice(output, [0, 10, 10, 0], tf.stack(
            [-1, oriHeight - 20, oriWidth - 20, -1]))

        output.set_shape([batch_size, oriHeight - 20, oriWidth - 20, channel])

        return output
