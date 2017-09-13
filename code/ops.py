from tfImport import *

# Helpers


def _weights(name, shape, mean=0.0, stddev=0.02):
    """ Helper to create an initialized Variable
    Args:
      name: name of the variable
      shape: list of ints
      mean: mean of a Gaussian
      stddev: standard deviation of a Gaussian
    Returns:
      A trainable variable
    """
    var = tf.get_variable(
        name, shape,
        initializer=tf.random_normal_initializer(
            mean=mean, stddev=stddev, dtype=tf.float32))
    return var


def _biases(name, shape, constant=0.0):
    """ Helper to create an initialized Bias with constant
    """
    return tf.get_variable(name, shape,
                           initializer=tf.constant_initializer(constant))


def _leaky_relu(input, slope):
    return tf.maximum(slope * input, input)


def norm(input, is_training, norm='instance'):
    """ Use Instance Normalization or Batch Normalization or None
    """
    if norm == 'instance':
        return _instance_norm(input)
    elif norm == 'batch':
        return _batch_norm(input, is_training)
    else:
        return input


def _batch_norm(input, is_training):
    """ Batch Normalization
    """
    with tf.variable_scope("batch_norm"):
        return tf.contrib.layers.batch_norm(input,
                                            decay=0.9,
                                            scale=True,
                                            updates_collections=None,
                                            is_training=is_training)


def _instance_norm(input):
    """ Instance Normalization
    """
    with tf.variable_scope("instance_norm"):
        depth = input.get_shape()[3]
        scale = _weights("scale", [depth], mean=1.0)
        offset = _biases("offset", [depth])
        mean, variance = tf.nn.moments(input, axes=[1, 2], keep_dims=True)
        epsilon = 1e-5
        inv = tf.rsqrt(variance + epsilon)
        normalized = (input - mean) * inv
        return scale * normalized + offset


def safe_log(x, eps=1e-12):
    return tf.log(x + eps)


def conv2d_padhalfKernel(x, input_filters, output_filters, kernel, strides, mode='REFLECT',mean=0.0,):
    with tf.variable_scope('conv'):

        shape = [kernel, kernel, input_filters, output_filters]
        weight = _weights("weights", shape=shape,mean = mean)
        padsize = int(kernel / 2)
        padding = [[0, 0], [padsize, padsize], [padsize, padsize], [0, 0]]
        x_padded = tf.pad(x, padding, mode=mode)
        return tf.nn.conv2d(x_padded, weight, strides=[1, strides, strides, 1], padding='VALID', name='conv')


def conv2d(x, input_filters, output_filters, kernel, strides, padding='SAME'):
    with tf.variable_scope('conv'):
        shape = [kernel, kernel, input_filters, output_filters]
        weight = _weights("weights", shape=shape)
        return tf.nn.conv2d(x, weight, strides=[1, strides, strides, 1], padding=padding, name='conv')


def relu(input):
    relu = tf.nn.relu(input)
    # convert nan to zero (nan != nan)
    nan_to_zero = Nan_to_zero(relu)
    return nan_to_zero

def Nan_to_zero(input):
    '''convert nan to zero (nan != nan)'''
    nan_to_zero = tf.where(tf.equal(input, input), input, tf.zeros_like(input))
    return nan_to_zero

def leaky_relu(input, slope=0.2):
    return Nan_to_zero(tf.maximum(slope * input, input))


def residual_block(input, normtype='instance', is_training=True, name="residual", mean=0.0):
    """ A residual block that contains two 3x3 convolutional layers
        with the same number of filters on both layer
    Args:
        input: 4D Tensor
        k: integer, number of filters (output depth)
        reuse: boolean
        name: string
    Returns:
        4D tensor (same shape as input)
    """

    filters = input.get_shape()[3]
    with tf.variable_scope(name):
        with tf.variable_scope('layer1'):
            conv1 = conv2d_padhalfKernel(input, filters, filters, 3, 1,mean=mean)
            conv1 = relu(norm(conv1, is_training, normtype))

        with tf.variable_scope('layer2'):
            conv2 = conv2d_padhalfKernel(conv1, filters, filters, 3, 1,mean=mean)
            conv2 = norm(conv1, is_training, normtype)

    output = input + conv2
    return output


def n_res_blocks(input, reuse, norm='instance', is_training=True, n=6, mean=0.0):
    for i in range(1, n + 1):
        output = residual_block(
            input, norm, is_training, 'residual{}'.format(i),mean=mean)
        input = output
    return output


def resize_conv2d(x, input_filters, output_filters, kernel, strides, training, mean=0.0):
    '''
    An alternative to transposed convolution where we first resize, then convolve.
    See http://distill.pub/2016/deconv-checkerboard/

    For some reason the shape needs to be statically known for gradient propagation
    through tf.image.resize_images, but we only know that for fixed image size, so we
    plumb through a "training" argument
    '''
    with tf.variable_scope('resize_conv2d'):
        height = x.get_shape()[1].value if training else tf.shape(x)[1]
        width = x.get_shape()[2].value if training else tf.shape(x)[2]

        new_height = height * strides * 2
        new_width = width * strides * 2

        x_resized = tf.image.resize_images(
            x, [new_height, new_width], tf.image.ResizeMethod.NEAREST_NEIGHBOR)

        # shape = [kernel, kernel, input_filters, output_filters]
        # weight = tf.Variable(tf.truncated_normal(shape, stddev=0.1), name='weight')
        return conv2d_padhalfKernel(x_resized, input_filters, output_filters, kernel, strides,mean=mean)

def safe_log(x, eps=1e-12):
    return tf.log(x + eps)