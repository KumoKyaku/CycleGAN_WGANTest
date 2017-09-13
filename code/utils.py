import zipfile
import tensorflow as tf 
from tensorflow.python.platform.gfile import GFile,Copy
import os
import random

temppath = 'temp'

if not os.path.exists(temppath):
    os.makedirs(temppath)
    print('Make dir : %s'%temppath)

def Unzip(fileName):
    z = zipfile.ZipFile(fileName,mode="r")
    z.extractall(temppath)

def pai_copy(fileName,oldPath,newPath = temppath):
    newfileName = os.path.join(temppath,fileName)
    oldfileName = os.path.join(oldPath,fileName)
    Copy(oldfileName,newfileName)
    print('copy %s to %s'%(oldfileName,newfileName))
    return newfileName

class ImagePool:
    """ History of generated images
        Same logic as https://github.com/junyanz/CycleGAN/blob/master/util/image_pool.lua
    """
    def __init__(self, pool_size):
        self.pool_size = pool_size
        self.images = []

    def query(self, image):
        if self.pool_size == 0:
            return image

        if len(self.images) < self.pool_size:
            self.images.append(image)
            return image
        else:
            p = random.random()
            if p > 0.5:
                # use old image
                random_id = random.randrange(0, self.pool_size)
                tmp = self.images[random_id].copy()
                self.images[random_id] = image.copy()
                return tmp
            else:
                return image


def convert2int(image):
    """ Transfrom from float tensor ([-1.,1.]) to int image ([0,255])
    """
    return tf.image.convert_image_dtype((image+1.0)/2.0, tf.uint8)

def convert2float(image):
    """ Transfrom from int image ([0,255]) to float tensor ([-1.,1.])
    """
    #image = tf.image.convert_image_dtype(image, dtype=tf.float32) #It's wrong
    image = tf.cast(image, dtype=tf.float32)
    return (image/127.5) - 1.0

def batch_convert2int(images):
    """
    Args:
        images: 4D float tensor (batch_size, image_size, image_size, depth)
    Returns:
        4D int tensor
    """
    return tf.map_fn(convert2int, images, dtype=tf.uint8)