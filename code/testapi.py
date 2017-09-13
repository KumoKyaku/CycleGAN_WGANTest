from tfImport import *
from generator import Generator
from discriminator import Discriminator
from model import CycleGAN
from utils import ImagePool
import utils
import time


def main():
    sess = tf.InteractiveSession()
    testimg = np.asarray([[0,127,255],
                            [20,128,127],
                            [255,100,100]],dtype=np.float)
    print(testimg)
    img = tf.convert_to_tensor(testimg,dtype=tf.uint8)
    print(img)

    img = tf.expand_dims(img,axis=2)
    print(img)
    print(img.eval())
    #img = tf.image.convert_image_dtype(img,dtype= tf.float32)
    img = utils.convert2float(img)
    #img = (img/127.5)-1.0
    print(img)
    print(img.eval())
    img = (img +1.0)/2.0
    img = tf.image.convert_image_dtype(img,dtype=tf.uint8)
    print(img)
    print(img.eval())












if __name__ == '__main__':
    main()