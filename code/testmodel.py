from tfImport import *
from generator import Generator
from discriminator import Discriminator
from model import CycleGAN
from utils import ImagePool
import utils
import time

def main():
    modelfile = 'model\\model.ckpt-2000'


    picF = "picF"
    files = os.listdir(picF)

    sess = tf.InteractiveSession()
    ge = Generator('G',is_training = False)
    x = tf.placeholder(dtype = tf.float32,shape=(5,270,480,3))

    out = ge(x)

    sess.run(tf.global_variables_initializer())

    saver = tf.train.Saver(var_list=ge.variables)
    v = ge.variables[0]
    print(v.name)
    saver.restore(sess,modelfile)
    print(v.eval())

    for pic in files:
        img = tf.read_file(os.path.join(picF,pic))
        img = tf.image.decode_jpeg(img)
        img = utils.convert2float(img)
        img = tf.expand_dims(img,axis=0)
        shape = tf.shape(img).eval()
        img.set_shape(shape)
        out = ge(img)

        out = tf.unstack(out)[0]
        out = utils.convert2int(out)
        out = tf.image.encode_jpeg(out)

        out = out.eval()
        with tf.gfile.GFile(os.path.join('out\\picF',pic),'wb') as fw:
            fw.write(out)
            fw.flush()

    sess.close()

if __name__ == '__main__':
    main()