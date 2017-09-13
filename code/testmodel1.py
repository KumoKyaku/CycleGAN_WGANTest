from tfImport import *
from generator import Generator
from discriminator import Discriminator
from model import CycleGAN
from utils import ImagePool
import utils
import time

def main():

    picF = "picF"
    files = os.listdir(picF)[:1]

    sess = tf.InteractiveSession()

    global_step = tf.Variable(2501, name="global_step", trainable=False)

    sess.run(tf.global_variables_initializer())

    img = tf.read_file(os.path.join(picF,files[0]))
    img = tf.image.decode_jpeg(img)
    #img = utils.convert2float(img)
    img = tf.expand_dims(img,axis=0)
    
    tf.summary.image('real', img)
    tf.summary.scalar('test',global_step)
    outimg = utils.convert2float(img)
    tf.summary.image('out', utils.batch_convert2int(outimg))
    sdf = outimg.eval()
    print(sdf)
    sdf = utils.batch_convert2int(outimg).eval()
    print(sdf)
    summary_op = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter('C:\\log')

    summary,_,s,outimg2 = sess.run([summary_op,img,global_step,outimg])
    train_writer.add_summary(summary)
    train_writer.flush()      

    sess.close()

if __name__ == '__main__':
    main()