from tfImport import *
from generator import Generator
from discriminator import Discriminator
from model1 import CycleGAN
from utils import ImagePool
import utils
import time
import reader


def main():
    sess = tf.InteractiveSession(
        config=tf.ConfigProto(allow_soft_placement=True))
    num_epoch = 40000
    pool_size = 20
    batch_size = 1
    oldpath = FLAGS.buckets
    RealPicPath = 'picF'
    AnimaPicPaht = 'picG'
    useCopyfile = False

    # testimg = 'picF\\109999-COCO_train2014_000000330671.jpg'

    # testimg = tf.read_file(testimg)
    # testimg = tf.image.decode_jpeg(testimg,channels=3)
    # print(testimg)
    # print(testimg.eval())
    # print(utils.convert2float(testimg).eval())
    # print(utils.convert2int(utils.convert2float(testimg)).eval())

    readera = reader.Reader(RealPicPath,batch_size=1)
    Ga2b = Generator('Ga2b', True, mean=0.02)

    real_a = readera.feed()

    tf.summary.image("real_a",real_a)

    fake_b = Ga2b(real_a)

    tf.summary.image("fake_b",fake_b)
    summary_op = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter('C:\\log')

    sess.run([tf.global_variables_initializer(),
              tf.local_variables_initializer()])

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    print(fake_b.eval())
    print(utils.batch_convert2int(fake_b).eval())

    print('start train')

    summary,ra,fb = sess.run([summary_op,real_a,fake_b])
    #print(ra,fb)
    train_writer.add_summary(summary, 1)
    train_writer.flush()      

    coord.request_stop()
    coord.join(threads)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--buckets', type=str, default='',
                        help='input data path')

    parser.add_argument('--checkpointDir', type=str, default='',
                        help='output model path')
    FLAGS, _ = parser.parse_known_args()

    main()
