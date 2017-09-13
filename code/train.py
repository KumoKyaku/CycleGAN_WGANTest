from tfImport import *
from generator import Generator
from discriminator import Discriminator
from model import CycleGAN
from utils import ImagePool
import utils
import time


def main():

    num_epoch = 40000
    pool_size = 20
    batch_size = 1
    oldpath = FLAGS.buckets
    picFpath = 'picF'
    picGpath = 'picG'
    useCopyfile = True

    if useCopyfile:
        trainfiles = ['picf1.zip', 'picf2.zip', 'picg1.zip']
        # trainfiles.extend(['picf3.zip','picf4.zip','picg2.zip'])

        print(trainfiles)

        for f in trainfiles:
            fn = utils.pai_copy(f, oldpath)
            utils.Unzip(fn)

        picFpath = os.path.join('temp', picFpath)
        picGpath = os.path.join('temp', picGpath)

    print(picFpath)
    print(picGpath)

    sess = tf.InteractiveSession(
        config=tf.ConfigProto(allow_soft_placement=True))

    cycle_gan = CycleGAN(
        X_train_file=picGpath,
        Y_train_file=picFpath,
        batch_size=batch_size,
        image_size=(270, 480),
        use_lsgan=True,
        lossfunc = 'wgan',
        norm='instance',
        learning_rate=3e-3,
        start_decay_step = 5000,
        decay_steps = 350000
        #optimizer = 'RMSProp'
    )

    G_loss, D_Y_loss, F_loss, D_X_loss, fake_y, fake_x = cycle_gan.build()
    optimizers = cycle_gan.optimize(G_loss, D_Y_loss, F_loss, D_X_loss)

    summary_op = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(FLAGS.checkpointDir)
    saver = tf.train.Saver(max_to_keep=0)

    sess.run([tf.global_variables_initializer(),
              tf.local_variables_initializer()])

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    # save_path = saver.save(sess,os.path.join(FLAGS.checkpointDir,"model_pre.ckpt"))
    # print("Model saved in file: %s" % save_path)

    fake_Y_pool = ImagePool(pool_size)
    fake_X_pool = ImagePool(pool_size)
    print('start train')
    start_time = time.time()

    for step in range(1, num_epoch + 1):
        # get previously generated images
        fake_y_val, fake_x_val = sess.run([fake_y, fake_x])

        # train
        _, G_loss_val, D_Y_loss_val, F_loss_val, D_X_loss_val, summary = (
            sess.run(
                [optimizers, G_loss, D_Y_loss, F_loss, D_X_loss, summary_op],
                feed_dict={cycle_gan.fake_y: fake_Y_pool.query(fake_y_val),
                           cycle_gan.fake_x: fake_X_pool.query(fake_x_val)}
            )
        )

        elapsed_time = time.time() - start_time
        start_time = time.time()

        if step % 25 == 0:
            print('G_loss : %s--D_Y_loss : %s--F_loss : %s--D_X_loss : %s--' % (G_loss_val,
                                                                                D_Y_loss_val, F_loss_val, D_X_loss_val))

            print('step : %s --elapsed_time : %s' % (step, elapsed_time))
            print('adding summary...')
            train_writer.add_summary(summary, step)
            train_writer.flush()

        # if step % 100 == 0:
        #     print('-----------Step %d:-------------' % step)
        #     print('  G_loss   : {}'.format(G_loss_val))
        #     print('  D_Y_loss : {}'.format(D_Y_loss_val))
        #     print('  F_loss   : {}'.format(F_loss_val))
        #     print('  D_X_loss : {}'.format(D_X_loss_val))

        if step % 1000 == 0:
            save_path = saver.save(sess, os.path.join(
                FLAGS.checkpointDir, "model.ckpt"), global_step=step,write_meta_graph=False)
            print("Model saved in file: %s" % save_path)

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
