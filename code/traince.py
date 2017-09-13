from tfImport import *
from generator import Generator
from discriminator import Discriminator
from model1 import CycleGAN
from utils import ImagePool
import utils
import time


def main():

    num_epoch = 100000
    pool_size = 20
    batch_size = 1
    oldpath = FLAGS.buckets
    RealPicPath = 'picF'
    AnimaPicPaht = 'picG'
    useCopyfile = True

    if useCopyfile:
        trainfiles = ['picf1.zip', 'picf2.zip', 'picg1.zip']
        # trainfiles.extend(['picf3.zip','picf4.zip','picg2.zip'])

        print(trainfiles)

        for f in trainfiles:
            fn = utils.pai_copy(f, oldpath)
            utils.Unzip(fn)

        RealPicPath = os.path.join('temp', RealPicPath)
        AnimaPicPaht = os.path.join('temp', AnimaPicPaht)

    print(RealPicPath)
    print(AnimaPicPaht)

    sess = tf.InteractiveSession(
        config=tf.ConfigProto(allow_soft_placement=True))

    cycle_gan = CycleGAN(
        X_train_file=AnimaPicPaht,
        Y_train_file=RealPicPath,
        batch_size=batch_size,
        image_size=(270, 480),
        lossfunc = 'sigmoid_cross_entropy_with_logits',
        norm='instance',
        learning_rate=2e-3,
        start_decay_step = 10000,
        decay_steps = 100000
        #optimizer = 'RMSProp'
    )

    Ga2b_loss, Da2b_loss, Gb2a_loss, Db2a_loss, fake_a, fake_b,real_a,real_b = cycle_gan.build()

    optimizers = cycle_gan.optimize(Ga2b_loss, Da2b_loss, Gb2a_loss, Db2a_loss)

    summary_op = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(FLAGS.checkpointDir)
    saver = tf.train.Saver(max_to_keep=0)

    sess.run([tf.global_variables_initializer(),
              tf.local_variables_initializer()])

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    # save_path = saver.save(sess,os.path.join(FLAGS.checkpointDir,"model_pre.ckpt"))
    # print("Model saved in file: %s" % save_path)

    fake_a_pool = ImagePool(pool_size)
    fake_b_pool = ImagePool(pool_size)
    print('start train')
    start_time = time.time()

    for step in range(1, num_epoch + 1):
        # get previously generated images
        fake_a_val, fake_b_val = sess.run([fake_a, fake_b])

        # train
        _, Ga2b_loss_val, Da2b_loss_val, Gb2a_loss_val, Db2a_loss_val,real_a_val,real_b_val, summary = (
            sess.run(
                [optimizers, Ga2b_loss, Da2b_loss, Gb2a_loss, Db2a_loss,real_a,real_b, summary_op],
                feed_dict={cycle_gan.fake_a: fake_a_pool.query(fake_a_val),
                           cycle_gan.fake_b: fake_b_pool.query(fake_b_val)}
            )
        )

        elapsed_time = time.time() - start_time
        start_time = time.time()

        if step % 25 == 0:
            print('Ga2b_loss_val : %s--Da2b_loss_val : %s--Gb2a_loss_val : %s--Db2a_loss_val : %s--' % (Ga2b_loss_val,
                                                                                Da2b_loss_val, Gb2a_loss_val, Db2a_loss_val))

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
