from tfImport import *
from generator import Generator
from discriminator import Discriminator
import ops
from reader import Reader
import utils

REAL_LABEL = 9e-1
LAMBDA = 10

class CycleGAN:
    def __init__(self,
                 X_train_file='',
                 Y_train_file='',
                 batch_size=1,
                 image_size=(270,480),
                 use_lsgan=True,
                 norm='instance',
                 lambda1=10.0,
                 lambda2=10.0,
                 optimizer = 'Adam',
                 lossfunc = 'use_lsgan',
                 learning_rate=2e-4,
                 start_decay_step = 100000,
                 decay_steps = 100000,
                 beta1=0.5,
                 ngf=64
                 ):
        """
        Args:
        X_train_file: string, X tfrecords file for training
        Y_train_file: string Y tfrecords file for training
        batch_size: integer, batch size
        image_size: integer, image size
        lambda1: integer, weight for forward cycle loss (X->Y->X)
        lambda2: integer, weight for backward cycle loss (Y->X->Y)
        use_lsgan: boolean
        norm: 'instance' or 'batch'
        learning_rate: float, initial learning rate for Adam
        beta1: float, momentum term of Adam
        ngf: number of gen filters in first conv layer
        """
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.use_lsgan = use_lsgan
        use_sigmoid = not use_lsgan
        self.batch_size = batch_size
        self.image_size = image_size
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.start_decay_step = start_decay_step
        self.decay_steps = decay_steps
        self.beta1 = beta1
        self.X_train_file = X_train_file
        self.Y_train_file = Y_train_file
        self.lossfunc = lossfunc
        self.is_training = tf.placeholder_with_default(
            True, shape=[], name='is_training')
        self.is_training = True

        self.G = Generator('G', self.is_training,
                        norm=norm, image_size=image_size,mean=0.25)
        self.D_G = Discriminator('D_G', self.is_training, norm=norm)
    
        self.F = Generator('F', self.is_training,
                        norm=norm, image_size=image_size,mean=0.75)
        self.D_F = Discriminator('D_F', self.is_training, norm=norm)

        self.fake_x = tf.placeholder(tf.float32,
                                     shape=[batch_size, image_size[0], image_size[1], 3])
        self.fake_y = tf.placeholder(tf.float32,
                                     shape=[batch_size,  image_size[0], image_size[1], 3])

    def build(self):

        X_reader = Reader(self.X_train_file, name='F',
                          image_size=self.image_size, batch_size=self.batch_size)
        Y_reader = Reader(self.Y_train_file, name='G',
                          image_size=self.image_size, batch_size=self.batch_size)

        x = X_reader.feed()
        y = Y_reader.feed()

        # x = tf.placeholder(dtype = tf.float32,shape=(5,270,480,3))
        # y = tf.placeholder(dtype = tf.float32,shape=(5,270,480,3))
        
        cycle_loss = self.cycle_consistency_loss(self.G, self.F, x, y)

        with tf.device("/gpu:0"):
            # X -> Y
            fake_y = self.G(x)
            G_gan_loss = self.generator_loss(self.D_G, fake_y, use_lsgan=self.use_lsgan)
            G_loss = G_gan_loss + cycle_loss
            D_G_loss = self.discriminator_loss(self.D_G, y, self.fake_y, use_lsgan=self.use_lsgan)
        
        with tf.device("/gpu:1"):
            # Y -> X
            fake_x = self.F(y)
            F_gan_loss = self.generator_loss(self.D_F, fake_x, use_lsgan=self.use_lsgan)
            F_loss = F_gan_loss + cycle_loss
            D_F_loss = self.discriminator_loss(self.D_F, x, self.fake_x, use_lsgan=self.use_lsgan)

        # summary
        tf.summary.histogram('D_G/true', self.D_G(y))
        tf.summary.histogram('D_G/fake', self.D_G(self.G(x)))
        tf.summary.histogram('D_F/true', self.D_F(x))
        tf.summary.histogram('D_F/fake', self.D_F(self.F(y)))

        tf.summary.scalar('loss/G_gan_loss', G_gan_loss)
        tf.summary.scalar('loss/D_G', D_G_loss)
        tf.summary.scalar('loss/F_gan_loss', F_gan_loss)
        tf.summary.scalar('loss/D_F', D_F_loss)
        tf.summary.scalar('loss/cycle', cycle_loss)
        tf.summary.scalar('loss/G', G_loss)
        tf.summary.scalar('loss/F', F_loss)

        tf.summary.image('X/generated', utils.batch_convert2int(self.G(x)))
        tf.summary.image('X/reconstruction',
                          utils.batch_convert2int(self.F(self.G(x))))
        tf.summary.image('Y/generated', utils.batch_convert2int(self.F(y)))
        tf.summary.image('Y/reconstruction',
                          utils.batch_convert2int(self.G(self.F(y))))

        return G_loss, D_G_loss, F_loss, D_F_loss, fake_y, fake_x

    def optimize(self, G_loss, D_Y_loss, F_loss, D_X_loss):
        def make_optimizer(loss, variables, name='Adam',optimizer = self.optimizer):
            """ Adam optimizer with learning rate 0.0002 for the first 100k steps (~100 epochs)
                and a linearly decaying rate that goes to zero over the next 100k steps
            """
            global_step = tf.Variable(0, trainable=False)
            starter_learning_rate = self.learning_rate
            end_learning_rate = 0.0
            start_decay_step = self.start_decay_step
            decay_steps = self.decay_steps
            beta1 = self.beta1
            learning_rate = (
                tf.where(
                    tf.greater_equal(global_step, start_decay_step),
                    tf.train.polynomial_decay(starter_learning_rate, global_step - start_decay_step,
                                              decay_steps, end_learning_rate,
                                              power=1.0),
                    starter_learning_rate
                )

            )
            tf.summary.scalar('learning_rate/{}'.format(name), learning_rate)

            if optimizer == 'Adam':
                learning_step = (
                    tf.train.AdamOptimizer(learning_rate, beta1=beta1, name=name)
                            .minimize(loss, global_step=global_step, var_list=variables)
                )
                return learning_step
            elif optimizer == 'RMSProp':
                learning_step = (
                    tf.train.RMSPropOptimizer(learning_rate, beta1=beta1, name=name)
                            .minimize(loss, global_step=global_step, var_list=variables)
                )
                return learning_step
            else:
                return None

        with tf.device("/gpu:0"):
            G_optimizer = make_optimizer(G_loss, self.G.variables, name='%s_G'%self.optimizer)
            D_Y_optimizer = make_optimizer(
                D_Y_loss, self.D_G.variables, name='%s_D_Y'%self.optimizer)

        with tf.device("/gpu:1"):
            F_optimizer = make_optimizer(F_loss, self.F.variables, name='%s_F'%self.optimizer)
            D_X_optimizer = make_optimizer(
                D_X_loss, self.D_F.variables, name='%s_D_X'%self.optimizer)

        de = [G_optimizer, D_Y_optimizer, F_optimizer, D_X_optimizer]

        with tf.control_dependencies(de):
            return tf.no_op(name='optimizers')

    def discriminator_loss(self, D, y, fake_y, use_lsgan=True):
        """ Note: default: D(y).shape == (batch_size,5,5,1),
                            fake_buffer_size=50, batch_size=1
        Args:
            G: generator object
            D: discriminator object
            y: 4D tensor (batch_size, image_size, image_size, 3)
        Returns:
            loss: scalar
        """
        if  self.lossfunc == 'use_lsgan':
            # use mean squared error
            error_real = tf.reduce_mean(
                tf.squared_difference(D(y), REAL_LABEL))
            error_fake = tf.reduce_mean(tf.square(D(fake_y)))
            loss = (error_real + error_fake) / 2
        elif self.lossfunc == 'sigmoid_cross_entropy_with_logits':
            # use cross entropy
            logitsy = D(y)
            logitsfy = D(fake_y)
            # error_real = -tf.reduce_mean(ops.safe_log(tf.sigmoid(D(y))))
            # error_fake = -tf.reduce_mean(ops.safe_log(1 - tf.sigmoid(D(fake_y))))
            error_real = tf.nn.sigmoid_cross_entropy_with_logits(logits=logitsy,labels=tf.ones_like(logitsy))
            error_real = tf.reduce_mean(error_real)
            error_fake = tf.nn.sigmoid_cross_entropy_with_logits(logits=logitsfy,labels=tf.zeros_like(logitsfy))
            error_fake = tf.reduce_mean(error_fake)
            loss = (error_real + error_fake) / 2
        elif self.lossfunc == 'wgan':
            loss = tf.reduce_mean(D(fake_y)) - tf.reduce_mean(D(y))

            # Gradient penalty
            alpha = tf.random_uniform(
                shape=[self.batch_size,1], 
                minval=0.,
                maxval=1.
            )

            differences = fake_y - y
            interpolates = y + (alpha*differences)
            gradients = tf.gradients(D(interpolates), [interpolates])[0]
            slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
            gradient_penalty = tf.reduce_mean((slopes-1.)**2)
            loss += LAMBDA*gradient_penalty
        return loss

    def generator_loss(self, D, fake_y, use_lsgan=True):
        """  fool discriminator into believing that G(x) is real
        """
        if  self.lossfunc == 'use_lsgan':
            # use mean squared error
            loss = tf.reduce_mean(tf.squared_difference(D(fake_y), REAL_LABEL))
        elif self.lossfunc == 'sigmoid_cross_entropy_with_logits':
            # heuristic, non-saturating loss
            # loss = tf.sigmoid(D(fake_y))
            # loss = -tf.reduce_mean(ops.safe_log(loss)) / 2
            logitsfy = D(fake_y)
            loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=logitsfy,labels=tf.ones_like(logitsfy))
            loss = tf.reduce_mean(loss)
        elif self.lossfunc == 'wgan':
            loss = -tf.reduce_mean(D(fake_y))
        return loss

    def cycle_consistency_loss(self, G, F, x, y):
        """ cycle consistency loss (L1 norm)
        """
        with tf.device("/gpu:0"):
            gx = G(x)
        with tf.device("/gpu:1"):
            forward_loss = tf.reduce_mean(tf.abs(F(gx) - x))
            backward_loss = tf.reduce_mean(tf.abs(G(F(y)) - y))
            loss = self.lambda1 * forward_loss + self.lambda2 * backward_loss
        return loss
