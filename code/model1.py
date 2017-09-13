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

        self.Ga2b = Generator('Ga2b', self.is_training,
                              norm=norm, image_size=image_size, mean=0.0)
        self.Da2b = Discriminator('Da2b', self.is_training, norm=norm)
    
        self.Gb2a = Generator('Gb2a', self.is_training,
                              norm=norm, image_size=image_size, mean=0.0)
        self.Db2a = Discriminator('Db2a', self.is_training, norm=norm)

        self.fake_a = tf.placeholder(tf.float32,
                                     shape=[batch_size, image_size[0], image_size[1], 3])
        self.fake_b = tf.placeholder(tf.float32,
                                     shape=[batch_size,  image_size[0], image_size[1], 3])

    def build(self):

        A_reader = Reader(self.X_train_file, name='RA',
                          image_size=self.image_size, batch_size=self.batch_size)
        B_reader = Reader(self.Y_train_file, name='RB',
                          image_size=self.image_size, batch_size=self.batch_size)

        real_a = A_reader.feed()
        real_b = B_reader.feed()

        # x = tf.placeholder(dtype = tf.float32,shape=(5,270,480,3))
        # y = tf.placeholder(dtype = tf.float32,shape=(5,270,480,3))

        with tf.device("/gpu:0"):
            fake_b = self.Ga2b(real_a)

            label_b = self.Da2b(real_b)
            logits_a2b = self.Da2b(fake_b)
        with tf.device("/gpu:1"):
            fake_a = self.Gb2a(real_b)

            label_a = self.Db2a(real_a)
            logits_b2a = self.Db2a(fake_a)
   
        cycle_loss = self.cycle_consistency_loss(real_a,real_b,fake_a,fake_b)

        with tf.device("/gpu:0"):
            # a -> b
            a2b_gen_loss = self.generator_loss(logits_a2b)
            Ga2b_loss = a2b_gen_loss + cycle_loss

            # Gradient penalty
            alpha = tf.random_uniform(
                shape=[self.batch_size,1], 
                minval=0.,
                maxval=1.
            )

            differences = self.fake_b - real_b
            interpolates = real_b + (alpha*differences)
            gradients = tf.gradients(self.Da2b(interpolates), [interpolates])[0]
            slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
            gradient_penalty = tf.reduce_mean((slopes-1.)**2)

            Da2b_loss = self.discriminator_loss(self.Da2b(self.fake_b),label_b,gradient_penalty)
        
        with tf.device("/gpu:1"):
            # b -> a
            b2a_gen_loss = self.generator_loss(logits_b2a)
            Gb2a_loss = b2a_gen_loss + cycle_loss

            # Gradient penalty
            alpha = tf.random_uniform(
                shape=[self.batch_size,1], 
                minval=0.,
                maxval=1.
            )

            differences = self.fake_a - real_a
            interpolates = real_a + (alpha*differences)
            gradients = tf.gradients(self.Db2a(interpolates), [interpolates])[0]
            slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
            gradient_penalty = tf.reduce_mean((slopes-1.)**2)

            Db2a_loss = self.discriminator_loss(self.Db2a(self.fake_a),label_a,gradient_penalty)

        # summary
        tf.summary.histogram('Da2b/true', label_b)
        tf.summary.histogram('Da2b/fake', self.Da2b(self.fake_b))
        tf.summary.histogram('Db2a/true', label_a)
        tf.summary.histogram('Db2a/fake', self.Db2a(self.fake_a))

        tf.summary.scalar('loss/a2b_gen_loss', a2b_gen_loss)
        tf.summary.scalar('loss/Da2b_loss', Da2b_loss)
        tf.summary.scalar('loss/b2a_gen_loss', b2a_gen_loss)
        tf.summary.scalar('loss/Db2a_loss', Db2a_loss)
        tf.summary.scalar('loss/cycle', cycle_loss)
        tf.summary.scalar('loss/Ga2b_loss', Ga2b_loss)
        tf.summary.scalar('loss/Gb2a_loss', Gb2a_loss)

        tf.summary.image('a2b/real', utils.batch_convert2int(real_a))
        tf.summary.image('a2b/generated', utils.batch_convert2int(fake_b))
        tf.summary.image('a2b/reconstruction',
                         utils.batch_convert2int(self.Gb2a(fake_b)))

        tf.summary.image('b2a/real', utils.batch_convert2int(real_b))
        tf.summary.image('b2a/generated', utils.batch_convert2int(fake_a))
        tf.summary.image('b2a/reconstruction',
                         utils.batch_convert2int(self.Ga2b(fake_a)))

        return Ga2b_loss, Da2b_loss, Gb2a_loss, Db2a_loss, fake_a, fake_b,real_a,real_b

    def optimize(self, Ga2b_loss, Da2b_loss, Gb2a_loss, Db2a_loss):
        def make_optimizer(loss, variables, name='Adam',optimizer = self.optimizer,starter_learning_rate = self.learning_rate):
            """ Adam optimizer with learning rate 0.0002 for the first 100k steps (~100 epochs)
                and a linearly decaying rate that goes to zero over the next 100k steps
            """
            global_step = tf.Variable(0, trainable=False)
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
            Ga2b_optimizer = make_optimizer(Ga2b_loss, self.Ga2b.variables, name='%s_Ga2b' % self.optimizer)
            Da2b_optimizer = make_optimizer(
                Da2b_loss, self.Da2b.variables, name='%s_Da2b' % self.optimizer)

        with tf.device("/gpu:1"):
            Gb2a_optimizer = make_optimizer(Gb2a_loss, self.Gb2a.variables, name='%s_Gb2a' % self.optimizer)
            Db2a_optimizer = make_optimizer(
                Db2a_loss, self.Db2a.variables, name='%s_Db2a' % self.optimizer)

        de = [Ga2b_optimizer, Da2b_optimizer, Gb2a_optimizer, Db2a_optimizer]

        with tf.control_dependencies(de):
            return tf.no_op(name='optimizers')

    def discriminator_loss(self,logits,label,gradient_penalty):
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
                tf.squared_difference(label, REAL_LABEL))
            error_fake = tf.reduce_mean(tf.square(logits))
            loss = (error_real + error_fake) / 2
        elif self.lossfunc == 'sigmoid_cross_entropy_with_logits':
            # use cross entropy
                 
            error_real = tf.nn.sigmoid_cross_entropy_with_logits(logits=label,labels=tf.ones_like(label))
            error_real = tf.reduce_mean(error_real)
            error_fake = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits,labels=tf.zeros_like(logits))
            error_fake = tf.reduce_mean(error_fake)
            loss = (error_real + error_fake) / 2
        elif self.lossfunc == 'wgan':
            loss = tf.reduce_mean(logits) - tf.reduce_mean(label)

            # Gradient penalty
            
            loss += LAMBDA*gradient_penalty
        return loss

    def generator_loss(self, logits):
        """  fool discriminator into believing that G(x) is real
        """
        if  self.lossfunc == 'use_lsgan':
            # use mean squared error
            loss = tf.reduce_mean(tf.squared_difference(logits, REAL_LABEL))
        elif self.lossfunc == 'sigmoid_cross_entropy_with_logits':
            # heuristic, non-saturating loss
            # loss = tf.sigmoid(D(fake_y))
            # loss = -tf.reduce_mean(ops.safe_log(loss)) / 2
            
            loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits,labels=tf.ones_like(logits))
            loss = tf.reduce_mean(loss)
        elif self.lossfunc == 'wgan':
            loss = -tf.reduce_mean(logits)
        return loss

    def cycle_consistency_loss(self, ra, rb, fa, fb):
        """ cycle consistency loss (L1 norm)
        """
        with tf.device("/gpu:0"):
            backward_loss = tf.reduce_mean(tf.abs(self.Ga2b(fa) - rb))
        with tf.device("/gpu:1"):
            forward_loss = tf.reduce_mean(tf.abs(self.Gb2a(fb) - ra))
            loss = self.lambda1 * forward_loss + self.lambda2 * backward_loss
        return loss
