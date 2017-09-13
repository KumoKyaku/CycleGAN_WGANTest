import tensorflow as tf
import utils
import os

class Reader():
    def __init__(self, path, image_size=(270, 480),
        min_queue_examples=1000, batch_size=16, num_threads=8, name=''):
        """
        Args:
            tfrecords_file: string, tfrecords file path
            min_queue_examples: integer, minimum number of samples to retain in the queue that provides of batches of examples
            batch_size: integer, number of images per batch
            num_threads: integer, number of preprocess threads
        """
        self.path = path
        self.image_size = image_size
        self.min_queue_examples = min_queue_examples
        self.batch_size = batch_size
        self.num_threads = num_threads
        self.reader = tf.WholeFileReader()
        self.name = name

    def feed(self):
        """
        Returns:
        images: 4D tensor [batch_size, image_width, image_height, image_depth]
        """

        fileNames = [os.path.join(self.path,f) for f in os.listdir(self.path)]
        print('%s hava %d pic'%(self.name,len(fileNames)))
        filename_queue = tf.train.string_input_producer(fileNames)

        name, imgbytes = self.reader.read(filename_queue)
            
        image = tf.image.decode_jpeg(imgbytes, channels=3)
        
        image = self._preprocess(image)

        image = tf.image.random_flip_left_right(image)

        images = tf.train.shuffle_batch(
                [image], batch_size=self.batch_size, num_threads=self.num_threads,
                capacity=self.min_queue_examples + 3*self.batch_size,
                min_after_dequeue=self.min_queue_examples
            )

        return images

    def _preprocess(self, image):
        image = utils.convert2float(image)
        image.set_shape([self.image_size[0], self.image_size[1], 3])
        return image

def test_reader():

    with tf.Graph().as_default():
        reader1 = Reader('picF')
        reader2 = Reader('picG')
        images_op1 = reader1.feed()
        images_op2 = reader2.feed()
        print(images_op1)
        
        sess = tf.Session()
        init = tf.global_variables_initializer()
        sess.run( tf.local_variables_initializer())
        sess.run(init)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        try:
            step = 0
            while not coord.should_stop():
                batch_images1, batch_images2 = sess.run([images_op1, images_op2])
                # print("image shape: {}".format(batch_images1))
                # print("image shape: {}".format(batch_images2))
                print("="*10)
                print(step)
                step += 1
        except KeyboardInterrupt:
            print('Interrupted')
            coord.request_stop()
        except Exception as e:
            coord.request_stop(e)
        finally:
            # When done, ask the threads to stop.
            coord.request_stop()
            coord.join(threads)

if __name__ == '__main__':
    test_reader()
