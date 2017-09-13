from tfImport import *
from generator import Generator
from discriminator import Discriminator
from model import CycleGAN

sess = tf.InteractiveSession()

# Reads an image from a file, decodes it into a dense tensor, and resizes it
# to a fixed shape.
def _parse_function(filename):
    image_string = tf.read_file(os.path.join('picG',filename))
    image_decoded = tf.image.decode_jpeg(image_string)
    #image_resized = tf.image.resize_images(image_decoded, [28, 28])
    return image_decoded,filename

# A vector of filenames.
filenames = tf.constant([f for f in os.listdir('picG')])

dataset = tf.contrib.data.Dataset.from_tensor_slices(filenames)
dataset = dataset.map(_parse_function)
dataset = dataset.shuffle(buffer_size=3)
dataset = dataset.repeat(2)
it = dataset.make_initializable_iterator()

sess.run(it.initializer)

nextimg = it.get_next()

i = 0
while True:
    try:
        img = tf.image.encode_jpeg(tf.cast(nextimg,dtype=tf.uint8))
        img = sess.run(img)
        with tf.gfile.GFile(os.path.join('out','%d.jpg'%i),'wb') as target:
            target.write(img)
        i = i + 1
    except Exception as identifier:
        print(identifier)
        break