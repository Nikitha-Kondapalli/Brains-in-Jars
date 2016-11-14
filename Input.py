import tensorflow as tf
import os

from six.moves import xrange  

IMAGE_SIZE = 32 #resizing image from 48x48 to 32x32.

NUM_CLASSES = 7
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 28709 
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 3589 


def read_file(filename_queue):
  
  class FER2013Record(object):
    pass
  result = FER2013Record()

  
  result.height = 48
  result.width = 48
  result.depth = 1
  reader = tf.TextLineReader(skip_header_lines=0)
  key, value = reader.read(filename_queue)


  lst=[]
  for i in range(2305):
    lst.append([0])


  record_defaults = lst
  l = []
  l.extend(tf.decode_csv(value, record_defaults=record_defaults))


  features = tf.cast(tf.pack(l[1:len(l)]),dtype=tf.uint8)
  result.uint8image=tf.reshape(features,[result.height, result.width,result.depth])
  result.label=tf.cast(l[0],dtype=tf.int32)

  return result


def _generate_image_and_label_batch(image, label, min_queue_examples,
                                    batch_size):
  num_preprocess_threads = 16
  images, label_batch = tf.train.shuffle_batch(
      [image, label],
      batch_size=batch_size,
      num_threads=num_preprocess_threads,
      capacity=min_queue_examples + 3 * batch_size,
      min_after_dequeue=min_queue_examples)

  return images, tf.reshape(label_batch, [batch_size])


def distorted_inputs(data_dir, batch_size):
  filenames = [os.path.join(data_dir, 'train_batch.csv')]

  for f in filenames:
    if not tf.gfile.Exists(f):
      raise ValueError('Failed to find file: ' + f)

  filename_queue = tf.train.string_input_producer(filenames)

  read_input = read_file(filename_queue)
  reshaped_image = tf.cast(read_input.uint8image, tf.float32)

  height = IMAGE_SIZE
  width = IMAGE_SIZE

  distorted_image = tf.image.resize_image_with_crop_or_pad(reshaped_image, width, height) # 32 x 32 cropping.

  distorted_image = tf.image.random_brightness(distorted_image, max_delta=63)
  distorted_image = tf.image.random_contrast(distorted_image, lower=0.2, upper=1.8)

  float_image = tf.image.per_image_whitening(distorted_image)

  min_fraction_of_examples_in_queue = 0.4
  min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN *
                           min_fraction_of_examples_in_queue)
  print ('Filling queue with %d FER2013 images before starting to train. '
         'This might take a few minutes.' % min_queue_examples)

  return _generate_image_and_label_batch(float_image, read_input.label,
                                         min_queue_examples, batch_size)


def inputs(eval_data, data_dir, batch_size):
  if not eval_data:
    filenames = [os.path.join(data_dir, 'train_batch.csv')]
    num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
  else:
    filenames = [os.path.join(data_dir, 'test_batch.csv')]
    num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_EVAL

  print("Reading file:",filenames)

  for f in filenames:
    if not tf.gfile.Exists(f):
      raise ValueError('Failed to find file: ' + f)

  filename_queue = tf.train.string_input_producer(filenames)

  read_input = read_file(filename_queue)
  reshaped_image = tf.cast(read_input.uint8image, tf.float32)

  height = IMAGE_SIZE
  width = IMAGE_SIZE

  resized_image = tf.image.resize_image_with_crop_or_pad(reshaped_image,
                                                         width, height)

  float_image = tf.image.per_image_whitening(resized_image)

  min_fraction_of_examples_in_queue = 0.4
  min_queue_examples = int(num_examples_per_epoch *
                           min_fraction_of_examples_in_queue)

  return _generate_image_and_label_batch(float_image, read_input.label,
                                         min_queue_examples, batch_size)
