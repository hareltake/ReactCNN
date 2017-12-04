
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from abc import abstractmethod

VGG_MEAN = [103.939, 116.779, 123.68]
CIFAR10_MEAN = [125.30690002, 122.95014954, 113.86599731]  #not RGB! already switched!
IMAGENET_MODES = ['eval', 'strong', 'weak']
CIFAR10_MODES = ['train', 'eval']

INPUT_QUEUE_MEMORY_FACTOR=16

class ImagePreprocessor(object):

    def __init__(self, mode, dataset, image_size, batch_size, num_preprocess_threads=4, num_readers=None):
        self.mode = mode
        self.dataset = dataset
        self.image_size = image_size
        self.batch_size = batch_size
        self.num_preprocess_threads = num_preprocess_threads
        self.num_readers = num_readers

    @abstractmethod
    def get_batch_input_tensors(self):
        pass

    @abstractmethod
    def preprocess_fn(self, image_size, image_buffer, bbox, mode, thread_id=0):
        pass



class CIFAR10Preprocessor(ImagePreprocessor):

    def __init__(self, mode, dataset, image_size, batch_size, num_preprocess_threads=4, num_readers=None):
        super(CIFAR10Preprocessor, self).__init__(mode, dataset, image_size, batch_size, num_preprocess_threads, num_readers)
        assert self.mode in CIFAR10_MODES
        if self.num_readers is None:
            if mode == 'eval':
                self.num_readers = 1
            else:
                self.num_readers = 4

    def get_batch_input_tensors(self):
        with tf.name_scope('batch_processing'):
            data_files = self.dataset.data_files()
            if data_files is None:
                raise ValueError('No data files found for this dataset')

            # Create filename_queue
            if self.mode == 'train':
                filename_queue = tf.train.string_input_producer(data_files,
                                                                shuffle=True,
                                                                capacity=16)
            else:
                filename_queue = tf.train.string_input_producer(data_files,
                                                                shuffle=False,
                                                                capacity=1)

            if self.num_preprocess_threads % 4:
                raise ValueError('Please make num_preprocess_threads a multiple '
                                 'of 4 (%d % 4 != 0).', self.num_preprocess_threads)

            if self.num_readers < 1:
                raise ValueError('Please make num_readers at least 1')

            # Approximate number of examples per shard.
            examples_per_shard = 1024
            # Size the random shuffle queue to balance between good global
            # mixing (more examples) and memory use (fewer examples).
            # 1 image uses 299*299*3*4 bytes = 1MB
            # The default input_queue_memory_factor is 16 implying a shuffling queue
            # size: examples_per_shard * 16 * 1MB = 17.6GB
            min_queue_examples = examples_per_shard * INPUT_QUEUE_MEMORY_FACTOR
            if self.mode == 'train':
                examples_queue = tf.RandomShuffleQueue(
                    capacity=min_queue_examples + 3 * self.batch_size,
                    min_after_dequeue=min_queue_examples,
                    dtypes=[tf.string])
            else:
                examples_queue = tf.FIFOQueue(
                    capacity=examples_per_shard + 3 * self.batch_size,
                    dtypes=[tf.string])

            # Create multiple readers to populate the queue of examples.
            if self.num_readers > 1:
                enqueue_ops = []
                for _ in range(self.num_readers):
                    reader = self.dataset.reader()
                    _, value = reader.read(filename_queue)
                    enqueue_ops.append(examples_queue.enqueue([value]))

                tf.train.queue_runner.add_queue_runner(
                    tf.train.queue_runner.QueueRunner(examples_queue, enqueue_ops))
                example_serialized = examples_queue.dequeue()
            else:
                reader = self.dataset.reader()
                _, example_serialized = reader.read(filename_queue)

            images_and_labels = []
            for thread_id in range(self.num_preprocess_threads):
                # Parse a serialized Example proto to extract the image and metadata.
                image, label_index = self.cifar10_parse_example_proto(example_serialized)
                # image = self.cifar10_preprocess(image_buffer)
                images_and_labels.append([image, label_index])

            images, label_index_batch = tf.train.batch_join(
                images_and_labels,
                batch_size=self.batch_size,
                capacity=2 * self.num_preprocess_threads * self.batch_size)

            # Reshape images into these desired dimensions.
            height = self.image_size
            width = self.image_size
            depth = 3

            images = tf.cast(images, tf.float32)
            images = tf.reshape(images, shape=[self.batch_size, height, width, depth])

            # Display the training images in the visualizer.
            tf.summary.image('images', images)

            return images, tf.reshape(label_index_batch, [self.batch_size])


    def cifar10_distort(self, image):
        image = tf.image.resize_image_with_crop_or_pad(image, 38, 38)
        image = tf.random_crop(image, [self.image_size, self.image_size, 3])
        image = tf.image.random_flip_left_right(image)
        return image

    def cifar10_parse_example_proto(self, example_serialized):
        features = tf.parse_single_example(
            example_serialized,
            features={
                'image': tf.FixedLenFeature([], tf.string),
                'label': tf.FixedLenFeature([], tf.int64),
            })
        label = tf.cast(features['label'], dtype=tf.int32)
        image = tf.decode_raw(features['image'], tf.uint8)
        image.set_shape([3 * self.image_size * self.image_size])
        # CHW -> HWC
        image = tf.cast(tf.transpose(tf.reshape(image, [3, self.image_size, self.image_size]), [1, 2, 0]), tf.float32)

        red, green, blue = tf.split(value=image, num_or_size_splits=[1, 1, 1], axis=2)
        assert red.get_shape().as_list() == [32, 32, 1]
        assert green.get_shape().as_list() == [32, 32, 1]
        assert blue.get_shape().as_list() == [32, 32, 1]
        image = tf.concat(values=[
            blue - CIFAR10_MEAN[0],
            green - CIFAR10_MEAN[1],
            red - CIFAR10_MEAN[2],
        ], axis=2)

        if self.mode == 'train':
            print('distort image')
            image = self.cifar10_distort(image)

        # image = tf.subtract(image, tf.constant(np.array(CIFAR10_MEAN, dtype=np.float32), dtype=tf.float32))
        return image, label



    # def parser(self, serialized_example):
    #     """Parses a single tf.Example into image and label tensors."""
    #     # Dimensions of the images in the CIFAR-10 dataset.
    #     # See http://www.cs.toronto.edu/~kriz/cifar.html for a description of the
    #     # input format.
    #     features = tf.parse_single_example(
    #         serialized_example,
    #         features={
    #             'image': tf.FixedLenFeature([], tf.string),
    #             'label': tf.FixedLenFeature([], tf.int64),
    #         })
    #     image = tf.decode_raw(features['image'], tf.uint8)
    #     image.set_shape([3 * self.image_size * self.image_size])
    #
    #     # Reshape from [depth * height * width] to [depth, height, width].
    #     image = tf.cast(
    #         tf.transpose(tf.reshape(image, [3, self.image_size, self.image_size]), [1, 2, 0]),
    #         tf.float32)
    #     label = tf.cast(features['label'], tf.int32)
    #
    #     image = tf.subtract(image, tf.constant(np.array(CIFAR10_MEAN, dtype=np.float32),dtype=tf.float32))
    #
    #     # Custom preprocessing.
    #     if self.mode == 'train':
    #         image = self.distort(image)
    #
    #     return image, label
    #
    # def get_batch_input_tensors(self):
    #     filenames = self.dataset.data_files()
    #     num_examples_per_epoch = self.dataset.num_examples_per_epoch()
    #     tfrecord_dataset = tf.contrib.data.TFRecordDataset(filenames).repeat()
    #     tfrecord_dataset = tfrecord_dataset.map(
    #         self.parser, num_threads=self.batch_size, output_buffer_size=2 * self.batch_size)
    #
    #     if self.mode == 'train':
    #         min_queue_examples = int(num_examples_per_epoch * 0.4)
    #         # Ensure that the capacity is sufficiently large to provide good random
    #         # shuffling.
    #         tfrecord_dataset = tfrecord_dataset.shuffle(buffer_size=min_queue_examples + 3 * self.batch_size)
    #
    #     # Batch it up.
    #     tfrecord_dataset = tfrecord_dataset.batch(self.batch_size)
    #     iterator = tfrecord_dataset.make_one_shot_iterator()
    #     image_batch, label_batch = iterator.get_next()
    #
    #     return image_batch, label_batch
    #
    #





class ImageNetPreprocessor(ImagePreprocessor):

    def __init__(self, mode, dataset, image_size, batch_size, num_preprocess_threads=4, num_readers=None):
        super(ImageNetPreprocessor, self).__init__(mode, dataset, image_size, batch_size, num_preprocess_threads, num_readers)
        assert self.mode in IMAGENET_MODES
        if self.num_readers is None:
            if mode == 'eval':
                self.num_readers = 1
            else:
                self.num_readers = 4

    def get_batch_input_tensors(self):
        if self.mode == 'eval':
            return inputs(self.dataset, batch_size=self.batch_size, image_size=self.image_size, num_preprocess_threads=self.num_preprocess_threads)
        elif self.mode == 'strong':
            return distorted_inputs(self.dataset, batch_size=self.batch_size, image_size=self.image_size, num_readers=self.num_readers)
        elif self.mode == 'weak':
            return inputs_with_weak_augment(self.dataset, batch_size=self.batch_size, image_size=self.image_size, num_readers=self.num_readers)
        else:
            assert False





def inputs(dataset, batch_size, image_size, num_preprocess_threads=4):
    """Generate batches of ImageNet images for evaluation.

    Use this function as the inputs for evaluating a network.

    Note that some (minimal) image preprocessing occurs during evaluation
    including central cropping and resizing of the image to fit the network.

    Args:
      dataset: instance of Dataset class specifying the dataset.
      batch_size: integer, number of examples in batch
      image_size
      num_preprocess_threads: integer, total number of preprocessing threads but
        None defaults to FLAGS.num_preprocess_threads.

    Returns:
      images: Images. 4D tensor of size [batch_size, image_size,
                                         image_size, 3].
      labels: 1-D integer Tensor of [batch_size].
    """

    # Force all input processing onto CPU in order to reserve the GPU for
    # the forward inference and back-propagation.
    with tf.device('/cpu:0'):
        images, labels = batch_inputs(
            dataset, batch_size=batch_size, image_size=image_size, train=False,
            num_preprocess_threads=num_preprocess_threads,
            num_readers=1, mode='eval')
    return images, labels

def distorted_inputs(dataset, image_size, batch_size, num_readers):
    with tf.device('/cpu:0'):
        images, labels = batch_inputs(
            dataset, batch_size=batch_size, image_size=image_size, train=True,
            num_readers=num_readers, mode='strong')
    return images, labels

def inputs_with_weak_augment(dataset, image_size, batch_size, num_readers):
    with tf.device('/cpu:0'):
        images, labels = batch_inputs(
            dataset, batch_size=batch_size, image_size=image_size, train=True,
            num_readers=num_readers, mode='weak')
    return images, labels


def decode_jpeg(image_buffer, scope=None):
    """Decode a JPEG string into one 3-D float image Tensor.

    Args:
      image_buffer: scalar string Tensor.
      scope: Optional scope for name_scope.
    Returns:
      3-D float Tensor with values ranging from [0, 1).
    """
    with tf.name_scope(values=[image_buffer], name=scope,
            default_name='decode_jpeg'):
        # Decode the string as an RGB JPEG.
        # Note that the resulting image contains an unknown height and width
        # that is set dynamically by decode_jpeg. In other words, the height
        # and width of image is unknown at compile-time.
        image = tf.image.decode_jpeg(image_buffer, channels=3)

        # After this point, all image pixels reside in [0,1)
        # until the very end, when they're rescaled to (-1, 1).  The various
        # adjust_* ops all require this range for dtype float.
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        return image


def distort_color(image, thread_id=0, scope=None):
    """Distort the color of the image.

    Each color distortion is non-commutative and thus ordering of the color ops
    matters. Ideally we would randomly permute the ordering of the color ops.
    Rather then adding that level of complication, we select a distinct ordering
    of color ops for each preprocessing thread.

    Args:
      image: Tensor containing single image.
      thread_id: preprocessing thread ID.
      scope: Optional scope for name_scope.
    Returns:
      color-distorted image
    """
    with tf.name_scope(values=[image], name=scope, default_name='distort_color'):
        color_ordering = thread_id % 2

        if color_ordering == 0:
            image = tf.image.random_brightness(image, max_delta=32. / 255.)
            image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
            image = tf.image.random_hue(image, max_delta=0.2)
            image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        elif color_ordering == 1:
            image = tf.image.random_brightness(image, max_delta=32. / 255.)
            image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
            image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
            image = tf.image.random_hue(image, max_delta=0.2)

        # The random_* ops do not necessarily clamp.
        image = tf.clip_by_value(image, 0.0, 1.0)
        return image


def distort_image(image, height, width, bbox, thread_id=0, scope=None):
    """Distort one image for training a network.

    Distorting images provides a useful technique for augmenting the data
    set during training in order to make the network invariant to aspects
    of the image that do not effect the label.

    Args:
      image: 3-D float Tensor of image
      height: integer
      width: integer
      bbox: 3-D float Tensor of bounding boxes arranged [1, num_boxes, coords]
        where each coordinate is [0, 1) and the coordinates are arranged
        as [ymin, xmin, ymax, xmax].
      thread_id: integer indicating the preprocessing thread.
      scope: Optional scope for name_scope.
    Returns:
      3-D float Tensor of distorted image used for training.
    """
    with tf.name_scope(values=[image, height, width, bbox], name=scope,
            default_name='distort_image'):
        # Each bounding box has shape [1, num_boxes, box coords] and
        # the coordinates are ordered [ymin, xmin, ymax, xmax].

        # Display the bounding box in the first thread only.
        if not thread_id:
            image_with_box = tf.image.draw_bounding_boxes(tf.expand_dims(image, 0),
                bbox)
            tf.summary.image('image_with_bounding_boxes', image_with_box)

            # A large fraction of image datasets contain a human-annotated bounding
            # box delineating the region of the image containing the object of interest.
            # We choose to create a new bounding box for the object which is a randomly
            # distorted version of the human-annotated bounding box that obeys an allowed
            # range of aspect ratios, sizes and overlap with the human-annotated
            # bounding box. If no box is supplied, then we assume the bounding box is
            # the entire image.
        sample_distorted_bounding_box = tf.image.sample_distorted_bounding_box(
            tf.shape(image),
            bounding_boxes=bbox,
            min_object_covered=0.1,
            aspect_ratio_range=[0.75, 1.33],
            area_range=[0.05, 1.0],
            max_attempts=100,
            use_image_if_no_bounding_boxes=True)
        bbox_begin, bbox_size, distort_bbox = sample_distorted_bounding_box
        if not thread_id:
            image_with_distorted_box = tf.image.draw_bounding_boxes(
                tf.expand_dims(image, 0), distort_bbox)
            tf.summary.image('images_with_distorted_bounding_box',
                image_with_distorted_box)

        # Crop the image to the specified bounding box.
        distorted_image = tf.slice(image, bbox_begin, bbox_size)

        # This resizing operation may distort the images because the aspect
        # ratio is not respected. We select a resize method in a round robin
        # fashion based on the thread number.
        # Note that ResizeMethod contains 4 enumerated resizing methods.
        resize_method = thread_id % 4
        distorted_image = tf.image.resize_images(distorted_image, [height, width],
            method=resize_method)
        # Restore the shape since the dynamic slice based upon the bbox_size loses
        # the third dimension.
        distorted_image.set_shape([height, width, 3])
        if not thread_id:
            tf.summary.image('cropped_resized_image',
                tf.expand_dims(distorted_image, 0))

        # Randomly flip the image horizontally.
        distorted_image = tf.image.random_flip_left_right(distorted_image)

        # Randomly distort the colors.
        distorted_image = distort_color(distorted_image, thread_id)

        if not thread_id:
            tf.summary.image('final_distorted_image',
                tf.expand_dims(distorted_image, 0))
        return distorted_image


def eval_image(image, height, width, scope=None):
    """Prepare one image for evaluation.

    Args:
      image: 3-D float Tensor
      height: integer
      width: integer
      scope: Optional scope for name_scope.
    Returns:
      3-D float Tensor of prepared image.
    """
    with tf.name_scope(values=[image, height, width], name=scope,
            default_name='eval_image'):
        # Crop the central region of the image with an area containing 87.5% of
        # the original image.
        image = tf.image.central_crop(image, central_fraction=0.875)

        # Resize the image to the original height and width.
        image = tf.expand_dims(image, 0)
        image = tf.image.resize_bilinear(image, [height, width],
            align_corners=False)
        image = tf.squeeze(image, [0])
        return image

def random_crop_and_flip_image(image, height, width, thread_id=0, scope=None):
    with tf.name_scope(values=[image, height, width], name=scope,
            default_name='distort_image'):

        distorted_image = tf.image.resize_images(image, [255, 255])
        distorted_image = tf.random_crop(distorted_image, [height, width, 3])

        distorted_image.set_shape([height, width, 3])
        if not thread_id:
            tf.summary.image('cropped_resized_image',
                tf.expand_dims(distorted_image, 0))

        # Randomly flip the image horizontally.
        distorted_image = tf.image.random_flip_left_right(distorted_image)
        if not thread_id:
            tf.summary.image('final_distorted_image',
                tf.expand_dims(distorted_image, 0))
        return distorted_image


def image_preprocessing(image_size, image_buffer, bbox, mode, thread_id=0):

    if bbox is None:
        assert mode == 'weak'

    image = decode_jpeg(image_buffer)
    height = image_size
    width = image_size

    if mode == 'strong':
        image = distort_image(image, height, width, bbox, thread_id)
    elif mode == 'eval':
        image = eval_image(image, height, width)
    elif mode == 'weak':
        image = random_crop_and_flip_image(image, height, width, thread_id)

    # Finally, rescale to [-1,1] instead of [0, 1)
    #Shawn: important!
    # image = tf.subtract(image, 0.5)
    # image = tf.multiply(image, 2.0)
    image = tf.multiply(image, 255.)
    red, green, blue = tf.split(value=image, num_or_size_splits = [1,1,1], axis=2)
    assert red.get_shape().as_list() == [224, 224, 1]
    assert green.get_shape().as_list() == [224, 224, 1]
    assert blue.get_shape().as_list() == [224, 224, 1]
    bgr = tf.concat(values=[
        blue - VGG_MEAN[0],
        green - VGG_MEAN[1],
        red - VGG_MEAN[2],
    ], axis=2)
    return bgr


def parse_example_proto(example_serialized):
    """Parses an Example proto containing a training example of an image.

    The output of the build_image_data.py image preprocessing script is a dataset
    containing serialized Example protocol buffers. Each Example proto contains
    the following fields:

      image/height: 462
      image/width: 581
      image/colorspace: 'RGB'
      image/channels: 3
      image/class/label: 615
      image/class/synset: 'n03623198'
      image/class/text: 'knee pad'
      image/object/bbox/xmin: 0.1
      image/object/bbox/xmax: 0.9
      image/object/bbox/ymin: 0.2
      image/object/bbox/ymax: 0.6
      image/object/bbox/label: 615
      image/format: 'JPEG'
      image/filename: 'ILSVRC2012_val_00041207.JPEG'
      image/encoded: <JPEG encoded string>

    Args:
      example_serialized: scalar Tensor tf.string containing a serialized
        Example protocol buffer.

    Returns:
      image_buffer: Tensor tf.string containing the contents of a JPEG file.
      label: Tensor tf.int32 containing the label.
      bbox: 3-D float Tensor of bounding boxes arranged [1, num_boxes, coords]
        where each coordinate is [0, 1) and the coordinates are arranged as
        [ymin, xmin, ymax, xmax].
      text: Tensor tf.string containing the human-readable label.
    """
    # Dense features in Example proto.
    feature_map = {
        'image/encoded': tf.FixedLenFeature([], dtype=tf.string,
            default_value=''),
        'image/class/label': tf.FixedLenFeature([1], dtype=tf.int64,
            default_value=-1),
        'image/class/text': tf.FixedLenFeature([], dtype=tf.string,
            default_value=''),
    }
    sparse_float32 = tf.VarLenFeature(dtype=tf.float32)
    # Sparse features in Example proto.
    feature_map.update(
        {k: sparse_float32 for k in ['image/object/bbox/xmin',
                                     'image/object/bbox/ymin',
                                     'image/object/bbox/xmax',
                                     'image/object/bbox/ymax']})

    features = tf.parse_single_example(example_serialized, feature_map)
    label = tf.cast(features['image/class/label'], dtype=tf.int32)

    xmin = tf.expand_dims(features['image/object/bbox/xmin'].values, 0)
    ymin = tf.expand_dims(features['image/object/bbox/ymin'].values, 0)
    xmax = tf.expand_dims(features['image/object/bbox/xmax'].values, 0)
    ymax = tf.expand_dims(features['image/object/bbox/ymax'].values, 0)

    # Note that we impose an ordering of (y, x) just to make life difficult.
    bbox = tf.concat(axis=0, values=[ymin, xmin, ymax, xmax])

    # Force the variable number of bounding boxes into the shape
    # [1, num_boxes, coords].
    bbox = tf.expand_dims(bbox, 0)
    bbox = tf.transpose(bbox, [0, 2, 1])

    return features['image/encoded'], label, bbox, features['image/class/text']


def batch_inputs(dataset, batch_size, image_size, train, num_preprocess_threads=4,
                 num_readers=1, input_queue_memory_factor=16, mode='strong'):
    """Contruct batches of training or evaluation examples from the image dataset.

    Args:
      dataset: instance of Dataset class specifying the dataset.
        See dataset.py for details.
      batch_size: integer
      train: boolean
      num_preprocess_threads: integer, total number of preprocessing threads
      num_readers: integer, number of parallel readers

    Returns:
      images: 4-D float Tensor of a batch of images
      labels: 1-D integer Tensor of [batch_size].

    Raises:
      ValueError: if data is not found
    """
    with tf.name_scope('batch_processing'):
        data_files = dataset.data_files()
        if data_files is None:
            raise ValueError('No data files found for this dataset')

        # Create filename_queue
        if train:
            filename_queue = tf.train.string_input_producer(data_files,
                shuffle=True,
                capacity=16)
        else:
            filename_queue = tf.train.string_input_producer(data_files,
                shuffle=False,
                capacity=1)

        if num_preprocess_threads % 4:
            raise ValueError('Please make num_preprocess_threads a multiple '
                             'of 4 (%d % 4 != 0).', num_preprocess_threads)

        if num_readers < 1:
            raise ValueError('Please make num_readers at least 1')

        # Approximate number of examples per shard.
        examples_per_shard = 1024
        # Size the random shuffle queue to balance between good global
        # mixing (more examples) and memory use (fewer examples).
        # 1 image uses 299*299*3*4 bytes = 1MB
        # The default input_queue_memory_factor is 16 implying a shuffling queue
        # size: examples_per_shard * 16 * 1MB = 17.6GB
        min_queue_examples = examples_per_shard * input_queue_memory_factor
        if train:
            examples_queue = tf.RandomShuffleQueue(
                capacity=min_queue_examples + 3 * batch_size,
                min_after_dequeue=min_queue_examples,
                dtypes=[tf.string])
        else:
            examples_queue = tf.FIFOQueue(
                capacity=examples_per_shard + 3 * batch_size,
                dtypes=[tf.string])

        # Create multiple readers to populate the queue of examples.
        if num_readers > 1:
            enqueue_ops = []
            for _ in range(num_readers):
                reader = dataset.reader()
                _, value = reader.read(filename_queue)
                enqueue_ops.append(examples_queue.enqueue([value]))

            tf.train.queue_runner.add_queue_runner(
                tf.train.queue_runner.QueueRunner(examples_queue, enqueue_ops))
            example_serialized = examples_queue.dequeue()
        else:
            reader = dataset.reader()
            _, example_serialized = reader.read(filename_queue)

        images_and_labels = []
        for thread_id in range(num_preprocess_threads):
            # Parse a serialized Example proto to extract the image and metadata.
            image_buffer, label_index, bbox, _ = parse_example_proto(
                example_serialized)
            image = image_preprocessing(image_size, image_buffer, bbox, mode, thread_id)
            images_and_labels.append([image, label_index])

        images, label_index_batch = tf.train.batch_join(
            images_and_labels,
            batch_size=batch_size,
            capacity=2 * num_preprocess_threads * batch_size)

        # Reshape images into these desired dimensions.
        height = image_size
        width = image_size
        depth = 3

        images = tf.cast(images, tf.float32)
        images = tf.reshape(images, shape=[batch_size, height, width, depth])

        # Display the training images in the visualizer.
        tf.summary.image('images', images)

        return images, tf.reshape(label_index_batch, [batch_size])
