import tensorflow as tf
from tfm_model import TFModel
import math
from datetime import datetime
import numpy as np
from tf_dataset import CIFAR10Data
from tf_vgg import VFSFullSurveyBuilder
import PIL.Image as Image
from tfm_image_processor import CIFAR10_MEAN
import os
import time
import glob

cifar_mean_array = np.array(CIFAR10_MEAN)

CONFIG_FILE = 'config.cfg'

RECORD_FILE_PATTERN = 'survey_example_{}.csv'

CORR_FILE_PATTERN = 'corr_layer_{}.csv'

IMAGE_FILE_PATTERN = 'image_{}.png'

EXAMPLE_CACHE_SIZE = 10

EXAMPLE_SAVE_EVERY_STEP = 1

CORR_CACHE_SIZE = 1000

CORR_SAVE_EVERY_STEP = 10

SLEEP_SECONDS = 1

class SurveyExampleRecord(object):

    def __init__(self, idx):
        # self.index = idx
        self.layer_filter_mean_output_list = []
        self.image_path = IMAGE_FILE_PATTERN.format(idx)
        self.csv_path = RECORD_FILE_PATTERN.format(idx)
        self.has_saved = False

    # save the input image, label, and mean outputs of every filter in every layer
    def save(self):
        Image.fromarray((self.output_image + cifar_mean_array)[:, :, [2, 1, 0]].astype(np.uint8), mode='RGB').save(self.image_path)
        with open(self.csv_path, 'w') as f:
            print(int(self.label), file=f)
            for array in self.layer_filter_mean_output_list:
                # print(np.array2string(array, precision=5, separator=',')[1:-1].replace(' ','').replace('\n',''), file=f)
                print(format_array(array), file=f)
        self.has_saved = True

    def __del__(self):
        if os.path.exists(self.image_path):
            os.remove(self.image_path)
        if os.path.exists(self.csv_path):
            os.remove(self.csv_path)

def cache_push(cache_list, element, cache_size):
    assert len(cache_list) <= cache_size
    if len(cache_list) == cache_size:
        cache_list.pop(0)
    cache_list.append(element)

def format_array(array):
    result = ''
    if array.ndim == 1:
        for a in array:
            result += '%.4f,' % a
    elif array.ndim == 2:
        for i in range(array.shape[0]):
            line = ''
            for j in range(array.shape[1]):
                line += '%.4f,' % array[i,j]
            result += line[:-1] + '\n'
    else:
        assert False, 'WTF'
    return result[:-1]






# def save_int_list(file, int_list):
#     np.savetxt(file, np.asarray(int_list, dtype=np.int32), fmt='%d', delimiter=',')
#
# def save_float_list(file, float_list):
#     np.savetxt(file, np.asarray(float_list), fmt='%.4e', delimiter=',')
#
# def save_image_list(img_list):
#     for i,img in enumerate(img_list):
#         Image.fromarray((img+cifar_mean_array)[:,:,[2,1,0]].astype(np.uint8), mode='RGB').save(IMAGE_FILE_PATTERN.format(i))

def launch_backend(model, num_examples=10000):

    for file in glob.glob("*.csv"):
        os.remove(file)
    for file in glob.glob("*.png"):
        os.remove(file)

    with model.graph.as_default():
        images, labels = model.input_images, model.input_labels
        output = model.output
        num_survey_layers = len(output)
        print(num_survey_layers, ' layers to survey')
        fetches = []

        for layer_out in output:
            fetches.append(tf.reduce_mean(layer_out, axis=[1, 2]))

        fetches.append(labels)
        fetches.append(images)

        with model.sess as sess:

            coord = tf.train.Coordinator()
            try:
                threads = []
                for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
                    threads.extend(qr.create_threads(sess, coord=coord, daemon=True,
                        start=True))

                num_iter = int(math.ceil(num_examples / model.batch_size))

                step = 0

                print('%s: starting survey.' % (datetime.now()))
                start_time = time.time()

                survey_examples_cache = []
                corr_cache_list = []
                for i in range(num_survey_layers):
                    corr_cache_list.append([])

                while step < num_iter and not coord.should_stop():
                    fet = sess.run(fetches)
                    new_example_record = SurveyExampleRecord(idx=step)
                    for i in range(num_survey_layers):
                        new_example_record.layer_filter_mean_output_list.append(fet[i].ravel()) # only works for batch size 1
                        cache_push(corr_cache_list[i], fet[i], CORR_CACHE_SIZE)
                    new_example_record.output_image = fet[-1][0,:,:,:]
                    new_example_record.label = fet[-2].ravel()[0]
                    cache_push(survey_examples_cache, new_example_record, EXAMPLE_CACHE_SIZE)

                    step += 1
                    if step % 20 == 0:
                        duration = time.time() - start_time
                        sec_per_batch = duration / 20.0
                        examples_per_sec = model.batch_size / sec_per_batch
                        print('%s: [%d batches out of %d] (%.1f examples/sec; %.3f'
                              'sec/batch)' % (datetime.now(), step, num_iter,
                                              examples_per_sec, sec_per_batch))
                        start_time = time.time()

                    if step % EXAMPLE_SAVE_EVERY_STEP == 0:
                        for record in survey_examples_cache:
                            if not record.has_saved:
                                record.save()
                    if step % CORR_SAVE_EVERY_STEP == 0:
                        for layer_idx, corr_cache in enumerate(corr_cache_list):
                            output_array = np.concatenate(corr_cache, axis=0)
                            corr_array = np.corrcoef(output_array, rowvar=False)
                            corr_array = np.exp(20*corr_array)              # mapping function
                            with open(CORR_FILE_PATTERN.format(layer_idx), 'w') as f:
                                print(format_array(corr_array), file=f)
                        print('corr file saved')

                    time.sleep(SLEEP_SECONDS)

                print('finished!')

            except Exception as e:  # pylint: disable=broad-except
                coord.request_stop(e)

            coord.request_stop()
            coord.join(threads, stop_grace_period_secs=10)

if __name__ == '__main__':
    vgg_deps = [64, 64, 128, 128, 256, 256, 256, 512, 512, 512, 512, 512, 512]
    fn = VFSFullSurveyBuilder(training=False).build_full_outs

    target = 'origin_vfs.npy'

    dataset = CIFAR10Data('validation', './')
    model = TFModel(dataset, fn, 'eval', batch_size=1, image_size=32)
    model.load_weights_from_np(target)

    launch_backend(model, num_examples=10000)