from tf_utils import *
import copy

class TFModel(object):

    def __init__(self, dataset, inference_fn, mode, batch_size, image_size, need_validation=False, validation_batch_size=64):
        self.dataset = dataset
        self.mode = mode
        preprocessor_type = dataset.preprocessor_type()
        self.data_preprocessor = preprocessor_type(mode, dataset, image_size=image_size, batch_size=batch_size, num_preprocess_threads=8)
        self.graph = tf.Graph()
        self.batch_size = batch_size
        self.inference_fn = inference_fn
        self.need_initialization = True

        if need_validation:
            assert mode != 'eval'
            self.support_validation = True
            self.validation_has_initialized = False
            self.validation_batch_size = validation_batch_size
            val_dataset = copy.deepcopy(dataset)
            val_dataset.subset = 'validation'
            self.num_validation_examples = val_dataset.num_examples_per_epoch()
            self.val_data_preprocessor = preprocessor_type('eval', val_dataset,
                image_size=image_size, batch_size=validation_batch_size, num_preprocess_threads=4, num_readers=1)

        with self.graph.as_default():
            # self.var_training = tf.get_variable('var_training', shape=(), dtype=tf.bool, trainable=False)
            self.input_images, self.input_labels = self.data_preprocessor.get_batch_input_tensors()
            # just compile
            self.output = inference_fn(self.input_images)   # for eval only
            config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
            self.sess = tf.Session(config=config)

    def initialize(self):
        with self.graph.as_default():
            init = tf.global_variables_initializer()
            self.sess.run(init)
        # if self.mode == 'train':
        #     self.set_var_training(True)
        # else:
        #     self.set_var_training(False)
        self.need_initialization = False

    def validation_initialize(self):
        assert self.support_validation
        self.validation_has_initialized = True
        validation_images, self.validation_labels = self.val_data_preprocessor.get_batch_input_tensors()
        tf.get_variable_scope().reuse_variables()
        self.validation_output = self.inference_fn(validation_images)

    def _get_variables_by_keyword(self, keyword):
        result = []
        for t in self.get_global_variables():
            if keyword in t.name:
                result.append(t)
        return result

    def get_global_variables(self):
        return self.graph.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)

    def get_kernel_tensors(self):
        return self._get_variables_by_keyword('kernel')

    def get_bias_tensors(self):
        return self._get_variables_by_keyword('bias')

    def get_moving_mean_tensors(self):
        return self._get_variables_by_keyword('moving_mean')

    def get_moving_variance_tensors(self):
        return self._get_variables_by_keyword('moving_variance')

    def get_variable_values(self, variables):
        result = {}
        for v in variables:
            value = self.sess.run(v)
            result[v.name] = value
        return result

    def set_variable_values(self, variables, values):
        cnt = 0
        for v in variables:
            if v.name in values:
                self.set_value(v, values[v.name])
                cnt += 1
        print('set values for {} variables'.format(cnt))





    def get_pred_loss_and_acc(self, image_batch, label_batch):
        tf.get_variable_scope().reuse_variables()
        logits = self.inference_fn(image_batch)
        pred = tf.argmax(logits, 1)
        equ = tf.equal(tf.cast(label_batch, tf.int32), tf.cast(pred, tf.int32))
        acc_op = tf.reduce_mean(tf.cast(equ, tf.float32))

        sparse_labels = tf.reshape(label_batch, [self.batch_size, 1])
        indices = tf.reshape(tf.range(self.batch_size), [self.batch_size, 1])
        concated = tf.concat(axis=1, values=[indices, sparse_labels])
        num_classes = logits[0].get_shape()[-1].value
        dense_labels = tf.sparse_to_dense(concated,
            [self.batch_size, num_classes],
            1.0, 0.0)
        return tf.losses.softmax_cross_entropy(dense_labels, logits), acc_op

    def get_tower_loss_and_acc(self, scope, image_batch, label_batch, tower_name='tower', l2_factor=5e-4):
        accuracy_loss, acc_op = self.get_pred_loss_and_acc(image_batch, label_batch)
        l2_loss = calculate_l2_loss(tf_extract_kernel_tensors())
        total_loss = tf.add(accuracy_loss, l2_factor * l2_loss, name='total_loss')
        tf.losses.add_loss(l2_loss)
        tf.losses.add_loss(total_loss)
        losses = tf.losses.get_losses(scope=scope)
        for l in losses:
            loss_name = re.sub('%s_[0-9]*/' % tower_name, '', l.op.name)
            tf.summary.scalar(loss_name, l)
        return total_loss, acc_op

    def clear(self):
        self.sess.close()
        tf.reset_default_graph()
        print('model cleared')

    def load_weights_from_np(self, np_file):
        self.np_file = np_file
        if self.need_initialization:
            self.initialize()
        with self.graph.as_default():
            with self.sess.as_default():
                assign_vars_from_np_dict_by_name(self.get_global_variables(), np_file)

    def set_value(self, t, value):
        with self.graph.as_default():
            with self.sess.as_default():
                ph = tf.placeholder(t.dtype, t.shape)
                op = t.assign(ph)
                tf.get_default_session().run(op, feed_dict={ph: value})

    def get_value(self, t):
        return self.sess.run(t)

    # def set_var_training(self, is_training):
    #     op = tf.assign(self.var_training, is_training)
    #     self.sess.run(op)
    #     print('set var_training to ', is_training)


    def __del__(self):
        self.clear()


