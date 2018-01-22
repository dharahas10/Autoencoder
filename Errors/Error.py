import tensorflow as tf
from Networks.Network import Model
from helper import *


class Error:

    def __init__(self):
        self._mae = 0
        self._rms = 0
        self._count = 0


    def _post_init(self, config, info, train_data):
        print("2. Post Initializing Error Configuration....")
        if config['type'] == 'U':
            self._input_nuerons = info['nV']
            self._train = train_data['U']['data']
            self._test = test_data['U']['data']

        else:
            self._input_nuerons = info['nU']
            self._train = train_data['V']['data']
            self._test = test_data['V']['data']

        self._output_nuerons = self._input_nuerons
        self._batch_size = config['batch_size']


    def _loadCheckpointSaver(self, config, saver, sess):

        if find_dir(config['save_model']['path']):
            # Found saved model
            print("Restoring From the previously Found Model .......")
            saver.restore(sess, config['save_model']['path']+config['save_model']['name'])
            print("Previous Model Found and restored Succesfully")
            return True
        else:
            print("No previously saved model found Please Train and save the model first")
            return False



    def _error_mini_batch(self, test_Y, predict_Y):
        with tf.variable_scope("error_mini_batch") as scope:
            nonzero_mat = tf.cast(
                tf.not_equal(test_Y, tf.constant(0, tf.float32)),
                tf.float32
            )
            predict_Y_nonzero = tf.multiply(predict_Y, nonzero_mat)

            with tf.variable_scope("mae") as sub_scope:
                mae = tf.reduce_sum(
                    tf.abs(
                        tf.subtract(predict_Y_nonzero, test_Y)
                    )
                , name=sub_scope.name)

            with tf.variable_scope("rms") as sub_scope:
                rms = tf.reduce_sum(
                    tf.square(
                        tf.subtract(predict_Y_nonzero, test_Y)
                    )
                , name=sub_scope.name)

        return mae, rms


    def measure(self, config, info, train_data, test_data):
        # Post Initializing
        self._post_init(config, info, train_data, test_data)

        model = Model(config)

        with tf.Graph().as_default():

            train_indices = tf.placeholder(tf.int32, name='train_indices')
            train_ratings = tf.placeholder(tf.float32, name='train_ratings')
            train_shape = [self._batch_size, self._input_nuerons]

            test_indices = tf.placeholder(tf.int32, name='test_indices')
            test_ratings = tf.placeholder(tf.float32, name='test_ratings')
            test_shape = [self._batch_size, self._input_nuerons]
            # Actual Train Input
            X = tf.sparse_to_dense(train_indices, train_shape, train_ratings)
            test_Y = tf.sparse_to_dense(test_indices, test_shape, test_ratings)

            Y_out = model.inference(self._input_nuerons,
                                    self._output_nuerons,
                                    X)

            mae_op, rms_op = self._error_mini_batch(test_Y, Y_out)

            init = tf.global_variables_initializer()
            saver = tf.train.Saver()

            with tf.Session() as sess:
                sess.run(init)

                if self._loadCheckpointSaver(config, saver, sess):
                    for training_indices, training_ratings,
                        testing_indices, testing_ratings in
                        iterate_test_mini_batch(self._train,
                                                self._test,
                                                self._batch_size):

                        curr_mae, curr_rms = sess.run([mae_op, rms_op],
                                                      {train_indices: training_indices,
                                                       train_ratings: training_ratings,
                                                       test_indices: testing_indices,
                                                       test_ratings: testing_ratings})

                        self._mae += curr_mae
                        self._rms += curr_rms
                        self._count += len(testing_ratings)

        self._mae = (self._mae*2)/self._count
        self._rms = (self._rms*2)/self._count


        print("Test Errors for trained model are MAE: {} and RMS: {} ".format(self._mae, self._rms))
