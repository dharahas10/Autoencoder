import tensorflow as tf
from new_Network import Model
from helper import *

class TrainNetwork:

    def __init__(self):
        print("\n\n1. Initializing Train Network")
        self._mae = 10 # Random value since no max_int in python3
        self._rms = 10 # Random value since no max_int in python3
        self._mae_list = []
        self._rms_list = []


    def _post_init(self, config, info, train_data):
        print("2. Post Initializing Trainer")
        if config['type'] == 'U':
            self._input_nuerons = info['nV']
            self._train = train_data['U']['data']

        else:
            self._input_nuerons = info['nU']
            self._train = train_data['V']['data']

        self._output_nuerons = self._input_nuerons
        self._batch_size = config['batch_size']
        self._epochs = config['epochs']


    def _noisyInput(self, noiseRatio, indices, values, shape):
        with tf.variable_scope("Corrupt_Input") as scope:
            rand_val = tf.random_uniform(tf.shape(values))
            noise_ratio = tf.constant(noiseRatio, tf.float32,
                                      name="nosie_ratio")
            rand_val_noiseRatio = tf.less_equal(rand_val, noise_ratio)
            noise_idx = tf.cast(rand_val_noiseRatio, tf.float32)
            noise_val = tf.multiply(values, noise_idx)
            noise_X = tf.sparse_to_dense(indices, shape, noise_val, name=scope.name)
            return noise_X


    def _loadCheckpointSaver(self, config, saver, sess):

        if find_dir(config['save_model']['path']):
            # Found saved model
            print("Restoring From the previously Found Model .......")
            saver.restore(sess, config['save_model']['path']+config['save_model']['name'])
            print("Previous Model Found and restored Succesfully")
        else:
            print("No previously saved model found")


    def _saveCheckpoint(self, config, saver, sess):
        # Save the variables to disk.
        save_path = saver.save(sess, config['save_model']['path']+config['save_model']['name'])
        print("Model saved in file: %s" % save_path)


    def _error_parts(self, l_out, output):
        with tf.variable_scope("MAE_RMS") as scope:
            nonzero_matrix = tf.cast(
                tf.not_equal(output, tf.constant(0, tf.float32)),
                tf.float32
            )

            l_out_nonzero = tf.multiply(l_out, nonzero_matrix)
            nonzero_count = tf.reduce_sum(nonzero_matrix)
            mae = tf.reduce_sum(
                        tf.abs(
                            tf.subtract(l_out_nonzero, output)))

            rms = tf.reduce_sum(tf.square(
                        tf.subtract(l_out_nonzero, output)))

            return mae, rms, nonzero_count

    def error(self, config, info, test_data):
        if config['type'] == 'U':
            self._test = test_data['U']['data']

        else:
            self._test = test_data['V']['data']

        model = Model(config)
        with tf.Graph().as_default():
            test_indices = tf.placeholder(tf.int32, name='indices')
            test_ratings = tf.placeholder(tf.float32, name='ratings')
            shape = [self._batch_size, self._input_nuerons]

            X = tf.sparse_to_dense(test_indices, shape, test_ratings)

            global_step = tf.Variable(0, name='global_step', trainable=False)

            l_out = model.inference(self._input_nuerons,
                                    self._output_nuerons,
                                    X)

            mae_op, rms_op, nonzero_count= self._error_parts(l_out, X)

            init = tf.global_variables_initializer()

            saver = tf.train.Saver()

            final_mae = 0
            final_rms = 0
            total_ratings = 0
            with tf.Session() as sess:
                sess.run(init)

                self._loadCheckpointSaver(config, saver, sess)

                print("3. Starting Evaluation....")


                for indices, ratings in iterate_mini_batch(self._test,
                                                          self._batch_size):

                    curr_mae, curr_rms, curr_nonzero_count = sess.run([mae_op, rms_op, nonzero_count],
                                            {test_indices: indices,
                                             test_ratings: ratings})

                    final_mae += curr_mae
                    final_rms += curr_rms
                    total_ratings += len(ratings)


                final_mae = (final_mae*2)/total_ratings
                final_rms = (final_rms*2)/total_ratings

                print("Final Errors are MAE: {}, RMS: {}".format(final_mae, final_rms))



            print("Completed Succesfully!!!!")


    def error_multi(self, config, info, test_data):
        if config['type'] == 'U':
            self._test = test_data['U']['data']

        else:
            self._test = test_data['V']['data']

        model = Model(config)
        with tf.Graph().as_default():

            test_multi_indices = tf.placeholder(tf.int32, name="multi_indices")
            test_multi_ratings = tf.placeholder(tf.float32, name='multi_ratings')
            multi_shape = [self._batch_size, self._input_nuerons*self._n_multi_ratings]

            output_indices = tf.placeholder(tf.int32, name='indices')
            output_ratings = tf.placeholder(tf.float32, name='ratings')
            output_shape = [self._batch_size, self._input_nuerons]

            X = tf.sparse_to_dense(test_multi_indices,
                                   multi_shape,
                                   test_multi_ratings)
            X = tf.random_shuffle(X)

            output = tf.sparse_to_dense(output_indices, output_shape, output_ratings)
            output = tf.random_shuffle(output)

            global_step = tf.Variable(0, name='global_step', trainable=False)

            l_out = model.inference_multi(self._n_multi_ratings,
                                          self._input_nuerons,
                                          self._output_nuerons, X)

            mae_op, rms_op, nonzero_count = self._error_parts(l_out, output)

            init = tf.global_variables_initializer()

            saver = tf.train.Saver()

            final_mae = 0
            final_rms = 0
            total_ratings = 0
            with tf.Session() as sess:
                sess.run(init)

                self._loadCheckpointSaver(config, saver, sess)

                print("3. Starting Evaluation....")


                for multi_indices, multi_ratings, indices, ratings in iterate_mini_batch_multi(self._test,
                                                 self._batch_size,
                                                 self._n_multi_ratings):

                    curr_mae_op, curr_rms_op, curr_nonzero_count = sess.run([mae_op, rms_op, nonzero_count],
                                            {
                                                test_multi_indices: multi_indices,
                                                test_multi_ratings: multi_ratings,
                                                output_indices: indices,
                                                output_ratings: ratings
                                            })

                    final_mae += curr_mae
                    final_rms += curr_rms
                    total_ratings += len(ratings)


                final_mae = (final_mae*2)/total_ratings
                final_rms = (final_rms*2)/total_ratings

                print("Final Errors are MAE: {}, RMS: {}".format(final_mae, final_rms))



            print("Completed Succesfully!!!!")


    def train(self, config, info, train_data):
        # Post initilization
        self._post_init(config, info, train_data)
        # Creating Model
        model = Model(config)
        with tf.Graph().as_default():
            train_indices = tf.placeholder(tf.int32, name='indices')
            train_ratings = tf.placeholder(tf.float32, name='ratings')
            shape = [self._batch_size, self._input_nuerons]

            X = tf.sparse_to_dense(train_indices, shape, train_ratings)
            # X = tf.random_shuffle(X)

            noise_X = self._noisyInput(config['noiseRatio'], train_indices, train_ratings, shape)

            global_step = tf.Variable(0, name='global_step', trainable=False)

            l_out = model.inference(self._input_nuerons,
                                    self._output_nuerons,
                                    X)

            loss_op = model.loss_noisy(X, noise_X, l_out)
            train_op = model.train(loss_op, global_step)
            mae_op, rms_op = model.error(l_out, X)

            init = tf.global_variables_initializer()

            saver = tf.train.Saver()

            counter = 0
            with tf.Session() as sess:
                sess.run(init)

                self._loadCheckpointSaver(config, saver, sess)

                print("3. Starting Training....")
                for epoch in range(self._epochs):
                    print("\n Current Running Epoch is {}/{}"
                          .format(epoch+1, self._epochs))
                    for indices, ratings in iterate_mini_batch(self._train,
                                                              self._batch_size):

                        _, curr_loss = sess.run([train_op, loss_op],
                                                {train_indices: indices,
                                                 train_ratings: ratings})

                        counter += 1
                        if counter % 1000 == 0:
                            curr_mae, curr_rms = sess.run([mae_op, rms_op],
                                                  {train_indices: indices,
                                                   train_ratings: ratings})
                            print("MAE: {}, RMS: {}".format(curr_mae, curr_rms))


                    self._saveCheckpoint(config, saver, sess)
            print("4.Trained Succesfully!!!!")



    def train_multi(self, config, info, train_data):
        # Post Initializing
        self._post_init(config, info, train_data)

        self._n_multi_ratings = config['nMultiRatings']

        model = Model(config)
        with tf.Graph().as_default():

            train_multi_indices = tf.placeholder(tf.int32, name="multi_indices")
            train_multi_ratings = tf.placeholder(tf.float32, name='multi_ratings')
            multi_shape = [self._batch_size, self._input_nuerons*self._n_multi_ratings]

            output_indices = tf.placeholder(tf.int32, name='indices')
            output_ratings = tf.placeholder(tf.float32, name='ratings')
            output_shape = [self._batch_size, self._input_nuerons]

            X = tf.sparse_to_dense(train_multi_indices,
                                   multi_shape,
                                   train_multi_ratings)
            X = tf.random_shuffle(X)

            output = tf.sparse_to_dense(output_indices, output_shape, output_ratings)
            output = tf.random_shuffle(output)

            global_step = tf.Variable(0, name='global_step', trainable=False)

            l_out = model.inference_multi(self._n_multi_ratings,
                                          self._input_nuerons,
                                          self._output_nuerons, X)

            loss_op = model.loss_multi(l_out, output)
            train_op = model.train(loss_op, global_step)

            mae_op, rms_op = model.error_multi(l_out, output)

            init = tf.global_variables_initializer()
            saver = tf.train.Saver()
            counter = 0
            with tf.Session() as sess:
                sess.run(init)
                self._loadCheckpointSaver(config, saver, sess)

                for epoch in range(self._epochs):
                    print("Current Training at Epoch : {}/{}".format(epoch+1, self._epochs))
                    for multi_indices, multi_ratings, indices, ratings in iterate_mini_batch_multi(self._train,
                                                     self._batch_size,
                                                     self._n_multi_ratings):

                        _, curr_loss = sess.run([train_op, loss_op],
                                                {
                                                    train_multi_indices: multi_indices,
                                                    train_multi_ratings: multi_ratings,
                                                    output_indices: indices,
                                                    output_ratings: ratings
                                                })
                        counter += 1
                        if counter % 1 == 0:
                            curr_mae, curr_rms = sess.run([mae_op, rms_op],
                                                {
                                                    train_multi_indices: multi_indices,
                                                    train_multi_ratings: multi_ratings,
                                                    output_indices: indices,
                                                    output_ratings: ratings
                                                })
                            print("MAE: {}, RMS: {}".format(curr_mae, curr_rms))

                    self._saveCheckpoint(config, saver,sess)
            print("Trained Sucessfully")
