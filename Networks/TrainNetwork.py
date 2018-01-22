import tensorflow as tf
from Networks.Network import Model
from helper import *

class TrainNetwork:

    def __init__(self):
        print("\n\n1. Initializing Train Network...")
        self._mae = 10 # Random value since no max_int in python3
        self._rms = 10 # Random value since no max_int in python3


    def _post_init(self, config, info, train_data):
        print("2. Post Initializing Trainer Configuration....")
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
        if not find_dir(config['save_model']['path']):
            make_dir(config['save_model']['path'])

        save_path = saver.save(sess, config['save_model']['path']+config['save_model']['name'])
        print("Model saved in file: %s" % save_path)


    def train(self, config, info, train_data):
        # Post initilization
        self._post_init(config, info, train_data)
        # Creating Model
        model = Model(config)
        with tf.Graph().as_default():
            train_indices = tf.placeholder(tf.int32, name='train_indices')
            train_ratings = tf.placeholder(tf.float32, name='train_ratings')
            shape = [self._batch_size, self._input_nuerons]
            # Actual Train Input
            X = tf.sparse_to_dense(train_indices, shape, train_ratings)
            # Noisy Train Input
            noise_X = self._noisyInput(config['noiseRatio'],
                                       train_indices, train_ratings, shape)

            global_step = tf.Variable(0, name='global_step', trainable=False)
            # Model Inference
            Y_out = model.inference(self._input_nuerons,
                                    self._output_nuerons,
                                    X)

            loss_op = model.loss_noisy(noise_X, X, Y_out)
            train_op = model.train_gradient(loss_op, global_step)
            mae_op, rms_op = model.error(X, Y_out)

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

                        _ = sess.run(train_op,
                                     {train_indices: indices,
                                      train_ratings: ratings})

                        counter += 1
                        if counter % 1000 == 0:
                            curr_mae, curr_rms = sess.run([mae_op, rms_op],
                                                  {train_indices: indices,
                                                   train_ratings: ratings})

                            print("Ater {} steps of training => MAE: {}, RMS: {}".format(counter, curr_mae, curr_rms))
                            self._mae = curr_mae if curr_mae < self._mae else self._mae
                            self._rms = curr_rms if curr_rms < self._rms else self._rms

                    self._saveCheckpoint(config, saver, sess)
            print("4.Model Trained Succesfully!!!!")

            print("Best Errors for trained model are MAE: {} and RMS: {} ".format(self._mae, self._rms))
