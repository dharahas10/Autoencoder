import tensorflow as tf
import Network as NN
import os
import pickle

class TrainNetwork:

    def __init__(self):
        self._mae = 0
        self._rms = 0
        self._mae_list = []
        self._rms_list = []

    def find_model(self, name, path):
        for root, dirs, files in os.walk(path):
            if name in files:
                return True

    def error_fn(self, predict, size, batch_size):
        # print([batch_size, size])
        indices = tf.placeholder(tf.int32, name="indices")
        ratings = tf.placeholder(tf.float32, name="ratings")
        # predict = tf.placeholder(tf.float32, name = "predict")
        test = tf.sparse_to_dense(indices, [batch_size, size], ratings)

        bool_indices = tf.cast(tf.not_equal(test, tf.constant(0, tf.float32)), tf.float32)
        predict_test_values = tf.multiply(predict, test)
        count_nonzero = tf.reduce_sum(bool_indices)

        mae = tf.reduce_sum(tf.abs(tf.subtract(test, predict_test_values)))
        rms = tf.reduce_sum(tf.square(tf.subtract(test, predict_test_values)))

        return {'mae': mae, 'rms': rms, 'count_nonzero': count_nonzero, "indices": indices, "ratings": ratings }


    def _iterate_mini_batch(self, data, batch_size):

        indices = []
        ratings = []
        currCount = 0
        for key, values in data.items():
            if currCount < batch_size:
                for value in values:
                    indices.append([currCount, value[0]-1])
                    ratings.append(value[1])
                currCount = currCount+1
            else:
                yield (indices, ratings)

                currCount = 0
                indices = []
                ratings = []

                for value in values:
                    indices.append([currCount, value[0]-1])
                    ratings.append(value[1])
                currCount = currCount+1

        yield (indices, ratings)


    def _iterate_test_train(self, train, test, batch_size):
        indices_train = []
        indices_test = []

        ratings_train = []
        ratings_test = []

        currCount = 0

        for key, values in test.items():
            if currCount < batch_size:
                for value in values:
                    indices_test.append([currCount, value[0]-1])
                    ratings_test.append(value[1])

                for value in train[key]:
                    indices_train.append([currCount, value[0]-1])
                    ratings_train.append(value[1])
                currCount = currCount+1

            else:
                yield (indices_train, indices_test, ratings_train, ratings_test)

                currCount = 0
                indices_train = []
                indices_test = []
                ratings_train = []
                ratings_test = []

                for value in values:
                    indices_test.append([currCount, value[0]-1])
                    ratings_test.append(value[1])

                for value in train[key]:
                    indices_train.append([currCount, value[0]-1])
                    ratings_train.append(value[1])
                currCount = currCount+1

        yield (indices_train, indices_test, ratings_train, ratings_test)



    def train(self, conf, train_data, test_data, info):
        #
        print("Starting the training")

        if conf['type'] == 'U':
            self._input_nuerons = info['nV']
            self._train = train_data['U']['data']
            self._test = test_data['U']['data']

        else:
            self._input_nuerons = info['nU']
            self._train = train_data['V']['data']
            self._test = test_data['V']['data']


        model = NN.model(conf, self._input_nuerons, self._input_nuerons)

        # saver = tf.train.Saver()
        # Start the trainig
        count_tmp = 1
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            # if self.find_model(conf['save_model']['file'], conf['save_model']["path"]):
            #     # Found saved model
            #     saver.restore(sess, conf['save_model']['path']+conf['save_model']["file"])
            #     print("Found and restored saved model")

            print("Training Started")
            for epoch in range(conf['epochs']):
                print("Current Epoch Training: {}".format(epoch+1))
                for batch_indices, batch_ratings in self._iterate_mini_batch(self._train, conf['batch_size']):

                    sess.run(model["optimize"], {model["indices"]: batch_indices, model["ratings"]: batch_ratings})
                    # tmp_loss = sess.run(model['loss'], {model["indices"]: batch_indices, model["ratings"]: batch_ratings})
                    # print("Counter: {} Loss is : {}".format(count_tmp, tmp_loss))
                    # count_tmp = count_tmp+1

                # break

                error = { 'mae': 0, 'rms': 0}
                count_nonzero = 0
                # Calculate error after each epoch
                for indices_train, indices_test, ratings_train, ratings_test in self._iterate_test_train(self._train, self._test, conf['batch_size']):

                    tmp_predict = sess.run(model["predict_test"], {model["indices"]: indices_train, model["ratings"]: ratings_train})
                    tmp_err = self.error_fn(tmp_predict, self._input_nuerons, conf['batch_size'])

                    tmp_mae = sess.run(tmp_err['mae'], {tmp_err['indices']: indices_test, tmp_err['ratings']: ratings_test})
                    tmp_rms = sess.run(tmp_err['rms'], {tmp_err['indices']: indices_test, tmp_err['ratings']: ratings_test})
                    tmp_count = sess.run(tmp_err['count_nonzero'], {tmp_err['indices']: indices_test, tmp_err['ratings']: ratings_test})

                    error['mae'] = error['mae'] + tmp_mae
                    error['rms'] = error['rms'] + tmp_rms
                    count_nonzero = count_nonzero+tmp_count

                error['mae'] = (error['mae'] / count_nonzero) * 2
                error['rms'] = error['rms'] / count_nonzero

                self._mae = error['mae']
                self._rms = error['rms']
                self._mae_list.append(error['mae'])
                self._rms_list.append(error['rms'])
                print("Error at Epoch : {}/{} are MAE: {}  and RMS: {}".format(epoch+1, conf['epochs'], error['mae'], error['rms']))

                    # print(indices_train, indices_test)
                    # break

                # break

            print("Succesfully Trained")


            # # Save the variables to disk.
            # save_path = saver.save(sess, conf['save_model']['path']+conf['save_model']['file'])
            # print("Model saved in file: %s" % save_path)


    def save_errors(self, conf):
        errors = { 'mae' : self._mae, 'rms' : self._rms, 'rms_list': self._rms_list, 'mae_list': self._mae_list}
        with open(conf['save_errors'], 'wb') as output:
            pickle.dump(errors, output)

        print("Saved Succesfully to location: {}".format(conf['save_errors']))

# if __name__ == '__main__':
