import tensorflow as tf
import math

class Model():

    def __init__(self, config):
        print("\t 1. Initializing Model Configuration...")
        self._hidden_layers = config["hidden_layers"]
        self._hidden_nuerons = config["hidden_nuerons"]
        self._nCriteria = config["nCriteria"]
        self._learning_rate = config["learning_rate"]
        self._batch_size = config["batch_size"]
        self._alpha = config["alpha"]
        self._beta = config["beta"]
        self._isNoisy = config["isNoisy"]


    def _create_weights(self, shape):
        return tf.get_variable("Weights",
                               dtype=tf.float32,
                               initializer=tf.random_uniform(shape,
                                    minval= -1.0/math.sqrt(shape[0]),
                                    maxval= 1.0/math.sqrt(shape[0])))


    def _create_bias(self, shape):
        return tf.get_variable("Bias",
                                dtype=tf.float32,
                                initializer=tf.constant(1.0, shape=[shape[1]],
                                    dtype=tf.float32))


    def _multi_weight_mask(self, multi_nuerons, input_nuerons):

        shape = [multi_nuerons, input_nuerons]
        values = []
        indices = []

        for i in range(input_nuerons):
            for j in range(self._nCriteria):
                indices.append([i*self._nCriteria+j, i])
                values.append(1.0)

        indices.sort(key=lambda x:x[0])
        return tf.sparse_to_dense(indices, shape, values)

    def _pooling_layer(self, data):
        #  data shape: [10, 30, 6] (nuerons, batch_size, nCriteria) weights shape : [10, 6, 1]
        with tf.variable_scope("Pooling_Layer") as scope:
            weights = self._create_weights([self._input_nuerons, self._nCriteria, 1])
            pool_X = tf.matmul(data, weights)
            pool_X = tf.transpose(
                        tf.reshape(pool_X, [self._input_nuerons,self._batch_size]),
                        name=scope.name)

        return pool_X





    def inference(self, input_nuerons, output_nuerons, input_data):

        for layer in range(self._hidden_layers+1):
            # Computing Shapes of Weights and Biases for each layer
            if layer == 0:
                shape = [self._input_nuerons, self._hidden_nuerons[layer]]
                input_X = input_data # Data input for each Layer
            elif layer == self._hidden_layers:
                shape = [self._hidden_nuerons[layer-1], self._output_nuerons]
            else:
                shape = [self._hidden_nuerons[layer], self._hidden_nuerons[layer+1]]

            # creating layers for model
            with tf.variable_scope("Layer_"+str(layer)) as scope:
                weights = self._create_weights(shape)
                # self._weights.append(weights) # Adding weights for later calling
                biases = self._create_bias(shape)
                # self._biases.append(biases) # Adding biases for later calling
                l_out = tf.tanh(tf.matmul(input_X,weights) + biases, name=scope.name)
                input_X = l_out

        return l_out


    def inference1(self, input_nuerons, output_nuerons, input_data):
        print("\t 2. Plotting the Network....")
        self._input_nuerons = input_nuerons
        self._output_nuerons = output_nuerons

        self._multi_input_nuerons = self._input_nuerons*self._nCriteria
        weight_mask = self._multi_weight_mask(self._multi_input_nuerons,
                                                self._input_nuerons)

        with tf.variable_scope("Pooling_Layer") as scope:
            shape = [self._multi_input_nuerons, self._input_nuerons]
            multi_weights = self._create_weights(shape)
            multi_pool = tf.multiply(multi_weights, weight_mask)
            pool_X = tf.tanh(tf.matmul(input_data, multi_pool), name=scope.name)

        l_out = self.inference(self._input_nuerons, self._output_nuerons, pool_X)
        return l_out


    def inference2(self, input_nuerons, output_nuerons, input_data):
        print("\t 2. Plotting the Network....")
        self._input_nuerons = input_nuerons
        self._output_nuerons = output_nuerons

        self._multi_input_nuerons = self._input_nuerons*self._nCriteria

        pool_X = self._pooling_layer(input_data)
        l_out = self.inference(self._input_nuerons, self._output_nuerons, pool_X)
        return l_out


    def loss(self,output, l_out):
        with tf.variable_scope('loss') as scope:
            nonzero_bool = tf.not_equal(output, tf.constant(0, tf.float32))
            nonzero_mat = tf.cast(nonzero_bool, tf.float32)
            l_out_nonzero = tf.multiply(l_out, nonzero_mat)
            cross_entropy = tf.square(tf.subtract(l_out_nonzero, output))
            cost = tf.reduce_mean(cross_entropy, name=scope.name)

        return cost


    def train_gradient(self, loss, global_step):
        optimizer_op = tf.train.\
                GradientDescentOptimizer(self._learning_rate).\
                minimize(loss, global_step=global_step)
        return optimizer_op


    def error(self, output, l_out):
        with tf.variable_scope("Error") as scope:
            nonzero_matrix = tf.cast(
                tf.not_equal(output, tf.constant(0, tf.float32)),
                tf.float32
            )

            l_out_nonzero = tf.multiply(l_out, nonzero_matrix)
            nonzero_count = tf.reduce_sum(nonzero_matrix)
            with tf.variable_scope("MAE") as sub_scope:
                mae = tf.multiply(
                        tf.truediv(
                            tf.reduce_sum(
                                tf.abs(
                                    tf.subtract(l_out_nonzero,
                                                output))),
                            nonzero_count),
                        tf.constant(2, tf.float32),
                        name=sub_scope.name)

            with tf.variable_scope("RMS") as sub_scope:
                rms = tf.multiply(
                        tf.truediv(
                            tf.reduce_sum(
                                tf.square(
                                    tf.subtract(l_out_nonzero,
                                                output))),
                            nonzero_count),
                        tf.constant(2, tf.float32),
                        name=sub_scope.name)

            return mae, rms
