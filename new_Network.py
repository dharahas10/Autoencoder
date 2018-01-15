import tensorflow as tf
import math

class Model():

    def __init__(self, config):
        print("\t 1. Initializing Model...")
        self._hidden_layers = config["hidden_layers"]
        self._hidden_nuerons = config["hidden_nuerons"]
        self._learning_rate = config["learning_rate"]
        self._batch_size = config["batch_size"]
        self._alpha = config["alpha"]
        self._beta = config["beta"]
        self._isNoisy = config["isNoisy"]
        self._weights = []
        self._biases = []


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


    def _multi_weight_mask(self, nRatings, multi_nuerons, input_nuerons):

        shape = [multi_nuerons, input_nuerons]
        values = []
        indices = []

        for i in range(input_nuerons):
            for j in range(nRatings):
                indices.append([i*4+j, i])
                values.append(1.0)

        indices.sort(key=lambda x:x[0])
        return tf.sparse_to_dense(indices, shape, values)


    def inference(self, input_nuerons, output_nuerons, input_data):
        self._input_nuerons = input_nuerons
        self._output_nuerons = output_nuerons

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
                self._weights.append(weights) # Adding weights for later calling
                biases = self._create_bias(shape)
                self._biases.append(biases) # Adding biases for later calling
                l_out = tf.tanh(tf.matmul(input_X,weights) + biases, name=scope.name)
                input_X = l_out

        return l_out


    def inference_multi(self, nRatings, input_nuerons, output_nuerons, input_data):
        self._input_nuerons = input_nuerons
        self._output_nuerons = output_nuerons

        self._multi_input_nuerons = self._input_nuerons*nRatings
        weight_mask = self._multi_weight_mask(nRatings,
                                                self._multi_input_nuerons,
                                                self._input_nuerons)

        with tf.variable_scope("Pooling_Layer") as scope:
            shape = [self._multi_input_nuerons, self._input_nuerons]
            multi_weights = self._create_weights(shape)
            multi_pool = tf.multiply(multi_weights, weight_mask)
            pool_X = tf.tanh(tf.matmul(input_data, multi_pool), name=scope.name)

        l_out = self.inference(input_nuerons, output_nuerons, pool_X)
        return l_out



    def loss(self, l_out, output):
        with tf.variable_scope('loss') as scope:
            nonzero_bool = tf.not_equal(output, tf.contant(0, tf.float32))
            nonzero_mat = tf.cast(nonzero_bool, tf.float32)
            l_out_nonzero = tf.multiply(l_out, nonzero_mat)
            cross_entropy = tf.square(tf.subtract(l_out_nonzero, output))
            cost = tf.reduce_mean(cross_entropy, name=scope.name)

        return cost

    def loss_multi(self, l_out, output):
        with tf.variable_scope('loss') as scope:
            nonzero_bool = tf.not_equal(output, tf.constant(0, tf.float32))
            nonzero_mat = tf.cast(nonzero_bool, tf.float32)
            l_out_nonzero = tf.multiply(l_out, nonzero_mat)
            cross_entropy = tf.square(tf.subtract(l_out_nonzero, output))
            cost = tf.reduce_mean(cross_entropy, name=scope.name)

        return cost


    def loss_noisy(self, Y, X, l_out):
        with tf.variable_scope("loss") as scope:
            Y_nonzero = tf.not_equal(Y, tf.constant(0, tf.float32))
            Y_nonzero_mat = tf.cast(Y_nonzero, tf.float32)
            l_out_Y = tf.multiply(l_out, Y_nonzero_mat)

            with tf.variable_scope("alpha_loss") as alpha_scope:
                X_alpha_idx = tf.equal(X, tf.constant(0, tf.float32))
                X_alpha_mat = tf.cast(X_alpha_idx, tf.float32)
                Y_alpha = tf.multiply(Y, X_alpha_mat)
                l_out_alpha = tf.multiply(l_out, X_alpha_mat)

                loss_alpha = tf.reduce_mean(tf.square(Y_alpha-l_out_alpha),
                                            name=alpha_scope.name)

            with tf.variable_scope("beta_loss") as beta_scope:
                Y_beta = tf.subtract(Y, Y_alpha)
                l_out_beta = tf.subtract(l_out, l_out_alpha)

                loss_beta = tf.reduce_mean(tf.square(Y_beta-l_out_beta),
                                           name=beta_scope.name)

            cost = tf.add(
                    tf.multiply(loss_alpha, self._alpha),
                    tf.multiply(loss_beta, self._beta), name=scope.name)

        return cost


    def train(self, loss, global_step):

        optimizer_op = tf.train.\
                GradientDescentOptimizer(self._learning_rate).\
                minimize(loss, global_step=global_step)
        return optimizer_op


    def error(self, l_out, output):
        with tf.variable_scope("Accuracy") as scope:
            nonzero_matrix = tf.cast(
                tf.not_equal(output, tf.constant(0, tf.float32)),
                tf.float32
            )

            l_out_nonzero = tf.multiply(l_out, nonzero_matrix)
            nonzero_count = tf.reduce_sum(nonzero_matrix)
            mae = tf.truediv(tf.reduce_sum(
                        tf.abs(
                            tf.subtract(l_out_nonzero, output))), nonzero_count)*2

            rms = tf.truediv(tf.reduce_sum(tf.square(
                        tf.subtract(l_out_nonzero, output))), nonzero_count)*2

            return mae, rms


    def error_multi(self, l_out, output):
        with tf.variable_scope("Accuracy") as scope:
            nonzero_matrix = tf.cast(
                tf.not_equal(output, tf.constant(0, tf.float32)),
                tf.float32
            )

            l_out_nonzero = tf.multiply(l_out, nonzero_matrix)
            nonzero_count = tf.reduce_sum(nonzero_matrix)
            mae = tf.truediv(tf.reduce_sum(
                        tf.abs(
                            tf.subtract(l_out_nonzero, output))), nonzero_count)*2

            rms = tf.truediv(tf.reduce_sum(tf.square(
                        tf.subtract(l_out_nonzero, output))), nonzero_count)*2

            return mae, rms
