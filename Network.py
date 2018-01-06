import tensorflow as tf
import math

def init_weights_and_biases(n_layers, hidden_layers, n_input, n_output):
    weights = [] # Weights for different layers
    biases = [] # Biases for different layers

    weights_biases_counter = 1

    # First Layer Weights and biases
    weights.append(tf.get_variable("Weights_"+str(weights_biases_counter), initializer= tf.random_uniform([n_input, hidden_layers[0]], -1.0/math.sqrt(n_input), 1.0/math.sqrt(n_input))))
    biases.append(tf.get_variable("Biases_"+str(weights_biases_counter), initializer= tf.random_uniform([hidden_layers[0]])))
    weights_biases_counter = weights_biases_counter+1
    # Hidden Layers Weights and Biases initialisation
    if n_layers > 1:
        for layer in range(n_layers-1):
            weights.append(tf.get_variable("Weights_"+str(weights_biases_counter), initializer= tf.random_uniform([hidden_layers[layer], hidden_layers[layer+1]], -1.0/math.sqrt(hidden_layers[layer]), 1.0/math.sqrt(hidden_layers[layer]))))
            biases.append(tf.get_variable("Biases_"+str(weights_biases_counter), initializer= tf.random_uniform([hidden_layers[layer+1]])))
            weights_biases_counter = weights_biases_counter+1
    # Output Layer weights
    weights.append(tf.get_variable("Weights_"+str(weights_biases_counter), initializer= tf.random_uniform([hidden_layers[n_layers-1], n_output], -1.0/math.sqrt(hidden_layers[n_layers-1]), 1.0/math.sqrt(hidden_layers[n_layers-1]))))
    biases.append(tf.get_variable("Biases_"+str(weights_biases_counter), initializer= tf.random_uniform([n_output])))
    weights_biases_counter = weights_biases_counter+1

    return weights, biases


def loss_fn(conf, X, predict, noise_X):

    X_nonzero = tf.not_equal(X, tf.constant(0, tf.float32))
    X_nonzero_mat = tf.cast(X_nonzero, tf.float32)
    predict_X = tf.multiply(predict, X_nonzero_mat)

    if conf['isNoisy']:
        #  loss = alpha(noisy difference square) + beta(non_noisy difference suqare)
        # alpha - phase
        noise_X_alpha_idx = tf.equal(noise_X,tf.constant(0, tf.float32))
        noise_X_mat_alpha = tf.cast(noise_X_alpha_idx, tf.float32)
        X_alpha = tf.multiply(X, noise_X_mat_alpha)
        predict_X_alpha = tf.multiply(predict_X, noise_X_mat_alpha)
        # beta - phase
        X_beta = tf.subtract(X, X_alpha)
        predict_X_beta = tf.subtract(predict_X, predict_X_alpha)
        # loss
        loss_alpha = tf.reduce_mean(tf.square(tf.subtract(X_alpha, predict_X_alpha)))
        loss_beta = tf.reduce_mean(tf.square(tf.subtract(X_beta, predict_X_beta)))
        loss = tf.add( tf.multiply(conf['alpha'], loss_alpha), tf.multiply(conf['beta'], loss_beta))
    else:

        loss = tf.reduce_sum(tf.square(tf.subtract(X, predict_X)))

    return loss


def corruptInput(conf, indices, values, shape):
    rand_val = tf.random_uniform(tf.shape(values))
    rand_less_noiseRatio = tf.less_equal(rand_val, tf.constant(conf['noiseRatio'], tf.float32))
    rand_less_idx = tf.cast(rand_less_noiseRatio, tf.float32)
    noise_values = tf.multiply(values, rand_less_idx)
    noise_X = tf.sparse_to_dense(indices, shape, noise_values)
    return tf.random_shuffle(noise_X)


def optimizer_fn(conf, loss):
    return tf.train.GradientDescentOptimizer(conf['learning_rate']).minimize(loss)


def build_network(conf, X, n_layers, weights, biases):
    # Network
    if conf['activate_fn'] == 'tanh':
        final_layer = tf.tanh(tf.matmul(X,weights[0]) + biases[0])

        if n_layers > 1 :
            for layer in range(n_layers-1):
                final_layer = tf.tanh(tf.matmul(final_layer, weights[layer+1]) + biases[layer+1])

        predict = tf.tanh(tf.matmul(final_layer, weights[n_layers]) + biases[n_layers])

    else:
        final_layer = tf.sigmoid(tf.matmul(X,weights[0]) + biases[0])

        if n_layers > 1 :
            for layer in range(n_layers-1):
                final_layer = tf.sigmoid(tf.matmul(final_layer, weights(layer+1)) + biases[layer+1])

        predict = tf.tanh(tf.matmul(final_layer, weights[n_layers]) + biases[n_layers])

    return predict


def model(conf,n_input, n_output):
    n_layers = len(conf['hidden_layers'])

    # Placeholder for datasets
    indices = tf.placeholder(tf.int32, name="indices")
    ratings = tf.placeholder(tf.float32, name="ratings")
    dense_X = tf.sparse_to_dense(indices, [conf['batch_size'], n_input], ratings)
    dense_X = tf.random_shuffle(dense_X)
    weights, biases = init_weights_and_biases(n_layers, conf['hidden_layers'], n_input, n_output)

    if conf['isNoisy']:
        noise_X = corruptInput(conf, indices, ratings, [conf['batch_size'], n_input])
    else:
        noise_X = dense_X

    Y_predict = build_network(conf, noise_X, n_layers, weights, biases)
    Y_test = build_network(conf, dense_X, n_layers, weights, biases)

    loss = loss_fn(conf, dense_X, Y_predict, noise_X)

    optimize = optimizer_fn(conf, loss)

    model = {
            "indices": indices,
            "ratings": ratings,
            'weights' : weights,
            'biases': biases,
            'predict_train': Y_predict,
            'loss': loss,
            'optimize': optimize,
            'predict_test': Y_test
            }

    return model
