import tensorflow as tf

def init_weights_and_biases(n_layers, hidden_layers, n_input, n_output):
    weights = [] # Weights for different layers
    biases = [] # Biases for different layers

    weights_biases_counter = 1

    # First Layer Weights and biases
    weights.append(tf.get_variable("Weights_"+str(weights_biases_counter), initializer= tf.random_normal([n_input, hidden_layers[0]])))
    biases.append(tf.get_variable("Biases_"+str(weights_biases_counter), initializer= tf.random_normal([hidden_layers[0]])))
    weights_biases_counter = weights_biases_counter+1
    # Hidden Layers Weights and Biases initialisation
    if n_layers > 1:
        for layer in range(n_layers-1):
            weights.append(tf.get_variable("Weights_"+str(weights_biases_counter), initializer= tf.random_normal([hidden_layers[layer], hidden_layers[layer+1]])))
            biases.append(tf.get_variable("Biases_"+str(weights_biases_counter), initializer= tf.random_normal([hidden_layers[layer+1]])))
            weights_biases_counter = weights_biases_counter+1
    # Output Layer weights
    weights.append(tf.get_variable("Weights_"+str(weights_biases_counter), initializer= tf.random_normal([hidden_layers[n_layers-1], n_output])))
    biases.append(tf.get_variable("Biases_"+str(weights_biases_counter), initializer= tf.random_normal([n_output])))
    weights_biases_counter = weights_biases_counter+1

    return weights, biases

def loss_fn(Y, predict):

    return tf.reduce_mean(tf.square(tf.subtract(Y, predict)))

def corruptInput(X):
    return X

def optimizer_fn(conf, loss):

    return tf.train.GradientDescentOptimizer(conf['learning_rate']).minimize(loss)

def error_fn(predict, test):

    bool_indices = tf.cast(tf.not_equal(test, tf.constant(0, tf.float32)), tf.float32)
    predict_test_values = tf.multiply(predict, test)
    count_nonzero = tf.reduce_sum(bool_indices)

    mae = tf.reduce_sum(tf.abs(tf.subtract(test, predict_test)))
    rms = tf.reduce_sum(tf.square(tf.subtract(test, predict_test)))

    return {'mae': mae, 'rms': rms, 'count_nonzero': count_nonzero}

def build_network(conf, X, n_layers, weights, biases):
    # Network

    if conf['activate_fn'] == 'tanh':
        final_layer = tf.tanh(tf.matmul(X,weights[0]) + biases[0])

        if n_layers > 1 :
            for layer in range(n_layers-1):
                final_layer = tf.tanh(tf.matmul(final_layer, weights(layer+1)) + biases[layer+1])

        predict = tf.tanh(tf.matmul(final_layer, weights[n_layers]) + biases[n_layers])

    else:
        final_layer = tf.sigmoid(tf.matmul(X,weights[0]) + biases[0])

        if n_layers > 1 :
            for layer in range(n_layers-1):
                final_layer = tf.sigmoid(tf.matmul(final_layer, weights(layer+1)) + biases[layer+1])

        predict = tf.sigmoid(tf.matmul(final_layer, weights[n_layers]) + biases[n_layers])

    return predict



def model(conf,n_input, n_output):
    n_layers = len(conf['hidden_layers'])

    # Placeholder for datasets
    indices = tf.placeholder(tf.int32, name="indices")
    ratings = tf.placeholder(tf.float32, name="ratings")
    dense_X = tf.sparse_to_dense(indices, [conf['batch_size'], n_input], ratings)
    weights, biases = init_weights_and_biases(n_layers, conf['hidden_layers'], n_input, n_output)

    noise_X = corruptInput(dense_X)

    Y_predict = build_network(conf, noise_X, n_layers, weights, biases)
    Y_test = build_network(conf, dense_X, n_layers, weights, biases)

    loss = loss_fn(dense_X, Y_predict)

    optimize = optimizer_fn(conf, loss)

    model = {"indices": indices, "ratings": ratings, 'weights' : weights, 'biases': biases, 'predict_train': Y_predict, 'loss': loss, 'optimize': optimize, 'predict_test': Y_test}

    return model


# if __name__ == '__main__':
