import tensorflow as tf
import numpy as np
import pickle
import timeit
from TrainNetwork import TrainNetwork


if __name__ == '__main__':
    start = timeit.default_timer()
    conf = {
        'file': './data/ml-1m.p',
        'type': 'U',
        'seed': 0,
        'hidden_layers': [500],
        'activate_fn' : 'tanh',
        'epochs': 10,
        'learning_rate': 0.01,
        'batch_size': 30,
        'save_model': {
            'path': './models/',
            'file': 'ml-1m.ckpt'
        },
        'save_errors': './errors/ml-1m.p'

    }

    tf.set_random_seed(conf['seed'])

    # Load saved data format
    with open(conf['file'], 'rb') as datafile:

        data = pickle.load(datafile)
        train, test, info = data['train'], data['test'], data['info']

    # print(info)

    trainer = TrainNetwork()

    trainer.train(conf, train, test, info)
    time = timeit.default_timer() - start
    print("Time taken to complete (mm:ss): {}:{}  ".format(time//60, time%60))
