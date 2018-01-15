import tensorflow as tf
import json
import timeit
import pickle
from new_TrainNetwork import TrainNetwork
from pprint import pprint
from helper import duration

if __name__ == '__main__':

    start = timeit.default_timer()

    # conf_filename = './Config/Models/ml-1m_config.json'
    # conf_filename = './Config/Models/tripAdvisor_config.json'
    conf_filename = './Config/Models/tripAdvisor_multi.json'

    print("1. Loading Config...")
    with open(conf_filename, 'r') as conf_file:
        config = json.load(conf_file)

    tf.set_random_seed(config['seed'])

    print("2. Loading Data...")
    with open(config['dataset'], 'rb') as dataFile:
        data = pickle.load(dataFile)
        train, test, info = data['train'], data['test'], data['info']

    print("Dataset Info: {}".format(info))
    # pprint(test)

    trainer = TrainNetwork()
    if not config['multi_ratings']:
        trainer.train(config, info, train)
        trainer.error(config, info, test)
    else:
        trainer.train_multi(config, info, train)
        trainer.error_multi(config, info, test)

    end = timeit.default_timer()
    mm, ss = duration(start, end)
    print("Time taken to complete (mm:ss): {}:{}  ".format(mm, ss))
