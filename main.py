import tensorflow as tf
import numpy as np
import json
import pickle
import timeit
from TrainNetwork import TrainNetwork
from pprint import pprint
from helper import *

if __name__ == '__main__':
    start = timeit.default_timer()

    conf_filename = './Config/Models/ml-1m_config.json'
    # Loading Config
    with open(conf_filename, 'r') as conf_file:
        conf = json.load(conf_file)

    tf.set_random_seed(conf['seed'])
    # Loading the Foramted dataset
    if not find_file(conf['dataset']):
        print("Data file not found. First run the data.py to generate the pickle file.")

    else:
        # Load saved data format
        print("Loading Data .....")
        with open(conf['dataset'], 'rb') as datafile:
            data = pickle.load(datafile)
            train, test, info = data['train'], data['test'], data['info']
        print("Information on dataset:")
        pprint(info)
        # Train Network
        trainer = TrainNetwork()
        trainer.train(conf, train, test, info)

    end = timeit.default_timer()
    mm, ss = duration(start, end)
    print("Time taken to complete (mm:ss): {}:{}  ".format(mm, ss))
