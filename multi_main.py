import tensorflow as tf
import json
import timeit
import pickle
from Networks.MultiTrainNetwork import MultiTrainNetwork
from Errors.MultiError import MultiError
from pprint import pprint
from helper import *

if __name__ == '__main__':

    start = timeit.default_timer()

    # conf_filename = './Config/Models/ml-1m_config.json'
    # conf_filename = './Config/Models/tripAdvisor_config.json'
    conf_filename = './Config/Models/tripAdvisor_multi.json'

    type_1 = False

    print("0. Loading Config...")
    with open(conf_filename, 'r') as conf_file:
        config = json.load(conf_file)

    tf.set_random_seed(config['seed'])

    print("0.1. Loading Data...")
    # Loading the Foramted dataset
    if not find_file(config['dataset']):
        print("Data file not found. First run the data.py to generate the pickle file.")

    else:
        with open(config['dataset'], 'rb') as dataFile:
            data = pickle.load(dataFile)
            train, test, info = data['train'], data['test'], data['info']
            pprint("Dataset Info: {}".format(info))

        if type_1:
            trainNetwork = MultiTrainNetwork()
            trainer = trainNetwork.train1(config, info, train)

            error = MultiError()
            error_measure = error.measure1(config, info, train, test)
        else:
            trainNetwork = MultiTrainNetwork()
            trainer = trainNetwork.train2(config, info, train)

            error = MultiError()
            error_measure = error.measure2(config, info, train, test)

    end = timeit.default_timer()
    mm, ss = duration(start, end)
    print("Time taken to complete (mm:ss): {}:{}  ".format(mm, ss))
