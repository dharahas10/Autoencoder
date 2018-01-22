import tensorflow as tf
import json
import timeit
import pickle
from Networks.TrainNetwork import TrainNetwork
from Errors.Error import Error
from pprint import pprint
from helper import duration

if __name__ == '__main__':

    start = timeit.default_timer()

    conf_filename = './Config/Models/ml-1m_config.json'
    # conf_filename = './Config/Models/tripAdvisor_config.json'
    # conf_filename = './Config/Models/tripAdvisor_multi.json'

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
            print("Dataset Info: {}".format(info))

        trainNetwork = TrainNetwork()
        trainer = trainNetwork.train(config, info, train)

        error = Error()
        error_measure = error.measure(config, info, train, test)

    end = timeit.default_timer()
    mm, ss = duration(start, end)
    print("Time taken to complete (mm:ss): {}:{}  ".format(mm, ss))
