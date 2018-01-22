from Data.SingleRatingDataLoader import SingleRatingDataLoader
from Data.MultiRatingDataLoader import MultiRatingDataLoader
from helper import *
import pickle
import json

import numpy as np

if __name__ == '__main__':

    np.random.seed(1)

    conf_filename = "./Config/Datasets/tripAdvisor.json"

    with open(conf_filename, 'r') as conf_file:
        conf = json.load(conf_file)

    if find_file(conf['out']):
        with open(conf['out'], 'rb') as datafile:
            data = pickle.load(datafile)
            print(data['info'])
    else:
        if not conf['multi_rating']:
            dataloader = SingleRatingDataLoader()
            dataloader.loadData(conf)
        else:
            dataloader = MultiRatingDataLoader()
            dataloader.loadData(conf)
