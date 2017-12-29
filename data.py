import numpy as np
from DataLoader import DataLoader
import pickle

if __name__ == '__main__':

    conf = {
        'ratings' : '../dataset/ml-1m/ratings.csv',
        'trainingRatio' : 0.9,
        'out' : './data/ml-1m.p'
    }

    # dataloader = DataLoader()
    #
    # dataloader.loadData(conf)

    # Uncomment to test save file
    with open(conf['out'], 'rb') as datafile:

        data = pickle.load(datafile)
        print(data['info'])
