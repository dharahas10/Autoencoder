from DataLoader import DataLoader
import pickle

from helper import *

if __name__ == '__main__':

    conf = {
        'ratings' : '../dataset/ml-1m/ratings.csv',
        'trainingRatio' : 0.9,
        'out' : './data/ml-1m.p'
    }

    if find_file(conf['out']):
        with open(conf['out'], 'rb') as datafile:
            data = pickle.load(datafile)
            print(data['info'])
    else:
        dataloader = DataLoader()
        dataloader.loadData(conf)
