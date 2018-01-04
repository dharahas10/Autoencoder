from DataLoader import DataLoader
import pickle
import json
from helper import *

if __name__ == '__main__':

    conf_filename = "./Config/Data/ml-1m.json"

    with open(conf_filename, 'r') as conf_file:
        conf = json.load(conf_file)

    if find_file(conf['out']):
        with open(conf['out'], 'rb') as datafile:
            data = pickle.load(datafile)
            print(data['info'])
    else:
        dataloader = DataLoader()
        dataloader.loadData(conf)
