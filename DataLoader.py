import numpy as np
import timeit
import pickle
import os

class DataLoader:

    def __init__(self):

        self.train = {
            'U' : { 'data' : {} },
            'V' : { 'data': {} }
        }

        self.test = {
            'U' : { 'data' : {} },
            'V' : { 'data': {} }
        }
        # Details about the Dataset
        self._Usize = 0
        self._Vsize = 0
        self._noRatings = 0
        self._nTrain = 0
        # User and Item counters
        self._userHash = {}
        self._userCounter = 1
        self._itemHash = {}
        self._itemCounter = 1
        np.random.seed(1)

    def _preprocess(self, x):
        return (x-3)/2

    def _postprocess(self, x):
        return 2*x+3


    def _sortAscById(self, X):

        for key, value in X.items():
            value.sort(key=lambda x:x[0])

    def _postprocessing(self):
        self._sortAscById(self.train['U']['data'])
        self._sortAscById(self.train['V']['data'])
        self._sortAscById(self.test['U']['data'])
        self._sortAscById(self.test['V']['data'])


    def _appendTrain(self, userIndex, itemIndex, rating):
        if userIndex not in self.train['U']['data'] :
            self.train['U']['data'][userIndex] = []
        if itemIndex not in self.train['V']['data'] :
            self.train['V']['data'][itemIndex] = []

        self.train['U']['data'][userIndex].append([itemIndex, rating])
        self.train['V']['data'][itemIndex].append([userIndex, rating])

        self._nTrain = self._nTrain + 1


    def _appendTest(self, userIndex, itemIndex, rating):
        if userIndex not in self.test['U']['data'] :
            self.test['U']['data'][userIndex] = []
        if itemIndex not in self.test['V']['data'] :
            self.test['V']['data'][itemIndex] = []

        self.test['U']['data'][userIndex].append([itemIndex, rating])
        self.test['V']['data'][itemIndex].append([userIndex, rating])


    def _appendRating(self, userIndex, itemIndex, rating):
        if np.random.uniform() < self._trainingRatio:
            self._appendTrain(userIndex, itemIndex, rating)
        else:
            self._appendTest(userIndex, itemIndex, rating)

        self._noRatings = self._noRatings + 1
        self._Usize = userIndex if self._Usize < userIndex else self._Usize
        self._Vsize = itemIndex if self._Vsize < itemIndex else self._Vsize


    def _getUserIndex(self, id):
        if id not in self._userHash:
            self._userHash[id] = self._userCounter
            self._userCounter = self._userCounter+1

        return self._userHash[id]


    def _getItemIndex(self, id):
        if id not in self._itemHash:
            self._itemHash[id] = self._itemCounter
            self._itemCounter = self._itemCounter+1

        return self._itemHash[id]


    def loadRatings(self, conf):

        with open(conf['ratings']) as file:
            for line in file:
                userId, itemId, rating, _ = line.split("::")
                userId = int(userId)
                itemId = int(itemId)
                rating = int(rating)

                userIndex = self._getUserIndex(userId)
                itemIndex = self._getItemIndex(itemId)

                rating = self._preprocess(rating)
                self._appendRating(userIndex, itemIndex, rating)


    def loadData(self, conf):

        start = timeit.default_timer()

        self._trainingRatio = conf['trainingRatio']
        self.loadRatings(conf)
        # Sorting the itemId's in user and vice versa
        self._postprocessing()
        # save to local disk for faster access next time
        self._save(conf)

        print("Time Taken to complete : {}"
                                .format(timeit.default_timer() - start))


    def _save(self,conf):

        sparsity = str(round(100 - (self._nTrain/(self._Usize*self._Vsize))*100, 3)) + "%"

        info = {
            'nRatings' : self._noRatings,
            'nTrain' : self._nTrain,
            'trainRatio' : self._trainingRatio,
            'nU' : self._Usize,
            'nV' : self._Vsize,
            'sparsity' : sparsity
        }
        data = {'train': self.train, 'test': self.test, 'info': info}

        with open(conf['out'], 'wb') as output:
            pickle.dump(data, output)

        print("Saved Succesfully to location: {}".format(conf['out']))
        print(info)
