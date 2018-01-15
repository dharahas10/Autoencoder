import numpy as np
import timeit
import pickle
import os

from pprint import pprint


class DataLoader():

    def __init__(self):
        np.random.seed(1)
        self._train = {
            'U': {
                'data': {}
            },
            'V': {
                'data': {}
            }
        }

        self._test = {
            'U': { 'data': {} },
            'V': { 'data': {} }
        }

        self._Usize = 0
        self._Vsize = 0
        self._noRatings = 0
        self._nTrain = 0
        self._nTest = 0

        self._userHash = {}
        self._userCounter = 1
        self._itemHash = {}
        self._itemCounter = 1


    def _preprocess(self, x):
        return (x-3)/2


    def _postprocess(self, x):
        return 2*x + 3

    def _isDup(self, alist, id):
        for x in alist:
            if x[0] == id:
                print("Found duplicate")
                return True

        return False


    def _getUserIndex(self, id):
        if id not in self._userHash:
            self._userHash[id] = self._userCounter
            self._userCounter += 1

        return self._userHash[id]


    def _getItemIndex(self, id):
        if id not in self._itemHash:
            self._itemHash[id] = self._itemCounter
            self._itemCounter += 1

        return self._itemHash[id]


    def _sortAscById(self, X):
        for key, value in X.items():
            value.sort(key=lambda x:x[0])


    def _postprocessing(self):
        self._sortAscById(self._train['U']['data'])
        self._sortAscById(self._train['V']['data'])
        self._sortAscById(self._test['U']['data'])
        self._sortAscById(self._test['V']['data'])


    def _appendRating(self, userIndex, itemIndex, rating):
        # np.random.seed(1)
        if np.random.uniform() <= self._trainingRatio:
            self._appendTrain(userIndex, itemIndex, rating)
        else:
            self._appendTest(userIndex, itemIndex, rating)

        self._noRatings += 1
        self._Usize = userIndex if self._Usize < userIndex else self._Usize
        self._Vsize = itemIndex if self._Vsize < itemIndex else self._Vsize

    def _appendMultiRating(self, userIndex, itemIndex, overallRating, otherRatings):

        k = np.random.uniform()
        if k <= self._trainingRatio:
            # print(k)
            self._appendMultiTrain(userIndex, itemIndex, overallRating, otherRatings)
        else:
            self._appendMultiTest(userIndex, itemIndex, overallRating, otherRatings)

        self._noRatings += 1
        self._Usize = userIndex if self._Usize < userIndex else self._Usize
        self._Vsize = itemIndex if self._Vsize < itemIndex else self._Vsize

    def _appendTrain(self, userIndex, itemIndex, rating):
        if userIndex not in self._train['U']['data']:
            self._train['U']['data'][userIndex] = []
        if itemIndex not in self._train['V']['data']:
            self._train['V']['data'][itemIndex] = []

        # new method to check duplicates
        if not self._isDup(self._train['U']['data'][userIndex], itemIndex):
            self._train['U']['data'][userIndex].append([itemIndex, rating])
            self._nTrain += 1
        if not self._isDup(self._train['V']['data'][itemIndex], userIndex):
            self._train['V']['data'][itemIndex].append([userIndex, rating])




    def _appendMultiTrain(self, userIndex, itemIndex, overallRating, otherRatings):
        if userIndex not in self._train['U']['data']:
            self._train['U']['data'][userIndex] = []
        if itemIndex not in self._train['V']['data']:
            self._train['V']['data'][itemIndex] = []

        if not self._isDup(self._train['U']['data'][userIndex], itemIndex):
            self._train['U']['data'][userIndex].append([itemIndex, overallRating, otherRatings])
            self._nTrain += 1
        if not self._isDup(self._train['V']['data'][itemIndex], userIndex):
            self._train['V']['data'][itemIndex].append([itemIndex, overallRating, otherRatings])


    def _appendTest(self, userIndex, itemIndex, rating):
        if userIndex not in self._test['U']['data']:
            self._test['U']['data'][userIndex] = []
        if itemIndex not in self._test['V']['data']:
            self._test['V']['data'][itemIndex] = []

        if not self._isDup(self._test['U']['data'][userIndex], itemIndex):
            self._test['U']['data'][userIndex].append([itemIndex, rating])
            self._nTest += 1
        if not self._isDup(self._test['V']['data'][itemIndex], userIndex):
            self._test['V']['data'][itemIndex].append([userIndex, rating])


    def _appendMultiTest(self, userIndex, itemIndex, overallRating, otherRatings):
        if userIndex not in self._test['U']['data']:
            self._test['U']['data'][userIndex] = []
        if itemIndex not in self._test['V']['data']:
            self._test['V']['data'][itemIndex] = []

        # self._test['U']['data'][userIndex].append([itemIndex, overallRating, otherRatings])
        # self._test['V']['data'][itemIndex].append([userIndex, overallRating, otherRatings])
        if not self._isDup(self._test['U']['data'][userIndex], itemIndex):
            self._test['U']['data'][userIndex].append([itemIndex, overallRating, otherRatings])
            self._nTest += 1
        if not self._isDup(self._test['V']['data'][itemIndex], userIndex):
            self._test['V']['data'][itemIndex].append([itemIndex, overallRating, otherRatings])


    def _save(self, conf):

        sparsity = str(round(100 - (self._nTrain/(self._Usize*self._Vsize))*100, 3)) + "%"
        info = {
            'nRatings' : self._noRatings,
            'nTrain' : self._nTrain,
            "nTest" : self._nTest,
            'trainRatio' : self._trainingRatio,
            'nU' : self._Usize,
            'nV' : self._Vsize,
            'sparsity' : sparsity
        }
        data = {'train': self._train, 'test': self._test, 'info': info}

        with open(conf['out'], 'wb') as output:
            pickle.dump(data, output)

        print("Saved Succesfully to location: {}".format(conf['out']))
        pprint(info)


    def loadRatings(self, conf):
        pass


    def loadData(self, conf):
        pass
