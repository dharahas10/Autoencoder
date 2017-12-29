import numpy as np
import timeit
import pickle
from pprint import pprint

def testData():
    indices = [[0, 0], [0, 2], [0, 4],
                [1,1], [1,2], [1,4],
                [2,0], [2,1], [2,3],
                [3,3],
                [4,1], [4, 4]
                ]
    values = [3, 4, 2, 3, 5, 1, 2, 4, 4, 1, 2, 4]

    return tf.Session().run(tf.sparse_to_dense(indices, [5, 5], values))


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

        self.__Usize = 0
        self.__Vsize = 0
        self.__noRatings = 0
        self.__nTrain = 0

        self._userHash = {}
        self._userCounter = 1
        self._itemHash = {}
        self._itemCounter = 1
        np.random.seed(1)

    def preprocess(self, x):
        return (x-3)/2

    def postprocess(self, x):
        return 2*x+3


    def sortAscById(self, X):

        for key, value in X.items():
            value.sort(key=lambda x:x[0])

    def __postprocessing(self):
        self.sortAscById(self.train['U']['data'])
        self.sortAscById(self.train['V']['data'])
        self.sortAscById(self.test['U']['data'])
        self.sortAscById(self.test['V']['data'])


    def _appendTrain(self, userIndex, itemIndex, rating):
        if userIndex not in self.train['U']['data'] :
            self.train['U']['data'][userIndex] = []
        if itemIndex not in self.train['V']['data'] :
            self.train['V']['data'][itemIndex] = []

        self.train['U']['data'][userIndex].append([itemIndex, rating])
        self.train['V']['data'][itemIndex].append([userIndex, rating])

        self.__nTrain = self.__nTrain + 1


    def _appendTest(self, userIndex, itemIndex, rating):
        if userIndex not in self.test['U']['data'] :
            self.test['U']['data'][userIndex] = []
        if itemIndex not in self.test['V']['data'] :
            self.test['V']['data'][itemIndex] = []

        self.test['U']['data'][userIndex].append([itemIndex, rating])
        self.test['V']['data'][itemIndex].append([userIndex, rating])


    def _appendRating(self, userIndex, itemIndex, rating):
        if np.random.uniform() < self.__trainingRatio:
            self._appendTrain(userIndex, itemIndex, rating)

        else:
            self._appendTest(userIndex, itemIndex, rating)

        self.__noRatings = self.__noRatings + 1

        self.__Usize = userIndex if self.__Usize < userIndex else self.__Usize
        self.__Vsize = itemIndex if self.__Vsize < itemIndex else self.__Vsize


    def getUserIndex(self, id):
        if id not in self._userHash:
            self._userHash[id] = self._userCounter
            self._userCounter = self._userCounter+1

        return self._userHash[id]


    def getItemIndex(self, id):
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

                userIndex = self.getUserIndex(userId)
                itemIndex = self.getItemIndex(itemId)

                rating = self.preprocess(rating)
                self._appendRating(userIndex, itemIndex, rating)


    def loadData(self, conf):

        start = timeit.default_timer()

        self.__trainingRatio = conf['trainingRatio']
        self.loadRatings(conf)

        # Sorting the itemId's in user and vice versa for easy use of spareTensor
        self.__postprocessing()

        # save to local disk for faster access
        self.__save(conf)

        print("Time Taken to complete : {}".format(timeit.default_timer() - start))


    def __save(self,conf):

        info = {
            'nRatings' : self.__noRatings,
            'nTrain' : self.__nTrain,
            'trainRatio' : self.__trainingRatio,
            'nU' : self.__Usize,
            'nV' : self.__Vsize
        }
        data = {'train': self.train, 'test': self.test, 'info': info}

        with open(conf['out'], 'wb') as output:
            pickle.dump(data, output)

        print("Saved Succesfully to location: {}".format(conf['out']))
        # print(data)
        pprint(info)


if __name__ == '__main__':

    print(testData())
