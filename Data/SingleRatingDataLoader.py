import numpy as np
import timeit
import pickle
import os

from Data.DataLoader import DataLoader

class SingleRatingDataLoader(DataLoader):

    def __init__(self):
        DataLoader.__init__(self)


    def loadRatings(self, conf):

        with open(conf['ratings']) as file:
            for line in file:
                userId, itemId, rating, *_ = line.split(conf['split'])
                userId = str(userId)
                itemId = str(itemId)
                rating = float(rating)

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
