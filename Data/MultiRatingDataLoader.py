import numpy as np
import pickle
import timeit
from Data.DataLoader import DataLoader


class MultiRatingDataLoader(DataLoader):
    def __init__(self):
        DataLoader.__init__(self)


    def loadRatings(self, conf):

        with open(conf['ratings']) as file:
            for line in file:
                userId, itemId, overallRating, *otherRatings = line.split(conf['split'])
                userId = str(userId)
                itemId = str(itemId)
                overallRating = float(overallRating)
                otherRatings = [float(x) for x in otherRatings]

                userIndex = self._getUserIndex(userId)
                itemIndex = self._getItemIndex(itemId)

                overallRating = self._preprocess(overallRating)
                otherRatings = [self._preprocess(x) for x in otherRatings]
                self._appendMultiRating(userIndex, itemIndex, overallRating, otherRatings)


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



# if __name__ == '__main__':
#
#     conf = {
#           "ratings" : "../dataset/Trip Advisor/tripadvisor_reviews.csv",
#           "split" : ";",
#           "trainingRatio" : 0.9,
#           "out" : "./data/tripAdvisor.p"
#     }
#
#     dataLoader = MultiDataLoader()
#     data = dataLoader.loadData(conf)
