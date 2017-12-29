import tensorflow as tf


class ErrorTest:


    def __init__(self):
        self.__mae = 0
        self.__rms = 0
        self.__mae_list = []
        self.__rms_list = []

    def iterate_mini_batch(self, data, size, batch_size):

        indices = []
        ratings = []
        currCount = 0
        for key, values in data.items():
            if currCount < batch_size:
                for value in values:
                    indices.append([currCount, value[0]-1])
                    ratings.append(value[1])
                currCount = currCount+1
            else:
                yield (indices, ratings)

                currCount = 0
                indices = []
                ratings = []

                for value in values:
                    indices.append([currCount, value[0]-1])
                    ratings.append(value[1])
                currCount = currCount+1

        yield (indices, ratings)


    def error_fn(self, model, train, test):

        predict = model['predict_test']


        
