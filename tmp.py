import numpy as np
import pickle
from pprint import pprint
import tensorflow as tf


def iterate_mini_batch(data, size, batch_size):
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


if __name__ == '__main__':





    conf = {
        'file': './data/ml-1m.p',
        'type': 'U',
        'seed': 0,
        'hidden_layers': [500],
        'activate_fn' : 'tanh',
        'epochs': 10,
        'learning_rate': 0.01,

    }

    with open(conf['file'], 'rb') as datafile:

        data = pickle.load(datafile)
        train, test, info = data['train'], data['test'], data['info']

    # # print(info)
    size = info['nV']



    # Tensorflow code

    tf_indices = tf.placeholder(tf.int32, name = "indices")
    tf_ratings = tf.placeholder(tf.float32, name="ratings")

    tf_sparse = tf.sparse_to_dense(tf_indices, [30, size], tf_ratings)



    # print(size)
    test_data = train['U']['data']
    # print(test_data)
    count = 0
    for batch_indices, batch_ratings in iterate_mini_batch(test_data, size, 30):
        # count = count+1
        # print("<<<<<<<<<<<<<<<<<<<<<<<<<<<indices>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        # print(batch_indices)
        # print("******************************************ratings**********************************")
        # print(batch_ratings)

        print(tf.Session().run(tf_sparse, {tf_indices: batch_indices, tf_ratings: batch_ratings}))

        # break
        # if count == 10:
        #     break

    # print(count*30)




    # print(train['U']['data'][1])
