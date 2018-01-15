import os

def find_file(filename):
    path_list = filename.split('/')
    path = ''
    for val in path_list[:-1]:
        path += val+'/'

    try:
        files = os.listdir(path)
        if path_list[-1] in files:
            return True
    except FileNotFoundError:
        os.makedirs(path)

    return False

def find_dir(folder):
    try:
        files = os.listdir(folder)
        return True
    except FileNotFoundError:
        os.makedirs(folder)
        print("Directory Created : {}".format(folder))

    return False

def duration(start, end):
    duration_ = end - start
    minutes = int(duration_//60)
    seconds = int(duration_%60)
    return minutes, seconds


def iterate_mini_batch(data, batch_size):

    indices = []
    ratings = []
    currCount = 0

    for key, values in data.items():
        if currCount >= batch_size:
            yield(indices, ratings)

            currCount = 0
            indices = []
            ratings = []

        for value in values:
            indices.append([currCount, value[0]-1])
            ratings.append(value[1])
        currCount += 1

    yield (indices, ratings)

def iterate_mini_batch_multi(data, batch_size, nRatings):
    indices_multi = []
    ratings_multi = []

    indices = []
    ratings = []

    currCount = 0

    for key, values in data.items():
        if currCount >= batch_size:
            yield indices_multi, ratings_multi, indices, ratings

            currCount = 0
            indices = []
            ratings = []
            indices_multi = []
            ratings_multi = []

        for value in values:
            indices.append([currCount, value[0]-1])
            ratings.append(value[1])

            for i in range(nRatings):
                indices_multi.append([currCount, (value[0]-1)*nRatings+i])
                ratings_multi.append(value[2][i])
        currCount+=1

    yield indices_multi, ratings_multi, indices, ratings


# def clean_indices(indices, ratings):
#
#     for index in
