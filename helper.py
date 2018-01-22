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
        print("{} Directory not found".format(folder))

    return False

def make_dir(folder):
    os.makedirs(folder)
    print("{} is Created.".format(folder))
    return True

def duration(start, end):
    duration_ = end - start
    minutes = int(duration_//60)
    seconds = int(duration_%60)
    return minutes, seconds

def solidify_dict(data):

    solid_data = []
    for key, value in data.items():
        solid_data.append(value)

    return solid_data


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


def iterate_test_mini_batch(train_data, test_data, batch_size):

    train_indices = []
    train_ratings = []

    test_indices = []
    test_ratings = []
    currCount = 0

    for key, values in test_data.items():
        if currCount >= batch_size:
            yield (train_indices, train_ratings, test_indices, test_ratings)

            currCount = 0
            train_indices = []
            train_ratings = []

            test_indices = []
            test_ratings = []

        for value in values:
            test_indices.append([currCount, value[0]-1])
            test_ratings.append(value[1])

        for value in train_data[key]:
            train_indices.append([currCount, value[0]-1])
            train_ratings.append(value[1])

        currCount += 1

    yield (train_indices, train_ratings, test_indices, test_ratings)


def iterate_multi_mini_batch(data, batch_size, nRatings):
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


def iterate_multi_mini_batch_2(data, batch_size, nRatings, nItems):

    input_data = {}
    # set all items to 0 lists
    for i in range(nItems):
        input_data[i]= []

    output_indices = []
    output_ratings = []

    currCount = 0

    for key, values in data.items():
        if currCount >= batch_size:
            yield solidify_dict(input_data), output_indices, output_ratings

            currCount = 0
            # set all items to 0 lists
            input_data = {}
            for i in range(nItems):
                input_data[i]= []

            output_indices = []
            output_ratings = []


        seen_items = []
        for value in values:
            output_indices.append([currCount, (value[0]-1)])
            output_ratings.append(value[1])

            # Train_data
            input_data[value[0]-1].append(value[2])
            seen_items.append(value[0]-1)

        #  adding unkonw rated item of users to '0' ratings
        for i in range(nItems):
            if i not in seen_items:
                input_data[i].append([float(0) for _ in range(nRatings)])

        currCount+=1

    yield solidify_dict(input_data), output_indices, output_ratings


def iterate_multi_test_mini_batch(nCriteria, train_data, test_data, batch_size):

    train_indices = []
    train_ratings = []

    test_indices = []
    test_ratings = []
    currCount = 0

    for key, values in test_data.items():
        if currCount >= batch_size:
            yield (train_indices, train_ratings, test_indices, test_ratings)

            currCount = 0
            train_indices = []
            train_ratings = []

            test_indices = []
            test_ratings = []

        for value in values:
            test_indices.append([currCount, value[0]-1])
            test_ratings.append(value[1])

        for value in train_data[key]:
            for i in range(nCriteria):
                train_indices.append([currCount, (value[0]-1)*nCriteria+i])
                train_ratings.append(value[2][i])

        currCount += 1

    yield (train_indices, train_ratings, test_indices, test_ratings)


def iterate_multi_test_mini_batch_2(nCriteria, train_data, test_data, batch_size, nItems):
    result = []
    input_data = {}
    # set all items to 0 lists
    for i in range(nItems):
        input_data[i]= []

    output_indices = []
    output_ratings = []

    currCount = 0

    for key, values in test_data.items():
        if currCount >= batch_size:
            result.append([solidify_dict(input_data), output_indices, output_ratings])

            currCount = 0
            # set all items to 0 lists
            for i in range(nItems):
                input_data[i]= []

            output_indices = []
            output_ratings = []


        seen_items = []
        if key in train_data:
            for value in values:
                output_indices.append([currCount, value[0]-1])
                output_ratings.append(value[1])

            # Train_data
            for value in train_data[key]:
                input_data[value[0]-1].append(value[2])
                seen_items.append(value[0]-1)

        #  adding unkonw rated item of users to '0' ratings
        for i in range(nItems):
            if i not in seen_items:
                input_data[i].append([float(0) for _ in range(nCriteria)])

        currCount+=1

    while currCount < batch_size-1:
        for i in range(nItems):
            input_data[i].append([float(0) for _ in range(nCriteria)])
        currCount += 1

    result.append([solidify_dict(input_data), output_indices, output_ratings])

    return result
