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

    return False

def duration(start, end):
    duration_ = end - start
    minutes = int(duration_//60)
    seconds = int(duration_%60)
    return minutes, seconds
