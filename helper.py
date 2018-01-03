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
