import numpy as np
import pickle
from pprint import pprint
import tensorflow as tf

import os


if 'dataeeee' in os.listdir('.'):
    print("True")
else:
    os.makedirs('dataeeee')
    print(os.listdir('.'))
