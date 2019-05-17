import os
import pickle
from datetime import datetime


def create_path_with_variables(**kwargs):

    to_concatenate = list()

    for cur_param, cur_val in kwargs.items():

        to_concatenate.append(f'{cur_param}={cur_val}')

    concatenated = ','.join(to_concatenate)
    concatenated = f'{concatenated}, {datetime.now()}'

    return concatenated


def load_pickle_safely(path_to_pickle):

    assert os.path.isfile(path_to_pickle)

    with open(path_to_pickle, 'rb') as f:
        return pickle.load(f)
