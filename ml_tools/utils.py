from datetime import datetime


def create_path_with_variables(**kwargs):

    to_concatenate = list()

    for cur_param, cur_val in kwargs.items():

        to_concatenate.append(f'{cur_param}={cur_val}')

    concatenated = ','.join(to_concatenate)
    concatenated = f'{concatenated}, {datetime.now()}'

    return concatenated
