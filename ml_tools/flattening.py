import numpy as np
from collections import namedtuple, OrderedDict


array_info = namedtuple('array_info', 'shape')


def extract_info(array):

    return array_info(shape=array.shape)


def flatten_and_summarise(**input_arrays):

    input_arrays = OrderedDict(input_arrays)

    summaries = OrderedDict({x: extract_info(y) for x, y in
                             input_arrays.items()})
    flattened = np.concatenate([y.reshape(-1) for y in input_arrays.values()])

    return flattened, summaries


def reconstruct(flat_array, summaries, reshape_fun=np.reshape):

    # Base case
    if len(summaries) == 0:

        return {}

    cur_name, cur_summary = list(summaries.items())[0]

    # Cast to int is there to have this definitely work with TF
    cur_elements = int(np.prod(cur_summary.shape))

    cur_result = {
        cur_name: reshape_fun(flat_array[:cur_elements], cur_summary.shape)
    }

    remaining_summaries = OrderedDict({x: y for x, y in summaries.items()
                                       if x != cur_name})

    return {**cur_result, **reconstruct(flat_array[cur_elements:],
                                        remaining_summaries)}
