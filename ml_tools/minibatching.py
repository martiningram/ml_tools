import numpy as np
from functools import partial
from tqdm import tqdm
from typing import Dict, Tuple, Callable, Any, Optional
from os.path import join
import os


def get_batch_indices(indices: np.ndarray, batch_size: int, cur_start: int) \
        -> Tuple[np.ndarray, int]:
    """
    Moves through a sequence of indices, extracting a batch each time and
    looping back at the end.

    Args:
        indices: The array of indices
        batch_size: The number of elements to pick
        cur_start: The current position in the array of indices

    Example:
        get_batch_indices(np.array([1, 2, 3, 4]), 3, 2) should return
        np.array([3, 4, 1]) and 1 [we're looping back to the start].
    """

    array_length = len(indices)

    cur_end = cur_start + batch_size

    if cur_end < array_length:

        picked = indices[cur_start:cur_end]

        new_start = cur_end

    else:

        # First pick whatever we can
        picked = indices[cur_start:]

        still_to_pick = batch_size - len(picked)

        picked = np.concatenate([picked, indices[:still_to_pick]])

        new_start = still_to_pick

    return picked, new_start


def next_batch(array_dict: Dict[str, np.ndarray],
               shuffled_indices: np.ndarray,
               cur_position: int,
               batch_size: int) -> Tuple[Dict[str, np.ndarray], int]:
    """
    Returns the next batch of arrays.

    Args:
        array_dict: A dictionary of arrays.
        shuffled_indices: The order of indices to traverse.
        cur_position: Where in the indices we currently are.
        batch_size: The size of batches to return.

    Returns:
        A tuple with the current batch of arrays and the next position.
    """

    indices, next_position = get_batch_indices(
        shuffled_indices, batch_size, cur_position)

    subset_arrays = {x: y[indices] for x, y in array_dict.items()}

    return subset_arrays, next_position


def optimise_minibatching(
        data_dict: Dict[str, np.ndarray],
        to_optimise: Callable[[np.ndarray, Any],
                              Tuple[np.ndarray, np.ndarray]],
        opt_step_fun: Callable[[Any, np.ndarray, np.ndarray],
                               Tuple[Any, np.ndarray]],
        theta: np.ndarray,
        batch_size: int,
        n_steps: int,
        n_data: int,
        log_file: Optional[str] = None,
        append_to_log_file: bool = True,
        opt_state: Any = None):
    """
    Optimises a function using minibatching.

    Args:
        data_dict: The full dataset in the form of a dictionary of arrays.
        to_optimise: The function to optimise. It takes the current setting
            of parameters, theta, as its first arguments, and the elements
            of the data dictionary as its others. It returns the objective
            and its gradient.
        opt_step_fun: Takes the current state of the optimiser, the current
            parameter setting, and the gradient of the objective. It produces
            an updated state of the optimiser and the new parameter settings.
        theta: The initial parameter setting.
        batch_size: The batch size to use.
        n_steps: How many steps to run the optimisation for
        n_data: How many data points there are in total
        log_file: If given, writes the sequence of losses to the log file.
        append_to_log_file: Appends to log file if true, otherwise creates a
            new one.
        opt_state: The initial state of the optimiser.

    Returns:
        A tuple containing the final setting of parameters theta and the
        list of objective values during optimisation.
    """

    # Pick shuffled indices
    indices = np.random.permutation(n_data)
    cur_position = 0

    if log_file is not None:

        if append_to_log_file:
            log_file_handle = open(log_file, 'a')
        else:
            log_file_handle = open(log_file, 'w')

    else:

        log_file_handle = None

    loss_log = list()

    for i in tqdm(range(n_steps)):

        # Get the array subset
        cur_arrays, cur_position = next_batch(
            data_dict, indices, cur_position, batch_size)

        cur_opt_fun = partial(to_optimise, **cur_arrays)
        obj, grad = cur_opt_fun(theta)

        theta, opt_state = opt_step_fun(opt_state, theta, grad)

        if log_file_handle is not None:

            log_dir = os.path.split(log_file)[0]

            np.savez(join(log_dir, f'adam_state_{i}'), **opt_state._asdict())
            log_file_handle.write(f'{obj}\n')
            log_file_handle.flush()

        loss_log.append(obj)

    if log_file is not None:
        log_file_handle.close()

    return theta, loss_log
