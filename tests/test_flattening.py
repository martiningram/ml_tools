import numpy as np
import tensorflow as tf
from ml_tools.flattening import (flatten_and_summarise, reconstruct_np,
                                 reconstruct_tf)  # NOQA


def generate_arrays():

    input_arrays = {
        'x': np.random.randn(3, 3),
        'y': np.random.randn(3),
        'z': np.random.randn(4, 2),
        'f': np.random.randn(1)
    }

    return input_arrays


def test_end_to_end():

    input_arrays = generate_arrays()

    flat_array, summaries = flatten_and_summarise(**input_arrays)

    reconstructed = reconstruct_np(flat_array, summaries)

    assert all([np.array_equal(reconstructed[x], input_arrays[x])
                for x in input_arrays])


def test_with_tf():

    input_arrays = generate_arrays()

    # Turn into TF version
    tf_arrays = {x: tf.constant(y) for x, y in input_arrays.items()}

    flat_array, summaries = flatten_and_summarise(**input_arrays)

    reconstructed = reconstruct_tf(flat_array, summaries,
                                   reshape_fun=tf.reshape)

    checks = [tf.reduce_all(tf.equal(reconstructed[x], tf_arrays[x])) for x in
              input_arrays]

    assert tf.reduce_all(checks).numpy()
