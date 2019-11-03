import numpy as np
from ml_tools.flattening import flatten_and_summarise, reconstruct


def test_end_to_end():

    input_arrays = {
        'x': np.random.randn(3, 3),
        'y': np.random.randn(3),
        'z': np.random.randn(4, 2),
        'f': np.random.randn(1)
    }

    flat_array, summaries = flatten_and_summarise(**input_arrays)

    reconstructed = reconstruct(flat_array, summaries)

    print(reconstructed)

    assert all([np.array_equal(reconstructed[x], input_arrays[x])
                for x in input_arrays])
