import numpy as np
import pandas as pd
from ml_tools.scaling import (standardise, apply_standardisation,
                              invert_standardisation)


def test_standardise_all():

    X = pd.DataFrame(np.random.randn(100, 4))

    X_standard, _ = standardise(X)

    assert np.allclose(X_standard.mean().values, np.zeros(4))
    assert np.allclose(X_standard.std().values, np.ones(4))


def test_standardise_some():

    X = pd.DataFrame(np.random.randn(100, 4))

    to_standardise = [0, 2]

    X_standard, _ = standardise(X, to_standardise)

    assert(X_standard.shape == X.shape)
    assert np.allclose(X_standard[to_standardise].mean().values, np.zeros(2))
    assert np.allclose(X_standard[to_standardise].std().values, np.ones(2))
    assert np.allclose(X_standard[[1, 3]].values, X[[1, 3]].values)


def test_invert_standardisation():

    X = pd.DataFrame(np.random.randn(100, 4))

    to_standardise = [0, 2]

    X_standard, scaling_dict = standardise(X, to_standardise)

    inverted = invert_standardisation(X_standard, scaling_dict)

    assert np.allclose(X.values, inverted.values)


def test_apply_standardisation():

    X = pd.DataFrame(np.random.randn(100, 4))

    to_standardise = [0, 2]

    X_standard, scaling_dict = standardise(X, to_standardise)

    X_standard_alt = apply_standardisation(X, scaling_dict)

    assert np.allclose(X_standard.values, X_standard_alt.values)
