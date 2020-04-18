import numpy as np
import tensorflow as tf
from ml_tools.lin_alg import generate_random_pos_def
from ml_tools.tensorflow import covar_to_corr_and_scales


def test_covar_to_corr_and_scales():

    pos_def = generate_random_pos_def(10)

    corr, scales = covar_to_corr_and_scales(pos_def)

    # Check this matches the original matrix:
    diag_scales = tf.linalg.diag(scales)

    recovered_mat = diag_scales @ corr @ diag_scales

    assert np.allclose(recovered_mat.numpy(), pos_def)
    assert np.allclose(np.abs(corr.numpy()).max(), 1.)
