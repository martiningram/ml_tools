import tensorflow as tf


def ard_rbf_kernel(x1, x2, lengthscales, alpha, jitter=1e-5):

    # x1 is N1 x D
    # x2 is N2 x D (and N1 can be equal to N2)

    # Must have same number of dimensions
    assert(x1.get_shape()[1] == x2.get_shape()[1])

    # Also must match lengthscales
    assert(lengthscales.get_shape()[0] == x1.get_shape()[1])

    # Use broadcasting
    # X1 will be (N1, 1, D)
    x1_expanded = tf.expand_dims(x1, axis=1)
    # X2 will be (1, N2, D)
    x2_expanded = tf.expand_dims(x2, axis=0)

    # These will be N1 x N2 x D
    scaled_diffs = ((x1_expanded - x2_expanded) / lengthscales)**2

    # Use broadcasting to do a dot product
    exponent = tf.reduce_sum(scaled_diffs, axis=2)

    kern = alpha**2 * tf.exp(-0.5 * exponent)

    # Jitter this a little bit
    kern = tf.matrix_set_diag(kern, tf.matrix_diag_part(kern) + jitter)

    return kern


def bias_kernel(x1, x2, sd, jitter=1e-5):

    output_rows = int(x1.get_shape()[0])
    output_cols = int(x2.get_shape()[0])

    to_multiply = tf.ones((output_rows, output_cols), dtype=x1.dtype)

    kern = sd**2 * to_multiply
    kern = tf.matrix_set_diag(kern, tf.matrix_diag_part(kern) + jitter)

    return kern


def ard_rbf_kernel_batch(x1, x2, lengthscales, alpha, jitter=1e-5):

    # x1 is N1 x D
    # x2 is N2 x D (and N1 can be equal to N2)
    # lengthscales is B x D [B is batch dim]
    # alpha is B,

    # Must have same number of dimensions
    assert(x1.get_shape()[1] == x2.get_shape()[1])

    # Also must match lengthscales
    assert(lengthscales.get_shape()[1] == x1.get_shape()[1])

    l_expanded = tf.expand_dims(lengthscales, axis=1)

    # Divide x1 by lengthscales
    # Gives (B x N x D)
    x1 = x1 / l_expanded
    x2 = x2 / l_expanded

    # This will be (B x N)
    x1s = tf.reduce_sum(tf.square(x1), axis=2)
    x2s = tf.reduce_sum(tf.square(x2), axis=2)

    # This produces an (N x N) matrix
    cross_prods = -2 * tf.matmul(x1, x2, transpose_b=True)

    # This should produce a (B x N x N) distance mat
    dist = (cross_prods + tf.expand_dims(x1s, axis=2) +
            tf.expand_dims(x2s, axis=1))

    # Multiply all of these
    kern = (tf.expand_dims(tf.expand_dims(alpha**2, axis=1), axis=1) *
            tf.exp(-0.5 * dist))

    # Jitter this a little bit
    kern = tf.matrix_set_diag(kern, tf.matrix_diag_part(kern) + jitter)

    return kern
