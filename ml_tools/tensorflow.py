import tensorflow as tf


def newton_optimize(start_f, fun, jac, hess, solve_fun=tf.linalg.solve,
                    tolerance=1e-5):

    # TODO: Consider adding a maxiter
    # FIXME: The float casts are egregious.
    # TODO: Also, we need to make sure that the shapes are as expected.
    def body(f, difference):

        cur_hess = hess(f)

        # Ensure jac is a (column) vector
        cur_jac = tf.reshape(jac(f), (-1, 1))
        sol = tf.squeeze(solve_fun(cur_hess, cur_jac))

        new_f = f - sol

        difference = tf.linalg.norm(f - new_f)

        return (new_f, difference)

    init_val = (start_f, tf.constant(1., dtype=tf.float64))

    result = tf.while_loop(lambda f, difference: difference > tolerance, body,
                           init_val)

    return result[0]
