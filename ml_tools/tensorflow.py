import tensorflow as tf


def newton_optimize(start_f, fun, jac, hess, solve_fun=tf.linalg.solve,
                    tolerance=1e-5, debug=False, float_dtype=tf.float64):

    # TODO: Consider adding a maxiter
    # FIXME: The float casts are egregious.
    # TODO: Also, we need to make sure that the shapes are as expected.
    def body(f, difference):

        cur_hess = hess(f)

        # Ensure jac is a (column) vector
        cur_jac = tf.reshape(jac(f), (-1, 1))

        sol = tf.squeeze(solve_fun(cur_hess, cur_jac))

        new_f = f - sol

        if debug:
            # TODO: Not the neatest -- is there another way?
            print_tensors = [tf.print(cur_hess), tf.print(f)]
            hess_evals, _ = tf.linalg.eigh(cur_hess)
            print_tensors += [tf.print(tf.reduce_max(hess_evals) /
                                       tf.reduce_min(hess_evals))]
            with tf.control_dependencies(print_tensors):
                difference = tf.linalg.norm(f - new_f)
        else:
            difference = tf.linalg.norm(f - new_f)

        return (new_f, difference)

    init_val = (start_f, tf.constant(1., dtype=float_dtype))

    result = tf.while_loop(lambda f, difference: difference > tolerance, body,
                           init_val)

    return result[0]
