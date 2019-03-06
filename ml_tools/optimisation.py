import numpy


def newton_optimise(start_f, fun, jac, hess, solve_fun=numpy.linalg.solve):

    f = start_f

    n_iter = 0

    # Write a Newton routine
    difference = 1.

    while difference > 1e-5:

        cur_hess = hess(f)
        cur_jac = jac(f)

        sol = solve_fun(cur_hess, cur_jac)

        new_f = f - sol
        difference = numpy.linalg.norm(f - new_f)

        f = new_f

        n_iter = n_iter + 1

        if n_iter % 10 == 0:
            print(f"On iteration {n_iter}. Difference is {difference}")

    print(f"Converged after {n_iter} iterations.")

    return f
