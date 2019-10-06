import matplotlib.pyplot as plt


def plot_with_error_bars(x, lower, mean, upper, ax=None, **kwargs):

    if ax is None:
        _, ax = plt.subplots(1, 1)

    # Plot the median
    ax.plot(x, mean, **kwargs)
    ax.fill_between(x, lower, upper, alpha=0.5)

    return ax


def plot_with_error_bars_sd(x, mean, sd, ax=None, **kwargs):

    return plot_with_error_bars(x, mean - 2 * sd, mean, mean + 2 * sd,
                                ax=ax, **kwargs)


def add_legend_on_right(ax):

    # Shrink current axis by 20%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

    # Put a legend to the right of the current axis
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    return ax
