import numpy as np


def uniform_random_points(n_points, x_bounds, y_bounds):

    x = np.random.uniform(x_bounds[0], x_bounds[1], n_points)
    y = np.random.uniform(y_bounds[0], y_bounds[1], n_points)

    return np.column_stack((x, y))
