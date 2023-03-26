from scipy.spatial import KDTree
import itertools
import math
import numpy as np


def get_query_points(
    data_points, k_nearest_neighbours, x_bounds, y_bounds, size_of_grid_search
):

    # do some calculations to work out computational complexity
    theoretical_min = math.ceil(len(data_points) / k_nearest_neighbours)
    number_of_unique_neighbour_sets = math.comb(len(data_points), k_nearest_neighbours)
    number_of_combinations_of_size_theoretical_min = math.comb(
        number_of_unique_neighbour_sets, theoretical_min
    )

    assert (
        number_of_combinations_of_size_theoretical_min < 10000
    ), "Too many combinations to search, grid search not appropriate"

    tree = KDTree(data_points.to_numpy())

    # split the space into a grid
    x_grid = np.linspace(x_bounds[0], x_bounds[1], size_of_grid_search)
    y_grid = np.linspace(y_bounds[0], y_bounds[1], size_of_grid_search)

    search_grid = itertools.product(x_grid, y_grid)

    print(
        "Iterating through grid of",
        len(x_grid) * len(y_grid),
        "points to find unique nearest neighbours",
    )
    search_grid = np.array(list(itertools.product(x_grid, y_grid)))

    # only keep unique points to save memory
    search_points = []
    indices_of_search_points = []

    for i, (x_point, y_point) in enumerate(search_grid):
        new_neighbours = tree.query([x_point, y_point], k_nearest_neighbours)[
            1
        ].tolist()
        if new_neighbours not in search_points:
            search_points += [new_neighbours]
            indices_of_search_points += [i]

    print("Found", len(search_points), "unique nearest neighbour sets")

    search_points = np.array(search_points)
    indices_of_search_points = np.array(indices_of_search_points)

    # find the maximum number of points to cover the entire space
    theoretical_max = len(data_points)

    # now work our way through all combinations returning the search point
    for i in range(theoretical_min, theoretical_max):
        print("\n")
        print("Searching solutions with", i, "search points")
        number_of_combinations = math.comb(len(search_points), i)
        for j, combination in enumerate(
            itertools.combinations(enumerate(search_points), i)
        ):
            print("Checking combination", j, "of", number_of_combinations, end="\r")
            indices_of_product = [i[0] for i in combination]
            data_points_found = [i[1] for i in combination]
            if len(set(np.concatenate(data_points_found))) == len(data_points):
                print("\n")
                print("Solution found, stopping")
                best_solution_indices = indices_of_search_points[indices_of_product]
                return search_grid[best_solution_indices, :]

        print("\n")

    return None
