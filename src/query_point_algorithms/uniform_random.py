import numpy as np
from scipy.spatial import KDTree


def sample_next_query_point(remaining_data_points):
    # sample a point from the remaining data points
    point = remaining_data_points.sample(1).iloc[0].to_dict()

    return np.array([point['x'], point['y']])


# add points uniformly at random until we cover all our points
def get_query_points(data_points, k_nearest_neighbours, verbose=False):

    tree = KDTree(data_points[['x', 'y']].to_numpy())

    # get the initial point
    intial_point = sample_next_query_point(data_points)

    # get the neighbours of the initial point
    neighbours = tree.query(intial_point, k_nearest_neighbours)[1]
    number_of_unique_neighbours = len(set(neighbours))

    # now iterate until we have all the points
    query_points = [intial_point]

    while number_of_unique_neighbours < len(data_points):
        if verbose:
            print(
                f"Found {number_of_unique_neighbours} out of {len(data_points)} points"
            )

        # remove the neighbours from the data points
        remaining_data = data_points.drop(neighbours).copy()

        # sample a new point
        new_point = sample_next_query_point(remaining_data)
        query_points += [new_point]

        # get the neighbours of the new point
        new_neighbours = tree.query(new_point, k_nearest_neighbours)[1]

        # add the new neighbours to the list of neighbours
        neighbours = set(neighbours).union(set(new_neighbours))
        number_of_unique_neighbours = len(neighbours)

    return np.array(query_points)
