from scipy.spatial import KDTree
import numpy as np


def extreme_point(data_points, k_nearest_neighbours):
    # defined as the point appearing least in other points k nearest neighbours
    tree = KDTree(data_points[["x", "y"]].to_numpy())

    # find the closest k points to each point
    closest_k = tree.query(data_points[["x", "y"]].to_numpy(), k_nearest_neighbours)

    # find how many times each point appears in the k nearest neighbours or other points
    counts_for_each_point = np.unique(closest_k[1], return_counts=True)[1]

    # select the one which appears the least
    extreme_point = data_points.index[np.argmin(counts_for_each_point)]

    return extreme_point


def get_query_points(data_points, k_nearest_neighbours, verbose=False):

    all_points_tree = KDTree(data_points[["x", "y"]].to_numpy())
    returned_points = set()
    query_points = []

    # recursively set the extreme point as the centre of the search query and then find the
    # next extreme point
    while len(returned_points) < len(data_points):

        if verbose:
            print(
                "Returned points =",
                len(returned_points),
                "out of",
                len(data_points),
                "points",
                end="\r",
            )

        # get uuids of possible points
        possible_points = [
            point for point in data_points.index if point not in returned_points
        ]

        # # filter coordinates to possible points
        updated_data_points = data_points.loc[possible_points].copy()

        # # find extreme point
        point = extreme_point(updated_data_points, k_nearest_neighbours)

        # find the k nearest in the original to this one
        k_nearest = all_points_tree.query(
            data_points.loc[point].to_numpy().reshape(-1), k_nearest_neighbours
        )[1]
        k_nearest_uuids = set(data_points.index[k_nearest])

        # add to list of query points
        query_points += [point]

        # add neighbours to list of returned points
        returned_points = returned_points.union(k_nearest_uuids)

    if verbose:
        print("\n")

    return data_points.loc[query_points][["x", "y"]].to_numpy()
