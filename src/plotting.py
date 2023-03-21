import matplotlib.pyplot as plt
from scipy.spatial import KDTree
import numpy as np


def plot(
    query_points: np.array, data_points: np.array, k_nearest_neighbours, figsize=(6, 6)
):

    plt.figure(figsize=figsize)
    plt.scatter(data_points[:, 0], data_points[:, 1], c="b", label="Data points")
    plt.scatter(
        query_points[:, 0], query_points[:, 1], c="r", label="Query points", marker="x"
    )

    # add lines from query points to data points
    tree = KDTree(data_points)
    closest_points = tree.query(query_points, k_nearest_neighbours)

    for point, knn in zip(query_points, closest_points[1]):
        for neighbour in knn:
            plt.plot(
                [point[0], data_points[neighbour][0]],
                [point[1], data_points[neighbour][1]],
                c="r",
                linestyle="--",
            )

    plt.legend()
