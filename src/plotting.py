import matplotlib.pyplot as plt
from scipy.spatial import KDTree
import numpy as np
from matplotlib.pyplot import cm


def plot(
    data_points: np.array,
    query_points: np.array = None,
    k_nearest_neighbours: int = None,
    figsize=(6, 6),
    query_point_labels=None,
):

    plt.figure(figsize=figsize)
    plt.scatter(data_points[:, 0], data_points[:, 1], c="b", label="Data points")

    if query_points is not None:
        plt.scatter(
            query_points[:, 0],
            query_points[:, 1],
            c="r",
            label="Query points",
            marker="x",
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

    if query_point_labels is not None:
        for i, txt in enumerate(query_point_labels):
            plt.annotate(txt, (query_points[i][0], query_points[i][1]))


def plot_feasible_area(
    A, b, x_bounds, y_bounds, figsize=(12, 3), save_to=None, grid_size=300
):

    fig = plt.figure(figsize=(12, 6))
    ax = plt.subplot(1, 1, 1)

    x_range = np.linspace(x_bounds[0], x_bounds[1], grid_size)
    y_range = np.linspace(y_bounds[0], y_bounds[1], grid_size)

    x_space, y_space = np.meshgrid(x_range, y_range)

    def _in_feasible_region(x, y, A, b):
        return np.all(np.dot(np.column_stack([x, y]), A.T) >= b, axis=1)

    feasible_region = []
    for x, y in zip(x_space, y_space):
        feasible_region += [_in_feasible_region(x, y, A, b)]

    feasible_region = np.array(feasible_region)

    ax.imshow(
        feasible_region.astype(int),
        extent=np.concatenate([x_bounds, y_bounds]),
        origin="lower",
        cmap="Greys",
        alpha=0.3,
    )

    color = cm.rainbow(np.linspace(0, 1, len(A)))

    # plot the lines
    for i, A_row, b_row in zip(color, A, b):
        # if it is a vertical line then plot as such
        if A_row[1] == 0:
            symbol = ">" if A_row[0] > 0 else "<"
            ax.axvline(
                x=b_row / A_row[0],
                color=i,
                label=f"x {symbol} {round(b_row/A_row[0], 2)}",
            )
        # if it is a horizontal line then plot as such
        elif A_row[0] == 0:
            symbol = ">" if A_row[1] > 0 else "<"
            ax.axhline(
                y=b_row / A_row[1],
                color=i,
                label=f"y {symbol} {round(b_row/A_row[1], 2)}",
            )
        else:
            m = -A_row[0] / A_row[1]
            c = b_row / A_row[1]
            symbol = ">" if A_row[1] > 0 else "<"
            ax.plot(
                x_range,
                m * x_range + c,
                color=i,
                label=f"y {symbol} {round(m, 2)}x + {round(c, 2)}",
            )

    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.xlim(x_bounds[0], x_bounds[1])
    plt.ylim(y_bounds[0], y_bounds[1])

    if save_to is not None:
        plt.savefig(save_to, bbox_inches="tight")

    return ax
