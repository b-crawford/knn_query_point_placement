import numpy as np
import scipy.spatial as scsp
import pandas as pd
import uuid


def generate_query_points(
    min_query_points, max_query_points, approx_x_bounds, approx_y_bounds
):

    # generate the query points, the idea is that if these are selected as a point
    # at which to conduct knn they will return the relevant points
    if min_query_points == max_query_points:
        n_query_points = min_query_points
    else:
        n_query_points = np.random.randint(min_query_points, max_query_points)
    query_points_x = np.random.uniform(
        approx_x_bounds[0], approx_x_bounds[1], n_query_points
    )
    query_points_y = np.random.uniform(
        approx_y_bounds[0], approx_y_bounds[1], n_query_points
    )

    return pd.DataFrame(
        {
            "query_point_uuid": [str(uuid.uuid4()) for _ in range(n_query_points)],
            "x": query_points_x,
            "y": query_points_y,
        }
    )


def distances_between_query_points(query_points):
    distance_matrix = scsp.distance.cdist(
        query_points[["x", "y"]], query_points[["x", "y"]], "euclidean"
    )

    distance_df = pd.DataFrame(
        distance_matrix,
        columns=query_points["query_point_uuid"],
        index=query_points["query_point_uuid"],
    )
    distance_df = (
        distance_df.reset_index()
        .melt(id_vars="query_point_uuid", var_name="target", value_name="distance")
        .rename(columns={"query_point_uuid": "source"})
    )

    distance_df = distance_df[distance_df["source"] != distance_df["target"]]

    return distance_df


def sample_uniformly_from_circle(centre, radius, n_points):

    distance = np.sqrt(np.random.uniform(0, radius, n_points))
    angle = np.random.uniform(0, 2 * np.pi, n_points)

    return np.array([distance * np.cos(angle), distance * np.sin(angle)]).T + centre


def sample_data_points_from_query_points(
    query_points, points_per_query_point, scale=0.4
):

    distances = distances_between_query_points(query_points)

    radius = distances.groupby("source").distance.min().reset_index()
    query_points = query_points.merge(
        radius, left_on="query_point_uuid", right_on="source"
    )

    points_dfs = []
    for query_point in query_points.to_dict("records"):
        points = sample_uniformly_from_circle(
            np.array([query_point["x"], query_point["y"]]),
            query_point["distance"] * scale,
            points_per_query_point,
        )

        points_dfs += [
            pd.DataFrame(
                {
                    "query_point_uuid": query_point["query_point_uuid"],
                    "point_uuid": [
                        str(uuid.uuid4()) for _ in range(points_per_query_point)
                    ],
                    "x": points[:, 0],
                    "y": points[:, 1],
                }
            )
        ]

    return pd.concat(points_dfs).reset_index(drop=True)


def simulation(
    min_query_points,
    max_query_points,
    approx_x_bounds,
    approx_y_bounds,
    points_per_query_point,
):

    query_points = generate_query_points(
        min_query_points, max_query_points, approx_x_bounds, approx_y_bounds
    )

    data_points = sample_data_points_from_query_points(
        query_points, points_per_query_point
    )

    return query_points, data_points
