import numpy as np
import pandas as pd
import uuid
import scipy.spatial as scsp


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
    return distance_df


def distances_to_query_points(current_point_x, current_point_y, query_points):
    query_points = query_points.copy()
    query_points["distance"] = scsp.distance.cdist(
        [[current_point_x, current_point_y]], query_points[["x", "y"]], "euclidean"
    )[0]
    return query_points[["query_point_uuid", "distance"]]


def distances_to_furthest_knn(current_points):
    return (
        current_points.copy()
        .groupby("query_point_uuid")["distance"]
        .max()
        .reset_index()
        .rename(columns={"distance": "distance_to_furthest_nn"})
    )


# def sample_new_data_point(
#     query_points,
#     query_point_uuid,
#     current_points,
#     verbose,
#     reccursion_count=0,
# ):

#     assert reccursion_count < 100, (
#         "Recusion count exceeded 100, it is reccomended to reduce the normal_sd or "
#         "increase the bounds when sampling the query points"
#     )

#     query_point = query_points.loc[
#         query_points["query_point_uuid"] == query_point_uuid
#     ].copy()
#     query_point_x = query_point["x"].iloc[0]
#     query_point_y = query_point["y"].iloc[0]

#     # sample a new point
#     distance = np.random.uniform(0, 1, 1)[0]

#     x = np.random.uniform(query_point_x, normal_sd, 1)[0]
#     y = np.random.uniform(query_point_y, normal_sd, 1)[0]

#     # check if the new plot is closer to any of the other query points than their
#     # current furthest neighbour
#     distances_to_furthest_nn = distances_to_furthest_knn(current_points)
#     distances_from_current_point = distances_to_query_points(x, y, query_points)

#     compare = pd.merge(
#         distances_to_furthest_nn.loc[
#             distances_to_furthest_nn.query_point_uuid != query_point_uuid
#         ],
#         distances_from_current_point.loc[
#             distances_from_current_point.query_point_uuid != query_point_uuid
#         ],
#         on="query_point_uuid",
#     )

#     not_in_other_points_knn = len(compare) == 0 or all(
#         compare["distance"] > compare["distance_to_furthest_nn"]
#     )

#     # also check if the new point closer to the current query point than any of the
#     # other query points
#     closest_to_current_query_point = (
#         distances_from_current_point.sort_values("distance").iloc[0]["query_point_uuid"]
#         == query_point_uuid
#     )

#     # if the new point is closer than the furthest neighbour, sample again:
#     if not_in_other_points_knn and closest_to_current_query_point:
#         distance = distances_from_current_point.loc[
#             distances_from_current_point.query_point_uuid == query_point_uuid
#         ]["distance"].iloc[0]
#         x, y, distance
#         return x, y, distance
#     else:
#         if verbose:
#             print("Resampling")
#         return sample_new_data_point(
#             query_points=query_points,
#             query_point_uuid=query_point_uuid,
#             current_points=current_points,
#             normal_sd=normal_sd,
#             verbose=verbose,
#         )


def sample_new_data_point(
    query_points,
    current_points,
    distances_query_points,
    query_point_uuid,
    verbose,
    reccursion_count=0,
):
    """Function to sample uniformly at random (i think) from the Voronoi cell of the query point in question."""

    # Calculate the maximum distance between query point of interest and all
    # other query points
    max_distance = distances_query_points.loc[
        distances_query_points.source == query_point_uuid
    ].distance.max()
    max_distance_to_sample = max_distance / 2

    # sample a distance
    distance = np.sqrt(np.random.uniform(0, max_distance_to_sample, 1)[0])

    # Choose a direction in which to sample
    angle = np.pi * np.random.uniform(0, 2)
    direction_vec = np.array([np.cos(angle), np.sin(angle)])
    direction_vec_norm = direction_vec / np.linalg.norm(direction_vec)

    # now we have a new point
    query_point = query_points.loc[
        query_points["query_point_uuid"] == query_point_uuid
    ][["x", "y"]].values[0]
    new_point = query_point + direction_vec_norm * distance

    # reject the point if it is closer to any other query point
    distances_from_current_point = distances_to_query_points(
        new_point[0], new_point[1], query_points
    )
    distances_to_other_query_points = distances_from_current_point.loc[
        distances_from_current_point.query_point_uuid != query_point_uuid
    ]
    closer_to_another_query_point = (
        len(
            distances_to_other_query_points.loc[
                distances_to_other_query_points.distance <= distance
            ]
        )
        > 0
    )

    # also reject if it is closer to another query point than its current furthest neighbour
    distances_to_furthest_nn = distances_to_furthest_knn(current_points)

    compare = pd.merge(
        distances_to_furthest_nn.loc[
            distances_to_furthest_nn.query_point_uuid != query_point_uuid
        ],
        distances_to_other_query_points,
        on="query_point_uuid",
    )

    in_other_points_knn = len(compare) > 0 and any(
        compare["distance"] < compare["distance_to_furthest_nn"]
    )

    if closer_to_another_query_point or in_other_points_knn:
        return sample_new_data_point(
            query_points,
            current_points,
            distances_query_points,
            query_point_uuid,
            verbose,
            reccursion_count + 1,
        )
    else:
        return new_point[0], new_point[1], distance


def generate_points_from_query_points(query_points, points_per_query_point, verbose):

    current_points = pd.DataFrame(
        {
            "point_uuid": np.nan,
            "query_point_uuid": np.nan,
            "x": np.nan,
            "y": np.nan,
            "distance": np.nan,
        },
        index=[],
    )

    distances = distances_between_query_points(query_points)

    for query_point_uuid in query_points["query_point_uuid"]:
        if verbose:
            print("Query point: {}".format(query_point_uuid))
        for i in range(points_per_query_point):
            if verbose:
                print("Generating point {} of {}".format(i + 1, points_per_query_point))
            x, y, distance = sample_new_data_point(
                query_points, current_points, distances, query_point_uuid, verbose
            )
            current_points = pd.concat(
                [
                    current_points,
                    pd.DataFrame(
                        {
                            "point_uuid": str(uuid.uuid4()),
                            "query_point_uuid": query_point_uuid,
                            "x": x,
                            "y": y,
                            "distance": distance,
                        },
                        index=[0],
                    ),
                ],
                ignore_index=True,
            )

    return current_points

    return np.array(points)


def simulation(
    min_query_points,
    max_query_points,
    points_per_query_point,
    approx_x_bounds,
    approx_y_bounds,
    verbose=False,
):
    query_points = generate_query_points(
        min_query_points,
        max_query_points,
        approx_x_bounds,
        approx_y_bounds,
    )
    data_points = generate_points_from_query_points(
        query_points, points_per_query_point, verbose
    )

    return query_points, data_points
