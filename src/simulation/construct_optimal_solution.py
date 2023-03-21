import numpy as np
import pandas as pd
import uuid
import scipy.spatial as scsp


def generate_query_points(
    min_query_points, max_query_points, approx_x_bounds, approx_y_bounds, max_normal_sd
):

    # generate the query points, the idea is that if these are selected as a point
    # at which to conduct knn they will return the relevant points
    n_query_points = np.random.randint(min_query_points, max_query_points)
    query_points_x = np.random.uniform(
        approx_x_bounds[0], approx_x_bounds[1], n_query_points
    )
    query_points_y = np.random.uniform(
        approx_y_bounds[0], approx_y_bounds[1], n_query_points
    )

    query_points_sd = np.random.uniform(0.1, max_normal_sd, n_query_points)

    return pd.DataFrame(
        {
            "query_point_uuid": [str(uuid.uuid4()) for _ in range(n_query_points)],
            "x": query_points_x,
            "y": query_points_y,
            "normal_sd": query_points_sd,
        }
    )


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


def sample_new_data_point(
    query_points,
    query_point_uuid,
    current_points,
    normal_sd,
    verbose,
    reccursion_count=0,
):

    assert reccursion_count < 100, (
        "Recusion count exceeded 100, it is reccomended to reduce the normal_sd or "
        "increase the bounds when sampling the query points"
    )

    query_point = query_points.loc[
        query_points["query_point_uuid"] == query_point_uuid
    ].copy()
    query_point_x = query_point["x"].iloc[0]
    query_point_y = query_point["y"].iloc[0]

    # sample a new point
    x = np.random.normal(query_point_x, normal_sd, 1)[0]
    y = np.random.normal(query_point_y, normal_sd, 1)[0]

    # check if the new plot is closer to any of the other query points than their
    # current furthest neighbour
    distances_to_furthest_nn = distances_to_furthest_knn(current_points)
    distances_from_current_point = distances_to_query_points(x, y, query_points)

    compare = pd.merge(
        distances_to_furthest_nn.loc[
            distances_to_furthest_nn.query_point_uuid != query_point_uuid
        ],
        distances_from_current_point.loc[
            distances_from_current_point.query_point_uuid != query_point_uuid
        ],
        on="query_point_uuid",
    )

    not_in_other_points_knn = len(compare) == 0 or all(
        compare["distance"] > compare["distance_to_furthest_nn"]
    )

    # also check if the new point closer to the current query point than any of the
    # other query points
    closest_to_current_query_point = (
        distances_from_current_point.sort_values("distance").iloc[0]["query_point_uuid"]
        == query_point_uuid
    )

    # if the new point is closer than the furthest neighbour, sample again:
    if not_in_other_points_knn and closest_to_current_query_point:
        distance = distances_from_current_point.loc[
            distances_from_current_point.query_point_uuid == query_point_uuid
        ]["distance"].iloc[0]
        x, y, distance
        return x, y, distance
    else:
        if verbose:
            print("Resampling")
        return sample_new_data_point(
            query_points=query_points,
            query_point_uuid=query_point_uuid,
            current_points=current_points,
            normal_sd=normal_sd,
            verbose=verbose,
        )


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
    for query_point_uuid in query_points["query_point_uuid"]:
        if verbose:
            print("Query point: {}".format(query_point_uuid))
        for i in range(points_per_query_point):
            if verbose:
                print("Generating point {} of {}".format(i + 1, points_per_query_point))
            normal_sd = query_points.loc[
                query_points.query_point_uuid == query_point_uuid
            ]["normal_sd"].iloc[0]
            x, y, distance = sample_new_data_point(
                query_points, query_point_uuid, current_points, normal_sd, verbose
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


def simulation(
    min_query_points,
    max_query_points,
    points_per_query_point,
    approx_x_bounds,
    approx_y_bounds,
    max_normal_sd,
    verbose=False,
):
    query_points = generate_query_points(
        min_query_points,
        max_query_points,
        approx_x_bounds,
        approx_y_bounds,
        max_normal_sd,
    )
    data_points = generate_points_from_query_points(
        query_points, points_per_query_point, verbose
    )

    return query_points, data_points
