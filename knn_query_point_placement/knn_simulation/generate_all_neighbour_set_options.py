import argparse
import os
import pandas as pd
import json
import uuid
import random_points
import sys

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(root_dir)
from query_point_algorithms import nth_degree_voronoi

# parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("-n", "--num_data_points", type=int, default=9)
parser.add_argument("-k", "--k_nearest_neighbours", type=int, default=3)
parser.add_argument("-N", "--num_simulations", type=int, default=1)
parser.add_argument("-x", "--x_bounds", type=float, nargs=2, default=[0, 10])
parser.add_argument("-y", "--y_bounds", type=float, nargs=2, default=[0, 10])
parser.add_argument(
    "-d",
    "--out_directory",
    type=str,
    default=os.path.abspath(os.path.join(root_dir, "../data/neighbour_set_options")),
)
args = parser.parse_args()

# check out directory exists and if not create it
out_path = os.path.abspath(args.out_directory)


for i in range(args.num_simulations):

    print("Simulation {}".format(i + 1))

    # create a subdirectory of the outpath for this simulation
    run_outpath = os.path.join(out_path, str(uuid.uuid4()))
    os.makedirs(run_outpath)

    # generate the data points
    data_points = random_points.uniform_random_points(
        args.num_data_points, args.x_bounds, args.y_bounds
    )

    # calculate voronoi cells
    data_points_uuid = pd.DataFrame(
        data_points,
        columns=["x", "y"],
        index=[str(uuid.uuid4()) for i in range(len(data_points))],
    )
    voronoi_cells = nth_degree_voronoi.VoronoiCells(
        data_points_uuid, nth_order=args.k_nearest_neighbours
    )

    # format the data for saving
    out_data = [
        {
            "query_point": cell.find_viable_point().tolist(),
            "neighbours": cell.nearest_neighbours_uuids,
        }
        for cell in voronoi_cells.voronoi_cells
    ]

    # save both to file
    data_points = data_points_uuid.to_csv(os.path.join(run_outpath, "data_points.csv"))
    with open(os.path.join(run_outpath, "neighbour_options.json"), "w") as f:
        f.write(json.dumps(out_data, indent=4))
