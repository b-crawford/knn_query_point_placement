import numpy as np
import matplotlib.pyplot as plt
import os
from knn_simulation import random_points as rp
import knn_plotting as kp

# get directory of assets folder
assets_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), "../assets"))

# set seed
np.random.seed(8)

# set up parameters
k_nearest_neighbours = 3
x_bounds = [-0.5, 0.6]
y_bounds = [51.2, 52.3]

# create first plot
n_data_points = 9

data_points = rp.uniform_random_points(n_data_points, x_bounds, y_bounds)
kp.plot(data_points)

plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.savefig(assets_directory + "/readme_image_1.png")

# choose query points manually
query_points = np.array([[0.5, 51.75], [0, 52.1], [-0.3, 51.6]])

kp.plot(data_points, query_points, k_nearest_neighbours)

plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.savefig(assets_directory + "/readme_image_2.png")

# now plot with many more points
data_points = rp.uniform_random_points(1000, x_bounds, y_bounds)
kp.plot(data_points)

plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.savefig(assets_directory + "/readme_image_3.png")
