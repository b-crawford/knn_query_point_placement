import uuid
import numpy as np
from pulp import LpProblem, LpVariable
from scipy.spatial import Voronoi
import knn_plotting as plot
import matplotlib.pyplot as plt


# functions to extract information from scipy.spatial.Voronoi on individual cells
# we will then create objects to represent these cells.
def _extract_cell_information(vor_output, data_point_index):

    # copy and adapt code from voronoi_plot_2d to extract info on the cells
    # into an easier to use format
    out_dict = {
        "finite_segments": [],
        "infinite_segments": [],
        "closest_data_points": [vor_output.points[data_point_index]],
    }

    for pointidx, simplex in zip(vor_output.ridge_points, vor_output.ridge_vertices):
        if data_point_index in pointidx:
            simplex = np.asarray(simplex)
            if np.all(simplex >= 0):
                out_dict["finite_segments"].append(vor_output.vertices[simplex])
            else:
                i = simplex[simplex >= 0][0]  # finite end Voronoi vertex

                t = (
                    vor_output.points[pointidx[1]] - vor_output.points[pointidx[0]]
                )  # tangent

                out_dict["infinite_segments"].append(
                    {"vertex": vor_output.vertices[i], "direction": t}
                )

    return out_dict


def _extract_all_cell_information(vor_output, data_points):

    all_cell_info = []
    for i in range(len(data_points)):
        info = _extract_cell_information(vor_output, i)
        all_cell_info.append({**info, **{"closest_point_uuid": data_points.index[i]}})

    return all_cell_info


# the voroni cells can be represented by a set of halfspaces, we define some helper
# functions here to generate those halfspaces.
def _halfspace_from_segment(line_segment, point):
    # return the equation of the halfspace on the side of the line segment
    # that the point is on, in the format Ax >=b

    # first check whether the line is vertical
    if line_segment[1][0] == line_segment[0][0]:
        # see which side of the line the point is on
        sign = np.sign(point[0] - line_segment[0][0])
        assert sign != 0, "Need point not on the line segment to determine half space"
        return np.array([sign, 0]), sign * line_segment[0][0]

    # now if we do not have vertical line then extract the equation of the line
    tangent = line_segment[1] - line_segment[0]
    m = tangent[1] / tangent[0]
    c = line_segment[0][1] - m * line_segment[0][0]

    # now find which side of the line the point is on
    sign = np.sign(point[1] - m * point[0] - c)
    assert sign != 0, "Need point not on the line segment to determine half space"
    return np.array([-sign * m, sign]), sign * c


def _halfspace_from_vector(direction, point_on_line, point_in_halfspace):
    # return the equation of the halfspace on the side of the line
    # that the point is on, in the format Ax >=b

    b = np.sum(direction * point_on_line)
    A = direction

    sign = np.sign(np.sum(A * point_in_halfspace) - b)

    assert sign != 0, "Need point not on the line to determine half space"

    return sign * A, sign * b


class VoronoiCell:
    def __init__(self, A, b, nearest_neighbours_uuids):

        self.cell_uuid = uuid.uuid4()
        self.nearest_neighbours_uuids = list(set(nearest_neighbours_uuids))
        self.A = A
        self.b = b

    def find_viable_point(self):
        # find a point that is in the shape and satisfies the linear constraints
        # we will do this by solving a linear program
        # we will use the pulp package for this

        # first create the linear program
        prob = LpProblem("finite_cell_viable_point")

        # create the variables
        x = LpVariable("x")
        y = LpVariable("y")

        # add the constraints
        for i in range(len(self.A)):
            prob += self.A[i, 0] * x + self.A[i, 1] * y >= self.b[i]

        # solve the problem
        prob.solve()

        # check if the problem is infeasible
        if prob.status == -1:
            return None

        # return the solution
        return np.array([x.value(), y.value()])

    def plot(self, x_bounds, y_bounds):
        ax = plot.plot_feasible_area(self.A, self.b, x_bounds, y_bounds)
        return ax

    # define a factory method to create from a list of edges
    @classmethod
    def from_polygon_edges(cls, edges):

        centre = edges.sum(axis=0).sum(axis=0) / (len(edges) * 2)

        half_spaces = [_halfspace_from_segment(edge, centre) for edge in edges]

        A = np.array([h[0] for h in half_spaces])
        b = np.array([h[1] for h in half_spaces])

        # create the object
        return cls(A=A, b=b)

    # define a factory method to create from a polygon, this will assume we have a
    # convex polygon and we want the interior as the viable space. We only have
    # the vertices rather than line segment. We assume the vertices are in the
    # correct order, i.e. there is an edge between each point and one edge back
    # to the beginning
    @classmethod
    def from_polygon_vertices(cls, vertices):

        # center of the shape
        centre = np.mean(vertices, axis=0)

        shape_vertices = np.concatenate([vertices, [vertices[0]]])

        half_spaces = [
            _halfspace_from_segment(shape_vertices[i : i + 2], centre)
            for i in range(len(shape_vertices) - 1)
        ]

        A = np.array([h[0] for h in half_spaces])
        b = np.array([h[1] for h in half_spaces])

        # create the object
        return cls(A=A, b=b)

    # define a factory method to create from a set of lines, some infinite and some
    # finite. We use the first closest neighbour to determine which side of the
    # line the half space is on
    @classmethod
    def from_lines(
        cls,
        finite_line_segments,
        infinite_lines,
        point_in_cell,
        nearest_neighbours_uuids=[],
    ):

        finite_half_spaces = [
            _halfspace_from_segment(edge, point_in_cell)
            for edge in finite_line_segments
        ]

        # for the infinte lines we have a vertex and a direction, we have defined
        # a function to create the half space from this
        infinite_half_spaces = [
            _halfspace_from_vector(line["direction"], line["vertex"], point_in_cell)
            for line in infinite_lines
        ]

        A = np.array([h[0] for h in finite_half_spaces + infinite_half_spaces])
        b = np.array([h[1] for h in finite_half_spaces + infinite_half_spaces])

        return cls(A=A, b=b, nearest_neighbours_uuids=nearest_neighbours_uuids)


# helper functions for dealing with collections of cells
def _combine_cells(cell1, cell2):
    A_combined = np.concatenate([cell1.A, cell2.A])
    b_combined = np.concatenate([cell1.b, cell2.b])

    combined_cell = VoronoiCell(
        A=A_combined,
        b=b_combined,
        nearest_neighbours_uuids=cell1.nearest_neighbours_uuids
        + cell2.nearest_neighbours_uuids,
    )

    if combined_cell.find_viable_point() is None:
        return None
    else:
        return combined_cell


def _data_points_to_cells(data_points, current_uuid_list=[]):

    # run voronoi algorithm
    vor = Voronoi(data_points)

    # extract information on cells
    cell_info = _extract_all_cell_information(vor, data_points)

    # return cell objects
    return [
        VoronoiCell.from_lines(
            cell["finite_segments"],
            cell["infinite_segments"],
            cell["closest_data_points"][0],
            nearest_neighbours_uuids=[cell["closest_point_uuid"]] + current_uuid_list,
        )
        for cell in cell_info
    ]


def _next_order_voronoi(voronoi_cells, data_points):

    print("Finding next order voronoi cells")
    out_cells = []
    for cell in voronoi_cells:
        relevant_data_points = data_points.drop(cell.nearest_neighbours_uuids).copy()
        new_cells = _data_points_to_cells(
            relevant_data_points, current_uuid_list=cell.nearest_neighbours_uuids
        )

        for new_cell in new_cells:
            new_cell = _combine_cells(cell, new_cell)
            if new_cell is not None:
                out_cells.append(new_cell)

    return out_cells


def nth_order_voronoi(data_points, n=1):

    # create the first order voronoi cells
    voronoi_cells = _data_points_to_cells(data_points)

    for i in range(n - 1):
        voronoi_cells = _next_order_voronoi(voronoi_cells, data_points)

    return voronoi_cells


class VoronoiCells:
    def __init__(self, data_points, nth_order=1):

        self.data_points = data_points
        self.voronoi_cells = nth_order_voronoi(data_points, n=nth_order)

    def plot(self, region_index, x_bounds, y_bounds):

        ax = self.voronoi_cells[region_index].plot(x_bounds, y_bounds)

        # add all the data points
        ax.scatter(
            self.data_points["x"],
            self.data_points["y"],
            c="k",
            s=5,
            label="All data Points",
        )

        # add the data points that define the cell
        defining_points = self.data_points.loc[
            self.voronoi_cells[region_index].nearest_neighbours_uuids
        ]
        ax.scatter(
            defining_points["x"],
            defining_points["y"],
            c="r",
            s=20,
            label="Nearest neighbours",
        )

        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)

        return ax
