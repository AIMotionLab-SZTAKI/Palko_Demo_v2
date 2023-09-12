import pickle
import numpy as np
from scipy import interpolate
from scipy.spatial import distance_matrix


def pickle_save(place, data):
    pickle_out = open(place, "wb")
    pickle.dump(data, pickle_out)
    pickle_out.close()


def pickle_load(file):
    pickle_in = open(file, "rb")
    data = pickle.load(pickle_in)
    return data


def print_WARNING(message: str) -> None:
    """
    Make the input string yellow and print it out to the console with WARNING: message

    :param message: The message to be printed out to the console.
    :return: None
    """
    print("\033[93mWARNING:", message, "\033[0m ")


def fit_spline(points: np.ndarray) -> list:
    """
    Fit a B-spline to the given 3D coordinate sequence.

    :param points: [[x,y,z]...[x,y,z]]
    :return: spline: A tuple, (t,c,k) containing the vector of knots, the B-spline coefficients, and the degree of the
                     spline.
    """
    points = np.array(points)
    X = points[:, 0]
    Y = points[:, 1]
    Z = points[:, 2]
    if len(X) == 2:
        spline, *_ = interpolate.splprep([X, Y, Z], k=1, s=0)
    elif len(X) == 3:
        spline, *_ = interpolate.splprep([X, Y, Z], k=2, s=0)
    else:
        spline, *_ = interpolate.splprep([X, Y, Z], k=3, s=0)

    return spline


def evaluate_splie(spline):
    u = np.arange(0, 1.01, 0.01)
    spline_points = interpolate.splev(u, spline)
    return spline_points


def parametrize_by_path_length(spline):
    """
    Modify a given spline to have length accurare values.
    During evaluation the resulted slpine will give back positions for given path lengths
    """
    u = np.arange(0, 1.01, 0.01)
    path = interpolate.splev(u, spline)
    length = np.linalg.norm(np.diff(path)) * 10
    s_params = np.linspace(0, length, 101)
    spline, *_ = interpolate.splprep(path, k=2, s=0, u=s_params)
    return spline, length


def calculate_eplis_rel_dist(positions, point_cloud, downwash, radius, safety_distance):
    """
    gives the relative distances of the points from the surface of the sphere in diferent time points
    """
    elipsoid_x = distance_matrix(positions[:, :1], point_cloud[:, :1])
    elipsoid_y = distance_matrix(positions[:, 1:2], point_cloud[:, 1:2])
    elipsoid_z = distance_matrix(positions[:, 2:3], point_cloud[:, 2:3])
    rel_distances = np.sqrt(((np.square(elipsoid_x) + np.square(elipsoid_y)) / ((2*radius+safety_distance) ** 2)) +
                            (np.square(elipsoid_z) / ((2*downwash+safety_distance) ** 2)))
    return rel_distances
