import functools
import os
import zipfile
import shutil
import json
from typing import List, Tuple, Union, Optional
import matplotlib.pyplot as plt
import numpy as np
import scipy.interpolate as interpolate
import bisect
from matplotlib.animation import FuncAnimation
from functools import partial
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def rotate_point_around_origo(angle_deg: float, point: np.ndarray) -> np.ndarray:
    """Function that takes a point and an angle, and returns that point rotated by the angle around the Z axis."""
    angle_rad = np.radians(angle_deg)
    rotmat = np.array([[np.cos(angle_rad), -np.sin(angle_rad), 0],
                       [np.sin(angle_rad), np.cos(angle_rad), 0],
                       [0, 0, 1]])
    return np.dot(rotmat, point)


def pose_to_vertices(pose: List[float]) -> List[List[np.ndarray]]:
    """Function that takes a pose and returns the vertices that the drone's model will contain in that pose. The pose
    may or may not contain a heading. If it doesn't, then the default angle shall be used."""
    corners = [[-L / 2, -L / 2, -H / 2],
               [L / 2, -L / 2, -H / 2],
               [L / 2, L / 2, -H / 2],
               [-L / 2, L / 2, -H / 2],
               [L / 2, -L / 2, H / 2],
               [-L / 2, -L / 2, H / 2],
               [-L / 2, L / 2, H / 2],
               [L / 2, L / 2, H / 2],
               [L / 2 + P, 0, 0]]  # the 3D coordinates of the drone model, which is a box with a triangular tip on its front
    faces = [[3, 2, 1, 0],
             [7, 6, 5, 4],
             [0, 5, 6, 3],
             [0, 1, 4, 5],
             [2, 3, 6, 7],
             [1, 8, 4],
             [1, 2, 8],
             [4, 8, 7],
             [2, 7, 8]]  # the corners defining the faces of the drone model, with normal vector pointing outward
    angle_deg = pose[3] if len(pose) == 4 else DEFAULT_ANGLE  # check if pose includes yaw
    # we first rotate *then* translate, else the translation gets rotated as well
    rotated_corners = np.array([rotate_point_around_origo(angle_deg, np.array(corner)) for corner in corners])
    rotated_translated_corners = rotated_corners + np.array(pose[:3])
    return [[rotated_translated_corners[vertex] for vertex in face] for face in faces]  # extract the vertices


def cleanup(skyc_filename: str) -> None:
    '''Function that deletes the folder from which we extracted the data.'''
    folder_name = os.path.splitext(skyc_filename)[0]  # first element of the list is the file name, second is ".skyc"
    if os.path.exists(folder_name):
        shutil.rmtree(folder_name)


def unpack_skyc_file(skyc_filename: str) -> str:
    '''Function that takes a skyc file and extracts its contents neatly into a folder, as if we used winrar. Returns
    the name of this folder.'''
    folder_name = os.path.splitext(skyc_filename)[0]  # first element of the list is the file name, second is ".skyc"
    if os.path.exists(folder_name):  # if there is a leftover folder from a previous run, delete it!
        shutil.rmtree(folder_name)
    os.makedirs(folder_name)  # make a new folder, named after the skyc file
    with zipfile.ZipFile(skyc_filename, 'r') as zip_ref:  # then extract everything into it
        zip_ref.extractall(folder_name)
    return folder_name


def get_traj_data(skyc_file: str) -> List[dict]:
    '''Function that extracts the contents of the trajectory.json files in the provided skyc file. Returns the
    dictionary containing this data.'''
    folder_name = unpack_skyc_file(skyc_file)  # unpack the skyc file (it's like a zip)
    drones_folder = os.path.join(folder_name, "drones")  # within it, there should be a 'drones' folder for trajectories
    traj_data = []
    for root, dirs, files in os.walk(drones_folder):
        # iterating over the files and folders in the drones folder, we are looking for trajectory files
        if 'trajectory.json' in files:
            with open(os.path.join(root, 'trajectory.json'), 'r') as json_file:
                data = json.load(json_file)
                points = data.get("points")
                assert points is not None
                data["has_yaw"] = True if len(points[0][1]) == 4 else False  # determine if there is a yaw trajectory
                traj_data.append(data)
                traj_type = data.get("type", "COMPRESSED").upper()
                # compressed trajectories can only be of degree 1, 3 and 7 as per the bitcraze documentation
                # if a trajectory is not compressed, it is poly4d, which (for now) can only have degrees up to 5
                ctrl_point_num = [0, 2, 6] if traj_type == "COMPRESSED" else [0, 1, 2, 3, 4]
                for point in points:
                    assert len(point[2]) in ctrl_point_num  # throw an error if the degree is not matching the type!
    cleanup(skyc_file)
    return traj_data


def extend_takeoff_land(traj_data: List[dict]) -> Tuple[float, float]:
    '''Function that takes the trajectories and adds a segment to their end or start, so that they start and end at
     the same time. Returns this start and end time. This is important for the animation later.'''
    last_land_time = 0.0
    first_takeoff_time = 0.0
    # first we loop over the trajectories to determine first_takeoff_time and last_land_time
    for trajectory in traj_data:
        landingTime = trajectory.get("landingTime")
        takeoffTime = trajectory.get("takeoffTime")
        last_land_time = landingTime if landingTime > last_land_time else last_land_time
        first_takeoff_time = takeoffTime if takeoffTime < first_takeoff_time else first_takeoff_time
    # then we loop over the trajectories again in order to insert the extension that ensures that they are all the same
    # duration by adding an extension to their beginning or end as needed
    for trajectory in traj_data:
        if trajectory.get("landingTime") < last_land_time:
            last_point = trajectory["points"][-1]
            extension = [last_land_time, last_point[1], []]
            trajectory["points"].append(extension)
        if trajectory.get("takeoffTime") > first_takeoff_time:
            first_point = trajectory["points"][0]
            extension = [first_takeoff_time, first_point[1], []]
            trajectory["points"].insert(0, extension)
    return first_takeoff_time, last_land_time


def evaluate_segment(points: List[List[float]], start_time: float, end_time: float,
                     eval_time, has_yaw: bool) -> Tuple[float, ...]:
    # TODO: find a more efficient method
    '''Function that takes the control points of a bezier curve, creates an interpolate.BPoly object for each
    dimension of the curve, evaluates them at the given time and returns a tuple with the time, x, y, z and yaw.'''
    # The bernstein coefficients are simply the coordinates of the control points for each dimension.
    x_coeffs = [point[0] for point in points]
    y_coeffs = [point[1] for point in points]
    z_coeffs = [point[2] for point in points]
    x_BPoly = interpolate.BPoly(np.array(x_coeffs).reshape(len(x_coeffs), 1), np.array([start_time, end_time]))
    y_BPoly = interpolate.BPoly(np.array(y_coeffs).reshape(len(y_coeffs), 1), np.array([start_time, end_time]))
    z_BPoly = interpolate.BPoly(np.array(z_coeffs).reshape(len(z_coeffs), 1), np.array([start_time, end_time]))
    X = x_BPoly(eval_time)
    Y = y_BPoly(eval_time)
    Z = z_BPoly(eval_time)
    # Make sure that the trajectory doesn't take the drone outside the limits of the optitrack system!
    assert LIMITS[0][0] < X < LIMITS[0][1] and LIMITS[1][0] < Y < LIMITS[1][1] and LIMITS[2][0] < Z < LIMITS[1][1]
    retval = [float(eval_time), float(X), float(Y), float(Z)]
    if has_yaw:
        yaw_coeffs = [point[3] for point in points]
        yaw_BPoly = interpolate.BPoly(np.array(yaw_coeffs).reshape(len(yaw_coeffs), 1), np.array([start_time, end_time]))
        retval.append(float(yaw_BPoly(eval_time)))
    return tuple(retval)


def evaluate_trajectory(trajectory: dict, times: List[float]) -> List[List[float]]:
    '''Function that looks at which bezier curve each timestamp falls into, then evaluates the curve at that
    timestamp, and returns the result for each timestamp.'''
    segments = trajectory.get("points")
    assert segments is not None
    eval = []
    for t in times:
        # check which segment the current timestamp falls into
        i = bisect.bisect_left([segment[0] for segment in segments], t)
        if i == 0:
            eval.append(tuple([segments[0][0]] + segments[0][1]))
        elif i == len(segments):
            eval.append(tuple([segments[-1][0]] + segments[-1][1]))
        else:
            prev_segment = segments[i-1]
            start_point = prev_segment[1]
            start_time = prev_segment[0]
            segment = segments[i]
            end_point = segment[1]
            end_time = segment[0]
            ctrl_points = segment[2]
            # points will contain all points of the bezier curve, including the start and end, unlike in trajectory.json
            points = [start_point, *ctrl_points, end_point] if ctrl_points else [start_point, end_point]
            eval.append(evaluate_segment(points, start_time, end_time, t, trajectory.get("has_yaw", False)))
    # 'Flip' the dimensions of eval. Currently it's a list of tuples, with each tuple containing x,y,z,yaw, but later on
    # it will be easier to work with it if we return it as a list of lists, so that t=eval[0], x=eval[1] and so on
    eval = [list(item) for item in zip(*eval)]
    return eval


def get_derivative(txyz_yaw: List[List[float]]) -> List[List[float]]:
    '''Function that takes a list of coordinates with timestamps and calculates their derivatives. Returns a
    List[List[float]] of the same dimension as the input, where the first timestamps' derivatives are the same as the
    second for ease of plotting later.'''
    output = [txyz_yaw[0]]
    for xyz_yaw in txyz_yaw[1:]:
        derivative = list(np.diff(xyz_yaw) / np.diff(txyz_yaw[0]))
        output.append([derivative[0]] + derivative)
    return output


def assert_no_collisions(traj_eval: List[List[List[float]]]) -> None:
    '''Function that asserts that for every drone, paired with every other drone, their timestamps are identical and
    their euclidean distance is smaller than 0.2, which is apprixmately the size of a drone.'''
    num_of_timestamps = len(traj_eval[0][0])
    num_of_drones = len(traj_eval)
    for i in range(num_of_drones):
        for j in range(i+1, num_of_drones):
            for idx in range(num_of_timestamps):
                t1 = traj_eval[i][0][idx]
                t2 = traj_eval[j][0][idx]
                xyz1 = (traj_eval[i][1][idx], traj_eval[i][2][idx], traj_eval[i][3][idx])
                xyz2 = (traj_eval[j][1][idx], traj_eval[j][2][idx], traj_eval[j][3][idx])
                distance = np.sqrt((xyz1[0] - xyz2[0])**2 + (xyz1[1] - xyz2[1])**2 + (xyz1[2] - xyz2[2])**2)
                assert distance > 0.2
                assert abs(t1-t2) < TIMESTEP / 50


def update_drones(frame_idx: int, anim_data: List[List[List[float]]], drones: List[Poly3DCollection]) -> None:
    """Function that updates the vertices of the drones according to the current frame's index."""
    for i, drone in enumerate(drones):
        pose = [float(lst[frame_idx]) for lst in anim_data[i][1:]]
        drone.set_verts(pose_to_vertices(pose))


def animate(limits: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]],
            fps: float, speed: float, timestep: float, traj_eval: List[List[List[float]]]) -> FuncAnimation:
    """Function that takes initializes the 3D plot for the animation, as well as the animation itself then returns the
    animation object."""
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.cla()
    ax.set_xlim(*limits[0])
    ax.set_ylim(*limits[1])
    ax.set_zlim(*limits[2])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    anim_interval = round(1000 / fps / speed)  # this is the delay between frames in msec
    # this line below seems scary, but all it does is takes every Nth element from the high resolution evaluation
    # results, where N is determined so that it fits the FPS requested
    anim_traj_eval = [[lst[::round(1 / timestep / fps)] for lst in drone] for drone in traj_eval]
    anim_length = len(anim_traj_eval[0][0])
    start_poses = [[float(lst[0]) for lst in drone[1:]] for drone in traj_eval]
    # we initialize the drones with their starting positions, and later update their vertices in the animation
    drones = [Poly3DCollection(pose_to_vertices(start_pose), facecolors=COLORS[idx], edgecolors='black') for
              idx, start_pose in enumerate(start_poses)]
    for drone in drones:
        ax.add_collection3d(drone)
    update_func = partial(update_drones, anim_data=anim_traj_eval, drones=drones)
    animation = FuncAnimation(fig, update_func, frames=anim_length, interval=anim_interval,
                              repeat=False)
    return animation


def plot_data(traj_eval: List[List[List[float]]],
              first_deriv: List[List[List[float]]],
              second_deriv: List[List[List[float]]]) -> None:
    """Function that takes cares of the plotting. It has one figure, with a column of subplots for each drone. There
    will be 3 rows: pose/angle, their derivatives and their 2nd derivatives"""
    fig, subplots = plt.subplots(3, len(traj_eval))
    if len(traj_eval) == 1:  # double indexing wouldn't work when we only have 1 drone, hence this
        subplots = np.array(subplots).reshape((3, 1))
    for idx in range(len(traj_eval)):
        subplots[0, idx].plot(traj_eval[idx][0], traj_eval[idx][1], label='x')
        subplots[0, idx].plot(traj_eval[idx][0], traj_eval[idx][2], label='y')
        subplots[0, idx].plot(traj_eval[idx][0], traj_eval[idx][3], label='z')
        subplots[0, idx].set_title(f"Drone{idx} pose [m, degrees]", fontsize=10)
        subplots[0, idx].grid(True)
        subplots[0, idx].set_xlabel('t [s]')

        subplots[1, idx].plot(first_deriv[idx][0], first_deriv[idx][1], label='x')
        subplots[1, idx].plot(first_deriv[idx][0], first_deriv[idx][2], label='y')
        subplots[1, idx].plot(first_deriv[idx][0], first_deriv[idx][3], label='z')
        subplots[1, idx].set_title(f"Drone{idx}, 1st derivatives [m/s, deg/s]", fontsize=10)
        subplots[1, idx].grid(True)
        subplots[1, idx].set_xlabel('t [s]')

        subplots[2, idx].plot(second_deriv[idx][0], second_deriv[idx][1], label='x')
        subplots[2, idx].plot(second_deriv[idx][0], second_deriv[idx][2], label='y')
        subplots[2, idx].plot(second_deriv[idx][0], second_deriv[idx][3], label='z')
        subplots[2, idx].set_title(f"Drone{idx}, 2nd derivatives [m/s2, degrees/s2]", fontsize=10)
        subplots[2, idx].grid(True)
        subplots[2, idx].set_xlabel('t [s]')
        if all([traj.get("has_yaw", False) for traj in traj_data]):
            # when yaw also has to be plotted, it's sensible to plot it on a different y axis, since it's a different
            # unit of measurement from the rest
            twin_subplot = subplots[0, idx].twinx()
            twin_subplot.plot(traj_eval[idx][0], traj_eval[idx][4], label='yaw', color='r')
            # we need to combine the legends from the two y axes since they will be on the same subplot
            xyz_lines, xyz_labels = subplots[0, idx].get_legend_handles_labels()
            yaw_lines, yaw_labels = twin_subplot.get_legend_handles_labels()
            subplots[0, idx].legend(xyz_lines + yaw_lines, xyz_labels + yaw_labels, fontsize=10)

            twin_subplot = subplots[1, idx].twinx()
            twin_subplot.plot(first_deriv[idx][0], first_deriv[idx][4], label='yaw', color='r')
            # we need to combine the legends from the two y axes since they will be on the same subplot
            xyz_lines, xyz_labels = subplots[1, idx].get_legend_handles_labels()
            yaw_lines, yaw_labels = twin_subplot.get_legend_handles_labels()
            subplots[1, idx].legend(xyz_lines + yaw_lines, xyz_labels + yaw_labels, fontsize=10)

            twin_subplot = subplots[2, idx].twinx()
            twin_subplot.plot(second_deriv[idx][0], second_deriv[idx][4], label='yaw', color='r')
            # we need to combine the legends from the two y axes since they will be on the same subplot
            xyz_lines, xyz_labels = subplots[2, idx].get_legend_handles_labels()
            yaw_lines, yaw_labels = twin_subplot.get_legend_handles_labels()
            subplots[1, idx].legend(xyz_lines + yaw_lines, xyz_labels + yaw_labels, fontsize=10)
        else:
            subplots[0, idx].legend(fontsize=10)
            subplots[1, idx].legend(fontsize=10)
            subplots[2, idx].legend(fontsize=10)
    fig.subplots_adjust(hspace=0.4)


LIMITS = ((-2, 2), (-2, 2), (-0.05, 3.95))  # physical constraints of the optitrack system
TIMESTEP = 0.005  # we keep this relatively constant for the sake of the animation coming later
SKYC_FILE = "nice_skyc_files/3_drones/2.skyc"
traj_data = get_traj_data(SKYC_FILE)  # this will be a list of the dictionaries in the trajectory.json files
takeoff_time, land_time = extend_takeoff_land(traj_data)  # make every trajectory start and end at the same time
eval_times = list(np.linspace(takeoff_time, land_time, round((land_time - takeoff_time) / TIMESTEP)))
traj_eval = [evaluate_trajectory(trajectory, eval_times) for trajectory in traj_data]
assert_no_collisions(traj_eval)
first_deriv = [get_derivative(item) for item in traj_eval]
second_deriv = [get_derivative(item) for item in first_deriv]
ANIM_FPS = 50
ANIM_SPEED = 5  # this is the factor by which we speed the animation up in case it's slow due to calculations
COLORS = ['r', 'b', 'g', 'y', 'c', 'm'][:len(traj_eval)]
DRONE_SCALE = 3
L = 0.1 * DRONE_SCALE
H = 0.05 * DRONE_SCALE
P = 0.04 * DRONE_SCALE
DEFAULT_ANGLE = 0
plot_data(traj_eval, first_deriv, second_deriv)
animation = animate(LIMITS, ANIM_FPS, ANIM_SPEED, TIMESTEP, traj_eval)
plt.show()


