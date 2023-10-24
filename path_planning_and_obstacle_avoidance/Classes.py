import numpy as np
import math
from scipy import interpolate
from typing import List


class Construction:
    """
    Set params in Scene construction
    """
    def __init__(self, real_obstacles: bool = True,
                 live_demo: bool = False, simulated_obstacles: bool = False,
                 get_new_measurement: bool = True,
                 fix_vertex_layout: int = 4,
                 N=0):
        # GENERAL PARAMETERS:

        # The dimensions of the flying area. (x_min, x_max, y_min, y_max, z_min, z_max)
        self.N = N
        if N < 10:
            self.dimensions = np.array([-1.5, 1.5, -1.5, 1.5, 0.3, 1.2])
        else:
            self.dimensions = np.array([-1, (N+3)*0.5, -1, (N+3)*0.5, 0.3, 1.5])
        # The safety distance aruond each obstacle. (It can be modified for the individual obstacles)
        self.general_safety_distance = 0.1

        #...............................................................................................................
        # STATIC OBSTACLE RELATED PARAMETERS:
        self.live_demo = live_demo

        # Select which type of static obstacles will be present
        # (real obstacles from measurement, priori given simulated obstacles or both).
        self.simulated_obstacles = simulated_obstacles
        # If true then the position of the obstacles are measured, otherwise use a saved layout.
        self.real_obstacles = real_obstacles
        self.get_new_measurement = get_new_measurement
        # Select one of the predefined simulated layouts. (0 or not defined layout number results in an empty space)
        # More layouts can be defined in the Scene_constuction.py.
        self.static_obstacle_layout = 3
        # The side lengths of the real obstacles can not be measured by the Optitrack, so they have to be given manualy.
        # Do not change the names, just the values. TODO: Makse a guide to how to add more real obstacle types.
        self.real_obstacles_side_lengths = {"buildings": 0.3,
                                            "landing_pads": 0.2,
                                            "poles": 0.2}
        # The distance from the top of the osstacles, where targetponts will be placed
        self.howering_heigt = 0.2

        #...............................................................................................................
        # DYNAMIC OBSTACLE RELATED PARAMETERS:

        # Ask for new paths during construction.
        self.ask_for_new_paths = False
        # Keep the previousli saved paths or wipe clean the saves.
        self.keep_existing_paths = False
        # !!! During construction as many obstacles will be maid as many paths are given here !!!
        # Set the desired paths for the dynamic obstacles.
        # The places of the values representing the path indexes are matched to the drone IDs. [ID=0, ID=1, ....]
        self.path_of_dynamic_obstacles = np.array([0, 1, 2])
        # The speeds of the static obstacles can be set individuali by providing a drone ID, speed pair.
        self.speed_of_individual_dynamic_obstacles = np.array([[0, 0.5],   # [ID, speed]
                                                               [1, 0.5],
                                                               [2, 0.5],
                                                               [-1, 0.5]]) # default
        # The radius of the static obstacles can be set individuali by providing a drone ID, radius pair.
        self.radii_of_individual_dynamic_obstacles = np.array([[0, 0.1],   # [ID, radius]
                                                               [1, 0.1],
                                                               [2, 0.1],
                                                               [-1, 0.1]]) # default
        # The time points when the obstacles start thier movement.
        self.movement_start = np.array([[0, 5],  # [ID, start time]
                                        [1, 10],
                                        [2, 15],
                                        [-1, 0]]) # default

        #...............................................................................................................
        # GRAPH RELATED PARAMETERS:

        # Select a predefined layout for the manualy added vertices.
        # (0 or not defined layout number results in an empty space)
        # More layouts can be defined in the Scene_constuction.py.
        self.fix_vertex_layout = fix_vertex_layout
        # The maximum distance between the points in the point cloud of a graph.
        self.point_cloud_density = 0.05
        # The generation of the densed graph can be switched off to reduce the run time of Scene_construction
        self.generate_dense_graph = False
        # Set the random seed for the vertex placement
        self.base_rand_seed = 444
        self.dense_rand_seed = 445
        # Set the number of vertices
        self.base_vertex_number = 100
        self.dense_vertex_number = 500
        # Minimal distance between the vertices
        self.base_minimal_vertex_distance = 0.25
        self.base_maximal_edge_length = 1.0
        self.dense_minimal_vertex_distance = 0.5
        self.dense_maximal_edge_length = 1.0

        #...............................................................................................................
        # PLOT RELATED PARAMETERS:

        self.resoulution_of_obstacles = 11
        self.obstacles_visibility = 1
        self.safety_zone_visibility = 0.2
        self.show_fix_vertices = True
        self.show_base_graph = False
        self.show_dense_graph = False
        self.show_paths_of_dynamic_obstacles = True

        # ..............................................................................................................
        # HANDLE WITH CARE:
        # The time unified time resolution which is used for time discretization everywhere in the algorithm
        # (eg.: collision matrix, speed-profile optimization)
        self.Ts = 0.1
        # The minimal and maximal costs wich could be assigned to a point in the point cloud of the edges.
        self.cmin = 5000
        self.cmax = 50000

        # ..............................................................................................................
        self.free_targets = np.array([])
        self.home_positions = np.array([])
        self.time_in_pause = 0


class Static_obstacles:
    """
    Set new layouts in Scene construction
    """
    def __init__(self):
        self.corners = np.array([]) # Set in the Scene_construction
        self.corners_of_safe_zone = np.array([]) # Set in the Scene_construction
        self.enclosed_space = np.array([]) # Set in the Scene_construction
        self.enclosed_space_of_safe_zone = np.array([]) # Set in the Scene_construction


class Dynamic_obstacle:
    """
    Spherical movig obstacles
    """
    def __init__(self, path_tck, path_length, speed, radius, start_time): # Define in class Construction
        self.speed = speed
        self.path_tck = path_tck
        self.path_length = path_length
        self.path_time = math.ceil((path_length/speed)*10)/10 # TODO: 1/Ts not 10
        self.radius = radius
        self.start_time = start_time

        self.collision_matrix = None
        self.collision_matrix_compressed = None
        self.surface = None
        self.position = None
        self.plot_face = None

    def move(self, t: np.ndarray) -> np.ndarray:
        """
        Gives the positions of the obstacles at given time points.
        """
        t = t-self.start_time
        t = np.where(t > 0, t, 0) # set to 0 the negative time values
        s = self.speed*t
        s = np.where(s < self.path_length, s, self.path_length)  # set maximum value for the path length
        position = np.transpose(interpolate.splev(s, self.path_tck))
        if np.size(t) == 1:
            print(position)
            position[2] = 0
        else:
            position[:, 2] = 0
        return position


class Drone:
    def __init__(self):
        self.cf_id = "00"
        # CHANGEABLE PARAMETERS:
        self.radius = 0.1
        self.DOWNWASH = 2
        self.constant_speeds = 2 * np.array([0.4, 0.5, 0.6])
        self.MAX_ACCELERATION = 0.6
        self.MAX_SPEED = 2 * 1
        self.rest_time = 3

        # DO NOT MODIFY IT:
        self.serial_number = None
        self.start_time = 0
        self.start_vertex = None
        self.target_vertex = None
        self.trajectory = None
        self.fligth_time = 0
        self.surface = None
        self.position = None
        self.plot_face = None
        self.stl_surface = None
        self.plot_stl = None
        self.collision_matrix = None
        self.collision_matrix_compressed = None
        self.emergency = False

    def move(self, t: np.ndarray) -> np.ndarray:
        t = np.where(t > self.trajectory['speed_profile'][0][0], t, self.trajectory['speed_profile'][0][0])
        t = np.where(t < self.trajectory['speed_profile'][0][-1], t, self.trajectory['speed_profile'][0][-1])
        s = interpolate.splev(t, self.trajectory['speed_profile'])
        position = interpolate.splev(s, self.trajectory['spline_path'])
        return np.transpose(position)
