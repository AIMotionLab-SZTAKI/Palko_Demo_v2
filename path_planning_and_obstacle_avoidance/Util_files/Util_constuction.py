import copy
from scipy.spatial import Delaunay
import networkx as nx
import motioncapture
from typing import Tuple
import matplotlib.pyplot as plt
from gurobipy import *

from path_planning_and_obstacle_avoidance.Util_files.Util_general import *
from path_planning_and_obstacle_avoidance.Classes import Static_obstacles, Dynamic_obstacle, Drone


#=======================================================================================================================
# FOR: STATIC OBSTACLES

def get_obstacles_positions_from_optitrack(new_measurement: bool) -> dict:
    """
    Measure the positions of the obstacles and return a dictinary as {'name':position}.
    It evaluates the mean value of "sample_size" number of measurements.
    Gives a warning if the deviation is bigger than the "maximum_measurement_deviation".

    :param new_measurement: True -> make a new optittrack measurement and save it too
                            False -> use the last measurements
    :return: obstacles_dict:  {'name':position}, where: names -> optitrack rigidBody names,
                                                        positions -> coordinates of the top center of the obstacles [x,y,z]
    """
    dir_path = os.getcwd()

    if new_measurement:
        SAMPLE_SIZE = 100
        obstacles_dict = get_measurement(SAMPLE_SIZE)
        check_maximum_deviation(obstacles_dict)
        for name in obstacles_dict:
            obstacles_dict[name] = np.sum(obstacles_dict[name], axis=0) / len(obstacles_dict[name])
        pickle_save(dir_path + "/path_planning_and_obstacle_avoidance/Pickle_saves/Construction_saves/obstacle_measurement.pickle", obstacles_dict)

    else:
        try:
            obstacles_dict = pickle_load(dir_path + "/path_planning_and_obstacle_avoidance/Pickle_saves/Construction_saves/obstacle_measurement.pickle")
        except FileNotFoundError:
            print_WARNING("No existing obstacle measurements available!!!")
            obstacles_dict = {}

    return obstacles_dict


def get_measurement(sample_size: int) -> dict:
    """
    Connect to the Optitrack data stream and collect "sample_size" number of measurements.

    :param sample_size: number of measurements
    :return: obstacles_dict: {'name':position}, where: names -> optitrack rigidBody names,
                                                       positions -> list of measured coordinates of the top center of
                                                                    the obstacles np.array([[x,y,z]...[x,y,z]])
    """
    mc = motioncapture.MotionCaptureOptitrack("192.168.2.141")
    obstacles_dict = {}
    for _ in range(sample_size):
        mc.waitForNextFrame()
        for name, obj in mc.rigidBodies.items():
            if name in obstacles_dict:
                obstacles_dict[name] = np.row_stack((obstacles_dict[name], obj.position))
            else:
                obstacles_dict[name] = obj.position

    return obstacles_dict


def check_maximum_deviation(obstacles_dict: dict) -> None:
    """
    Check the deviation of the measurements and give a warning if it is bigger than the "maximum_measurement_deviation".

    :param obstacles_dict: {'name':position}, where: names -> optitrack rigidBody names,
                                                     positions -> list of measured coordinates of the top center of
                                                                  the obstacles np.array([[x,y,z]...[x,y,z]])
    """
    max_deviation = 0
    PERMISSIBLE_DEVIATION = 1
    for name in obstacles_dict:
        deviation = max([max(obstacles_dict[name][:, 0]) - min(obstacles_dict[name][:, 0]),
                         max(obstacles_dict[name][:, 1]) - min(obstacles_dict[name][:, 1]),
                         max(obstacles_dict[name][:, 2]) - min(obstacles_dict[name][:, 2])])
        if deviation > max_deviation:
            max_deviation = deviation
    if max_deviation > PERMISSIBLE_DEVIATION:
        print_WARNING("The maximum deviation in the obstacle position measurments is " + str(max_deviation) +
                      "mm, which is bigger than the given "+str(PERMISSIBLE_DEVIATION) + "mm maximum!!!")


def add_dimension_to_obstacles(obstacle_measurements: dict, obstacles_side_lengths: dict) -> np.ndarray:
    """
    Since the optitrack measurements only give the top-center coordinates of the static obstacles
    the width of them has to be set manually. This function sets the width of the obstacles based on their names.

    The width of the obstacles can be set in Classes/Construction/self.real_obstacles_side_lengths

    :param obstacle_measurements: {'name':position}, where: names -> optitrack rigidBody names,
                                                     positions -> list of measured coordinates of the top center of
                                                                  the obstacles as np.array([[x,y,z]...[x,y,z]])
    :param obstacles_side_lengths: {'name':width}, where: names -> static oostacle names,
                                                          width -> float(w)
    :return: enclosed_space: main parameters of the obstacles as np.array([[x,y,z,w,w]...[x,y,z,w,w]])
    """
    enclosed_space = []
    start_points = []
    for obstacle in obstacle_measurements:
        x = obstacle_measurements[obstacle][0]
        y = obstacle_measurements[obstacle][1]
        z = obstacle_measurements[obstacle][2]
        if obstacle[0:2] == "bu":
            w = obstacles_side_lengths["buildings"] / 2
            enclosed_space.append([x, y, z, w, w])
        if obstacle[0:2] == "ob":
            w = obstacles_side_lengths["poles"] / 2
            enclosed_space.append([x, y, z, w, w])
        if obstacle[0:2] == "cf":
            w = obstacles_side_lengths["landing_pads"] / 2
            start_points.append([x, y, z, w, w])
    enclosed_space.extend(start_points)
    return np.array(enclosed_space)


def match_dimensions(obstacle_dimensions: np.ndarray, obstacle_positions: np.ndarray) -> np.ndarray:
    """
    If the widths of the obstacles are the same for all abstacles, generate an obstacle dimension set which number of
    rows is as the obstacles positions.

    :param obstacle_dimensions: np.array([[w_x, w_y]...[w_x, w_y]]) OR np.array([[w_x, w_y]]), width in x and y directions
    :param obstacle_positions: np.array([[x, y, z]...[x, y, z]]), top center positions
    :return: obstacle_dimensions: np.array([[w_x, w_y]...[w_x, w_y]]), width in x and y directions
    """
    if len(obstacle_dimensions) == 1 and len(obstacle_positions) > 1:
        obstacle_dimensions = obstacle_dimensions * np.ones((len(obstacle_positions), 1))
    elif 1 < len(obstacle_dimensions) != len(obstacle_positions):
        sys.exit("Not matching obstacle positions and dimensions")

    return obstacle_dimensions


def calculate_corners(enclosed_spaces: np.ndarray) -> np.ndarray:
    """
    Give the coordinates of the corners of the obstacles.

    :param enclosed_spaces: np.array([[x,y,z,w_x,w_y]...[x,y,z,w_x,w_y]])
    :return: corners_of_static_obstacles: np.array([[c1...c8],[c1...c8]]), where c=[x,y,z]
    """
    corners_of_static_obstacles = []
    for obstacle in enclosed_spaces:
        obstacle_corners = []
        for corner_x, corner_y, corner_z in zip([-1, 1, 1, -1, -1, -1, 1, 1], [-1, -1, 1, 1, 1, -1, -1, 1],
                                                [-1, -1, -1, -1, 0, 0, 0, 0]):
            obstacle_corners.append([obstacle[0] + corner_x * obstacle[3], obstacle[1] + corner_y * obstacle[4],
                                     obstacle[2] + corner_z * obstacle[2]])
        corners_of_static_obstacles.append(obstacle_corners)
    return np.array(corners_of_static_obstacles)


def add_safety_zone_to_static_obstacles(enclosed_spaces: np.ndarray, safety_distance: float) -> np.ndarray:
    """
    The size of the obstacles are enlarged by the safety zone.
    If there are no static obstacles present return with an empty array.

    :param enclosed_spaces: array([[x,y,z,w_x,w_y]...[x,y,z,w_x,w_y]])
                            -> the positions of the obstacles and their widths
    :param safety_distance: float
                            -> the minimal distance between the drones and the obstacles (incuding other drones)
    :return: enclosed_space_of_safe_zone: array([[x,y,Z,W_x,W_y]...[x,y,Z,W_x,W_y]])
                                          -> the positions of the obstacles and their increased height and widths
    """

    if len(enclosed_spaces) == 0:
        return enclosed_spaces

    enclosed_space_of_safe_zone = copy.deepcopy(enclosed_spaces)
    enclosed_space_of_safe_zone[:, 2:] = enclosed_spaces[:, 2:]+[safety_distance, safety_distance, safety_distance]

    return enclosed_space_of_safe_zone


#=======================================================================================================================
# FOR: GRAPH GENERATION

def add_vertices_above_obstacles(enclosed_spaces: np.ndarray, V_fix: np.ndarray, z_max: float,
                                 hover_heigth: float) -> np.ndarray:
    """
    Create vertices above the static obstacles and add them to the fix vertices.
    Vertices above the flying zone are ignored and a warning is raised.

    :param enclosed_spaces: array([[x,y,Z,W_x,W_y]...[x,y,Z,W_x,W_y]])
                            -> the positions of the obstacles and their increased height and widths
    :param V_fix: array([[x,y,z]...[x,y,z]]) -> positoins of the manualy added vertices
    :param z_max: float -> the top border of the flight zone
    :param hover_heigth: float -> the hovering height above the static obstacles
    :return: array([[x,y,z]...[x,y,z],[x,y,z]...[x,y,z]]) -> V_fix and the newly added vertices
    """
    if len(enclosed_spaces) == 0:
        return V_fix

    V_above_obstacles = enclosed_spaces[:, :3] + [0, 0, hover_heigth]
    V_inside_zone = V_above_obstacles[V_above_obstacles[:, 2] < z_max]

    if len(V_inside_zone) != len(V_above_obstacles):
        print_WARNING("Some obstacles are too tall to place a vertex above them!!!")

    if len(V_fix) == 0:
        return V_inside_zone

    vertices = np.row_stack((V_fix, V_inside_zone))

    return vertices


def generate_base_graph(dimensions: np.ndarray, static_obstacles: Static_obstacles, min_vertex_distance: float,
                        max_edge_length: float, number_of_vertices: int, rand_seed: int,
                        V_fix: np.ndarray) -> Tuple[nx.Graph, np.ndarray]:
    """
    Fill the flight zone with the searchin-graph avoiding the static obstacles.

    :param dimensions: array([x_min, x_max, y_min, y_max,z_min, z_max]) -> the borders of the flight zone
    :param static_obstacles: Static_obstacles object
    :param min_vertex_distance: float -> The minimum distance between the vertices.
    :param number_of_vertices: int -> The number of vertices to be scattered inside the flight zone.
    :param rand_seed: int -> the ranom seed for placing the verices
    :param V_fix: array([[x,y,z]...[x,y,z]]) -> coordinates of the targets
    :returns: graph: The generated graph object.
              vertices: The vertices of the graph. #TODO: if the densed graph is not needed any more -> do not return it
    """
    graph = nx.Graph()
    check_fix_vertices(V_fix, static_obstacles.enclosed_space_of_safe_zone)
    vertices = generate_vertices(dimensions, static_obstacles.enclosed_space_of_safe_zone, min_vertex_distance, number_of_vertices,
                                 rand_seed, V_fix)
    graph = load_vertices_to_graph(graph, vertices)
    edges = generate_edges(graph, static_obstacles.enclosed_space_of_safe_zone,
                           static_obstacles.corners_of_safe_zone)
    graph = load_edges_to_graph(graph, edges, max_edge_length)

    return graph, vertices


def check_fix_vertices(V_fix: np.ndarray, enclosed_spaces: np.ndarray) -> None:
    """
    Check if any target is inside an obstacle, which can cause some inconvinience later.
    Print a WARNING if needed but do not exit.

    :param V_fix: array([[x,y,z]...[x,y,z]]) -> coordinates of the targets
    :param enclosed_spaces: array([[x,y,Z,W_x,W_y]...[x,y,Z,W_x,W_y]])
                            -> the positions of the obstacles and their increased height and widths
    :return: None
    """
    V_fix_ = remove_vertices_in_obstacles(V_fix, enclosed_spaces)
    if len(V_fix) != len(V_fix_):
        print_WARNING("Some fix vertices are inside the obstacles!!!")


def generate_vertices(dimensions: np.ndarray, enclosed_spaces: np.ndarray, thres: float, number_of_vertices: int,
                      rand_seed: int, V_fix: np.ndarray) -> np.ndarray:
    """
    Generate random vertices outside from the static obstacles with a minimum distance from each other.

    :param dimensions: array([x_min, x_max, y_min, y_max,z_min, z_max]) -> the borders of the flight zone
    :param enclosed_spaces: array([[x,y,Z,W_x,W_y]...[x,y,Z,W_x,W_y]])
                            -> the positions of the obstacles and their increased height and widths
    :param thres: float -> The minimum distance between the vertices.
    :param number_of_vertices: int -> The number of vertices to be scattered inside the flight zone.
    :param rand_seed: int -> the ranom seed for placing the verices
    :param V_fix:  array([[x,y,z]...[x,y,z]]) -> coordinates of the targets
    :return: vertices: array([[x,y,z]...[x,y,z]]) -> coordinates of the vertices
    """
    enclosing_vertices = generate_enclosing_mesh(dimensions)
    random_vertices = generate_random_vertices(number_of_vertices, dimensions, rand_seed)
    vertices = np.row_stack((enclosing_vertices, random_vertices, V_fix))
    vertices = remove_redundant_vertices(len(V_fix), vertices, thres)
    for i in range(5):
        random_vertices = generate_random_vertices(number_of_vertices, dimensions, rand_seed+i)
        prev_vertex_number = len(vertices)
        vertices = np.row_stack((random_vertices, vertices))
        vertices = remove_redundant_vertices(prev_vertex_number, vertices, thres)
    vertices = remove_vertices_in_obstacles(vertices, enclosed_spaces)

    return vertices


def generate_enclosing_mesh(dimensions: np.ndarray) -> np.ndarray:
    """
    Generate a mesh like border around the flying zone.
    Without it, there will be undesiredly long edges at the sides of the flying zone.

    :param dimensions: array([x_min, x_max, y_min, y_max,z_min, z_max]) -> the borders of the flight zone
    :return: enclosing_mesh: array([[x,y,z]...[x,y,z]]) -> coordinates of the mesh vertices
    """
    x_points = np.linspace(dimensions[0], dimensions[1], math.ceil((dimensions[1]-dimensions[0])*2))
    y_points = np.linspace(dimensions[2], dimensions[3], math.ceil((dimensions[3]-dimensions[2])*2))
    z_points = np.linspace(dimensions[4], dimensions[5], math.ceil((dimensions[5]-dimensions[4])*3))

    Xv, Yv = np.meshgrid(x_points, y_points)
    Xh, Zh = np.meshgrid(x_points, z_points)
    Yh, Zh = np.meshgrid(y_points, z_points)

    Xv = Xv.reshape((np.prod(Xv.shape),))
    Yv = Yv.reshape((np.prod(Yv.shape),))
    Zh = Zh.reshape((np.prod(Zh.shape),))
    Yh = Yh.reshape((np.prod(Yh.shape),))
    Xh = Xh.reshape((np.prod(Xh.shape),))

    bottom = np.ones(len(Xv)) * dimensions[4]
    enclosing_mesh = list(zip(Xv, Yv, bottom))
    top = np.ones(len(Xv)) * dimensions[5]
    enclosing_mesh = np.concatenate((enclosing_mesh, list(zip(Xv, Yv, top))))
    front = np.ones(len(Zh)) * dimensions[0]
    enclosing_mesh = np.concatenate((enclosing_mesh, list(zip(front, Yh, Zh))))
    back = np.ones(len(Zh)) * dimensions[1]
    enclosing_mesh = np.concatenate((enclosing_mesh, list(zip(back, Yh, Zh))))
    left = np.ones(len(Zh)) * dimensions[2]
    enclosing_mesh = np.concatenate((enclosing_mesh, list(zip(Xh, left, Zh))))
    right = np.ones(len(Zh)) * dimensions[3]
    enclosing_mesh = np.concatenate((enclosing_mesh, list(zip(Xh, right, Zh))))

    return enclosing_mesh


def generate_random_vertices(number_of_vertices: int, dimensions: np.ndarray, rand_seed: int) -> np.ndarray:
    """
    Generate given number of vertices scattered randomly inside the flight zone.

    :param number_of_vertices: int -> The number of vertices to be scattered inside the flight zone.
    :param dimensions: array([x_min, x_max, y_min, y_max,z_min, z_max]) -> the borders of the flight zone
    :param rand_seed: int -> the ranom seed for placing the verices
    :return: random_vertices: array([[x,y,z]...[x,y,z]]) -> coordinates of the randomly placed vertices
    """
    DIMENSION = 3
    np.random.seed(rand_seed)
    random_vertices = np.multiply(dimensions[1::2] - dimensions[0::2],
                                  np.random.rand(number_of_vertices, DIMENSION)) - [dimensions[1], dimensions[3],
                                                                                    - dimensions[4]]

    return random_vertices


def remove_redundant_vertices(number_of_fix_vertices: int, vertices: np.ndarray, thres: float) -> np.ndarray:
    """
    Remove vertices which are closer to other vertices than the threshold value. The fix vertices (targets) will be not
     removed.

    :param number_of_fix_vertices: int -> the number of vertices that should not be deleted
    :param vertices: array([[x,y,z]...[x,y,z]]) -> coordinates of all vertices (including targets)
    :param thres: float -> The minimum distance between the vertices
    :return: vertices: The remaining vertices with the desired minimum distace from each other.
    """

    removal = []
    for i in range(len(vertices) - number_of_fix_vertices - 1):  # Removing vertices that are too close
        if min(np.linalg.norm(vertices[i, :] - vertices[i + 1:, :], axis=1)) < thres:
            removal.append(i)
    vertices = np.delete(vertices, removal, axis=0)

    return vertices


def remove_vertices_in_obstacles(vertices: np.ndarray, enclosed_spaces: np.ndarray) -> np.ndarray:
    """
    Remove the vertices that are inside any static obstacle.

    :param vertices:  array([[x,y,z]...[x,y,z]]) -> the coordinates of the vertices
    :param enclosed_spaces: array([[x,y,Z,W_x,W_y]...[x,y,Z,W_x,W_y]])
                            -> the positions of the obstacles and their increased height and width
    :return: vertices: The input vertices without the ones in the obstacles.
    """
    for occupied_space in enclosed_spaces:
        # Sides of obstacle
        front_side = occupied_space[0] + occupied_space[3]
        back_side = occupied_space[0] - occupied_space[3]
        rigth_side = occupied_space[1] + occupied_space[4]
        left_side = occupied_space[1] - occupied_space[4]
        top = occupied_space[2]

        # check verteces outside from obstacle
        in_front_of = front_side < vertices[:, 0]
        behinde = back_side > vertices[:, 0]
        to_the_rigth = rigth_side < vertices[:, 1]
        to_the_left = left_side > vertices[:, 1]
        above = top < vertices[:, 2]

        outside = np.bitwise_or(in_front_of, behinde)
        outside = np.bitwise_or(outside, to_the_rigth)
        outside = np.bitwise_or(outside, to_the_left)
        outside = np.bitwise_or(outside, above)
        vertices = vertices[outside]

    return vertices


def load_vertices_to_graph(graph: nx.Graph, vertices: np.ndarray) -> nx.Graph:
    """
    Add the vertices to the nx.Graph object, with their index and positions.

    :param graph: nx.Graph object (empty)
    :param vertices: array([[x,y,z]...[x,y,z]]) -> the coordinates of the vertices
    :return: graph: nx.Graph object (with nodes)
    """
    index_start = len(graph) # needed for addind nodes to already exsting graph
    for i, position in enumerate(vertices):
        graph.add_node(i+index_start, pos=position)

    return graph


def generate_edges(graph: nx.Graph, enclosed_spaces: np.ndarray, corners_of_static_obstacles: np.ndarray) -> list:
    """
    Generate edges with delanuay triangulation and remove those which intersect any static obstacles.
    It gives back an adjacenci matrix wich is processeed by tho load_edges function.

    :param graph: nx.Graph object containing the vertices and their coordinates
    :param enclosed_spaces: array([[x,y,Z,W_x,W_y]...[x,y,Z,W_x,W_y]])
                            -> the positions of the obstacles and their increased height and width
    :param corners_of_static_obstacles: np.array([[c1...c8],[c1...c8]]), where c=[x,y,z]
    :return: graph_adj: [[neighbours of v_0]...[neighbours of v_N]] -> Lists of neighbours of the verteices which indices
                                                                       are the same as the indices of the lists
    """

    vertices = np.array(list(nx.get_node_attributes(graph, 'pos').values()))
    tri = Delaunay(vertices)  # Delaunay triangulation of a set of points
    # tri:[A,B,C],[A,B,D] -> Adj_graph:A[B,C,D],B[A,C,D],C[A,B],D[A,B]
    graph_adj = [list() for _ in range(len(vertices))]
    for simplex in tri.simplices:
        for j in range(4):
            a = simplex[j]
            graph_adj[a].extend(simplex)
            graph_adj[a].remove(a)
            graph_adj[a] = remove_duplicates(graph_adj[a])
            graph_adj[a] = [e for e in graph_adj[a] if not intersect(vertices[a], vertices[e], enclosed_spaces,
                                                                     corners_of_static_obstacles)]

    return graph_adj


def remove_duplicates(duplist: list) -> list:
    """
    Ensures that the adjacency matrix contains the vertices only once

    :param duplist: list of indeces with possible repetition
    :return: noduplist: list of unique indeces
    """
    noduplist = []
    for i in duplist:
        if i not in noduplist:
            noduplist.append(i)
    return noduplist


def intersect(v1: np.ndarray, v2: np.ndarray, enclosed_spaces: np.ndarray,
              corners_of_static_obstacles: np.ndarray) -> bool:
    """
    Check if an edge is intersecting a static obstacle or not.

    :param v1: array([x,y,z]) coordinates of a vertex
    :param v2: array([x,y,z]) coordinates of a vertex
    :param enclosed_spaces: array([[x,y,Z,W_x,W_y]...[x,y,Z,W_x,W_y]])
                            -> the positions of the obstacles and their increased height and width
    :param corners_of_static_obstacles: np.array([[c1...c8],[c1...c8]]), where c=[x,y,z]
    :return: True if the edg intersect with an obstacle, False otherwise.
    """
    xmin = min(v1[0], v2[0])
    xmax = max(v1[0], v2[0])
    ymin = min(v1[1], v2[1])
    ymax = max(v1[1], v2[1])
    zmin = min(v1[2], v2[2])

    for i, occupied_space in enumerate(enclosed_spaces):
        front_side = occupied_space[0] + occupied_space[3]
        back_side = occupied_space[0] - occupied_space[3]
        rigth_side = occupied_space[1] + occupied_space[4]
        left_side = occupied_space[1] - occupied_space[4]
        top = occupied_space[2]

        if xmax < back_side or xmin > front_side or ymax < left_side or ymin > rigth_side or zmin > top:
            continue

        a1, b1 = equation_plane(corners_of_static_obstacles[i][0], corners_of_static_obstacles[i][1], corners_of_static_obstacles[i][2])
        a2, b2 = equation_plane(corners_of_static_obstacles[i][4], corners_of_static_obstacles[i][6], corners_of_static_obstacles[i][5])
        a3, b3 = equation_plane(corners_of_static_obstacles[i][0], corners_of_static_obstacles[i][4], corners_of_static_obstacles[i][5])
        a4, b4 = equation_plane(corners_of_static_obstacles[i][0], corners_of_static_obstacles[i][5], corners_of_static_obstacles[i][6])
        a5, b5 = equation_plane(corners_of_static_obstacles[i][2], corners_of_static_obstacles[i][7], corners_of_static_obstacles[i][4])
        a6, b6 = equation_plane(corners_of_static_obstacles[i][7], corners_of_static_obstacles[i][1], corners_of_static_obstacles[i][6])

        dist = np.linalg.norm(v1-v2)
        q = (v2-v1)/dist

        opt_mod = Model("intersection")
        opt_mod.setParam('OutputFlag', False)
        opt_mod.setParam('TimeLimit', 0.01)
        lmd = opt_mod.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=dist, name='lmd')
        opt_mod.setObjective(1, GRB.MINIMIZE)
        opt_mod.addConstr(a1.dot(np.transpose(v1)) + lmd * (a1.dot(np.transpose(q))) <= b1)
        opt_mod.addConstr(a2.dot(np.transpose(v1)) + lmd * (a2.dot(np.transpose(q))) <= b2)
        opt_mod.addConstr(a3.dot(np.transpose(v1)) + lmd * (a3.dot(np.transpose(q))) <= b3)
        opt_mod.addConstr(a4.dot(np.transpose(v1)) + lmd * (a4.dot(np.transpose(q))) <= b4)
        opt_mod.addConstr(a5.dot(np.transpose(v1)) + lmd * (a5.dot(np.transpose(q))) <= b5)
        opt_mod.addConstr(a6.dot(np.transpose(v1)) + lmd * (a6.dot(np.transpose(q))) <= b6)
        opt_mod.optimize()

        try:
            lmd = opt_mod.objVal  # Inside
            return True
        except AttributeError as e:
            pass

    return False


def equation_plane(p1, p2, p3):
    # These two vectors are in the plane
    v1 = p3 - p1
    v2 = p2 - p1
    # the cross product is a vector normal to the plane
    cp = np.cross(v1, v2)
    a = cp
    # This evaluates a * x3 + b * y3 + c * z3 which equals d
    b = np.dot(cp, p3)
    return a, b


def load_edges_to_graph(graph: nx.Graph, edges: list, max_edge_length: float) -> nx.Graph:
    """
    Load the edges to the nx.Graph object with the edge lengths as weights

    :param graph: nx.Graph object without edges
    :param edges: [[neighbours of v_0]...[neighbours of v_N]] -> adjacency list
    :param max_edge_length: float -> the maximum length of the edges (longer edges are not added to the graph)
    :return: graph: nx.Graph object with edges
    """
    for vertex, neighbours in enumerate(edges):
        for neighbour in neighbours:
            length = np.linalg.norm(graph.nodes.data('pos')[vertex] - graph.nodes.data('pos')[neighbour])
            if length < max_edge_length:
                graph.add_edge(vertex, neighbour, weight=length)

    return graph


def extend_base_graph(scene, static_obstacles, min_vertex_distance, max_edge_length: float, number_of_vertices,
                      rand_seed, base_vertices):
    """
    Generate a graph with more vertices and edges based on the input graph.
    """
    graph = nx.Graph()
    graph = copy_vertices(scene.base_graph, graph)
    graph = dense_base_edges(scene.base_graph, graph)
    vertices_all, vertices_extra = generate_vertices_dense(scene, static_obstacles.enclosed_space_of_safe_zone,
                                                           min_vertex_distance, number_of_vertices, rand_seed,
                                                           base_vertices)
    graph = load_vertices_to_graph(graph, vertices_extra)
    edges = generate_edges(graph, static_obstacles.enclosed_space_of_safe_zone,
                           static_obstacles.corners_of_safe_zone)
    graph = load_edges_to_graph(graph, edges, max_edge_length)

    return graph


def copy_vertices(base_graph, graph):
    """
    Load the vertices of the original graph to the new nx.Graph object
    """
    for vertex in base_graph.nodes.data('pos'):
        graph.add_node(vertex[0], pos=vertex[1])

    return graph


def dense_base_edges(base_graph, graph):
    """
    Add edges and new vertices to the nx.Graph object based on the edges of the original  graph.
    The new edges are shorter and chained together to replace the original edges.
    """
    for edge in base_graph.edges.data('weight'):
        vertex_1 = base_graph.nodes.data('pos')[edge[0]]
        vertex_2 = base_graph.nodes.data('pos')[edge[1]]
        edge_length = edge[2]

        number_of_plus_vertices = math.ceil(edge_length / 0.1 + 0.7)
        plus_vertices_on_edge = np.linspace(vertex_1, vertex_2, number_of_plus_vertices)

        index_start = len(graph.nodes())
        index_stop = len(plus_vertices_on_edge) - 2

        # if there is no need for adding an extra vertex to the edge
        if index_stop == 0:
            graph.add_edge(edge[0], edge[1], weight=edge_length)
            continue

        for i, position in enumerate(plus_vertices_on_edge):
            if i == index_stop:
                graph.add_edge(index_start + i - 1, edge[1], weight=np.linalg.norm(vertex_2 - position))
                break
            graph.add_node(index_start + i, pos=position)
            if i == 0:
                graph.add_edge(edge[0], index_start, weight=np.linalg.norm(vertex_1 - position))
            else:
                graph.add_edge(index_start + i - 1, index_start + i, weight=np.linalg.norm(plus_vertices_on_edge[i - 1]
                                                                                           - position))

    return graph


def generate_vertices_dense(scene, occupied_spaces, thres, number_of_vertices, rand_seed, V_fix):
    """
    Generate random vertices outside from the static obstacles with a minimum distance from each other
    """
    random_vertices = generate_random_vertices(number_of_vertices, scene.dimensions, rand_seed)
    vertices = np.row_stack((random_vertices, V_fix))
    vertices = remove_redundant_vertices(V_fix, vertices, thres)
    vertices = remove_vertices_in_obstacles(vertices, occupied_spaces)
    # print vertex data
    print("Number of random extra vertices:", len(random_vertices))
    print("Number of all vertices:", len(vertices))
    print("Number of nonredundant vertices:", len(vertices))

    return vertices, vertices[:-len(V_fix)]


def create_point_cloud(graph: nx.Graph, density: float) -> Tuple[nx.Graph, np.ndarray]:
    """
    Divides the edges into individual points and stores them in a point cloud, while adding a new attribute to the
    edges of the graph that shows which rows of the point_cloud correspond to the edge.

    :param graph: nx.Graph object
    :param density: The approximate distance between the points that will be created along the edges of the graph
    :return: graph: nx.Graph object, where the edges get an artibute that shows which rows of the point_cloud correspond
                    to the edge
             point_cloud: array([[x,y,z]...[x,y,z]]) -> collection of points which represents the edges of the graph
    """
    first_iteration = True
    point_cloud = None
    for edge in list(graph.edges):
        v0 = graph.nodes.data('pos')[edge[0]]
        v1 = graph.nodes.data('pos')[edge[1]]
        edge_length = np.linalg.norm(v0-v1)
        point_number = math.ceil(edge_length / density)
        edge_points = np.linspace(v0, v1, point_number)
        if first_iteration:
            point_cloud = edge_points
            first_iteration = False
        else:
            point_cloud = np.row_stack((point_cloud, edge_points))
        graph[edge[0]][edge[1]]["point_range"] = [len(point_cloud)-len(edge_points), len(point_cloud)-1]

    return graph, point_cloud


def solve_target_point_collisions(graph: nx.Graph, point_cloud: np.ndarray, number_of_targets: int,
                                  safety_distance: float) -> None:
    """
    Check wich edges could be in collision with the drones in their target positions and mark them.
    This function adds a new atribute to the edges that contains the indeces of targetpoints where the edges are in
    collision with the drones.

    :param graph: nx.Graph object
    :param point_cloud: array([[x,y,z]...[x,y,z]]) -> collection of points which represents the edges of the graph
    :param number_of_targets: int
    :param safety_distance: float
    :return: None, but it modifies the graph
    """
    example_drone = Drone()
    radius = example_drone.radius
    downwash = example_drone.DOWNWASH
    vertices = np.array(list(nx.get_node_attributes(graph, 'pos').values()))
    dist_m = calculate_eplis_rel_dist(vertices[-number_of_targets:], point_cloud, downwash, radius, safety_distance)
    nx.set_edge_attributes(graph, None, 'touching_targets')
    for edge in graph.edges.data():
        touching_targets = np.unique(np.where(dist_m[:, edge[2]['point_range'][0]:edge[2]['point_range'][1]] <= 1)[0])
        edge[2]['touching_targets'] = touching_targets + (len(graph.nodes())-number_of_targets)


#=======================================================================================================================
# FOR: DYNAMIC OBSTACLES

def load_paths() -> list:
    """
    Try to load in previously generated moving obstacles paths
    """
    try:
        existing_paths = pickle_load("Pickle_saves/Construction_saves/paths_of_dynamic_obstacles.pickle")
        return existing_paths
    except (OSError, IOError):
        return []


def generate_dynamic_obstacles(paths_points: list, speeds: np.ndarray, radii: np.ndarray, desired_paths: np.ndarray,
                               start_times: np.ndarray) -> list:
    """
    Generate the dynamic obstacles.

    :param paths_points: [[p]...[p]], where p=[[x,y,z]...[x,y,z]] -> points of the paths of the obstacles
    :param speeds: array([obstacle_idx, speed]...[idx, speed]) -> define the speed of the obstacles
    :param radii: array([obstacle_idx, radius]...[idx, radius]) -> define the radius of the obstacles
    :param desired_paths: array([path_idx, path_idx, path_idx]) -> define the paths of the obstacles
    :param start_times: array([obstacle_idx, start_time]...[idx, start_time]) -> define the start of obstacle movements
    :return: dynamic_obstacles: list containing the generated dynamic_obstacle objects
    """

    dynamic_obstacles = []
    for i, points in enumerate(paths_points):
        if i not in desired_paths:
            continue

        spline = fit_spline(points)
        spline, length = parametrize_by_path_length(spline)

        if i in speeds[:, 0]:
            speed = speeds[speeds[:, 0] == i][0][1]
        else:
            speed = speeds[-1, 1]
        if i in radii[:, 0]:
            radius = radii[radii[:, 0] == i][0][1]
        else:
            radius = radii[-1, 1]
        if i in start_times[:, 0]:
            start_time = start_times[start_times[:, 0] == i][0][1]
        else:
            start_time = start_times[-1, 1]

        dynamic_obstacles.append(Dynamic_obstacle(path_tck=spline, path_length=length, speed=speed, radius=radius,
                                                  start_time=start_time))
    return dynamic_obstacles
