from queue import PriorityQueue
from typing import Tuple, List

import gurobipy.gurobipy
import matplotlib.pyplot as plt
import networkx as nx
import copy
import math

import numpy as np
from gurobipy import *

from path_planning_and_obstacle_avoidance.Util_files.Util_general import *
from path_planning_and_obstacle_avoidance.Util_files.Util_constuction import intersect
from path_planning_and_obstacle_avoidance.Util_files.Util_visualization import plot_arena
from path_planning_and_obstacle_avoidance.Util_files.Util_visualization import plot_trajectoty
from path_planning_and_obstacle_avoidance.Classes import Drone, Dynamic_obstacle


def add_coll_matrix_to_poles(obstacles: list, graph_dict: dict, Ts: float, cmin: int, cmax: int,
                             safety_distance: float) -> None:
    """
    Fill the collision matrix parameter of the obastacles.
    The collision matrix has the costs of the points of the graph edges in different time points.
    Handle the obstacles as floating spheres.

    :param obstacles: list of dynamic obstacle objects
    :param graph_dict:graph_dict: graph_dict['point_cloud'] = array([[x,y,z]...[x,y,z]]) -> collection of points which represents
                                                                                 the edges of the graph
    :param Ts: float -> sample time
    :param cmin: int -> the minimal cost of a collision
    :param cmax: int -> the maximal cost of a collision
    :param safety_distance: float -> minimal distance between the obstacles and the drones
    :return: None, but its modifies the dynamic obstacle objects
    """
    point_cloud = graph_dict['point_cloud']
    drone = Drone()
    for obs in obstacles:
        time_grid = np.arange(0, obs.path_time + Ts, Ts) + obs.start_time
        positions = obs.move(time_grid)
        dist_matrix = distance_matrix(positions[:, :2], point_cloud[:, :2]) # (time x edge)
        rel_distances = dist_matrix/(obs.radius+drone.radius+safety_distance)
        coll_matrix = (1 - rel_distances) * (cmax - cmin)
        coll_matrix = np.where(coll_matrix >= 0, cmin + coll_matrix, 0)
        obs.collision_matrix = np.column_stack((time_grid, coll_matrix))
        collision_matrix_compressed = ([np.append(points[2][0], np.max(obs.collision_matrix[:, points[2][0] + 1:points[2][1] + 2], axis=1))
                                        for points in graph_dict['graph'].edges.data('point_range')])
        collision_matrix_compressed = np.transpose(collision_matrix_compressed)
        obs.collision_matrix_compressed = np.column_stack((np.append(np.nan, time_grid), collision_matrix_compressed))


def add_coll_matrix_to_shepres(obstacles: list, point_cloud: np.ndarray, Ts: float, cmin: int, cmax: int,
                               safety_distance: float) -> None:
    """
    Fill the collision matrix parameter of the obastacles.
    The collision matrix has the costs of the points of the graph edges in different time points.
    Handle the obstacles as floating spheres.

    :param obstacles: list of dynamic obstacle objects
    :param point_cloud: array([[x,y,z]...[x,y,z]]) -> collection of points which represents the edges of the graph
    :param Ts: float -> sample time
    :param cmin: int -> the minimal cost of a collision
    :param cmax: int -> the maximal cost of a collision
    :param safety_distance: float -> minimal distance between the obstacles and the drones
    :return: None, but its modifies the dynamic obstacle objects
    """
    drone = Drone()
    for obs in obstacles:
        fligth_time = math.ceil((obs.path_length/obs.speed)*10)/10
        time_grid = np.arange(0, fligth_time + Ts, Ts)
        positions = obs.move(time_grid)
        dist_matrix = distance_matrix(positions, point_cloud) # (time x edge)
        rel_distances = dist_matrix/(obs.radius+drone.radius+safety_distance)
        coll_matrix = (1 - rel_distances) * (cmax - cmin)
        coll_matrix = np.where(coll_matrix >= 0, cmin + coll_matrix, 0)
        obs.collision_matrix = np.column_stack((time_grid, coll_matrix))


def add_coll_matrix_to_elipsoids(drones: list, graph_dict: dict, Ts: float, cmin: int, cmax: int,
                                 safety_distance: float) -> None:
    """
    Fill the collision matrix parameter of the drones.
    The collision matrix has the costs of the points of the graph edges in different time points.
    Handle the obstacles as floating elipsoids.

    :param drones: list of dynamic done objects
    :param graph_dict: graph_dict['point_cloud'] = array([[x,y,z]...[x,y,z]]) -> collection of points which represents
                                                                                 the edges of the graph
                       graph_dict['graph'] = nx.Graph object
    :param Ts: float -> sample time
    :param cmin: int -> the minimal cost of a collision
    :param cmax: int -> the maximal cost of a collision
    :param safety_distance: float -> minimal distance between the obstacles and the drones
    :return: None, but modifies the drone objects
    """
    point_cloud = graph_dict['point_cloud']
    for drone in drones:
        time_grid = np.arange(0, drone.fligth_time + Ts, Ts) + drone.start_time
        positions = drone.move(time_grid)
        rel_distances = calculate_eplis_rel_dist(positions, point_cloud, drone.DOWNWASH, drone.radius, safety_distance)
        coll_matrix = (1 - rel_distances) * (cmax - cmin)
        coll_matrix = np.where(coll_matrix >= 0, cmin + coll_matrix, 0)
        drone.collision_matrix = np.column_stack((time_grid, coll_matrix))
        collision_matrix_compressed = ([np.append(points[2][0], np.max(drone.collision_matrix[:, points[2][0] + 1:points[2][1] + 2], axis=1))
                                        for points in graph_dict['graph'].edges.data('point_range')])
        collision_matrix_compressed = np.transpose(collision_matrix_compressed)
        drone.collision_matrix_compressed = np.column_stack((np.append(np.nan, time_grid), collision_matrix_compressed))


def select_collision_matrix_time_window(moving_obstacles: list, time_max: float) -> float:
    """
    Select the time window for the collision cost matrix.

    :param moving_obstacles: list of the moving obstacles objects
    :param time_max: the latest time point when an obstacle is still in movement
    :return: time_max: the latest time point when an obstacle is still in movement

    """
    for moving_obstacle in moving_obstacles:
        if moving_obstacle.collision_matrix[-1][0] > time_max:
            time_max = moving_obstacle.collision_matrix[-1][0]

    return time_max


def summ_collision_matrices(other_drones: list, time_min: float, time_max: float, Ts: float) -> np.ndarray:
    """
    TODO
    :param other_drones:
    :param time_min:
    :param time_max:
    :param Ts:
    :return:
    """
    summed_coll_matrix = np.array([])

    # skip in case of the very first trajectory generation
    if len(other_drones) == 0:
        return summed_coll_matrix

    tgrid = np.arange(round(time_min/Ts), round(time_max/Ts)+1, 1)*Ts
    zero_M = np.zeros((len(tgrid), np.shape(other_drones[0].collision_matrix_compressed)[1]-1))
    summed_coll_matrix = np.column_stack((tgrid, zero_M))
    for other_drone in other_drones:
        added_coll_matrix = other_drone.collision_matrix_compressed[1:, 1:]
        later_times = round((time_max - other_drone.collision_matrix_compressed[-1][0])/Ts)
        if later_times < 0:
            print(f"LATER TIMES NEGATIVE??? {later_times}")
        if later_times: # When the obstacles finises its movement
            one_M = np.ones((later_times, np.shape(other_drones[0].collision_matrix_compressed)[1] - 1))
            coll_matrix_at_end = 99999999999 * one_M * other_drone.collision_matrix_compressed[-1][1:]
            added_coll_matrix = np.row_stack((added_coll_matrix, coll_matrix_at_end))
        # cut off the costs representing earlier time points than t_min
        earlier_times = round((time_min - other_drone.collision_matrix_compressed[1][0])/Ts)
        if earlier_times >= 0:
            added_coll_matrix = added_coll_matrix[earlier_times:]
        # during avoidance calculation it can happen that a drone is not yet started moving,
        # therefore adding earlier cost is needed
        else:
            one_M = np.ones((-earlier_times, np.shape(other_drones[0].collision_matrix_compressed)[1] - 1))
            coll_matrix_at_front = other_drone.collision_matrix_compressed[1][1:] * one_M
            added_coll_matrix = np.row_stack((coll_matrix_at_front, added_coll_matrix))
        summed_coll_matrix[:, 1:] = summed_coll_matrix[:, 1:]+added_coll_matrix
    # Add edge numbers
    summed_coll_matrix = np.row_stack((other_drones[0].collision_matrix_compressed[:][0], summed_coll_matrix))

    return summed_coll_matrix


def find_route(speeds: np.ndarray, graph: nx.Graph, dynamic_obstacles: list, other_drones: list, start_point: int,
               target_point: int, start_time: float, coll_matrix: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Find the best route in the graph from the start point to the target point.

    :param speeds: [v...v] -> possible constant speeds for the path planning
    :param graph: nx.Graph object
    :param dynamic_obstacles: list of dynamic obstacle objects
    :param other_drones: list of the drone objects except the drone which trajectory is currently generated
    :param start_point: the index of the vertex which from the drone will start its trajectory
    :param target_point: the index of the vertex which the drone targets
    :param start_time: float -> the time instance when the drone will start executing its trajectory
    :param coll_matrix: #TODO
    :return: final_route: array([start_idx...vertex_idx...target_idx]) -> series of vertices connecting the start and
                                                                          tartget vertices
             best_speed: float -> the speed which was used with the final_route
    """
    final_route = np.array([])
    time_of_final_route = np.inf
    best_speed = None
    for speed in speeds:
        route, time_of_route = A_star(graph, dynamic_obstacles, other_drones, start_point, target_point, start_time,
                                      speed, coll_matrix)
        if time_of_route < time_of_final_route:
            time_of_final_route = time_of_route
            final_route = route
            best_speed = speed

    return final_route, best_speed


def A_star(graph: nx.Graph, dynamic_obstacles: list, other_drones: list, start_point: int, target_point: int,
           start_time: float, drone_speed: float, coll_matrix: np.ndarray) -> Tuple[np.ndarray, float]:
    """
     A modified A* algorithm, which tries to avoid the dynamic obstacles.

    :param graph: nx.Graph object
    :param dynamic_obstacles: list of dynamic obstacle objects
    :param other_drones: list of the drone objects except the drone which trajectory is currently generated
    :param start_point: the index of the vertex which from the drone will start its trajectory
    :param target_point: the index of the vertex which the drone targets
    :param start_time: float -> the time instance when the drone will start executing its trajectory
    :param drone_speed: float -> used to approximate the drones possition to calculate the possible collisions
    :param coll_matrix: #TODO
    :return: route: array([start_idx...vertex_idx...target_idx]) -> series of vertices connecting the start and
                                                                    tartget vertices
             time_of_route: the time which needed for completing the route with the given constant speed
    """
    nv = max(graph.nodes) + 1
    visited = np.full(nv, False, dtype=bool)
    prev = np.zeros(nv)
    tt = np.full(nv, np.inf)
    tt[start_point] = 0
    cost = np.full(nv, np.inf)
    cost[start_point] = 0
    pq = PriorityQueue()
    pq.put((0, start_point))
    target_position = graph.nodes.data('pos')[target_point]

    while not pq.empty():
        (dist, current_vertex) = pq.get()
        if current_vertex == target_point:
            break
        visited[current_vertex] = True
        t = tt[current_vertex]
        for neighbour in list(graph[current_vertex].keys()):
            if visited[neighbour]:
                continue
            # Time to reach the target in a straight line
            neighbours_position = graph.nodes.data('pos')[neighbour]
            time_to_target = np.linalg.norm(neighbours_position-target_position)/drone_speed
            # Time cost of the neighbour
            dt = list(graph[neighbour][current_vertex].values())[0] / drone_speed
            # Collisions with dínamic obstacles while traversing the edge
            tspan = np.array([t, t + dt]) + start_time
            edege_collisions = graph[current_vertex][neighbour]['point_range']
            if not edege_collisions:
                cc = 0
            elif len(graph[current_vertex][neighbour]['touching_targets']) and \
                (not (any(start_point == graph[current_vertex][neighbour]["touching_targets"]) or
                      any(target_point == graph[current_vertex][neighbour]["touching_targets"]))):
                    cc = np.inf
            else:
                cc = summ_collision_costs_M(coll_matrix, tspan, edege_collisions[0])

            # Update route
            old_cost = cost[neighbour]
            new_cost = cost[current_vertex] + dt + cc + time_to_target
            if new_cost < old_cost:
                pq.put((new_cost, neighbour))
                cost[neighbour] = new_cost
                prev[neighbour] = current_vertex
                tt[neighbour] = tt[current_vertex] + dt

    if sum(prev) == 0:
        sys.exit("ERROR: A*\n"
                 "The A* did not find a suiteble route between the "
                 f"start_point: {start_point} {graph.nodes.data('pos')[start_point]} and the "
                 f"target_point: {target_point} {graph.nodes.data('pos')[target_point]}.\n\n"
                 "Possible solutions:\n"
                 "- Place further the target points\n"
                 "- Make the graph more dense\n"
                 "- reduce the radii of the drones and/or the safety distance (not recommended)")

    vk = target_point
    k = nv - 1
    route = np.zeros(nv)
    while vk != start_point:
        route[k] = vk
        k = k - 1
        vk = int(prev[vk])
    route[k] = vk
    route = route.astype(int)
    route = route[k:]
    time_of_route = tt[route][-1]

    return route, time_of_route


def summ_collision_costs_M(coll_matrix: np.ndarray, tspan: np.ndarray, edege: int):
    if len(coll_matrix) != 0:
        time_zone = np.where((coll_matrix[:, 0] >= (tspan[0] - 0.1)) & (coll_matrix[:, 0] <= (tspan[1] + 0.1)))[0]
        cc = np.sum(coll_matrix[time_zone, coll_matrix[0, :] == edege])
    else:
        cc = 0

    return cc


def extend_route(route: np.ndarray, graph: nx.Graph, LINE_BUFFER: int) -> np.ndarray:
    """
    Adds aditional points to the route to keep close the fitted spline

    :param route: array([start_idx...vertex_idx...target_idx]) -> series of vertices connecting the start and
                                                                    tartget vertices
    :param graph: nx.Graph object
    :param LINE_BUFFER: int -> the numberof points added to each edge
    :return: array([[x,y,z]...[x,y,z]]) -> thh coordinates of the points of the extended route
    """
    spline_points = np.array([])
    for i in range(len(route) - 1):
        v1 = graph.nodes.data('pos')[route[i]]
        v2 = graph.nodes.data('pos')[route[i+1]]
        if i == 0:
            spline_points = np.linspace(v1, v2, 2 + LINE_BUFFER)[:-1]
        else:
            spline_points = np.row_stack((spline_points, np.linspace(v1, v2, 2 + LINE_BUFFER)[:-1]))
    spline_points = np.row_stack((spline_points, graph.nodes.data('pos')[route[-1]]))

    return spline_points


def optimize_speed_profile(drone: Drone, other_drones: list, dynamic_obstacles: list, spline_path: list, length: float,
                           speed: float, Ts: float, safety_distance: float):
    """

    :param drone: Drone object
    :param other_drones: list of the drone objects except the drone which trajectory is currently generated
    :param dynamic_obstacles: list of the dynamic obstacles
    :param spline_path: A tuple, (t,c,k) containing the vector of knots, the B-spline coefficients, and the degree of
                        the spline.
    :param length: float -> the length of the spline
    :param speed: float -> the constant speed wich was selected during the route searching phase
    :param Ts: float -> sample time
    :param safety_distance: float -> the minimum distance between the drones and the obstacles
    :return: speed_profile: A tuple, (t,c,k) containing the vector of knots, the B-spline coefficients, and the degree
                            of the spline. Describes the t-s function
             flight_time: the duration of the trajetory
    """
    flight_time_multiplier = np.array([2, 4, 8, 16])

    time_windows = flight_time_multiplier * (length / speed)
    safety_time_window = 0
    for other_drone in other_drones:
        # The speed profile optimization has to consider the whole flight time of the other drones
        time_windows = time_windows[time_windows > (other_drone.fligth_time+other_drone.start_time-drone.start_time)]
        if other_drone.fligth_time > safety_time_window:
            safety_time_window = other_drone.fligth_time

    # If the maximum flight time is less than max(other_drone.fligth_time) then use the largest flight time
    if len(time_windows) == 0:
        time_windows = [safety_time_window]

    time_windows = np.append(time_windows,10)

    sgrid = np.linspace(0, length, math.ceil(length / (0.25 * drone.radius)))
    drone_pos = np.transpose(interpolate.splev(sgrid, spline_path))
    for time_window in time_windows:
        H = math.ceil(time_window / Ts)
        tgrid = np.arange(0, H * Ts, Ts) + drone.start_time
        table = np.zeros((len(tgrid), 201))
        table[:, 0] = np.transpose(tgrid)
        collision_counter = 0
        table, collision_counter = fill_table_poles(dynamic_obstacles, drone_pos, tgrid, sgrid, drone.radius,
                                                      safety_distance, table, collision_counter)
        table, collision_counter = fill_table_elipsoids(other_drones, drone_pos, tgrid, sgrid, drone.radius,
                                                        drone.DOWNWASH, safety_distance, table, collision_counter)

        opt_mod = run_gurobi(drone, collision_counter, table, H, Ts, length, tgrid, speed)

        try:
            ans = opt_mod.objVal
            break
        except AttributeError as e:
            print("Not enough time")
            if time_window == time_windows[-1]:
                #=======================================================
                print("DEBUG: DEADLOCK")

                with open("col_table.txt", 'w') as file:
                    np.set_printoptions(threshold=50000, linewidth=50000)
                    file.write(str(table))
                    np.set_printoptions(threshold=1000, linewidth=75)

                plot_arena([-1.5, 1.5, -1.5, 1.5, 0, 3])
                plot_trajectoty(spline_path, length, 1)
                plot_trajectoty(dynamic_obstacles[0].path_tck, dynamic_obstacles[0].path_length, 3)
                for other_drone in other_drones:
                    plot_trajectoty(other_drone.trajectory['spline_path'], other_drone.trajectory['spline_path'][0][-1], 4)
                plt.show()
                #=======================================================
                return None, None
            else:
                pass

    s_result = [var.x for var in opt_mod.getVars() if "s" in var.VarName]
    if len(tgrid) != len(s_result):
        tgrid = np.append(tgrid, tgrid[-1] + Ts)

    s_result, tgrid = trim_trajectory(s_result, tgrid)
    speed_profile = interpolate.splrep(tgrid, s_result, k=5)
    flight_time = tgrid[-1]-drone.start_time

    return speed_profile, flight_time


def fill_table_poles(dynamic_obstacles: list, drone_pos: np.ndarray, tgrid: np.ndarray, sgrid: np.ndarray,
                       drone_radius: float, safety_distance: float, table: np.ndarray,
                       collision_counter: int) -> Tuple[np.ndarray, int]:
    """
    Fills the table which contains which part of the path is occupied by the moving obstacles during the planning
    horizon. It handles the moving obstacles as spheres.

    :param dynamic_obstacles: list of the dynamic obstacles
    :param drone_pos: array([[x,y,z]...[x,y,z]]) ->  discretised path of the drone
    :param tgrid: array([t0...tH]) -> discretised time points
    :param sgrid: array([s0...sN]) -> discretised path length
    :param drone_radius: float -> the radius of the drone
    :param safety_distance: float -> the minimum distance between the drones and the obstacles
    :param table: NxM matrix containing the occupied parts of the path
    :param collision_counter: the number of instances when a moving obstacle touches the path
    :return: table, collision_counter: updated
    """
    for obs in dynamic_obstacles:
        active = False
        obs_pos = obs.move(tgrid)
        d_m = distance_matrix(obs_pos[:,:2], drone_pos[:,:2])
        for j in range(0, len(tgrid)):
            svals = sgrid[d_m[j, :] <= (drone_radius + obs.radius + safety_distance)]
            if svals.any():
                if not active:
                    active = True
                    collision_counter = collision_counter + 1
                table[j, 2 * collision_counter - 1], table[j, 2 * collision_counter] = np.amin(svals), np.amax(
                    svals)
            else:
                active = False
    return table, collision_counter


def fill_table_elipsoids(other_drones: list, drone_pos: np.ndarray, tgrid: np.ndarray, sgrid: np.ndarray,
                         drone_radius: float, drone_downwash: float, safety_distance: float, table: np.ndarray,
                         collision_counter: int) -> Tuple[np.ndarray, int]:
    """
    Fills the table which contains which part of the path is occupied by the moving obstacles during the planning
    horizon. It handles the moving obstacles as elipsoids.

    :param other_drones: list of the drone objects except the drone which trajectory is currently generated
    :param drone_pos: array([[x,y,z]...[x,y,z]]) ->  discretised path of the drone
    :param tgrid: array([t0...tH]) -> discretised time points
    :param sgrid: array([s0...sN]) -> discretised path length
    :param drone_radius: float -> the radius of the drone
    :param drone_downwash: float -> the height of the airflow under a drone
    :param safety_distance: float -> the minimum distance between the drones and the obstacles
    :param table: NxM matrix containing the occupied parts of the path
    :param collision_counter: the number of instances when a moving obstacle touches the path
    :return: table, collision_counter: updated
    """
    for obs in other_drones:
        active = False
        obs_pos = obs.move(tgrid)
        d_m = calculate_eplis_rel_dist(obs_pos, drone_pos, drone_downwash, drone_radius, safety_distance)
        for j in range(0, len(tgrid)):
            svals = sgrid[d_m[j, :] <= 1]
            if svals.any():
                if not active:
                    active = True
                    collision_counter = collision_counter + 1
                table[j, 2 * collision_counter - 1], table[j, 2 * collision_counter] = np.amin(svals), np.amax(svals)
            else:
                active = False
    return table, collision_counter


def run_gurobi(drone: Drone, collision_counter: int, table: np.ndarray, H: int, Ts: float, length: float,
               tgrid: np.ndarray, speed: float):
    """
    The optimization of the collision free speed profile.

    :param drone: Drone object
    :param collision_counter: the number of instances when a moving obstacle touches the path
    :param table: NxM matrix containing the occupied parts of the path
    :param H: int -> number of time points
    :param Ts: float -> sample time
    :param length: float -> the length of the spline
    :param tgrid: array([t0...tH]) -> discretised time points
    :param speed: float -> the constant speed wich was selected during the route searching phase
    :return: opt_mod: reasult of the optimization
    """
    opt_mod = Model(name="linear program")
    opt_mod.setParam('OutputFlag', False)

    a = opt_mod.addVars(H, name='a', vtype=GRB.CONTINUOUS, lb=-drone.MAX_ACCELERATION, ub=drone.MAX_ACCELERATION)
    v = opt_mod.addVars(H + 1, name='v', vtype=GRB.CONTINUOUS, lb=0, ub=drone.MAX_SPEED)
    s = opt_mod.addVars(H + 1, name='s', vtype=GRB.CONTINUOUS)
    adir = opt_mod.addVars(collision_counter, name='adir', vtype=GRB.BINARY)
    cost_binary = opt_mod.addVars(H + 1, name='c_binary', vtype=GRB.BINARY)

    sBM = length + 1
    bigM = 1000
    J_p = 5
    J_v = 1

    if drone.emergency:
        v_start = np.linalg.norm( drone.move(np.array(drone.start_time)) - drone.move(np.array(drone.start_time+0.001))  )/0.001
        if v_start > drone.MAX_SPEED:
            v_start = 0#drone.MAX_SPEED
    else:
        v_start = 0
    opt_mod.addConstr(s[0] == 0.0000)  # not 0 to negate the invincible start point effect
    opt_mod.addConstr(s[len(s) - 1] == length - 0.0000)
    opt_mod.addConstr(v[0] == v_start)
    opt_mod.addConstr(v[len(v) - 1] == 0)
    opt_mod.addConstrs(v[k + 1] == v[k] + a[k] * Ts
                       for k in range(H))
    opt_mod.addConstrs(s[k + 1] == s[k] + v[k] * Ts + 0.5 * (Ts ** 2) * a[k]
                       for k in range(H))
    opt_mod.addConstrs(s[k] - adir[obs] * sBM <= table[k, 2 * obs + 1]
                       for k in range(H)
                       for obs in range(collision_counter)
                       if table[k, 2 * obs + 2] > 0)
    opt_mod.addConstrs(s[k] + (1 - adir[obs]) * sBM >= table[k, 2 * obs + 2]
                       for k in range(H)
                       for obs in range(collision_counter)
                       if table[k, 2 * obs + 2] > 0)
    opt_mod.addConstrs(s[k] + 0.0001 - bigM * (1 - cost_binary[k]) <= length - 0.001
                       for k in range(H + 1))

    opt_mod.addConstrs(s[k] + bigM * cost_binary[k] >= length - 0.001
                       for k in range(H + 1))

    tgrid_H = np.append(tgrid, tgrid[-1]+Ts)-drone.start_time
    s_opt = [min(speed * t, length) for t in tgrid_H]
    difference_in_position = [(s[x] - s_opt[x]) for x in range(len(s))]
    difference_in_position = np.array(difference_in_position)
    difference_in_position = difference_in_position.dot(np.transpose(difference_in_position))

    difference_in_speed = [(v[x] - speed) for x in range(len(v))]
    difference_in_speed = np.array(difference_in_speed)
    difference_in_speed = difference_in_speed.dot(np.transpose(difference_in_speed))

    weighted_difference = J_p * difference_in_position + J_v * difference_in_speed
    d_compensation = 0
    for x in range(len(v)):
        d_compensation += speed ** 2 * (1 - cost_binary[x]) * J_v

    opt_mod.setObjective(weighted_difference - d_compensation)

    opt_mod.ModelSense = GRB.MINIMIZE
    opt_mod.optimize()

    return opt_mod


def trim_trajectory(s_result: list, tgrid: np.ndarray) -> Tuple[list, np.ndarray]:
    """
    Remove the end of the trajectory where the drone howers in its goal position.

    :param s_result: [s0...sH] -> the flight path lengths assigned to the time points
    :param tgrid: array([t0...tH]) -> discretised time points
    :return: s_result, tgrid: updated
    """
    standing = np.where(np.array(s_result) == s_result[-1])
    standing = np.flip(standing)[0]
    remove = False
    for i in range(len(standing)-1):
        if standing[i] - standing[i + 1] > 1:
            break
        remove = standing[i]
    if remove:
        if remove < 6: # for later spline generation m>k must hold
            remove = 6
        s_result = s_result[:remove]
        tgrid = tgrid[:remove]
    if len(s_result)<6:
        print_WARNING(f"Not enough point even with out trimming.\n s_result: {s_result} \n tgrid: {tgrid}")

    return s_result, tgrid


def choose_target(scene, drone, return_home: bool = False):
    """
    Assing a new target point for the given drone.

    :param scene: the free_targets contains those target points which are not start or goal ploints for either of the
                  drones
    :param drone: start_vertex -> the vertex which from the drone will start is trajectory
                  goal_vertex -> the vertex which to the drone has to go
    :return: None, but the free targets and the drone start and target pasitions are updated
    """
    if drone.start_vertex not in scene.home_positions:
        scene.free_targets = np.append(scene.free_targets, drone.start_vertex)
    drone.start_vertex = drone.target_vertex
    if return_home:
        drone.target_vertex = np.random.choice(scene.home_positions)
        scene.home_positions = np.delete(scene.home_positions, scene.home_positions == drone.target_vertex)
    else:
        drone.target_vertex = np.random.choice(scene.free_targets)
        scene.free_targets = np.delete(scene.free_targets, scene.free_targets == drone.target_vertex)


def check_collisions_with_new_obstacles(new_obstacle: Dynamic_obstacle, drones: list, Ts: float, safety_distance: float,
                                        collision_with: list) -> list:
    """
    TODO
    :param new_obstacle:
    :param drones:
    :param Ts:
    :param safety_distance:
    :param collision_with:
    :return:
    """

    obs_positions = new_obstacle.move(np.arange(new_obstacle.start_time,
                                                new_obstacle.start_time + new_obstacle.path_time + Ts, Ts))[:, :2]
    obs_start = new_obstacle.start_time
    for drn in drones:
        drone_arrive = drn.start_time + drn.fligth_time + Ts
        if drone_arrive <= obs_start:
            continue
        try:
            drn_position = drn.move(np.arange(obs_start, drone_arrive, Ts))[:, :2][:len(obs_positions)]
        except:
            print("ob_start, d_arrive",obs_start, drone_arrive)
        obs_positions_crop = obs_positions[:len(drn_position)]
        distances = np.linalg.norm(drn_position - obs_positions_crop, axis=1)

        # Select drones for imediate recalculation
        if any(distances <= (drn.radius + new_obstacle.radius + safety_distance)):
            collision_with.append(drn.serial_number)

    return collision_with


def check_collisions_with_single_obstacle(new_obstacle: Dynamic_obstacle, drones: List[Drone], Ts: float,
                                       safety_distance: float ) -> list:

    """
    new_obstacle.start_time
    new_obstacle.path_time = duration
    new_obstacle.radius
    """
    collision_with = []
    obs_positions = new_obstacle.move(np.arange(new_obstacle.start_time,
                                                new_obstacle.start_time + new_obstacle.path_time + Ts, Ts))[:, :2]
    obs_start = new_obstacle.start_time
    for drn in drones:
        drone_arrive = drn.start_time + drn.fligth_time + Ts
        if drone_arrive <= obs_start:
            continue
        drn_position = drn.move(np.arange(obs_start, drone_arrive, Ts))[:, :2][:len(obs_positions)]
        obs_positions_crop = obs_positions[:len(drn_position)]
        distances = np.linalg.norm(drn_position - obs_positions_crop, axis=1)

        # Select drones for imediate recalculation
        if any(distances <= (drn.radius + new_obstacle.radius + safety_distance)):
            collision_with.append(drn.cf_id)

    return collision_with


def expand_graph(graph, vertices, drone, time_now, static_obstacles):
    new_edge_num = 5
    drone_position_now = drone.move(time_now)
    dist = np.linalg.norm(vertices - drone_position_now, axis=1)
    idx_of_closest_vertices = np.argpartition(dist, new_edge_num)[:new_edge_num]
    idx_of_closest_vertices = [e for e in idx_of_closest_vertices if not intersect(drone_position_now,
                                                                                   vertices[e],
                                                                                   static_obstacles.enclosed_space_of_safe_zone,
                                                                                   static_obstacles.corners_of_safe_zone)]
    new_graph = copy.deepcopy(graph['graph'])
    new_idx = len(graph['graph'].nodes())
    new_graph.add_node(new_idx, pos=drone_position_now)
    for neighbour in idx_of_closest_vertices:
        edge_length = np.linalg.norm(new_graph.nodes.data('pos')[new_idx] - new_graph.nodes.data('pos')[neighbour])
        new_graph.add_edge(new_idx, neighbour, weight=edge_length, point_range=None)

    return new_graph, new_idx
