import cProfile
import pstats
import time

from path_planning_and_obstacle_avoidance.Classes import Construction, Drone
from path_planning_and_obstacle_avoidance.Util_files.Util_trajectory_planning import *


def generate_trajectory(drone, G: dict, dynamic_obstacles, other_drones, Ts, safety_distance):

    # ==================================================================================================================
    # SUMM COLLISION MATRICES
    time_min = drone.start_time
    time_max = time_min+30
    coll_matrix_summ = summ_collision_matrices(other_drones, time_min, time_max, Ts)
    if not len(dynamic_obstacles) == 0:
        coll_matrix_summ_obs = summ_collision_matrices(dynamic_obstacles, time_min, time_max, Ts)
        if len(coll_matrix_summ) !=0:
            coll_matrix_summ[1:, 1:] += coll_matrix_summ_obs[1:, 1:]
        else:
            coll_matrix_summ = coll_matrix_summ_obs
    # ==================================================================================================================
    # FIND ROUTE
    route, speed = find_route(drone.constant_speeds, G['graph'], dynamic_obstacles, other_drones,
                              drone.start_vertex, drone.target_vertex, drone.start_time, coll_matrix_summ)

    # ==================================================================================================================
    # FIT SPLINE
    line_buffer = 5
    spline_points = extend_route(route, G['graph'], line_buffer)
    spline = fit_spline(spline_points)
    spline_path, length = parametrize_by_path_length(spline)

    # ==================================================================================================================
    # DESIGN SPEED PROFILE
    speed_profile, flight_time = optimize_speed_profile(drone, other_drones, dynamic_obstacles, spline_path, length,
                                                        speed, Ts, safety_distance)

    # ==================================================================================================================
    # RETURN
    if flight_time is None:
        print_WARNING("DEADLOCK")
 
    return spline_path, speed_profile, flight_time, length


if __name__ == '__main__':
    """
    Measure detailed computational times.
    """
    profiler = cProfile.Profile()
    profiler.enable()

    graph = pickle_load("Pickle_saves/Construction_saves/base_graph.pickle")
    dynamic_obstacles = pickle_load("Pickle_saves/Construction_saves/dynamic_obstacles.pickle")
    static_obstacles = pickle_load("Pickle_saves/Construction_saves/static_obstacles.pickle")
    number_of_targets = pickle_load("Pickle_saves/Construction_saves/number_of_targets.pickle")
    scene = Construction()
    add_coll_matrix_to_poles(dynamic_obstacles, graph, scene.Ts, scene.cmin, scene.cmax,
                             scene.general_safety_distance)

    target_zero = len(graph['graph'].nodes()) - number_of_targets
    targets = np.arange(target_zero, len(graph['graph'].nodes()), 1)

    drone_num = 3
    past_drones = []

    for i in range(drone_num):
        t0 = time.time()
        drone = Drone()
        drone.serial_number = i
        drone.start_vertex = np.random.choice(targets)
        targets = np.delete(targets, targets == drone.start_vertex)
        drone.target_vertex = np.random.choice(targets)
        targets = np.delete(targets, targets == drone.target_vertex)
        spline_path, speed_profile, fligth_time, length = generate_trajectory(drone, graph, dynamic_obstacles,
                                                                              past_drones, scene.Ts,
                                                                              scene.general_safety_distance)

        drone.trajectory = {'spline_path': spline_path, 'speed_profile': speed_profile}
        drone.fligth_time = fligth_time
        add_coll_matrix_to_elipsoids([drone], graph, scene.Ts, scene.cmin, scene.cmax,
                                     scene.general_safety_distance)
        past_drones.append(drone)

        t1 = time.time()
        print("-------------------------------------------------------------------------------------")
        print("Trajectory generation:", t1 - t0, "sec")
    print("-------------------------------------------------------------------------------------")

    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats('cumtime')
    stats.print_stats()
