import cProfile
import os
import pstats
from typing import List, Dict

from path_planning_and_obstacle_avoidance.Util_files.Util_constuction import *
from path_planning_and_obstacle_avoidance.Util_files.Util_visualization import *
from path_planning_and_obstacle_avoidance.Classes import Construction, Static_obstacles

def get_id(string):
    '''takes a name as found in optitrack and returns the ID found in it, for example cf6 -> 06'''
    number_match = re.search(r'\d+', string)
    if number_match:
        number = number_match.group(0)
        if len(number) == 1:
            number = '0' + number
        return number
    return None

# ======================================================================================================================
    # CONSTRUCTION MAIN
def construction(real_drones: List[str], sim_drones: List[str], scene: Construction, start_pos):
    """
    The scene constructor sets up the static obstacles and the paths of the virtual moving obstacles.
    It also generates the searching graphs.

    The static obstacles can be given virtually or scanned with Optitrack.
    !!! The dimensions have to be set up manually either way at Classes -> Static_obstacles !!!
    """

    # Check if the directories made to contain generated files are exists. If not make them.
    dir_path = os.path.join(os.getcwd(), "path_planning_and_obstacle_avoidance")
    if not os.path.exists(dir_path + "/Pickle_saves"):
        os.mkdir(dir_path + "/Pickle_saves")
    if not os.path.exists(dir_path + "/Pickle_saves/Construction_saves"):
        os.mkdir(dir_path + "/Pickle_saves/Construction_saves")

# ======================================================================================================================
    # DIMENSIONS OF THE  FLYING ARENA
    plot_arena(scene.dimensions)

# ======================================================================================================================
    # PLACE STATIC OBSTACLES
    static_obstacles = Static_obstacles()

    obstacle_measurements = get_obstacles_positions_from_optitrack(scene.get_new_measurement)
    if not scene.real_obstacles:
        # if we don't want obstacles, remove everything other than the crazyflies themselves
        obstacle_measurements = {key: value for key, value in obstacle_measurements.items() if key.startswith('cf')}

    if scene.live_demo:
        # if the demo is live, we made a new measurement, which may contain drones that we don't want to fly
        # with: remove them
        extra_cfs = [key for key in obstacle_measurements.keys() if
                     key.startswith("cf") and get_id(key) not in real_drones]
        [obstacle_measurements.pop(cf) for cf in extra_cfs]
        # then add the simulated drones to the dictionary
        for idx, drone_ID in enumerate(sim_drones):
            obstacle_measurements[f"cf{drone_ID}"] = start_pos[idx]
    else:
        # remove cfs from the dictionary
        obstacle_measurements = {key: value for key, value in obstacle_measurements.items() if not key.startswith('cf')}
        # then add both the simulated and the 'real' drones to the dictionary
        for idx, drone_ID in enumerate(real_drones + sim_drones):
            obstacle_measurements[f"cf{drone_ID}"] = start_pos[idx]
    # Note: add_dimension_to_obstacles also has the side effect of organizing the obstacles in a way where the
    # cf-type objects are placed at the end of the array!
    enclosed_space_real = add_dimension_to_obstacles(obstacle_measurements, scene.real_obstacles_side_lengths)

    enclosed_space_simulated = []
    if scene.simulated_obstacles:
        enclosed_space_simulated = select_fix_obstacle_set(scene.static_obstacle_layout)

    if len(enclosed_space_simulated)>0 and len(enclosed_space_real)>0:
        static_obstacles.enclosed_space = np.row_stack((enclosed_space_real, enclosed_space_simulated))
    elif len(enclosed_space_real)>0:
        static_obstacles.enclosed_space = enclosed_space_real
    elif len(enclosed_space_simulated)>0:
        static_obstacles.enclosed_space = enclosed_space_simulated
    else:
        static_obstacles.enclosed_space = np.array([])

    static_obstacles.corners = calculate_corners(static_obstacles.enclosed_space)
    static_obstacles.enclosed_space_of_safe_zone = add_safety_zone_to_static_obstacles(static_obstacles.enclosed_space,
                                                                                       scene.general_safety_distance)
    static_obstacles.corners_of_safe_zone = calculate_corners(static_obstacles.enclosed_space_of_safe_zone)

    plot_static_obstacles(static_obstacles.corners, scene.obstacles_visibility)
    plot_static_obstacles(static_obstacles.corners_of_safe_zone, scene.safety_zone_visibility)

# ======================================================================================================================
    # GRAPH GENERATION
    V_fix = select_fix_vertex_set(scene.fix_vertex_layout)
    V_fix = add_vertices_above_obstacles(static_obstacles.enclosed_space_of_safe_zone, V_fix, scene.dimensions[-1],
                                         scene.howering_heigt)

    base_graph, base_vertices = generate_base_graph(scene.dimensions, static_obstacles,
                                                    scene.base_minimal_vertex_distance, scene.base_maximal_edge_length,
                                                    scene.base_vertex_number, scene.base_rand_seed, V_fix)
    base_graph, base_point_cloud = create_point_cloud(base_graph, scene.point_cloud_density)
    solve_target_point_collisions(base_graph, base_point_cloud, len(V_fix), scene.general_safety_distance)
    if scene.generate_dense_graph:
        densed_graph = extend_base_graph(scene, static_obstacles, scene.dense_minimal_vertex_distance,
                                         scene.dense_maximal_edge_length, scene.dense_vertex_number,
                                         scene.dense_rand_seed, base_vertices)
        densed_graph, densed_point_cloud = create_point_cloud(densed_graph, scene.point_cloud_density)
        solve_target_point_collisions(densed_graph, densed_point_cloud, len(V_fix), scene.general_safety_distance)
    else:
        densed_graph = None
        densed_point_cloud = None

    if scene.show_base_graph:
        plot_graph(graph=base_graph, size=1, color='black')
    if scene.show_dense_graph and scene.generate_dense_graph:
        plot_graph(graph=densed_graph, size=0.5, color='grey')
    if scene.show_fix_vertices:
        plot_vertices(V_set=V_fix, size=5, color='blue')

# ======================================================================================================================
    # ADD SIMULATED DYNAMIC OBSTACLES
    if scene.keep_existing_paths:
        paths_points = load_paths()
    else:
        paths_points = []

    if scene.show_paths_of_dynamic_obstacles:
        for points in paths_points:
            spline = fit_spline(np.array(points))
            plot_spline(spline)

    if scene.ask_for_new_paths:
        new_paths_points = ask_for_paths(scene.dimensions)
        paths_points.extend(new_paths_points[:])

    dynamic_obstacles = generate_dynamic_obstacles(paths_points, scene.speed_of_individual_dynamic_obstacles,
                                                   scene.radii_of_individual_dynamic_obstacles,
                                                   scene.path_of_dynamic_obstacles, scene.movement_start)

# ======================================================================================================================
    # SAVES

    pickle_save(dir_path+"/Pickle_saves/Construction_saves/number_of_targets.pickle", len(V_fix))
    pickle_save(dir_path+"/Pickle_saves/Construction_saves/base_graph.pickle", {'graph': base_graph,
                                                                                'point_cloud': base_point_cloud})
    pickle_save(dir_path+"/Pickle_saves/Construction_saves/densed_graph.pickle", {'graph': densed_graph,
                                                                                  'point_cloud': densed_point_cloud})
    pickle_save(dir_path+"/Pickle_saves/Construction_saves/static_obstacles.pickle", static_obstacles)
    pickle_save(dir_path+"/Pickle_saves/Construction_saves/paths_of_dynamic_obstacles.pickle", paths_points)
    pickle_save(dir_path+"/Pickle_saves/Construction_saves/dynamic_obstacles.pickle", dynamic_obstacles)
    # plt.show()
    graph = {
        "graph": base_graph,
        "point_cloud": base_point_cloud
    }
    return len(V_fix), graph, static_obstacles


# ======================================================================================================================
    # SELECTIONS

def select_fix_obstacle_set(index_of_obstacle_set: int) -> np.ndarray:
    """
    There can be defined the various static obstacle layouts. To add a new layout just add a new elif statment
    and define the ostacle layout there. The obstacles have to have their top center positions and their widths,
    where the widths can be either unified or unique for each obstacle.

    :param index_of_obstacle_set: int -> the serial number of the choosen obstacle set
    :return: enclosed_spaces: array([[x,y,z,w_x,w_y]...[x,y,z,w_x,w_y]])
                              -> the positions of the obstacles and their increased height and widths
    """
    if index_of_obstacle_set == 0:
        return np.array([])
    elif index_of_obstacle_set == 1:
        obstacle_positions = np.array([[1.2, 1.3, 2]])
        obstacle_dimensions = np.array([[0.6, 0.6]])/2
    elif index_of_obstacle_set == 2:
        obstacle_positions = np.array([[0, 0, 1], [0.5, 0.5, 2]])
        obstacle_dimensions = np.array([[0.75, 0.2], [0.5, 0.5]])/2
    elif index_of_obstacle_set == 3:
        obstacle_positions = np.array([[-0.75, -1.2, 2],[0, -1.2, 2], [-0.75, 1.2, 2], [0, 1.2, 2]])
        obstacle_dimensions = np.array([[0.01, 1]])/2
    else:
        return np.array([])

    obstacle_dimensions = match_dimensions(obstacle_dimensions, obstacle_positions)
    return np.column_stack((obstacle_positions, obstacle_dimensions))


def select_fix_vertex_set(index_of_verex_set: int) -> np.ndarray:
    """
    Define the target points which will be available for the drones to fly to. To add a new set of targets just
    add a new elif statement and define a set of cordinates.

    :param index_of_verex_set: int -> the selected target point set.
    :return: array([[x,y,z]...[x,y,z]]) -> the coordinates of the targets
    """
    if index_of_verex_set == 0:
        return np.array([])
    elif index_of_verex_set == 1:
        # Shape: X_________________________
        #           03/19 02/18 01/17 00/16|
        #           07/23 06/22 05/21 04/20|   2 layers
        #           11/27 10/26 09/25 08/24|
        #           15/31 14/30 13/29 12/28|
        #                                  Y
        inner_xy = 0.4
        outer_xy = 3 * inner_xy
        high_layer = 1.1
        low_layer = 0.4
        offset = 0.15

        V_fix = [[-outer_xy, -outer_xy, high_layer], [-inner_xy, -outer_xy, high_layer], [inner_xy, -outer_xy, high_layer], [outer_xy, -outer_xy, high_layer],
                 [-outer_xy, -inner_xy, high_layer], [-inner_xy - offset, -inner_xy - offset, high_layer], [inner_xy + offset, -inner_xy - offset, high_layer], [outer_xy, -inner_xy, high_layer],
                 [-outer_xy, inner_xy, high_layer], [-inner_xy - offset, inner_xy + offset, high_layer], [inner_xy + offset, inner_xy + offset, high_layer], [outer_xy, inner_xy, high_layer],
                 [-outer_xy, outer_xy, high_layer], [-inner_xy, outer_xy, high_layer], [inner_xy, outer_xy, high_layer], [outer_xy, outer_xy, high_layer]]
    elif index_of_verex_set == 2:
        # Shape: X_________________________
        #           03 02 01 00|
        #           07       04|   2 layer
        #           11       08|
        #           15 14 13 12|
        #                                  Y
        inner_xy = 0.4
        outer_xy = 3 * inner_xy
        high_layer = 1.1

        V_fix = [[-inner_xy, outer_xy, high_layer], [-outer_xy, outer_xy, high_layer],
                 [-outer_xy, inner_xy, high_layer], [-outer_xy, -inner_xy, high_layer],
                 [0.25, 1.1, high_layer]]

    elif index_of_verex_set == 3:
        # Shape: X_________________________
        #           03 02 01 00|
        #           07       04|   2 layer
        #           11       08|
        #           15 14 13 12|
        #                                  Y
        inner_xy = 0.4
        outer_xy = 3 * inner_xy
        high_layer = 1.1

        V_fix = [[-inner_xy, -outer_xy, high_layer],  [inner_xy, -outer_xy, high_layer],
                 [-inner_xy, outer_xy, high_layer], [-outer_xy, -outer_xy, high_layer],
                 [-outer_xy, outer_xy, high_layer], [inner_xy, outer_xy, high_layer]]
    elif index_of_verex_set == 4:
        # CAR DEMO
        V_fix = [[1.4,-1.3, 0.5],
                 [-1.35, 0.7, 0.7],
                 [-0.75, 1.35, 0.7],
                 [-0.35, 0, 1.2]]
    elif index_of_verex_set == 5:
        inner_xy = 0.4
        outer_xy = 3 * inner_xy
        high_layer = 0.5
        V_fix = [[-inner_xy, -outer_xy, high_layer], [inner_xy, -outer_xy, high_layer],
                 [-inner_xy, outer_xy, high_layer], [-outer_xy, -outer_xy, high_layer],
                 [-outer_xy, outer_xy, high_layer], [inner_xy, outer_xy, high_layer],
                 [-outer_xy, 0, high_layer], [-inner_xy, 0, high_layer], [inner_xy, 0, high_layer]]
    elif index_of_verex_set == 6:
        V_fix = [[-0.4, -1.4, 1],
                 [-1.35, 0.7, 0.7],
                 [-0.75, 1.35, 0.7],
                 [-0.35, 0, 1.2],
                 [-1.35, -0.4, 0.8],
                 [0, 1.35, 0.8]]
    else:
        return np.array([])
    return np.array(V_fix)


# ======================================================================================================================
    # MAIN
if __name__ == '__main__':
    """
    Measure detailed computational times.
    """
    profiler = cProfile.Profile()
    profiler.enable()
    construction()
    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats('cumtime')
    #stats.print_stats()
    plt.show()
