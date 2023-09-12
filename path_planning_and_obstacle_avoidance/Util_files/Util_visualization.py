import copy
import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
import stl


from path_planning_and_obstacle_avoidance.Util_files.Util_constuction import fit_spline
from path_planning_and_obstacle_avoidance.Util_files.Util_general import evaluate_splie


def plot_arena(dimensions: list) -> None:
    """
    Set up the dimensions of the plotted fligt zone, name the axes and set the initial view angle.

    :param dimensions: [x_min, x_max, y_min, y_max,z_min, z_max] -> the borders of the flight zone
    :return: None
    """
    ax = plt.axes(projection='3d')
    ax.set_xlim3d(dimensions[0], dimensions[1])
    ax.set_ylim3d(dimensions[2], dimensions[3])
    ax.set_zlim3d(0, dimensions[1] + dimensions[3])
    ax.set_aspect('auto', adjustable='box')
    plt.grid(True)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.view_init(elev=89.9999, azim=269.9999)


def plot_static_obstacles(corners_of_static_obstacles: np.ndarray, alpha: float) -> None:
    """
    Plot the static obstacles to the arena.

    :param corners_of_static_obstacles: np.array([[c1...c8],[c1...c8]]), where c=[x,y,z]
    :param alpha: float -> Defines the visibility of the obstacles. Use 1 for the obstacles
                           and a lower value for the safety zones.
    :return: None
    """
    ax = plt.gca()
    for i in range(corners_of_static_obstacles.shape[0]):
        p1 = corners_of_static_obstacles[i][0]
        p2 = corners_of_static_obstacles[i][1]
        p3 = corners_of_static_obstacles[i][2]
        p4 = corners_of_static_obstacles[i][3]
        p5 = corners_of_static_obstacles[i][4]
        p6 = corners_of_static_obstacles[i][5]
        p7 = corners_of_static_obstacles[i][6]
        p8 = corners_of_static_obstacles[i][7]

        if p8[2]>0.01:
            x = [p1[0], p2[0], p3[0], p4[0]]
            y = [p1[1], p2[1], p3[1], p4[1]]
            z = [p1[2], p2[2], p3[2], p4[2]]
            verts = [list(zip(x, y, z))]
            ax.add_collection3d(Poly3DCollection(verts, facecolors='black', alpha=alpha, edgecolor='k',linewidths=.1))
            x = [p1[0], p6[0], p5[0], p4[0]]
            y = [p1[1], p6[1], p5[1], p4[1]]
            z = [p1[2], p6[2], p5[2], p4[2]]
            verts = [list(zip(x, y, z))]
            ax.add_collection3d(Poly3DCollection(verts, facecolors='grey', alpha=alpha, edgecolor='k', linewidths=.1))
            x = [p1[0], p6[0], p7[0], p2[0]]
            y = [p1[1], p6[1], p7[1], p2[1]]
            z = [p1[2], p6[2], p7[2], p2[2]]
            verts = [list(zip(x, y, z))]
            ax.add_collection3d(Poly3DCollection(verts, facecolors='grey', alpha=alpha, edgecolor='k', linewidths=.1))
            x = [p3[0], p8[0], p5[0], p4[0]]
            y = [p3[1], p8[1], p5[1], p4[1]]
            z = [p3[2], p8[2], p5[2], p4[2]]
            verts = [list(zip(x, y, z))]
            ax.add_collection3d(Poly3DCollection(verts, facecolors='grey', alpha=alpha, edgecolor='k', linewidths=.1))
            x = [p8[0], p7[0], p2[0], p3[0]]
            y = [p8[1], p7[1], p2[1], p3[1]]
            z = [p8[2], p7[2], p2[2], p3[2]]
            verts = [list(zip(x, y, z))]
            ax.add_collection3d(Poly3DCollection(verts, facecolors='grey', alpha=alpha, edgecolor='k', linewidths=.1))
        x = [p5[0], p6[0], p7[0], p8[0]]
        y = [p5[1], p6[1], p7[1], p8[1]]
        z = [p5[2], p6[2], p7[2], p8[2]]
        verts = [list(zip(x, y, z))]
        ax.add_collection3d(Poly3DCollection(verts, facecolors='white', alpha=alpha, edgecolor='k', linewidths=.1))


def plot_vertices(V_set, size, color):
    ax = plt.gca()
    ax.scatter(V_set[:, 0], V_set[:, 1], V_set[:, 2], s=size, alpha=1, c=color)


def plot_graph(graph, size, color):
    ax = plt.gca()
    vertices = np.array([graph.nodes.data('pos')[i] for i in graph.nodes])
    ax.scatter(*vertices.T, s=size, alpha=1, c=color)
    edges = np.array([(graph.nodes.data('pos')[i], graph.nodes.data('pos')[j]) for i, j in graph.edges()])
    for edge in edges:
        ax.plot(*edge.T, color=color, linewidth=size/10)


def ask_for_paths(dimensions: np.ndarray) -> list:
    """
    Ask the user for the dynamic obstacle paths.

    :param dimensions: array([x_min, x_max, y_min, y_max,z_min, z_max]) -> the borders of the flight zone
    :return: paths_points: [[p]...[p]], where p=[[x,y,z]...[x,y,z]] -> points of the paths of the obstacles
    """
    fig = plt.gcf()  # Catch the current fig which previously created by create_view
    ax = fig.gca()
    plt.subplots_adjust(left=0.25)
    slider_ax = plt.axes([0.05, 0.25, 0.0225, 0.63])  # left, right, height, width
    x_slider = Slider(ax=slider_ax, label='X', orientation="vertical",
                      valmin=dimensions[0]-1, valmax=dimensions[1]+0.5, valinit=0)
    slider_ax = plt.axes([0.12, 0.25, 0.0225, 0.63])  # left, right, height, width
    y_slider = Slider(ax=slider_ax, label='Y', orientation="vertical",
                      valmin=dimensions[2]-1, valmax=dimensions[3]+0.5, valinit=0)
    slider_ax = plt.axes([0.19, 0.25, 0.0225, 0.63])  # left, right, height, width
    z_slider = Slider(ax=slider_ax, label='Z', orientation="vertical",
                      valmin=0, valmax=2.5, valinit=0)
    button_ax = plt.axes([0.05, 0.15, 0.12, 0.04])
    marker_button = Button(button_ax, 'Mark', hovercolor='0.975')
    button_ax = plt.axes([0.05, 0.08, 0.12, 0.04])
    draw_path_button = Button(button_ax, 'Draw path', hovercolor='0.975')

    surface = sphere_surface(radius=0.05, resolution=10)
    marker = ax.add_collection3d(Poly3DCollection(surface, facecolor='red'))

    marked = []
    points = []
    paths_points = []

    def update(_):
        marker.set_verts(surface + np.array([x_slider.val, y_slider.val, z_slider.val]))
        #marker = ax.plot_surface(x+x_slider.val, y+y_slider.val, z+z_slider.val, color='red')

    def set_marker(_):
        marked.append([ax.scatter(x_slider.val, y_slider.val, z_slider.val)])
        points.append([x_slider.val, y_slider.val, z_slider.val])

    def draw_3dspline(_):
        if len(points) < 2:  # Make sure that the user gives at least 2 points
            print('Select min 2 points with left mouse click\n'
                  'Stop adding points with right mouse click')
        else:
            paths_points.append(copy.deepcopy(points))
            spline = fit_spline(np.array(points))
            spline_points = evaluate_splie(spline)
            ax.plot(spline_points[0], spline_points[1], spline_points[2], 'black', lw=2)
            points.clear()

    x_slider.on_changed(update)
    y_slider.on_changed(update)
    z_slider.on_changed(update)
    marker_button.on_clicked(set_marker)
    draw_path_button.on_clicked(draw_3dspline)
    plt.show()

    return paths_points


def sphere_surface(radius, resolution):
    theta = np.linspace(0, np.pi, resolution)
    phi = np.linspace(0, np.pi, resolution)
    verts2 = []
    for i in range(len(phi) - 1):
        for j in range(len(theta) - 1):
            cp0 = radius * np.cos(phi[i])
            cp1 = radius * np.cos(phi[i + 1])
            sp0 = radius * np.sin(phi[i])
            sp1 = radius * np.sin(phi[i + 1])
            ct0 = np.cos(theta[j])
            ct1 = np.cos(theta[j + 1])
            st0 = radius * np.sin(theta[j])
            st1 = radius * np.sin(theta[j + 1])
            verts = [(cp0 * ct0, sp0 * ct0, st0), (cp1 * ct0, sp1 * ct0, st0), (cp1 * ct1, sp1 * ct1, st1),
                     (cp0 * ct1, sp0 * ct1, st1)]
            verts2.append(verts)
    return verts2


def cylinder_surface(radius: float, resolution: int) -> np.ndarray:
    phi = np.linspace(0, 360, resolution) / 180.0 * np.pi
    z = np.linspace(0, 1.5, 2)

    PHI, Z = np.meshgrid(phi, z)
    CP = radius * np.cos(PHI)
    SP = radius * np.sin(PHI)
    XYZ = np.dstack([CP, SP, Z])
    verts = np.stack([XYZ[:-1, :-1], XYZ[:-1, 1:], XYZ[1:, 1:], XYZ[1:, :-1]], axis=-2).reshape(-1, 4, 3)

    return verts


def place_cylinder(ax, cylinders, resolution, visibility, color):
    for cylinder in cylinders:
        cylinder.surface = cylinder_surface(cylinder.radius, resolution)
        cylinder.position = cylinder.move(t=np.array(0))
        cylinder.plot_face = ax.add_collection3d(Poly3DCollection(cylinder.surface + cylinder.position,
                                                                  alpha=visibility, facecolors=color))


def plot_spline(spline):
    fig = plt.gcf()
    ax = fig.gca()
    spline_points = evaluate_splie(spline)
    ax.plot(spline_points[0], spline_points[1], spline_points[2], 'black', lw=2)


def plot_trajectoty(spline, length, i):
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    fig = plt.gcf()
    ax = fig.gca()
    u = np.arange(0, length+0.01, 0.01)
    x, y, z = interpolate.splev(u, spline)
    points = np.array([x, y, z]).T.reshape(-1, 1, 3)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    ax.add_collection3d(Line3DCollection(segments, alpha=0.5, color=colors[i]))
    ax.scatter(points[0][0][0], points[0][0][1], points[0][0][2], s=5, alpha=1, c='green')
    ax.scatter(points[-1][0][0], points[-1][0][1], points[-1][0][2], s=5, alpha=1, c='red')


def place_spheres(ax, spheres, resolution, visibility, color):
    for sphere in spheres:
        sphere.surface = sphere_surface(sphere.radius, resolution)
        sphere.position = sphere.move(t=np.array(0))
        sphere.plot_face = ax.add_collection3d(Poly3DCollection(sphere.surface + sphere.position, alpha=visibility,
                                               facecolors=color))


def place_drones(ax, drones):
    stlmesh = stl.mesh.Mesh.from_file('Util_files/cf2_assembly.stl')
    # centering
    n = stlmesh.vectors.shape[0]
    offs = np.sum(np.sum(stlmesh.vectors, 0), 0) / (3 * n)
    faces = list((stlmesh.vectors - offs))
    for drone in drones:
        drone.stl_surface = faces
        drone.position = drone.move(t=np.array(0))
        drone.plot_stl = ax.add_collection3d(Poly3DCollection(drone.stl_surface + drone.position, facecolors=None,
                                                              edgecolor='k'))


def anim_ojbects(time, objects):
    for obj in objects:
        obj.position = obj.move(t=np.array(time))
        obj.plot_face.set_verts(obj.surface + obj.position)


def anim_drones(time, objects):
    for obj in objects:
        obj.position = obj.move(t=np.array(time))
        obj.plot_stl.set_verts(obj.stl_surface + obj.position)


def tellme(s: str) -> None:  # Helping function to write out instructions for the making the paths
    plt.title(s, fontsize=16)
    plt.draw()

