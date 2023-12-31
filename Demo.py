import matplotlib.pyplot as plt
import math
from path_planning_and_obstacle_avoidance.Scene_construction import construction
from path_planning_and_obstacle_avoidance.Classes import Construction, Drone, Static_obstacles
from path_planning_and_obstacle_avoidance.Trajectory_planning import *
import re
import sys
import trio
from trio import sleep, sleep_until, current_time, Event
import json
from scipy import interpolate
import numpy as np
import pickle
from typing import Tuple, List, Union, Any, Callable, Dict, Optional
import os
import shutil
import zipfile
from functools import partial
import motioncapture
from collections import namedtuple
import traceback
from queue import PriorityQueue, Queue
import socket

PRIORITY_HIGH = 1
PRIORITY_MID = 2
PRIORITY_LOW = 3


class PrioEvent:
    """A class that has wait() and set() like trio.Event, but is comparable. Here, a subclass would be the obvious
    soluition, instead of this. However, trio.Event doesn't allow subclassing. """
    def __init__(self, priority: int):
        self.event = Event()
        self.priority = priority

    async def wait(self):
        await self.event.wait()

    def set(self):
        self.event.set()

    def __lt__(self, other):
        return self.priority < other.priority


class Semaphore:
    def __init__(self):
        self.taken = False
        self.queue = PriorityQueue()

    async def wait_take(self, priority):
        if not self.taken:
            self.taken = True
        else:
            ticket = PrioEvent(priority)
            self.queue.put(ticket)
            await ticket.wait()

    def let_go(self):
        if self.queue.empty():
            self.taken = False
        else:
            ticket: PrioEvent = self.queue.get()
            ticket.set()


async def send_with_semaphore(data: bytes, semaphore: Semaphore, stream: trio.SocketStream, wait_ack: bool):
    await semaphore.wait_take(PRIORITY_MID)
    assert stream is not None
    await stream.send_all(data)
    ack = b""
    if wait_ack:
        while ack != b"ACK":
            ack = await stream.receive_some()
            await sleep(0.001)
    semaphore.let_go()


def send_skyc(file: str):
    ip = "127.0.0.1"
    port = (SETTINGS.get("SERVER_PORT") if SETTINGS.get("LIVE_DEMO") else SETTINGS.get("DUMMY_SERVER_PORT")) + 2
    Socket = socket.socket()
    Socket.connect((ip, port))
    Socket.settimeout(None)
    with open(file, 'rb') as file:
        data = file.read() + b'SKYC'
    Socket.sendall(data)


def evaluate_trajectories(drone_IDs: List[str]):
    """Function that takes a look at the saved trajectories, and evaluates them with a certain time interval
    (currently 1ms), paying attention to the fact that we may swap to a different trajectory before the previous is
    over, if an emergency trajectory was uploaded."""
    time_interval = 1/1000
    TXYZ = []
    for drone_ID in drone_IDs:
        trajectories: List[Tuple] = []  # the trajectories (as saved by the handler) are stored here
        txyz = [[], [], [], []]  # the final evaluations for the trajectory are stored here
        traj_folder = os.path.join(SETTINGS.get("traj_folder_name"), str(drone_ID))
        # extract the trajectories, which have numbers in order of the time when they were saved
        for file_name in sorted(os.listdir(traj_folder)):
            file_path = os.path.join(traj_folder, file_name)
            with open(file_path, 'rb') as file:
                traj_data: Tuple = pickle.load(file)
                trajectories.append(traj_data)
        for idx, trajectory in enumerate(trajectories):
            start_time = trajectory[0]  # when the trajectory was started, IN DEMO TIME
            spline_path = trajectory[1]  # distance travelled(time spent), IN TRIO.TIME
            speed_profile = trajectory[2]  # position(distance travelled)
            speed_profile_duration = speed_profile[0][-1] - speed_profile[0][0]
            next_start_time = trajectories[idx+1][0] if idx+1 < len(trajectories) else start_time + speed_profile_duration + time_interval  # when the following trajectory was started, IN DEMO TIME
            finished_normally = start_time + speed_profile_duration < next_start_time  # True if the trajectory was not aborted early
            if not finished_normally:
                warning(f"Drone {drone_ID} had an early trajectory recalculation. It was going to finish at "
                        f"{(start_time + speed_profile_duration):.3f}, but the new trajectory starts at {(next_start_time):.3f}.")
            end_time = start_time + speed_profile_duration if finished_normally else next_start_time  # when the trajectory actually ended, IN DEMO TIME
            duration = end_time - start_time  # what the trajectory's duration ended up being
            eval_times = np.arange(speed_profile[0][0], speed_profile[0][0] + duration, time_interval)  # timestamps where we evaluate the speed profile, IN TRIO.TIME
            distance_travelled = interpolate.splev(eval_times, speed_profile)
            spline_eval = interpolate.splev(distance_travelled, spline_path)  # the position data that will correspond to the timestamps
            timestamps = eval_times - eval_times[0] + start_time  # the timestamps, shifted from trio time into DEMO TIME
            if finished_normally:  # this means we will have a hover segment: add extra eval points for it
                extra_timestamps = np.arange(timestamps[-1]+time_interval, next_start_time, time_interval)
                extra_x = np.array([spline_eval[0][-1]]*len(extra_timestamps))
                extra_y = np.array([spline_eval[1][-1]] * len(extra_timestamps))
                extra_z = np.array([spline_eval[2][-1]] * len(extra_timestamps))
                timestamps = np.append(timestamps, extra_timestamps)
                spline_eval[0] = np.append(spline_eval[0], extra_x)
                spline_eval[1] = np.append(spline_eval[1], extra_y)
                spline_eval[2] = np.append(spline_eval[2], extra_z)
            txyz[0].extend(list(timestamps))
            txyz[1].extend(list(spline_eval[0]))
            txyz[2].extend(list(spline_eval[1]))
            txyz[3].extend(list(spline_eval[2]))
        TXYZ.extend([txyz])
    return TXYZ


def get_skyc_data(TXYZ):
    """Function that converts the raw t-x-y-z data into bezier curves."""
    compressed = SETTINGS.get("traj_type") == "COMPRESSED"
    number_of_bezier_segments = 100 if compressed else 25
    degree = 3 if compressed else 5
    Skyc_Data = []
    for txyz in TXYZ:
        t, x, y, z = txyz
        takeoff_time = t[0]
        knots = determine_knots(t, number_of_bezier_segments)[1:-1]
        # We need to give the splrep inside knots. I think [0] and [-1] should also technically be inside knots, but apparently
        # not. I seem to remember that the first k-1 and last k-1 knots are the outside knots. Anyway, slprep seems to add k
        # knots both at the end and at the beginning, instead of k-1 knots which is what would make sense to me. How it decides
        # what those knots should be is a mystery to me, but upon checking them, they are the exact first and last knots that I
        # would've added, so it works out kind of.
        w = [1] * len(t)  # if the fit is particularly bad a certain point in the path, we can adjust it here
        xSpline = interpolate.splrep(t, x, w, k=degree, task=-1, t=knots)
        ySpline = interpolate.splrep(t, y, w, k=degree, task=-1, t=knots)
        zSpline = interpolate.splrep(t, z, w, k=degree, task=-1, t=knots)
        Bezier_Data = [[0, [x[0], y[0], 0], []],
                       [takeoff_time, [x[0], y[0], z[0]], []]]
        # BPoly can be constructed from PPoly but not from BSpline. PPoly can be constructed from BSPline. BSpline can
        # be fitted to points. So Points->PPoly->BPoly. The coeffs of the BPoly representation are the control points.
        x_PPoly = interpolate.PPoly.from_spline(xSpline)
        y_PPoly = interpolate.PPoly.from_spline(ySpline)
        z_PPoly = interpolate.PPoly.from_spline(zSpline)
        x_BPoly = interpolate.BPoly.from_power_basis(x_PPoly)
        y_BPoly = interpolate.BPoly.from_power_basis(y_PPoly)
        z_BPoly = interpolate.BPoly.from_power_basis(z_PPoly)
        # These two lines below seem complicated but all they do is pack the data above into a convenient form: a list
        # of lists where each element looks like this: [t, (x,y,z), (x,y,z), (x,y,z)]. Note that this can almost
        # definitely be done in a simpler way :)
        BPoly = list(zip(list(x_BPoly.x)[degree + 1:-degree],
                         list(x_BPoly.c.transpose())[degree:-degree],
                         list(y_BPoly.c.transpose())[degree:-degree],
                         list(z_BPoly.c.transpose())[degree:-degree]))
        BPoly = [[element[0]]+list(zip(*list(element[1:]))) for element in BPoly]
        for Bezier_Curve in BPoly:
            curve_to_append = [Bezier_Curve[0],
                               Bezier_Curve[-1],
                               Bezier_Curve[2:-1]]
            Bezier_Data.append(curve_to_append)
        land_segment = [Bezier_Data[-1][0] + takeoff_time,
                        [Bezier_Data[-1][1][0], Bezier_Data[-1][1][1], 0],
                        []]
        Bezier_Data.append(land_segment)
        Skyc_Data.append(Bezier_Data)
    return Skyc_Data


def cleanup(files: List[str], folders: List[str]):
    """function meant for deleting unnecessary files"""
    for file in files:
        if os.path.exists(file):
            os.remove(file)
            print(f"Deleted {file}")
    for folder in folders:
        if os.path.exists(folder):
            shutil.rmtree(folder)
            print(f"Deleted {folder} folder")


def write_trajectory(Data):
    """Function that makes a trajectory.json file from the raw data."""
    json_dict = {
        "version": 1,
        "points": Data,
        "takeoffTime": Data[0][0],
        "landingTime": Data[-1][0],
        "type": SETTINGS.get("traj_type")
    }
    json_object = json.dumps(json_dict, indent=2)
    with open("trajectory.json", "w") as f:
        f.write(json_object)


def write_to_skyc(filename, Skyc_Data):
    """Function that generates a complete skyc file from the bezier curves."""
    # delete every file that we can generate that might have been left over from previous sessions
    name = filename
    cleanup(files=["show.json",
                   "cues.json",
                   f"{name}.zip",
                   f"{name}.skyc",
                   "trajectory.json"],
            folders=["drones"])
    # Create the 'drones' folder if it doesn't already exist
    os.makedirs('drones', exist_ok=True)
    drones = []
    for index, Data in enumerate(Skyc_Data):
        write_trajectory(Data)
        drone_settings = {
            "trajectory": {"$ref": f"./drones/drone_{index}/trajectory.json#"},
            "home": Data[0][1][0:3],
            "landAt": Data[-1][1][0:3],
            "name": f"drone_{index}"
        }
        drones.append({
            "type": "generic",
            "settings": drone_settings
        })
        drone_folder = os.path.join('drones', f'drone_{index}')
        os.makedirs(drone_folder, exist_ok=True)
        shutil.move('trajectory.json', drone_folder)
    # This wall of text below is just overhead that is required to make a skyc file.
    ########################################CUES.JSON########################################
    items = [{"time": Skyc_Data[0][0][0],
              "name": "start"}]
    cues = {
        "version": 1,
        "items": items
    }
    json_object = json.dumps(cues, indent=2)
    with open("cues.json", "w") as f:
        f.write(json_object)
    #######################################SHOW.JSON###########################################
    validation = {
        "maxAltitude": 2.0,
        "maxVelocityXY": 2.0,
        "maxVelocityZ": 1.5,
        "minDistance": 0.8
    }
    cues = {
        "$ref": "./cues.json"
    }
    settings = {
        "cues": cues,
        "validation": validation
    }
    meta = {
        "id": f"{name}.py",
        "inputs": [f"{name}.py"]
    }
    show = {
        "version": 1,
        "settings": settings,
        "swarm": {"drones": drones},
        "environment": {"type": "indoor"},
        "meta": meta,
        "media": {}
    }
    json_object = json.dumps(show, indent=2)
    with open("show.json", "w") as f:
        f.write(json_object)

    # Create a new zip file
    with zipfile.ZipFile(f"{name}.zip", 'w', zipfile.ZIP_DEFLATED) as zipf:
        # Add the first file to the zip
        zipf.write("show.json")

        # Add the second file to the zip
        zipf.write("cues.json")

        # Recursively add files from the specified folder and its sub-folders
        for root, _, files in os.walk("drones"):
            for file in files:
                file_path = os.path.join(root, file)
                zipf.write(file_path)

    print(f'Compression complete. The files and folder have been zipped as {name}.zip.')

    os.rename(f'{name}.zip', f'{name}.skyc')
    # Delete everything that's not 'trajectory.skyc'
    cleanup(files=["show.json",
                   "cues.json",
                   f"{name}.zip",
                   "trajectory.json"],
            folders=["drones"])
    print(f"{name}.skyc file ready!")


def generate_skyc_file(filename, drones: List[str]):
    TXYZ = evaluate_trajectories(drones)
    Skyc_Data = get_skyc_data(TXYZ)
    write_to_skyc(filename=filename, Skyc_Data=Skyc_Data)


async def async_generate_trajectory(drone, G, dynamic_obstacles, other_drones, Ts, safety_distance):
    """An asynchronous, and therefore interruptable version of the
    generate_trajectory function :)"""
    tmpfunc = partial(generate_trajectory, drone=drone, G=G, dynamic_obstacles=dynamic_obstacles,
                      other_drones=other_drones, Ts=Ts, safety_distance=safety_distance)
    spline_path, speed_profile, duration, length = await trio.to_thread.run_sync(tmpfunc)
    return spline_path, speed_profile, duration, length


def warning(text):
    red = "\033[91m"
    reset = "\033[0m"
    formatted_text = f"WARNING: {text}"
    print(red + formatted_text + reset)


def display_time():
    """returns the time since it was first called, in order to make time.time() more usable, since time.time() is
    a big number"""
    if not hasattr(display_time, 'start_time'):
        display_time.start_time = current_time()
    return current_time() - display_time.start_time


class DroneCommand:
    def __init__(self, command: Callable, deadline: float):
        self.command = command
        self.deadline = deadline


class DroneHandler:
    next_command: Union[DroneCommand, None] # which function to call next, and when it has to finish

    def __init__(self, socket: trio.SocketStream, drone: Drone, other_drones: List[Drone], color: str,
                 calc_semaphore: Semaphore):
        self.drone = drone
        self.drone_ID = drone.cf_id  # same as the key in the dictionary for the handler
        self.socket: Union[None, trio.SocketStream] = socket  # used for communication with the handler
        self.trajectory: str = splines_to_json(drone.trajectory['spline_path'], drone.trajectory['speed_profile'])
        self.return_to_home = False  # used to signal that the drone's next trajectory will be a RTH maneuver
        self.other_drones = other_drones
        self.interrupt = Event()  # used to wake the drone from sleep
        self.text_color = color if self.drone_ID != "10" else "\033[96m"
        self.traj_id = 0
        self.calc_semaphore = calc_semaphore
        self.stream_semaphore = Semaphore()


        # make folders for trajectory files and log files:
        traj_folder_path = os.path.join(os.getcwd(), SETTINGS.get("traj_folder_name"))
        log_folder_path = os.path.join(os.getcwd(), SETTINGS.get("log_folder_name"))
        self.traj_folder = os.path.join(traj_folder_path, self.drone_ID)
        os.makedirs(self.traj_folder)
        self.log_file_path = os.path.join(log_folder_path, self.drone_ID)
        if os.path.exists(self.log_file_path):
            os.remove(self.log_file_path)
        open(self.log_file_path, 'a').close()

    async def sim_send(self, data):
        pass

    def print(self, text):
        """Function that we can use instead of the regular pring in the context of a drone handler, to print with
        timestamp, drone annotation, and the drone's color to differentiate between drones. """
        reset_color = "\033[0m"
        formatted_text = f"[{display_time():.3f}] {self.drone_ID}: {text}"
        print(self.text_color + formatted_text + reset_color)

    def save_traj(self):
        """Function that writes the currently memorized drone trajectory to a local file, for skyc file construction."""
        # we save the timestamp when the trajectory was saved alongside the trajectory data
        data_to_save = (display_time(), self.drone.trajectory['spline_path'], self.drone.trajectory['speed_profile'])
        str_traj_id = '0'+str(self.traj_id) if self.traj_id < 10 else str(self.traj_id)
        with open(os.path.join(self.traj_folder, str_traj_id), 'wb') as file:
            pickle.dump(data_to_save, file)
        self.traj_id += 1

    async def interruptable_sleep_until(self, deadline) -> bool:
        """To be used instead of the regular trio.sleep_until, since this sleep can be interrupted by an event."""
        was_interrupted = False
        # if an interrupt doesn't arrive, we cancel this scope at the deadline, meaning that it functioned as sleep
        with trio.move_on_at(deadline) as cancel_scope:
            await self.interrupt.wait()
            was_interrupted = True
            self.print("Woke up from sleep due to interrupt!")
            cancel_scope.cancel()
        self.interrupt = trio.Event()
        return was_interrupted

    async def send_and_ack(self, data: bytes):
        """Function that sends a message to the server, and waits for a response if necessary."""
        with open(self.log_file_path, 'a') as log:  # note the command to the log file
            log.write(f"{display_time():.3f}: {data}\n")
        if self.drone_ID in SETTINGS.get("real_drones"):
            await send_with_semaphore(data=data, semaphore=self.stream_semaphore, stream=self.socket, wait_ack=True)
        else:
            # this function handles the sim semaphore by default
            await self.sim_send(f"{self.drone_ID}_".encode("utf-8") + data)
        with open(self.log_file_path, 'a') as log:
            log.write(f"{display_time():.3f}: {b'ACK'}\n")

    async def move_to_next_command(self, next_callable: Union[Callable, None], duration: float):
        """Function that prepares the next command in line: it appends the necessary command, and sleeps until the
        current one is over."""
        current_command_over = self.next_command.deadline
        if next_callable is not None:
            next_command = DroneCommand(next_callable, current_command_over + duration)
        else:
            next_command = None
        self.next_command = next_command
        await self.interruptable_sleep_until(current_command_over)

    async def takeoff(self):
        """Handles a takeoff command: send the necessary message and then prepare a start command."""
        height = interpolate.splev(0, self.drone.trajectory['spline_path'])[2]
        self.print(f"Got takeoff command, takeoff height is {height:.3f}.")
        data = f"CMDSTART_takeoff_{height:.4f}_EOF".encode()
        await self.send_and_ack(data)
        await self.move_to_next_command(next_callable=self.start, duration=self.drone.fligth_time)

    async def _calculate(self):
        # self.return_to_home = display_time() > SETTINGS.get("demo_time")
        global traj_counter
        traj_counter += 1
        self.return_to_home = traj_counter > SETTINGS.get("max_traj")

        if not self.return_to_home:
            self.print(f"Got calculate command. Trajectory should start in {SETTINGS.get('REST_TIME')} sec")
        else:
            self.print(f"Got RTH command. Last trajectory should start in {SETTINGS.get('REST_TIME')} sec")
        choose_target(scene, self.drone,
                      self.return_to_home)  # RTH will mean that the target chosen will be the home position
        self.next_command.deadline = current_time() + SETTINGS.get("REST_TIME")  # !
        self.drone.start_time = round(self.next_command.deadline, 1)  # required for trajectory calculation
        calc_start_time = current_time()
        spline_path, speed_profile, duration, length = await async_generate_trajectory(drone=self.drone,
                                                                                       G=graph,
                                                                                       dynamic_obstacles=dynamic_obstacles,
                                                                                       other_drones=self.other_drones,
                                                                                       Ts=scene.Ts,
                                                                                       safety_distance=scene.general_safety_distance)
        calc_end_time = current_time()
        self.drone.trajectory = {'spline_path': spline_path, 'speed_profile': speed_profile}
        self.drone.fligth_time = duration
        await trio.to_thread.run_sync(add_coll_matrix_to_elipsoids, [self.drone], graph, scene.Ts, scene.cmin,
                                      scene.cmax,
                                      scene.general_safety_distance)
        coll_mtrx_time = current_time()
        self.trajectory = splines_to_json(spline_path, speed_profile)
        self.print(f"Calculation took {(calc_end_time-calc_start_time):.3f}s, coll_matrix took {coll_mtrx_time-calc_end_time}")
        calc_times.append(calc_end_time-calc_start_time)
        if current_time() > self.next_command.deadline:
            self.print(f"Calculations took too long!")
            raise TimeoutError

    async def calculate_upload(self):
        """Handles the calculations regarding a new trajectory, and uploads it to the drone.
        Next command will be the start command for this trajectory."""
        # if the demo time is past, it's about time that we pick a home destination and go there in order to land
        calc_wait_start = current_time()
        await self.calc_semaphore.wait_take(PRIORITY_LOW)
        calc_wait_times.append(current_time() - calc_wait_start)
        await self._calculate()
        self.calc_semaphore.let_go()
        data = f"CMDSTART_upload_{self.trajectory}_EOF".encode()
        await self.send_and_ack(data)
        await self.move_to_next_command(next_callable=self.start, duration=self.drone.fligth_time)

    async def _emergency_calculate(self):
        self.print(f"Got emergency calculate command.")
        # grab the coordinates of the nodes in the graph currently
        vertices = np.array([graph['graph'].nodes.data('pos')[node_idx] for node_idx in graph['graph'].nodes])
        # this is the time at which the *NEW* emergency trajectory will start
        start_time = math.ceil(self.next_command.deadline * 10) / 10
        # add the position of the drone at time start_time to the graph so we can calculate a trajectory from it
        extended_graph, new_start_idx = await trio.to_thread.run_sync(expand_graph, graph, vertices, self.drone,
                                                                      start_time, static_obstacles)
        origin_vertex = self.drone.start_vertex  # save the previous start vertex
        self.drone.emergency = True  # signal to gurobi that drone will be moving at the start of the trajectory
        self.drone.start_vertex = new_start_idx
        self.drone.start_time = start_time
        spline_path, speed_profile, duration, length = await async_generate_trajectory(drone=self.drone,
                                                                                       G={'graph': extended_graph,
                                                                                          'point_cloud': graph['point_cloud']},
                                                                                       dynamic_obstacles=dynamic_obstacles,
                                                                                       other_drones=self.other_drones,
                                                                                       Ts=scene.Ts,
                                                                                       safety_distance=scene.general_safety_distance)
        self.drone.trajectory = {'spline_path': spline_path, 'speed_profile': speed_profile}
        self.drone.emergency = False
        self.drone.fligth_time = duration
        self.drone.start_vertex = origin_vertex  # reset the previous start vertex
        await trio.to_thread.run_sync(add_coll_matrix_to_elipsoids, [self.drone], graph, scene.Ts, scene.cmin,
                                      scene.cmax,
                                      scene.general_safety_distance)
        self.trajectory = splines_to_json(spline_path, speed_profile)
        if current_time() > self.next_command.deadline:
            self.print(f"Calculations took too long!")
            raise TimeoutError

    async def emergency_calc_upload(self):
        """Handles the calculations regarding an emergency avoidance trajectory, and uploads it to the drone.
        Next command will be the start command for this trajectory."""
        await self.calc_semaphore.wait_take(PRIORITY_HIGH)
        await self._emergency_calculate()
        self.calc_semaphore.let_go()
        data = f"CMDSTART_upload_{self.trajectory}_EOF".encode()
        await self.send_and_ack(data)
        await self.move_to_next_command(next_callable=self.start, duration=self.drone.fligth_time)

    async def start(self):
        """Handles a start command: send out the necessary message to the server, and prepare the next command.
        If we have already set the return to home variable, it means the trajectory we start is the last one, and we
        need to land afterwards. Else we need to start a calculation."""
        self.print(f"Got start command. Beginning trajectory lasting {self.drone.fligth_time:.3f} sec.")
        traj_type = "absolute" if SETTINGS.get("absolute_traj", True) else "relative"
        data = f"CMDSTART_start_{traj_type}_EOF".encode()
        await self.send_and_ack(data)
        self.save_traj()
        if self.return_to_home:
            await self.move_to_next_command(next_callable=self.land, duration=SETTINGS.get("TAKEOFF_DURATION"))
        else:
            await self.move_to_next_command(next_callable=self.calculate_upload, duration=SETTINGS.get("REST_TIME"))

    async def land(self):
        """Handles a land command, however, the next command is None since the demo is supposed to be over."""
        self.print(f"Got land command.")
        data = f"CMDSTART_land_EOF".encode()
        await self.send_and_ack(data)
        await self.move_to_next_command(next_callable=None, duration=SETTINGS.get("TAKEOFF_DURATION"))

    async def do_commands(self):
        """This is the 'main' function for a handler: it runs in an infinite loop, calling the callable parts of the
        commands, which all append the next command."""
        while self.next_command:
            await self.next_command.command()
        self.print("No more commands.")


def determine_knots(time_vector, N):
    '''returns knot vector for the BSplines according to the incoming timestamps and the desired number of knots'''
    if SETTINGS.get("equidistant_knots", False):  # knots are every N samples
        # Problems start to arise when part_length becomes way smaller than N, so generally keep them longer :)
        part_length = len(time_vector) // N
        result = time_vector[::part_length][:N]
        result.append(time_vector[-1])
    else:  # knots are placed evenly in time
        start = min(time_vector)
        end = max(time_vector)
        result = list(np.linspace(start, end, N))
    return result


def splines_to_json(spline_path: List[Union[np.ndarray, List[np.ndarray], int]],
                    speed_profile: Tuple[np.ndarray, np.ndarray, int]):
    # spline_path is a 3D BSpline with path length as its variable. 0: knots, 1: List of arrays x-y-z
    # speed_profile is a BSpline for distance at each time segment. 0: knots, 1: List of distances
    traj_type = SETTINGS.get("traj_type")
    granularity = None
    degree = 0
    if traj_type == 'COMPRESSED':
        degree = 3
        granularity = 1001
        num_of_segments = 20
    elif traj_type == 'POLY4D':
        degree = 5
        granularity = 1001
        num_of_segments = 20
        raise NotImplementedError
    assert degree > 0 and granularity is not None
    t_abs = list(np.linspace(speed_profile[0][0], speed_profile[0][-1], granularity))
    t = [x - t_abs[0] for x in t_abs]
    distances = interpolate.splev(t_abs, speed_profile)
    x_abs, y_abs, z_abs = interpolate.splev(distances, spline_path)
    if SETTINGS.get("absolute_traj", True):
        x = x_abs
        y = y_abs
        z = z_abs
    else:
        x = [element - x_abs[0] for element in x_abs]
        y = [element - y_abs[0] for element in y_abs]
        z = [element - z_abs[0] for element in z_abs]

    knots = determine_knots(t, num_of_segments)[1:-1]
    # We need to give the splrep inside knots. I think [0] and [-1] should also technically be inside knots, but apparently
    # not. I seem to remember that the first k-1 and last k-1 knots are the outside knots. Anyway, slprep seems to add k
    # knots both at the end and at the beginning, instead of k-1 knots which is what would make sense to me. How it decides
    # what those knots should be is a mystery to me, but upon checking them, they are the exact first and last knots that I
    # would've added, so it works out kind of.
    xSpline = interpolate.splrep(t, x, k=degree, task=-1, t=knots)
    ySpline = interpolate.splrep(t, y, k=degree, task=-1, t=knots)
    zSpline = interpolate.splrep(t, z, k=degree, task=-1, t=knots)

    # BPoly can be constructed from PPoly but not from BSpline. PPoly can be constructed from BSPline. BSpline can
    # be fitted to points. So Points->PPoly->BPoly. The coeffs of the BPoly representation are the control points.
    x_PPoly = interpolate.PPoly.from_spline(xSpline)
    y_PPoly = interpolate.PPoly.from_spline(ySpline)
    z_PPoly = interpolate.PPoly.from_spline(zSpline)
    x_BPoly = interpolate.BPoly.from_power_basis(x_PPoly)
    y_BPoly = interpolate.BPoly.from_power_basis(y_PPoly)
    z_BPoly = interpolate.BPoly.from_power_basis(z_PPoly)
    # These two lines below seem complicated but all they do is pack the data above into a convenient form: a list
    # of lists where each element looks like this: [t, (x,y,z), (x,y,z), (x,y,z)]. Note that this can almost
    # definitely be done in a simpler way :)
    BPoly = list(zip(list(x_BPoly.x)[degree + 1:-degree],
                     list(x_BPoly.c.transpose())[degree:-degree],
                     list(y_BPoly.c.transpose())[degree:-degree],
                     list(z_BPoly.c.transpose())[degree:-degree]))
    BPoly = [[element[0]] + list(zip(*list(element[1:]))) for element in BPoly]
    Data = [[t[0], [x[0], y[0], z[0]], []]]
    for Bezier_Curve in BPoly:
        curve_to_append = [Bezier_Curve[0],
                           Bezier_Curve[-1],
                           Bezier_Curve[2:-1]]
        Data.append(curve_to_append)
    json_dict = {
        "version": 1,
        "points": Data,
        "takeoffTime": Data[0][0],
        "landingTime": Data[-1][0],
        "type": traj_type
    }
    json_object = json.dumps(json_dict, indent=2)
    return json_object


def verify_drones(drone_IDs: List[str]):
    '''Function that asserts we have exactly the drones that we want in frame.'''
    mocap = motioncapture.MotionCaptureOptitrack("192.168.2.141")
    mocap.waitForNextFrame()
    items = mocap.rigidBodies.items()
    # drones = [(determine_id(name), list(obj.position[:-1])) for name, obj in items if 'cf' in name]
    # assert set(drone[0] for drone in drones) == set(drone_IDs)
    drones = [determine_id(name) for name, obj in items if 'cf' in name]
    for drone_ID in drone_IDs:
        assert drone_ID in drones


async def establish_connection_with_handler(drone_id: str):
    PORT = SETTINGS.get("SERVER_PORT") if SETTINGS.get("LIVE_DEMO") else SETTINGS.get("DUMMY_SERVER_PORT")
    drone_stream: trio.SocketStream = await trio.open_tcp_stream("127.0.0.1", PORT)
    request = f"REQ_{drone_id}"
    print(f"Requesting handler for drone {drone_id}")
    await drone_stream.send_all(request.encode('utf-8'))
    acknowledgement: bytes = await drone_stream.receive_some()
    if acknowledgement.decode('utf-8') == f"ACK_{drone_id}":
        print(f"successfully created server-side handler for drone {drone_id}")
        return drone_stream
    else:
        raise NotImplementedError # immediately stop if we couldn't reach one of the drones


def setup_logs():
    current_dir = os.getcwd()
    traj_folder_name = SETTINGS.get("traj_folder_name")
    traj_folder_path = os.path.join(current_dir, traj_folder_name)
    if os.path.exists(traj_folder_path):
        shutil.rmtree(traj_folder_path)
    os.makedirs(traj_folder_path)
    log_folder_name = SETTINGS.get("log_folder_name")
    log_folder_path = os.path.join(current_dir, log_folder_name)
    if os.path.exists(log_folder_path):
        shutil.rmtree(log_folder_path)
    os.makedirs(log_folder_path)


def setup_scene() -> Tuple[Construction, Dict, Static_obstacles]:
    """Function that builds the scene and graph for the trajectory generator."""
    drone_num = len(SETTINGS.get("real_drones")) + len(
        SETTINGS.get("sim_drones"))  # including both real and fake drones
    scene = Construction(real_obstacles=SETTINGS.get("real_obstacles"),
                         live_demo=SETTINGS.get("LIVE_DEMO"),
                         simulated_obstacles=SETTINGS.get("simulated_obstacles"),
                         get_new_measurement=SETTINGS.get("measure"),
                         fix_vertex_layout=SETTINGS.get("fix_vertex_layout"),
                         N=drone_num)

    start_pos: List[np.ndarray] = SETTINGS.get("start_pos")
    number_of_targets, graph, static_obstacles = construction(real_drones=SETTINGS.get("real_drones"),
                                                              sim_drones=SETTINGS.get("sim_drones"),
                                                              scene=scene, start_pos=start_pos)
    # now we must initialize the free targets
    target_zero = len(graph['graph'].nodes()) - number_of_targets
    scene.free_targets = np.arange(target_zero, len(graph['graph'].nodes()), 1)
    # the home positions are appended to the end of the nodes
    scene.home_positions = scene.free_targets[-drone_num:]
    # let's not fly to the home positions, only the 'normal' nodes
    scene.free_targets = scene.free_targets[:-drone_num]
    return scene, graph, static_obstacles


def unpack_car_trajectories():
    """Function that extracts the car's trajectories from its pickle, and unpacks them to separate trajectories.
    Not used in newer versions. TODO: delete once we're sure it won't be needed later."""
    current_dir = os.getcwd()
    car_folder_name = SETTINGS.get("car_folder_name")
    car_folder_path = os.path.join(current_dir, car_folder_name)
    if os.path.exists(car_folder_path):
        shutil.rmtree(car_folder_path)
    os.makedirs(car_folder_path)
    car_trajectories = []
    with open(os.path.join(current_dir, "city_demo_all_paths_tuple.pkl"), 'rb') as f:
        while True:
            try:
                car_trajectories.append(pickle.load(f))
            except EOFError:
                break
    for idx, car_trajectory in enumerate(car_trajectories):
        with open(os.path.join(car_folder_path, str(idx)), 'wb') as file:
            # add a 3rd dimension full of zeros for compatibility
            z_coeffs = np.zeros_like(car_trajectory[0][1][0])
            car_trajectory[0][1].append(z_coeffs)
            pickle.dump(car_trajectory, file)


def setup_demo() -> Tuple[Construction, Dict, Static_obstacles]:
    """Function that "lays the bed" for the setup_scene function."""
    print(f"Random seed: {SETTINGS.get('random_seed')}")
    np.random.seed(SETTINGS.get("random_seed"))
    warning(f"DEMO IS {'LIVE' if SETTINGS.get('LIVE_DEMO') else 'NOT LIVE'}")
    if SETTINGS.get("LIVE_DEMO"):
        verify_drones(SETTINGS.get("real_drones"))  # throw an error if the drones in frame are not exactly the drones we set above
    setup_logs()
    scene, graph, static_obstacles = setup_scene()
    return scene, graph, static_obstacles


def determine_id(string):
    '''takes a name as found in optitrack and returns the ID found in it, for example cf6 -> 06'''
    number_match = re.search(r'\d+', string)
    if number_match:
        number = number_match.group(0)
        if len(number) == 1:
            number = '0' + number
        return number
    return None


def determine_home_position(drone_ID: str, graph: Dict):
    """Function that takes a drone and the graph, and returns the index corresponding to the drone's start position in
    the graph."""
    if drone_ID in SETTINGS.get("sim_drones"):
        # in the case of simulated drones, we assign positions from the predetermined ones
        idx = -(SETTINGS.get("sim_drones").index(drone_ID)+1)
        drone_num = len(SETTINGS.get("sim_drones")) + len(SETTINGS.get("real_drones"))
        start_pos = SETTINGS.get("start_pos")[:drone_num]
        drone_pos = start_pos[idx][:-1]
    else:
        if SETTINGS.get("LIVE_DEMO"):
            mocap = motioncapture.MotionCaptureOptitrack("192.168.2.141")
            mocap.waitForNextFrame()
            items = mocap.rigidBodies.items()
            # let's put the rigid bodies containing cf index into a dictionary with their IDs.
            drones = {determine_id(name): list(obj.position[:-1]) for name, obj in items if 'cf' in name}
            # and then select the drone which we're inspecting from the dictionary
            drone_pos = drones[drone_ID]
        else:
            # in the case of the drones that we want to consider as 'real', but the demo is not live,
            # we also assign a position from the predetermined ones
            drone_pos = SETTINGS.get("start_pos")[SETTINGS.get("real_drones").index(drone_ID)][:-1]

    # this line below looks scary but what it does is it selects the x-y coordinates of the starting nodes, and packs
    # them into a tuple with their associated index in graph['graph'].nodes.data('pos')
    starting_positions = [(index, graph['graph'].nodes.data('pos')[index][:-1]) for index in scene.home_positions]

    def calculate_distance(xy1: List[float], xy2: List[float]) -> float:
        '''calculates eucledian distance between xy points '''
        return np.linalg.norm(np.array(xy1) - np.array(xy2))

    min_distance = float('inf')
    # what we want is not the starting position, but the index of the starting position's node in this ugly looking
    # variable: graph['graph'].nodes.data('pos'). So we look at which xy pair is closest to the drone's position and we
    # return its index
    for pos_idx, xy in starting_positions:
        distance = calculate_distance(drone_pos, xy)
        if distance < min_distance:
            min_distance = distance
            home_pos_index = pos_idx
    assert min_distance < 0.2  # if the closest starting position is further than 0.2m, we messed up probably!
    return home_pos_index


def initialize_drones(scene: Construction, graph: Dict) -> List[Drone]:
    """Function that generates the drones, already pre-installed with their first trajectory, collision matrix and
    start time."""
    drones: List[Drone] = []
    drone_IDs = SETTINGS.get("real_drones") + SETTINGS.get("sim_drones")
    for i, drone_ID in enumerate(drone_IDs):
        # let's calculate the first trajectories, since they should be handled differently from the rest
        drone = Drone()
        drone.rest_time = SETTINGS.get("REST_TIME")
        drone.cf_id = drone_ID
        # It could be possible, for example, that the home position associated witht he index home_positions[i] is
        # actually not the position where the drone in question is. The function below calculates which home position
        # is the closest.
        drone.start_vertex = determine_home_position(drone_ID, graph)
        drone.target_vertex = np.random.choice(scene.free_targets)
        # bar the other drones from selecting the node we're going to. i.e. delete the target vertex from the free vertices
        scene.free_targets = np.delete(scene.free_targets, scene.free_targets == drone.target_vertex)
        spline_path, speed_profile, duration, length = generate_trajectory(drone=drone, G=graph,
                                                                           dynamic_obstacles=dynamic_obstacles,
                                                                           other_drones=drones, Ts=scene.Ts,
                                                                           safety_distance=scene.general_safety_distance)
        drone.trajectory = {'spline_path': spline_path, 'speed_profile': list(speed_profile)}
        drone.fligth_time = duration
        add_coll_matrix_to_elipsoids([drone], graph, scene.Ts, scene.cmin, scene.cmax,
                                     scene.general_safety_distance)
        drones.append(
            drone)  # for the next drone, this current drone will count as part of 'other_drones': avoid collision
    return drones


async def get_handlers(handlers: Dict[str, Union[DroneHandler, None]], drones: List[Drone], calc_semaphore: Semaphore) -> Dict[str, DroneHandler]:
    colors = SETTINGS.get("text_colors")
    """Function that creates the drone handlers, and puts them in a dictionary with the IDs as keys."""
    for idx, drone in enumerate(drones):
        other_drones = [element for element in drones if element != drone]
        # run the initialization, where we establish a TCP socket for each drone if they aren't a simulated drone. If
        # they are simulated drones, then leave the socket as None
        if drone.cf_id in SETTINGS.get("sim_drones"):
            socket = None
        else:
            socket = await establish_connection_with_handler(drone.cf_id)
        # designate a TCP socket and an associated handler for each drone
        color = colors[idx % len(colors)]
        handler = DroneHandler(socket=socket, drone=drone, other_drones=other_drones, color=color,
                               calc_semaphore=calc_semaphore)
        handlers[drone.cf_id] = handler
        await sleep(0.01)
    return handlers


async def car_handler(stream: trio.SocketStream,
                      handlers: Dict[str, DroneHandler],
                      dynamic_obstacles: List[Dynamic_obstacle],
                      car_ready: Event,
                      takeoff: Event):
    """Function that handles the communication with the car, calculates whether a collision is about to occur, then
    instructs the handlers to calculate an emergency maneuver if necessary."""
    car_log = os.path.join(SETTINGS.get("log_folder_name"), "car")
    if os.path.exists(car_log):
        os.remove(car_log)
    open(car_log, 'a').close()
    car_safety_distance = SETTINGS.get("car_safety_distance")
    init_message: bytes = b""
    while init_message != b"ready":
        init_message = await stream.receive_some()
    print(f"Got init message!")
    car_ready.set()
    await takeoff.wait()
    await stream.send_all(b'6') # MATCH THIS TO THE SKYBRUSH SERVER!
    while True:
        try:
            data: bytes = await stream.receive_some(max_bytes=65536)
            if not data or data.startswith(b'-1'):
                break
            tck, speed = pickle.loads(data)
            with open(car_log, 'a') as log:
                car_start_display_time = display_time() + SETTINGS.get("EMERGENCY_TIME")
                log.write(f"{car_start_display_time:.3f}: {data}\n")
            car_start_time = current_time() + SETTINGS.get("EMERGENCY_TIME")
            car_start_time_display = display_time() + + SETTINGS.get("EMERGENCY_TIME")
            car = Dynamic_obstacle(path_tck=tck, path_length=tck[0][-1], speed=abs(speed), radius=SETTINGS.get("car_radius"),
                                   start_time=car_start_time)
            add_coll_matrix_to_poles(obstacles=[car], graph_dict=graph, Ts=scene.Ts, cmin=scene.cmin,
                                     cmax=scene.cmax, safety_distance=car_safety_distance)
            dynamic_obstacles.clear()
            dynamic_obstacles.append(car)
            drones = [handler.drone for handler in handlers.values()]
            collision_ids = check_collisions_with_single_obstacle(new_obstacle=car, drones=drones,
                                                                  Ts=scene.Ts, safety_distance=car_safety_distance)
            if len(collision_ids) > 0:
                warning(f"[{display_time():.3f}] For car trajectory starting at {car_start_time_display:.3f}, "
                        f"collisions were detected with the following drone(s): {collision_ids}. Their emergency "
                        f"trajectories should start in {SETTINGS.get('EMERGENCY_TIME')}")
            else:
                warning(f"[{display_time():.3f}] For car trajectory starting at {car_start_time_display:.3f}, "
                        f"no collisions were detected.")
            for id in collision_ids:
                handler = handlers[id]
                # instead of whatever the next command was going to be, do an emergency calculation
                handler.next_command = DroneCommand(handler.emergency_calc_upload, car_start_time)
                # wake up from the sleep in-between the commands
                handler.interrupt.set()
        except Exception as exc:
            print(f"[{display_time():.3f}] TCP connection to car handler crashed with exception: {exc!r}. TRACEBACK:\n")
            print(traceback.format_exc())
            break
    print(f"[{display_time():.3f}] Closing TCP car connection")
    await stream.send_all(b'-1EOF')  # signal for the server-side handler function to stop listening


async def demo(drones):
    print(f"Welcome to Palkovits Máté's drone demo++!")
    calc_semaphore = Semaphore()
    takeoff = Event()
    async with trio.open_nursery() as nursery:
        handlers: Dict[str, Union[DroneHandler, None]] = {}
        if SETTINGS.get("CAR"):
            car_ready = Event()
            PORT = SETTINGS.get("SERVER_PORT") if SETTINGS.get("LIVE_DEMO") else SETTINGS.get("DUMMY_SERVER_PORT")
            CAR_PORT = PORT+1
            car_stream: trio.SocketStream = await trio.open_tcp_stream("127.0.0.1", CAR_PORT)
            nursery.start_soon(partial(car_handler, stream=car_stream, handlers=handlers,
                                       dynamic_obstacles=dynamic_obstacles, car_ready=car_ready, takeoff=takeoff))
            print(f"WAITING FOR CAR!!!!")
            await car_ready.wait()
        handlers = await get_handlers(handlers, drones, calc_semaphore)
        # make the simulation connection
        if len(SETTINGS.get("sim_drones")) > 0:
            PORT = SETTINGS.get("SERVER_PORT") if SETTINGS.get("LIVE_DEMO") else SETTINGS.get("DUMMY_SERVER_PORT")
            SIM_PORT = PORT + 2
            sim_stream: trio.SocketStream = await trio.open_tcp_stream("127.0.0.1", SIM_PORT)
            sim_semaphore = Semaphore()
            for handler in handlers.values():
                handler.sim_send = partial(send_with_semaphore, semaphore=sim_semaphore, stream=sim_stream,
                                           wait_ack=SETTINGS.get("wait_sim_ack"))
        takeoff.set()
        display_time.start_time = current_time()  # reset display time to 0
        demo_start_time = current_time() + SETTINGS.get("TAKEOFF_DURATION")  # start time of first trajectory
        for drone in drones:  # shift the drone time (start time was left as 0 during initial calculations)
            drone.start_time = demo_start_time
            drone.trajectory['speed_profile'][0] = demo_start_time + drone.trajectory['speed_profile'][0]
            drone.collision_matrix_compressed[:, 0] += demo_start_time

        for ID, handler in handlers.items():
            handler.next_command = DroneCommand(handler.takeoff, demo_start_time)  # once that's done, take off
            nursery.start_soon(handler.do_commands)  # and start the infinite loop that executes commands
        for handler in handlers.values():
            data = f"CMDSTART_upload_{handler.trajectory}_EOF".encode()
            await handler.send_and_ack(data)  # upload the first trajectory before starting the demo
            await handler.sim_send(f"{handler.drone_ID}_".encode("utf-8") + data)

# Ideally, nothing at all has to be modified anywhere else to control the demo completely. Only here.
N = 30
calc_wait_times = []
calc_times = []
traj_counter = 0
SETTINGS = {
    "max_traj": 200,
    "real_drones": [str(i) for i in range(10, N+10)],
    "sim_drones": [],
    "wait_sim_ack": True,
    # "sim_drones": ["07", "08"],
    "random_seed": 13,
    "LIVE_DEMO": False,
    "demo_time": 40,
    "REST_TIME": 3,
    "TAKEOFF_DURATION": 4,
    "absolute_traj": True,
    "real_obstacles": False,
    "measure": False,
    "plot": False,
    "simulated_obstacles": False,
    "SERVER_PORT": 6000,
    "DUMMY_SERVER_PORT": 7000,
    "CAR": False,
    "start_pos": [np.array([x, 0, 0]) for x in list(np.arange(0.5, (N+1)*0.5, 0.5))],
    # "start_pos": [np.array([1.25, 0, 0]),  # these are the positions where the drones start if we don't use real drones
    #               np.array([1.25, -0.5, 0]),
    #               np.array([1, -1, 0]),
    #               np.array([1.25, 0.5, 0]),
    #               np.array([0.6, -1.3, 0]),
    #               np.array([0, -1.3, 0]),
    #               np.array([-0.5, -1.3, 0]),
    #               np.array([0, 0, 0])],
    "traj_folder_name": "trajectories",
    "log_folder_name": "logs",
    "car_folder_name": "car",
    "traj_type": "COMPRESSED",
    "car_radius": 0.15,
    "car_safety_distance": 0.15,
    "equidistant_knots": False,
    "EMERGENCY_TIME": 1,
    "fix_vertex_layout": 7, # if initialization doesn't succeeed, this is likely the issue
    "text_colors": ["\033[92m",
                    "\033[93m",
                    "\033[94m",
                    "\033[96m",
                    "\033[95m"],
}
print("Building graph...")
scene, graph, static_obstacles = setup_demo()
print("Built graph, calculating first trajectories...")
dynamic_obstacles = []
drones = initialize_drones(scene, graph)
if SETTINGS.get("plot", False):
    plt.show()
try:
    trio.run(demo, drones)
except Exception as exc:
    print(f"Exception: {exc!r}. TRACEBACK:\n")
    print(traceback.format_exc())
    # or
    # print(sys.exc_info()[2])
print(f"Demo is over, generating skyc file!")

try:
    if len(SETTINGS.get("real_drones")) > 0:
        generate_skyc_file(filename="Demo", drones=SETTINGS.get("real_drones"))
    if len(SETTINGS.get("sim_drones")) > 0:
        generate_skyc_file(filename="Sim", drones=SETTINGS.get("sim_drones"))
        send_skyc("Sim.skyc")
except Exception as exc:
    print(f"Exception: {exc!r}. TRACEBACK:\n")
    print(traceback.format_exc())
print(f"{len(calc_wait_times)} data points. Average wait time: {np.mean(calc_wait_times)} \nMaximum wait time: {np.max(calc_wait_times)} \nMinimum wait time: {np.min(calc_wait_times)}")
input("Press Enter to exit...")
