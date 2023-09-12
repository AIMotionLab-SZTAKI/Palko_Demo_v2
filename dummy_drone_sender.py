import trio
from path_planning_and_obstacle_avoidance.Util_files.Util_constuction import generate_dynamic_obstacles
from path_planning_and_obstacle_avoidance.Classes import Dynamic_obstacle
from path_planning_and_obstacle_avoidance.Util_files.Util_general import fit_spline, parametrize_by_path_length
import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt
import pickle
from trio import sleep_until, sleep
import json

async def establish_connection_with_handler(drone_id: str, port: int):
    drone_stream: trio.SocketStream = await trio.open_tcp_stream("127.0.0.1", port)
    await sleep(0.01)
    request = f"REQ_{drone_id}"
    print(f"Requesting handler for drone {drone_id}")
    await drone_stream.send_all(request.encode('utf-8'))
    acknowledgement: bytes = await drone_stream.receive_some()
    if acknowledgement.decode('utf-8') == f"ACK_{drone_id}":
        print(f"successfully created server-side handler for drone {drone_id}")
        return drone_stream
    else:
        return None

def spline_to_json(spline, speed):
    s = spline[0][2:-2]
    t = s / speed
    x, y, z = interpolate.splev(s, spline)
    degree = 3
    xSpline = interpolate.splrep(t, x, k=degree)
    ySpline = interpolate.splrep(t, y, k=degree)
    zSpline = interpolate.splrep(t, z, k=degree)
    x_PPoly = interpolate.PPoly.from_spline(xSpline)
    y_PPoly = interpolate.PPoly.from_spline(ySpline)
    z_PPoly = interpolate.PPoly.from_spline(zSpline)
    x_BPoly = interpolate.BPoly.from_power_basis(x_PPoly)
    y_BPoly = interpolate.BPoly.from_power_basis(y_PPoly)
    z_BPoly = interpolate.BPoly.from_power_basis(z_PPoly)
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
    # print(*Data, sep='\n')
    type = "COMPRESSED"
    json_dict = {
        "version": 1,
        "points": Data,
        "takeoffTime": Data[0][0],
        "landingTime": Data[-1][0],
        "type": type
    }
    json_object = json.dumps(json_dict, indent=2)
    return json_object, Data[-1][0]

async def send_and_ack(socket: trio.SocketStream, data):
    # print(f"sending:\n {data}")
    await socket.send_all(data)
    ack = b""
    while ack != b"ACK":
        ack = await socket.receive_some()
        await sleep(0.01)

async def drone_commander(start_time, takeoff_height, traj, duration, dummy_ID):
    server_port = 6000
    dummy_drone_socket: trio.SocketStream = await establish_connection_with_handler(dummy_ID, server_port)
    takeoff = f"CMDSTART_takeoff_{takeoff_height:.4f}_EOF".encode()
    upload = f"CMDSTART_upload_{traj}_EOF".encode()
    start = f"CMDSTART_start_absolute_EOF".encode()
    land = f"CMDSTART_land_EOF".encode()
    await send_and_ack(dummy_drone_socket, takeoff)
    await send_and_ack(dummy_drone_socket, upload)
    await sleep_until(start_time)
    await send_and_ack(dummy_drone_socket, start)
    await sleep(duration)
    await send_and_ack(dummy_drone_socket, land)

async def traj_sender(dummy_drone: Dynamic_obstacle, start_time):
    car_port = 6001
    try:
        stream: trio.SocketStream = await trio.open_tcp_stream("127.0.0.1", car_port)
    except Exception as exc:
        print(f"Exception: {exc!r}")
        return
    serialized_dummy_drone = pickle.dumps(dummy_drone) + b"EOF"
    async with stream:
        await trio.sleep_until(start_time)
        await stream.send_all(serialized_dummy_drone)
        print(f"SENT DUMMY DRONE TRAJ")

async def parent():
    start_time = trio.current_time() + 1
    dummy_ID = "04"
    p0 = [1.4, 0, 0.5]
    p1 = [-0.75, 0, 0.5]
    # p0 = [3, 0, 0.5]
    # p1 = [-3, 0, 0.5]
    points = np.array([p0, p1, p0, p1, p0, p1, p0])
    speed = 0.5
    spline = fit_spline(points)
    spline, length = parametrize_by_path_length(spline)
    traj, duration = spline_to_json(spline=spline, speed=speed)
    dummy_drone = Dynamic_obstacle(path_tck=spline, path_length=length, speed=speed, radius=0.2, start_time=0)
    takeoff_height = points[0][2]
    async with trio.open_nursery() as nursery:
        nursery.start_soon(drone_commander, start_time, takeoff_height, traj, duration, dummy_ID)
        nursery.start_soon(traj_sender, dummy_drone, start_time)


trio.run(parent)
input("TCP Sender is done, press Enter to exit...")
# p0 = [1.4, 0, 0.5]
# p1 = [-0.75, 0, 0.5]
# points = np.array([p0, p1, p0, p1, p0])
# speed = 0.3
# spline = fit_spline(points)
# spline, length = parametrize_by_path_length(spline)
# traj, duration = spline_to_json(spline=spline, speed=speed)
# dummy_drone = Dynamic_obstacle(path_tck=spline, path_length=length, speed=speed, radius=0.2, start_time=10201)
# print(dummy_drone.move(np.array([10202,10202])))