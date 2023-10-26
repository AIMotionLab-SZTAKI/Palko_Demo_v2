import socket
import time
import traceback
import os
import pickle
from scipy import interpolate
import matplotlib.pyplot as plt

import numpy as np

ip = "127.0.0.1"
dummy_port = 7001
skybrush_port = 6001

wait_time = 1
# speed = 0.5
# x1 = -1
# x2 = 4
# y1 = -1
# y2 = 4
# L = np.sqrt((x2-x1)**2 + (y2-y1)**2)
# s = np.linspace(0, L)
# x = np.linspace(x1, x2)
# y = np.linspace(y1, y2)
# z = np.linspace(0, 0)
# tck, *_ = interpolate.splprep(x=[x, y, z], u=s)

car_data = []
with open(os.path.join(os.getcwd(), "city_demo_all_paths_tuple.pkl"), 'rb') as f:
    while True:
        try:
            tck, speed = pickle.load(f)
            T = tck[0][-1] / abs(speed)
            z_coeffs = np.zeros_like(tck[1][0])
            tck[1].append(z_coeffs)
            car_data.append((tck, abs(speed), T))
        except EOFError:
            break

start_delay = 0
try:
    Socket = socket.socket()
    Socket.connect((ip, skybrush_port))
    print("Connected to skybrush server port.")
except ConnectionRefusedError:
    Socket = socket.socket()
    Socket.connect((ip, dummy_port))
    print("Connected to dummy server port.")
init_message = b'ready'
Socket.sendall(init_message)
start_delay = float(Socket.recv(1024).strip())
print(f"Waiting {start_delay} to launch!")
time.sleep(start_delay)
print(f"Starting trajectories.")
for tck, speed, duration in car_data:
    serialized_data = pickle.dumps((tck, speed))
    assert len(serialized_data) < 65536
    time.sleep(wait_time)
    Socket.sendall(serialized_data)
    print("Sent a trajectory!")
    time.sleep(duration)

Socket.sendall(b'-1')
input("Press Enter to exit...")
