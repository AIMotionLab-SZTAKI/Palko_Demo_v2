import socket
import time
import traceback
import os
import pickle

ip = "127.0.0.1"
dummy_port = 7001
skybrush_port = 6001

car_timings = []
with open(os.path.join(os.getcwd(), "city_demo_all_paths_tuple.pkl"), 'rb') as f:
    while True:
        try:
            car_data = pickle.load(f)
            car_timings.append(car_data[0][0][-1]/abs(car_data[1]))
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
for i, delay in enumerate(car_timings):
    Socket.sendall(f"{i}EOF".encode("utf-8"))
    print(f"Sent traj {i}.")
    time.sleep(delay + 1)
Socket.sendall(b'-1EOF')
input("Press Enter to exit...")
