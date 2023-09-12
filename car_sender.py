import socket
import time
import traceback

ip = "127.0.0.1"
demo_port = 6001
dummy_server_port = 6003
skybrush_server_port = 6002


def send_traj(traj_num: int):
    client_socket = socket.socket()
    print(f"Connecting to demo...")
    client_socket.connect((ip, demo_port))
    print(f"Connected to demo!")
    client_socket.sendall(f"{traj_num}EOF".encode("utf-8"))
    print(f"Sent traj!")
    client_socket.close()

client_socket = socket.socket()
try:
    client_socket.connect((ip, skybrush_server_port))
    print(f"Connected to skybrush server!")
except:
    client_socket.connect((ip, dummy_server_port))
    print(f"Connected to dummy server!")
wait_time = int(client_socket.recv(1024))
print(f"Got {wait_time}s wait time.")
try:
    time.sleep(wait_time)
    send_traj(0)
    time.sleep(7)
    send_traj(1)
    time.sleep(7)
    send_traj(2)
    time.sleep(7)
    send_traj(3)
    time.sleep(7)
    send_traj(4)
except Exception as exc:
    print(f"Exception: {exc!r}. TRACEBACK:\n")
    print(traceback.format_exc())
input("Press Enter to exit...")