import subprocess
import time
import platform

server = False
demo = True
car = True
skybrush_client = False
sim = True

if server:
    print(f"Starting server...")
    if platform.system() == "Windows":
        server = subprocess.Popen(["start", "cmd", "/K", "python", "Dummy_Server.py"], shell=True)
    elif platform.system() == "Linux":
        server = subprocess.Popen(["x-terminal-emulator", "-e", "python3", "Dummy_Server.py"])

if sim:
    time.sleep(1)
    print(f"Starting sim...")
    if platform.system() == "Windows":
        client = subprocess.Popen(["start", "cmd", "/K", "python3", "simulator.py"], shell=True)
    elif platform.system() == "Linux":
        client = subprocess.Popen(["x-terminal-emulator", "-e", "python3", "simulator.py"])

if demo:
    time.sleep(2)
    print(f"Starting demo...")
    if platform.system() == "Windows":
        client = subprocess.Popen(["start", "cmd", "/K", "python3",  "Demo.py"], shell=True)
    elif platform.system() == "Linux":
        client = subprocess.Popen(["x-terminal-emulator", "-e", "python3", "Demo.py"])

if car:
    time.sleep(2)
    print(f"Starting car sender...")
    if platform.system() == "Windows":
        car = subprocess.Popen(["start", "cmd", "/K", "python3",  "car_sender.py"], shell=True)
    elif platform.system() == "Linux":
        car = subprocess.Popen(["x-terminal-emulator", "-e", "python3", "car_sender.py"])

if skybrush_client:
    time.sleep(2)
    print(f"Starting client..")
    if platform.system() == "Windows":
        client = subprocess.Popen(["start", "cmd", "/K", "python3",  "Client/Client.py"], shell=True)
    elif platform.system() == "Linux":
        client = subprocess.Popen(["x-terminal-emulator", "-e", "python3", "Client/Client.py"])

# server.wait()
# client.wait()
# car.wait()
