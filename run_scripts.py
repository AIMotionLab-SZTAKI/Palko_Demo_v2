import subprocess
import time
import platform

print(f"Starting server...")
if platform.system() == "Windows":
    server = subprocess.Popen(["start", "cmd", "/K", "python",  "Dummy_Server.py"], shell=True)
elif platform.system() == "Linux":
    server = subprocess.Popen(["x-terminal-emulator", "-e", "python3", "Dummy_Server.py"])

time.sleep(0.5)

print(f"Starting dummy drone...")
if platform.system() == "Windows":
    car = subprocess.Popen(["start", "cmd", "/K", "python3",  "car_sender.py"], shell=True)
elif platform.system() == "Linux":
    car = subprocess.Popen(["x-terminal-emulator", "-e", "python3", "car_sender.py"])

time.sleep(2)
print(f"Starting demo...")
if platform.system() == "Windows":
    client = subprocess.Popen(["start", "cmd", "/K", "python3",  "Demo.py"], shell=True)
elif platform.system() == "Linux":
    client = subprocess.Popen(["x-terminal-emulator", "-e", "python3", "Demo.py"])


# time.sleep(12)
# print(f"Starting dummy drone...")
# if platform.system() == "Windows":
#     dummy_drone = subprocess.Popen(["start", "cmd", "/K", "python3",  "dummy_drone_sender.py"], shell=True)
# elif platform.system() == "Linux":
#     dummy_drone = subprocess.Popen(["x-terminal-emulator", "-e", "python3", "dummy_drone_sender.py"])



# print(f"Starting client..")
# if platform.system() == "Windows":
#     client = subprocess.Popen(["start", "cmd", "/K", "python3",  "Client/Client.py"], shell=True)
# elif platform.system() == "Linux":
#     client = subprocess.Popen(["x-terminal-emulator", "-e", "python3", "Client/Client.py"])


server.wait()
client.wait()
car.wait()
