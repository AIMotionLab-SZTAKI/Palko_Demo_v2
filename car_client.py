import socket
ip = "127.0.0.1"
port = 6002
client_socket = socket.socket()
print(f"Connecting to demo...")
client_socket.connect((ip, port))
print(f"Connected to demo!")
start = client_socket.recv(1024)
print(f"Received {start}")
client_socket.close()
