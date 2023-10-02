import socket
import time

ip = "127.0.0.1"
port = 7002

Socket = socket.socket()
Socket.connect((ip, port))
Socket.settimeout(None)
start = time.time()
while True:
    data = b''
    while not data.endswith(b'EOF'):
        data += Socket.recv(4096)
    data = data.decode("utf-8")
    if len(data) > 50:
        data = data[:45] + "..." + data[-5:]
    print(f"{(time.time()-start):.3f}: Received data: {data}")
