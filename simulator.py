import socket
import time
import traceback

ip = "127.0.0.1"
port = 7002

try:
    Socket = socket.socket()
    Socket.connect((ip, port))
    Socket.settimeout(None)
    start = time.time()
    while True:
        data = b''
        while not (data.endswith(b'EOF') or data.endswith(b'SKYC')):
            data += Socket.recv(4096)
        Socket.sendall(b'ACK')
        if data.endswith(b'SKYC'):
            data = data[:-4]
            print(f"{(time.time()-start):.3f}: Received skyc file.")
            with open('Received.skyc', 'wb') as file:
                file.write(data)
            break
        else:
            if len(data) > 50:
                data = data[:45] + b"..." + data[-5:]
            print(f"{(time.time()-start):.3f}: Received data: {data}")
except Exception as exc:
    print(f"Exception: {exc!r}. TRACEBACK:\n")
    print(traceback.format_exc())
input("Press Enter to exit...")
