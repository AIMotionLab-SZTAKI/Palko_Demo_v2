import trio
from typing import Tuple, Callable, Any, List, Optional, Dict, Union
from trio import sleep, sleep_until
import json
import time
from functools import partial


def warning(text: str):
    color = "\033[33m"
    reset = "\033[0m"
    print(f"{color}[{time.time()-start_time:.3f} WARNING] {text}{reset}")

def log(text: str):
    color = "\033[36m"
    reset = "\033[0m"
    print(f"{color}[{time.time() - start_time:.3f} LOG] {text}{reset}")

class DroneHandler:
    def __init__(self, uav_id: str, stream: trio.SocketStream):
        self.uav_id = uav_id
        self.stream = stream
        self.transmission_active = False
        self.stream_data = b''
        self.traj = b''

    def parse(self, raw_data: bytes, ) -> Tuple[Union[bytes, None], Union[bytes, None]]:
        data = raw_data.strip()
        if not data:
            return None, None
        data = data.split(b'_')
        if data[0] != b'CMDSTART':
            return b'NO_CMDSTART', None
        command = data[1]
        argument = data[2] if b'EOF' not in data[2] else None
        return command, argument

    @staticmethod
    def get_traj_type(self, arg: bytes) -> Tuple[bool, Union[bool, None]]:
        # trajectories can either be relative or absolute. This is determined by a string/bytes, but only these two
        # values are allowed. The return tuple tells whether the argument is valid (first part) and if it's
        # relative (second part). If it wasn't valid, we just get None for the second part.
        traj_type_lower = arg.lower()  # Bit of an allowance to not be case sensitive
        if traj_type_lower == b'relative' or traj_type_lower == b'rel':
            return True, True
        elif traj_type_lower == b'absolute' or traj_type_lower == b'abs':
            return True, False
        else:
            return False, None

    async def handle_transmission(self):
        log(f"drone{self.uav_id}: Transmission of trajectory started.")
        start_index = self.stream_data.find(b'{')
        # If the command was 'upload', then a json file must follow. If it doesn't (we can't find the beginning b'{'),
        # then the command or the file was corrupted.
        if start_index == -1:
            warning("Corrupted trajectory file.")
        else:
            self.traj = self.stream_data[start_index:]
            self.transmission_active = True  # signal we're in the middle of transmission
            while not self.traj.endswith(b'_EOF'):  # receive data until we see that the file has ended
                self.traj += await self.stream.receive_some()  # append the new data to the already received data
            self.traj = self.traj[:-len(b'_EOF')]  # once finished, remove the EOF indicator
            self.transmission_active = False  # then signal that the transmission has ended
            log(f"drone{self.uav_id}: Transmission of trajectory finished.")

    async def command(self, cmd: bytes, arg: bytes):
        log(f"drone{self.uav_id}: {cmd.decode('utf-8')} command received.")
        await self.tcp_command_dict[cmd](self, arg)

    async def takeoff(self, arg: bytes):
        try:
            arg = float(arg)
            log(f"drone{self.uav_id}: Takeoff command dispatched to drone.")
            await sleep(0.01)
            await self.stream.send_all(b'ACK')  # reply with an acknowledgement
        except ValueError:
            warning("Takeoff argument is not a float.")
        except Exception as exc:
            warning(f"drone{self.uav_id}: Couldn't take off because of this exception: {exc!r}. ")

    async def land(self, arg: bytes):
        log(f"{self.uav_id}: Land command dispatched..")
        await self.stream.send_all(b'ACK')  # reply with an acknowledgement

    async def upload(self, arg: bytes):
        await self.handle_transmission()
        trajectory_data = json.loads(self.traj.decode('utf-8'))
        f"Defined trajectory of length {trajectory_data.get('landingTime')} sec for drone {self.uav_id}"
        await sleep(0.5)
        await self.stream.send_all(b'ACK')  # reply with an acknowledgement

    async def start(self, arg: bytes):
        is_valid, is_relative = self.get_traj_type(self, arg=arg)
        if is_valid:
            log(f"drone{self.uav_id}: Started {'relative' if is_relative else 'absolute'} trajectory.")
            await self.stream.send_all(b'ACK')  # reply with an acknowledgement

    async def hover(self, arg: bytes):
        log(f"drone{self.uav_id}: Hover command dispatched.")
        await self.stream.send_all(b'ACK')  # reply with an acknowledgement

    tcp_command_dict: Dict[bytes, Callable] = {
        b"takeoff": takeoff,
        b"land": land,
        b"upload": upload,
        b"hover": hover,
        b"start": start
    }

    async def listen(self):
        while True:
            if not self.transmission_active:
                try:
                    self.stream_data: bytes = await self.stream.receive_some()
                    if not self.stream_data:
                        break
                    cmd, arg = self.parse(self.stream_data)
                    if cmd == b'NO_CMDSTART':
                        log(f"{self.uav_id}: Command is missing standard CMDSTART.")
                        break
                    elif cmd is None:
                        warning(f"{self.uav_id}: None-type command.")
                        break
                    else:
                        await self.command(cmd, arg)
                except Exception as exc:
                    warning(f"drone{self.uav_id}: TCP handler crashed: {exc!r}")
                    break

async def listen_and_broadcast(stream: trio.SocketStream, *, streams: List[trio.SocketStream]):
    streams.append(stream)
    print(f"Number of connections on the broadcast port changed to {len(streams)}")
    data = b''
    while not data.startswith(b'-1'):
        data = await stream.receive_some()
        if len(data) > 0:
            for target_stream in [other_stream for other_stream in streams if other_stream != stream]:
                await target_stream.send_all(data)
    streams.remove(stream)
    print(f"Number of connections on the broadcast port changed to {len(streams)}")


async def establish_drone_handler(stream: trio.SocketStream, *, handlers: List[DroneHandler]):
    taken_ids = [handler.uav_id for handler in handlers]
    available_ids = [drone_id for drone_id in uav_ids if drone_id not in taken_ids]
    if len(available_ids) != 0:
        log(f"TCP connection made. Valid drone IDs: {uav_ids}. "
              f"Of these the following are not yet taken: {available_ids}")
        request = await stream.receive_some()
        request = request.decode('utf-8')
        if 'REQ_' in request:
            requested_id = request.split('REQ_')[1]
            if requested_id not in available_ids:
                warning(f"ID {requested_id} taken already.")
                await stream.send_all(b'ACK_00')
                return
            handler = DroneHandler(requested_id, stream)
            handlers.append(handler)
            log(f"Made handler for drone {requested_id}. The following drones have handlers: {[handler.uav_id for handler in handlers]}")
            acknowledgement = f"ACK_{requested_id}"
            await stream.send_all(acknowledgement.encode('utf-8'))
            await handler.listen()
            handlers.remove(handler)
            log(f"Removing handler for drone {handler.uav_id}. "
                  f"Remaining handlers: {[handler.uav_id for handler in handlers]}")
        else:
            warning(f"Wrong request.")
            await stream.send_all(b'ACK_00')
            return
    else:
        warning("All drone IDs are accounted for.")
        await stream.send_all(b'ACK_00')
        return

uav_ids = ["04", "06", "07", "08", "09"]
handlers: List[DroneHandler] = []
streams: List[trio.SocketStream] = []
start_time = time.time()
log("DUMMY SERVER READY! :)")
ports: List[Tuple[int, Callable]] = [(7000, partial(establish_drone_handler, handlers=handlers)),(7001, partial(listen_and_broadcast, streams=streams))]

async def TCP_parent():
    async with trio.open_nursery() as nursery:
        for port, func in ports:
            # func is partial(establish_drone_handler, handlers=handlers), with one positional argument: stream
            serve_tcp = partial(trio.serve_tcp, handler=func, port=port, handler_nursery=nursery)
            nursery.start_soon(serve_tcp)
        # start = None
        # while start!= "start":
        #     start = await trio.to_thread.run_sync(input, 'Type "start" to start car in 4 seconds!!!\n')
        # try:
        #     for stream in streams:
        #         print("STARTING CAR WROOM WROOM")
        #         await stream.send_all(b'4')
        # except Exception as exc:
        #     print(f"Exception: {exc!r}")
trio.run(TCP_parent)