import numpy as np
import zmq
import msgpack
import time
from coppeliasim_zmqremoteapi_client import RemoteAPIClient

def main():
    print("Connecting to CoppeliaSim...")
    client = RemoteAPIClient()
    sim = client.getObject('sim')

    sensor_handle = sim.getObject('/UR5/visionSensor')

    context = zmq.Context()
    socket = context.socket(zmq.PUB)
    socket.bind("tcp://*:24000")

    print("Streaming depth data from /visionSensor...")

    while True:
        try:
            depth_bytes, resolution = sim.getVisionSensorDepth(sensor_handle)
            width, height = resolution 
            depth_array = np.frombuffer(depth_bytes, dtype=np.float32).reshape((height, width))

            msg = {
                'data': depth_array.astype(np.float32).tobytes(),
                'width': width,
                'height': height
            }

            socket.send_multipart([b'depth_image', msgpack.packb(msg)])

        except Exception as e:
            print(f"[Error] {e}")

        time.sleep(0.05)

if __name__ == '__main__':
    main()