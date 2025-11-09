import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose

import numpy as np
import threading
import time
from scipy.spatial.transform import Rotation as R, Slerp
from coppeliasim_zmqremoteapi_client import RemoteAPIClient


#tuning variables 

SMOOTH_TRANSFORM_DURATION = 1.0  # seconds for smooth_transform interpolation
SMOOTH_TRANSFORM_STEPS = 50      # number of interpolation steps
STAGE1_PAUSE = 0.5               # seconds to wait between stage1 and stage2


class TargetPoseController(Node):
    def __init__(self):
        super().__init__('target_pose_controller')

        self.subscription = self.create_subscription(
            Pose,
            'surface_pose',
            self.pose_callback,
            10)

        self.client = RemoteAPIClient()
        self.sim = self.client.getObject('sim')

        self.target_handle = self.sim.getObject('/UR5/target')
        self.get_logger().info("Connected to CoppeliaSim and UR5/target located.")

        self.lock = threading.Lock()
        self.target_pose_matrix = np.eye(4)

        self.motion_thread = None

    def pose_callback(self, msg: Pose):
        quat = [msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w]
        position = np.array([msg.position.x, msg.position.y, msg.position.z])

        euler = R.from_quat(quat).as_euler('xyz', degrees=True)
        euler[2] = (euler[2] - 180) % 360 

        rot_matrix = R.from_euler('xyz', euler, degrees=True).as_matrix()

        pose_matrix = np.eye(4)
        pose_matrix[:3, :3] = rot_matrix
        pose_matrix[:3, 3] = position

        with self.lock:
            self.target_pose_matrix = pose_matrix.copy()

        if self.motion_thread is None or not self.motion_thread.is_alive():
            self.motion_thread = threading.Thread(target=self.move_target, daemon=True)
            self.motion_thread.start()

    def move_target(self):
        with self.lock:
            target_matrix = self.target_pose_matrix.copy()

        parent_handle = self.sim.getObjectParent(self.target_handle)

        current_pose = self.sim.getObjectMatrix(self.target_handle, -1)
        current_matrix = np.eye(4)
        current_matrix[:3, :] = np.array(current_pose).reshape(3, 4)

        stage1_matrix = current_matrix.copy()
        stage1_matrix[:2, 3] = target_matrix[:2, 3]  
        stage1_matrix[:3, :3] = target_matrix[:3, :3]  
        stage1_matrix[2, 3] = current_matrix[2, 3]  

        self.smooth_transform(current_matrix, stage1_matrix, parent_handle, duration=SMOOTH_TRANSFORM_DURATION)

        time.sleep(STAGE1_PAUSE)

        stage2_matrix = target_matrix.copy()
        self.smooth_transform(stage1_matrix, stage2_matrix, parent_handle, duration=SMOOTH_TRANSFORM_DURATION)

        self.get_logger().info("movement complete.")

    def smooth_transform(self, start_matrix, end_matrix, parent_handle, duration=1.0, steps=50):
        r1 = R.from_matrix(start_matrix[:3, :3])
        r2 = R.from_matrix(end_matrix[:3, :3])
        slerp = Slerp([0, 1], R.concatenate([r1, r2]))

        for i in range(steps + 1):
            alpha = i / steps
            interp_matrix = np.eye(4)

            interp_matrix[:3, 3] = (1 - alpha) * start_matrix[:3, 3] + alpha * end_matrix[:3, 3]

            interp_matrix[:3, :3] = slerp([alpha])[0].as_matrix()

            if parent_handle != -1:
                parent_pose = self.sim.getObjectMatrix(parent_handle, -1)
                parent_matrix = np.array(parent_pose).reshape(3, 4)
                parent_matrix = np.vstack([parent_matrix, [0, 0, 0, 1]])
                parent_inv = np.linalg.inv(parent_matrix)
                relative_matrix = parent_inv @ interp_matrix
            else:
                relative_matrix = interp_matrix

            matrix_flat = relative_matrix[:3].flatten().tolist()
            self.sim.setObjectMatrix(self.target_handle, parent_handle, matrix_flat)

            time.sleep(duration / steps)


def main(args=None):
    rclpy.init(args=args)
    node = TargetPoseController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
