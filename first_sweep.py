import zmq
import msgpack
import numpy as np
import time
from scipy.spatial.transform import Rotation as R
import numpy.linalg as LA
import open3d as o3d
from sklearn.cluster import DBSCAN

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose, Vector3

from coppeliasim_zmqremoteapi_client import RemoteAPIClient

#tunable parameters
MAX_PLANES = 5
PLANE_DISTANCE_THRESHOLD = 0.0006
PLANE_MIN_POINTS = 300

HORIZONTAL_FOV_DEGREES = 120.0
NUM_STEPS = 840
STEP_Y_START_OFFSET = 0.02
STEP_Y_END_OFFSET = -0.63  # y_end is y_start - 0.65, so here combined as offset from start_pos[1]

STEP_DELAY_TOTAL = 20.0  # seconds total movement duration
STEP_DELAY = STEP_DELAY_TOTAL / NUM_STEPS

OUTLIER_NB_NEIGHBORS = 20
OUTLIER_STD_RATIO = 1.5

Z_THRESHOLD = 0.05

DBSCAN_EPS = 0.004
DBSCAN_MIN_SAMPLES = 100

CIRCLE_RADIUS_MIN = 0.009
CIRCLE_RADIUS_MAX = 0.03

X_DIFFERENCE_MULTIPLIER = 5
CENTER_3D_Z_OFFSET = -0.1
#End


def depth_image_to_point_cloud_transformed(depth_image, sensor_pose_matrix, fx):
    H, W = depth_image.shape
    cx = W / 2
    jj = np.arange(W)
    z = depth_image.flatten()
    x = (jj - cx) * z / fx
    y = np.zeros_like(z)
    pts_camera_frame = np.stack((x, y, z), axis=-1)
    mask = (z > 0) & (~np.isnan(z))
    pts_camera_frame_filtered = pts_camera_frame[mask]

    if pts_camera_frame_filtered.shape[0] == 0:
        return np.array([])

    pts_homogeneous = np.hstack((pts_camera_frame_filtered, np.ones((pts_camera_frame_filtered.shape[0], 1))))
    pts_world_frame_homogeneous = (sensor_pose_matrix @ pts_homogeneous.T).T

    return pts_world_frame_homogeneous[:, :3]


def segment_planes(points, max_planes=MAX_PLANES, distance_threshold=PLANE_DISTANCE_THRESHOLD, min_points=PLANE_MIN_POINTS):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    planes = []
    remaining = pcd
    for _ in range(max_planes):
        if len(remaining.points) < min_points:
            break
        model, inliers = remaining.segment_plane(distance_threshold, 3, 1000)
        if len(inliers) < min_points:
            break
        cloud = remaining.select_by_index(inliers)
        [a, b, c, d] = model
        points_inlier = np.asarray(cloud.points)
        dists = np.abs((points_inlier @ np.array([a, b, c])) + d) / np.linalg.norm([a, b, c])
        tight_mask = dists < 0.001
        filtered = points_inlier[tight_mask]
        if len(filtered) < min_points:
            break
        normal = np.array([a, b, c]) / np.linalg.norm([a, b, c])
        planes.append({
            "normal": normal,
            "centroid": np.mean(filtered, axis=0),
            "points": filtered,
            "plane_eq": model
        })
        remaining = remaining.select_by_index(inliers, invert=True)
    return planes


class NormalPublisher(Node):
    def __init__(self):
        super().__init__('normal_publisher')
        self.pose_publisher = self.create_publisher(Pose, 'surface_pose', 10)
        self.euler_publisher = self.create_publisher(Vector3, 'surface_euler', 10)

    def publish_normal_as_pose(self, centroid, normal):
        if normal[2] < 0:
            normal = -normal

        up = np.array([0, 1, 0])
        forward = -normal
        if abs(np.dot(forward, up)) > 0.99:
            up = np.array([1, 0, 0])
        right = np.cross(up, forward)
        right /= np.linalg.norm(right)
        up = np.cross(forward, right)
        R_mat = np.stack((right, up, forward), axis=-1)

        r = R.from_matrix(R_mat)
        euler = r.as_euler('zxy', degrees=True)

        euler_for_quat = euler.copy()
        euler_for_quat[0] = (euler[0] + 180) % 360  # yaw
        euler_for_quat[2] = (euler[2] - 180) % 360  # roll

        r_modified = R.from_euler('zxy', euler_for_quat, degrees=True)
        quat = r_modified.as_quat()

        pose = Pose()
        pose.position.x = float(centroid[0])
        pose.position.y = float(centroid[1])
        pose.position.z = float(centroid[2])
        pose.orientation.x = float(quat[0])
        pose.orientation.y = float(quat[1])
        pose.orientation.z = float(quat[2])
        pose.orientation.w = float(quat[3])
        self.pose_publisher.publish(pose)

        euler_msg = Vector3()
        euler_msg.x = float(euler[2])  
        euler_msg.y = float(euler[1])  
        euler_msg.z = float(euler[0])  
        self.euler_publisher.publish(euler_msg)

        r_inv = r_modified.inv()
        euler_inv = r_inv.as_euler('zxy', degrees=True)

        self.get_logger().info(
            f"""
            Published pose:
              Position              = {centroid}
              Quaternion            = {quat}
            """
        )


def main():
    rclpy.init()
    ros_node = NormalPublisher()

    ctx = zmq.Context()
    socket = ctx.socket(zmq.SUB)
    socket.connect("tcp://localhost:24000")
    socket.setsockopt(zmq.SUBSCRIBE, b"depth_image")

    client = RemoteAPIClient()
    sim = client.getObject('sim')
    dummy = sim.getObject('/UR5/target')
    vision_sensor_handle = sim.getObject('/UR5/visionSensor')
    res = sim.getVisionSensorResolution(vision_sensor_handle)
    W_sensor, H_sensor = res[0], res[1]

    fx = (W_sensor / 2) / np.tan(np.radians(HORIZONTAL_FOV_DEGREES / 2))

    all_transformed_points = []

    start_pos = np.array(sim.getObjectPosition(dummy, -1))
    scan_line_center_x = start_pos[0]

    y_start = start_pos[1] + STEP_Y_START_OFFSET
    y_end = y_start + STEP_Y_END_OFFSET
    step_delay = STEP_DELAY

    for step in range(NUM_STEPS):
        r = step / (NUM_STEPS - 1)
        y_current = y_start * (1 - r) + y_end * r
        sim.setObjectPosition(dummy, -1, [start_pos[0], y_current, start_pos[2]])
        time.sleep(step_delay)

        sensor_pose = sim.getObjectPose(vision_sensor_handle, -1)
        pos = np.array(sensor_pose[:3])
        quat = np.array(sensor_pose[3:])
        rotation = R.from_quat(quat)
        sensor_matrix = np.eye(4)
        sensor_matrix[:3, :3] = rotation.as_matrix()
        sensor_matrix[:3, 3] = pos

        if socket.poll(timeout=1000):
            _, data = socket.recv_multipart()
            m = msgpack.unpackb(data, raw=False)
            depth = np.frombuffer(m['data'], np.float32).reshape((m['height'], m['width']))
            pts = depth_image_to_point_cloud_transformed(depth, sensor_matrix, fx)
            if pts.size > 0:
                all_transformed_points.append(pts)

    if not all_transformed_points:
        print("No scan lines collected.")
        return

    points = np.vstack(all_transformed_points)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=OUTLIER_NB_NEIGHBORS, std_ratio=OUTLIER_STD_RATIO)
    filtered_points = np.asarray(pcd.points)

    planes = segment_planes(filtered_points)
    if not planes:
        print("No planes detected.")
        return

    valid_planes = [p for p in planes if p['centroid'][2] > Z_THRESHOLD]
    plane = max(valid_planes, key=lambda p: p['centroid'][2]) if valid_planes else max(planes, key=lambda p: p['centroid'][2])
    ctr, n, pts_plane = plane['centroid'], plane['normal'], plane['points']

    centered = pts_plane - ctr
    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    idx = np.argsort(eigvals)[::-1]
    basis_x = eigvecs[:, idx[0]]
    basis_x -= np.dot(basis_x, n) * n
    basis_x /= np.linalg.norm(basis_x)
    basis_y = np.cross(n, basis_x)
    basis_y /= np.linalg.norm(basis_y)

    clustering = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES).fit(pts_plane)
    labels = clustering.labels_
    found = False

    for label_id in np.unique(labels):
        if label_id == -1:
            continue
        cluster_points = pts_plane[labels == label_id]
        if len(cluster_points) < DBSCAN_MIN_SAMPLES:
            continue

        centered_cluster = cluster_points - ctr
        points_2d = np.stack([centered_cluster @ basis_x, centered_cluster @ basis_y], axis=-1)

        A = np.c_[2 * points_2d, np.ones(points_2d.shape[0])]
        b = np.sum(points_2d ** 2, axis=1)
        try:
            c, _, _, _ = LA.lstsq(A, b, rcond=None)
            center_2d = c[:2]
            radius = np.sqrt(c[2] + np.sum(center_2d ** 2))
        except:
            continue

        if not (CIRCLE_RADIUS_MIN <= radius <= CIRCLE_RADIUS_MAX):
            continue

        calculated_center_on_plane_raw = ctr + center_2d[0] * basis_x + center_2d[1] * basis_y
        x_difference = calculated_center_on_plane_raw[0] - scan_line_center_x

        center_3d = np.array([
            calculated_center_on_plane_raw[0] - X_DIFFERENCE_MULTIPLIER * x_difference,
            calculated_center_on_plane_raw[1],
            calculated_center_on_plane_raw[2] + CENTER_3D_Z_OFFSET
        ])

        ros_node.publish_normal_as_pose(center_3d, n)
        found = True
        break

    if not found:
        print("No valid circle detected.")
    ros_node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
