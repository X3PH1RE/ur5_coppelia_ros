import zmq
import msgpack
import numpy as np
import time
import os
from scipy.spatial.transform import Rotation as R
import numpy.linalg as LA
import open3d as o3d
from sklearn.cluster import DBSCAN
from skimage.measure import LineModelND, ransac

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose

from coppeliasim_zmqremoteapi_client import RemoteAPIClient

#tuning variables
MAX_PLANES = 5
PLANE_DISTANCE_THRESHOLD = 0.0006
PLANE_MIN_POINTS = 300

NB_NEIGHBORS_OUTLIER_REMOVAL = 20
STD_RATIO_OUTLIER_REMOVAL = 1.5

DBSCAN_EPS = 0.004
DBSCAN_MIN_SAMPLES = 100

RANSAC_MIN_SAMPLES = 5
RANSAC_RESIDUAL_THRESHOLD = 0.005
RANSAC_MAX_TRIALS = 1000

CIRCLE_RADIUS_MIN = 0.009
CIRCLE_RADIUS_MAX = 0.03

LINE_POINT_DISTANCE_THRESHOLD = 0.005

Z_THRESHOLD_PLANE_SELECTION = 0.05

MAX_ATTEMPTS = 10
NUM_STEPS_SCAN = 420
STEP_DELAY_TOTAL_TIME = 20.0  # seconds

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

    def publish_normal_as_pose(self, centroid, orientation_matrix):
        euler = R.from_matrix(orientation_matrix).as_euler('zxy', degrees=True)
        euler[0] = -euler[0]
        orientation_matrix = R.from_euler('zxy', euler, degrees=True).as_matrix()
        quat = R.from_matrix(orientation_matrix).as_quat()

        pose = Pose()
        pose.position.x = float(centroid[0])
        pose.position.y = float(centroid[1])
        pose.position.z = float(centroid[2])
        pose.orientation.x = float(quat[0])
        pose.orientation.y = float(quat[1])
        pose.orientation.z = float(quat[2])
        pose.orientation.w = float(quat[3])
        self.pose_publisher.publish(pose)

        self.get_logger().info(f"Published pose:\n  Position = {centroid}\n  Euler zxy (yaw inverted) = {euler}")


def visualize_with_circle_and_line(points, circle_points, line_anchor, line_dir, line_length=0.05):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.paint_uniform_color([0.7, 0.7, 0.7])

    circle_pcd = o3d.geometry.PointCloud()
    circle_pcd.points = o3d.utility.Vector3dVector(circle_points)
    circle_pcd.paint_uniform_color([1.0, 0.0, 0.0])

    line_start = line_anchor - line_dir * line_length / 2
    line_end = line_anchor + line_dir * line_length / 2
    line_points = [line_start, line_end]
    lines = [[0, 1]]
    colors = [[0, 1, 0]]
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(line_points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)

    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])

    o3d.visualization.draw_geometries([pcd, circle_pcd, line_set, coordinate_frame])


def save_and_visualize_output(output_dir, base_name, point_cloud, circle_points, line_anchor, line_dir, center_3d, orientation_matrix):
    os.makedirs(output_dir, exist_ok=True)

    def get_next_index(prefix):
        existing = [f for f in os.listdir(output_dir) if f.startswith(prefix) and f.endswith('.npz')]
        indices = [int(f.split("_")[-1].split(".")[0]) for f in existing if f.split("_")[-1].split(".")[0].isdigit()]
        return max(indices, default=0) + 1

    index = get_next_index(base_name)
    file_path = os.path.join(output_dir, f"{base_name}_{index}.npz")

    np.savez(file_path,
             point_cloud=point_cloud,
             circle_points=circle_points,
             line_anchor=line_anchor,
             line_dir=line_dir,
             center=center_3d,
             orientation=orientation_matrix)

    print(f"Saved output to {file_path}")
    visualize_with_circle_and_line(point_cloud, circle_points, center_3d, line_dir)


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
    W_sensor, H_sensor = sim.getVisionSensorResolution(vision_sensor_handle)
    fx = (W_sensor / 2) / np.tan(np.radians(120.0 / 2))

    original_position = sim.getObjectPosition(dummy, -1)
    current_z_offset = 0.0

    for attempt in range(MAX_ATTEMPTS):
        all_transformed_points = []

        start_pos = [original_position[0], original_position[1], original_position[2] + current_z_offset]
        sim.setObjectPosition(dummy, -1, start_pos)

        y_start = start_pos[1] + 0.06
        y_end = y_start - 0.11
        step_delay = STEP_DELAY_TOTAL_TIME / NUM_STEPS_SCAN

        for step in range(NUM_STEPS_SCAN):
            r = step / (NUM_STEPS_SCAN - 1)
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
            current_z_offset += 0.01
            continue

        points = np.vstack(all_transformed_points)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=NB_NEIGHBORS_OUTLIER_REMOVAL,
                                                std_ratio=STD_RATIO_OUTLIER_REMOVAL)
        filtered_points = np.asarray(pcd.points)

        planes = segment_planes(filtered_points)
        if not planes:
            print("No planes detected.")
            current_z_offset += 0.01
            continue

        valid_planes = [p for p in planes if p['centroid'][2] > Z_THRESHOLD_PLANE_SELECTION]
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

            circle_center_3d = ctr + center_2d[0] * basis_x + center_2d[1] * basis_y
            distances_from_circle_center = np.linalg.norm(cluster_points - circle_center_3d, axis=1)
            points_for_line_fit = cluster_points[distances_from_circle_center <= LINE_POINT_DISTANCE_THRESHOLD]
            if len(points_for_line_fit) < 2:
                continue

            try:
                model_robust, inliers_line = ransac(
                    points_for_line_fit, LineModelND, min_samples=RANSAC_MIN_SAMPLES,
                    residual_threshold=RANSAC_RESIDUAL_THRESHOLD, max_trials=RANSAC_MAX_TRIALS
                )
                line_direction = model_robust.params[1]
                line_direction /= np.linalg.norm(line_direction)
            except:
                continue

            line_dir_in_plane = line_direction - np.dot(line_direction, n) * n
            norm = np.linalg.norm(line_dir_in_plane)
            if norm < 1e-8:
                continue
            line_dir_in_plane /= norm

            x_axis = line_dir_in_plane
            z_axis = n
            if z_axis[2] < 0:
                z_axis *= -1
            y_axis = np.cross(z_axis, x_axis)
            y_axis /= np.linalg.norm(y_axis)
            x_axis = np.cross(y_axis, z_axis)
            x_axis /= np.linalg.norm(x_axis)

            orientation_matrix = np.stack((x_axis, y_axis, z_axis), axis=-1)
            orientation_matrix = orientation_matrix @ R.from_euler('z', 90, degrees=True).as_matrix()

            ros_node.publish_normal_as_pose(circle_center_3d, orientation_matrix)

            save_and_visualize_output("output_data", "output",
                                      filtered_points,
                                      cluster_points,
                                      circle_center_3d,
                                      line_dir_in_plane,
                                      circle_center_3d,
                                      orientation_matrix)
            found = True
            break

        if found:
            break
        else:
            print(f"No valid circle found in attempt {attempt + 1}. Retrying...")
            current_z_offset += 0.01
            sim.setObjectPosition(dummy, -1, [original_position[0], original_position[1], original_position[2] + current_z_offset])
            time.sleep(0.5)

    ros_node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
