#!/usr/bin/env python3
import os
# Đặt cấu hình FFMPEG TRƯỚC KHI import cv2 để đảm bảo triệt tiêu lag RTSP
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay"

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point
from nav_msgs.msg import Odometry
from std_msgs.msg import String
import cv2
import numpy as np
import math
import time
import os
import threading
import queue
import pupil_apriltags as apriltag
import yaml
from ament_index_python.packages import PackageNotFoundError, get_package_share_directory
from collections import deque
from datetime import datetime

class Kalman1D:
    def __init__(self, Q=1e-2, R=1e-5, P=1.0, x0=0.0):
        self.Q, self.R, self.P, self.x = Q, R, P, x0
        self.initial_P = P
        self.initialized = False

    def reset(self, value):
        self.x = float(value)
        self.P = self.initial_P
        self.initialized = True

    def update(self, measurement):
        measurement = float(measurement)
        if not self.initialized:
            self.reset(measurement)
            return self.x

        self.P += self.Q
        K = self.P / (self.P + self.R)
        self.x += K * (measurement - self.x)
        self.P *= (1 - K)
        return self.x

def normalize_angle_deg(angle):
    """Đưa góc về [-180, 180]"""
    while angle > 180:
        angle -= 360
    while angle < -180:
        angle += 360
    return angle

def angle_diff_deg(a, b):
    """Tính a - b nhưng kết quả nằm trong [-180, 180]"""
    d = a - b
    while d < -180: d += 360
    while d > 180: d -= 360
    return d

def yaw_from_rotation_matrix(rot):
    return math.atan2(rot[1, 0], rot[0, 0])

def quaternion_from_yaw(yaw_rad):
    half = yaw_rad * 0.5
    return 0.0, 0.0, math.sin(half), math.cos(half)

class CameraPoseEstimator(Node):
    def __init__(self):
        super().__init__('pose_estimation_publisher')
        
        # Publisher chuyển sang hệ Odometry để EKF có thể đọc được
        self.pose_pub = self.create_publisher(Odometry, '/odom_camera', 10)
        
        # Đổi đường dẫn yaml mặc định về thư mục hiện tại để tránh lỗi hardcode cũ
        self.declare_parameter('yaml_path', 'position.yaml')
        self.declare_parameter('draw_trajectory', True)
        self.declare_parameter('trajectory_topic', '/desired_trajectory')
        self.declare_parameter('trajectory_mode_topic', '/desired_trajectory_mode')
        self.declare_parameter('trajectory_max_points', 3000)
        self.declare_parameter('trajectory_mode', 'actual')
        self.declare_parameter('actual_min_pixel_step', 2.0)
        self.declare_parameter('circle_radius', 1.0)
        self.declare_parameter('trajectory_rotation_deg', 0.0)
        self.declare_parameter('trajectory_scale', 1.0)
        self.declare_parameter('trajectory_y_sign', -1.0)
        self.declare_parameter('trajectory_auto_align_yaw', False)
        self.declare_parameter('trajectory_pixels_per_meter', 0.0)
        self.declare_parameter('undistort_display', True)
        self.declare_parameter('undistort_alpha', 0.0)
        self.declare_parameter('warp_enabled', False)
        self.declare_parameter('warp_width', 960)
        self.declare_parameter('warp_height', 720)
        self.declare_parameter('warp_src_points', '')
        self.declare_parameter('warp_yaml_path', 'camera_warp.yaml')
        self.declare_parameter('display_resize', False)
        self.declare_parameter('display_width', 1280)
        self.declare_parameter('display_height', 720)
        self.declare_parameter('prefilter_pose', False)
        self.declare_parameter('camera_x_variance', 0.0004)
        self.declare_parameter('camera_y_variance', 0.0004)
        self.declare_parameter('camera_yaw_variance', 0.0003)
        self.declare_parameter('floor_homography_path', 'floor_homography.yaml')
        self.declare_parameter('homography_scale_x', 1.0)
        self.declare_parameter('homography_scale_y', 1.0)

        self.yaml_path = self.get_parameter('yaml_path').value
        self.draw_trajectory_enabled = bool(self.get_parameter('draw_trajectory').value)
        self.trajectory_topic = self.get_parameter('trajectory_topic').value
        self.trajectory_mode_topic = self.get_parameter('trajectory_mode_topic').value
        self.trajectory_mode = self.get_parameter('trajectory_mode').value
        self.actual_min_pixel_step = float(self.get_parameter('actual_min_pixel_step').value)
        self.circle_radius = float(self.get_parameter('circle_radius').value)
        self.trajectory_rotation = math.radians(
            float(self.get_parameter('trajectory_rotation_deg').value)
        )
        self.trajectory_scale = float(self.get_parameter('trajectory_scale').value)
        self.trajectory_y_sign = float(self.get_parameter('trajectory_y_sign').value)
        self.trajectory_auto_align_yaw = bool(
            self.get_parameter('trajectory_auto_align_yaw').value
        )
        self.trajectory_pixels_per_meter_param = float(
            self.get_parameter('trajectory_pixels_per_meter').value
        )
        self.undistort_display = bool(self.get_parameter('undistort_display').value)
        self.undistort_alpha = float(self.get_parameter('undistort_alpha').value)
        self.warp_enabled = bool(self.get_parameter('warp_enabled').value)
        self.warp_width = int(self.get_parameter('warp_width').value)
        self.warp_height = int(self.get_parameter('warp_height').value)
        self.warp_src_points_text = self.get_parameter('warp_src_points').value
        self.warp_yaml_path = self.get_parameter('warp_yaml_path').value
        self.display_resize = bool(self.get_parameter('display_resize').value)
        self.display_width = int(self.get_parameter('display_width').value)
        self.display_height = int(self.get_parameter('display_height').value)
        self.prefilter_pose = bool(self.get_parameter('prefilter_pose').value)
        self.camera_x_variance = max(
            float(self.get_parameter('camera_x_variance').value),
            1e-9,
        )
        self.camera_y_variance = max(
            float(self.get_parameter('camera_y_variance').value),
            1e-9,
        )
        self.camera_yaw_variance = max(
            float(self.get_parameter('camera_yaw_variance').value),
            1e-9,
        )
        self.floor_homography_path = self.get_parameter('floor_homography_path').value
        self.homography_scale_x = float(self.get_parameter('homography_scale_x').value)
        self.homography_scale_y = float(self.get_parameter('homography_scale_y').value)

        self.warp_matrix = None
        self.warp_src_points = None
        self.warp_calibrating = False
        self.warp_calibration_points = []

        self.homography_matrix = None
        self.load_floor_homography()

        max_points = max(2, int(self.get_parameter('trajectory_max_points').value))
        self.trajectory_points = deque(maxlen=max_points)
        self.actual_path_pixels = deque(maxlen=max_points)
        self.latest_desired = None
        self.trajectory_anchor_desired = None
        self.trajectory_anchor_desired_theta = None
        self.trajectory_anchor_pixel = None
        self.trajectory_anchor_camera_yaw = None
        self.trajectory_px_per_meter = None
        self.current_tag_pixel = None
        self.current_px_per_meter = None
        self.latest_tag_depth = None
        # Lưu thông số camera để dùng projectPoints vẽ quỹ đạo đúng phối cảnh
        self._latest_rvec = None
        self._latest_tvec = None
        self._latest_cam_matrix = None
        self._latest_dist_coeffs = None
        self.traj_sub = self.create_subscription(
            Point, self.trajectory_topic, self.trajectory_callback, 10
        )
        self.traj_mode_sub = self.create_subscription(
            String, self.trajectory_mode_topic, self.trajectory_mode_callback, 10
        )

        self.ip_url = f"rtsp://{os.getenv('CAMERA_USERNAME', 'admin')}:{os.getenv('CAMERA_PASSWORD', 'lab208b3')}@" \
                 f"{os.getenv('CAMERA_IP', '192.168.100.56')}:{os.getenv('CAMERA_PORT', '554')}/cam/realmonitor?channel=1&subtype=0"

        # Calibration & detector (Đã calibrate ở độ phân giải 1280x720)
        self.base_camera_matrix = np.array([[767.6786, 0., 637.4356],
                                  [0., 765.5082, 357.2588],
                                  [0., 0., 1.]], dtype=np.float32)
        self.dist_coeffs = np.array([-0.2374, 0.0734, 0.00345, -0.00824, -0.0514], dtype=np.float32)
        self.zero_dist_coeffs = np.zeros_like(self.dist_coeffs)
        self.detector = apriltag.Detector(families='tag36h11', nthreads=3, refine_edges=1)

        # Sửa lại kích thước tag cho khớp thực tế (15cm = 0.15m)
        self.marker_size = 0.150
        self.marker_id = 0
        self.marker_3D = np.array([
            [-self.marker_size/2, self.marker_size/2, 0],
            [self.marker_size/2, self.marker_size/2, 0],
            [self.marker_size/2, -self.marker_size/2, 0],
            [-self.marker_size/2, -self.marker_size/2, 0]
        ], dtype=np.float32)
        self.Rz_90 = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]], dtype=np.float32)

        self.kalman_x = Kalman1D(Q=0.005, R=0.01)
        self.kalman_y = Kalman1D(Q=0.005, R=0.01)
        self.kalman_yaw = Kalman1D(Q=0.5, R=2.0)

        self.pose = {'x': None, 'y': None, 'yaw': None}
        self.raw_yaw_unwrapped = None
        self.last_raw_yaw_deg = None
        self.pose_saved = False

        self.q = queue.Queue(maxsize=1)
        
        self.window_name = "AprilTag Navigation Monitor"
        self.load_warp_config()

        # Tạo cửa sổ OpenCV cho phép phóng to/thu nhỏ, giữ tỷ lệ ảnh gốc
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)

        # Bắt đầu luồng đọc camera
        self.camera_thread = threading.Thread(target=self.camera_stream_thread, daemon=True)
        self.camera_thread.start()

        # Timer loop (thay thế cho while loop & rate.sleep() của ROS 1)
        timer_period = 1.0 / 20  # 30 Hz
        self.timer = self.create_timer(timer_period, self.timer_callback)

        self.get_logger().info(
            f"Camera overlay enabled={self.draw_trajectory_enabled}, "
            f"trajectory_topic={self.trajectory_topic}, "
            f"trajectory_mode={self.trajectory_mode}, "
            f"auto_align_yaw={self.trajectory_auto_align_yaw}, "
            f"undistort_display={self.undistort_display}, "
            f"warp_enabled={self.warp_enabled}, "
            f"display_resize={self.display_resize}, "
            f"prefilter_pose={self.prefilter_pose}"
        )

    def camera_stream_thread(self):
        cap = None
        while rclpy.ok():
            if cap is None or not cap.isOpened():
                cap = cv2.VideoCapture(self.ip_url, cv2.CAP_FFMPEG)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                if not cap.isOpened():
                    self.get_logger().warn("Cannot connect to camera, retrying...")
                    time.sleep(1)
                    continue

            ret, frame = cap.read()
            if not ret:
                cap.release()
                cap = None
                continue

            if not self.q.full():
                self.q.put_nowait(frame)
            else:
                try:
                    self.q.get_nowait()
                except queue.Empty:
                    pass
                self.q.put_nowait(frame)
        if cap: cap.release()

    def write_pose_to_yaml(self):
        clean_data = {k: float(v) for k, v in self.pose.items() if k in ['x', 'y', 'yaw']}
        try:
            with open(self.yaml_path, "w") as f:
                yaml.dump(clean_data, f)
            self.get_logger().info(f"Initial position saved to {self.yaml_path}")
        except Exception as e:
            self.get_logger().error(f"Failed to save pose to yaml: {e}")

    def load_floor_homography(self):
        candidate_paths = [self.floor_homography_path]
        if not os.path.isabs(self.floor_homography_path):
            try:
                candidate_paths.append(
                    os.path.join(
                        get_package_share_directory('amr_control'),
                        'config',
                        os.path.basename(self.floor_homography_path),
                    )
                )
            except PackageNotFoundError:
                pass

        homography_path = next((p for p in candidate_paths if os.path.exists(p)), None)
        if homography_path is not None:
            try:
                with open(homography_path, "r") as f:
                    data = yaml.safe_load(f) or {}
                h_mat = data.get('homography_matrix')
                if h_mat and len(h_mat) == 3:
                    self.homography_matrix = np.array(h_mat, dtype=np.float32)
                    self.get_logger().info(f"Loaded FLOOR HOMOGRAPHY from {homography_path}")
                else:
                    self.get_logger().warn(f"Invalid homography matrix format in {homography_path}")
            except Exception as exc:
                self.get_logger().error(f"Failed to load floor homography yaml: {exc}")
        else:
            searched = ', '.join(candidate_paths)
            self.get_logger().warn(
                f"Floor homography file not found. Searched: {searched}. "
                "Using fallback solvePnP for XY."
            )

    def parse_warp_points(self, text):
        if not text:
            return None

        try:
            points = []
            for item in text.split(';'):
                x_str, y_str = item.split(',')
                points.append([float(x_str), float(y_str)])

            if len(points) != 4:
                raise ValueError("need exactly 4 points")

            return np.array(points, dtype=np.float32)
        except Exception as exc:
            self.get_logger().warn(f"Invalid warp_src_points: {exc}")
            return None

    def load_warp_config(self):
        points = self.parse_warp_points(self.warp_src_points_text)

        if points is None and os.path.exists(self.warp_yaml_path):
            try:
                with open(self.warp_yaml_path, "r") as f:
                    data = yaml.safe_load(f) or {}
                saved_points = data.get('src_points')
                if saved_points and len(saved_points) == 4:
                    points = np.array(saved_points, dtype=np.float32)
                    self.get_logger().info(f"Loaded camera warp from {self.warp_yaml_path}")
            except Exception as exc:
                self.get_logger().warn(f"Failed to load camera warp yaml: {exc}")

        if points is not None:
            self.set_warp_points(points, save=False)

    def set_warp_points(self, points, save=True):
        dst = np.array([
            [0.0, 0.0],
            [float(self.warp_width - 1), 0.0],
            [float(self.warp_width - 1), float(self.warp_height - 1)],
            [0.0, float(self.warp_height - 1)],
        ], dtype=np.float32)

        self.warp_src_points = np.array(points, dtype=np.float32)
        self.warp_matrix = cv2.getPerspectiveTransform(self.warp_src_points, dst)

        if save:
            try:
                with open(self.warp_yaml_path, "w") as f:
                    yaml.dump({'src_points': self.warp_src_points.tolist()}, f)
                self.get_logger().info(f"Saved camera warp to {self.warp_yaml_path}")
            except Exception as exc:
                self.get_logger().warn(f"Failed to save camera warp yaml: {exc}")

    def start_warp_calibration(self):
        self.warp_calibrating = True
        self.warp_calibration_points = []
        self.warp_enabled = False
        self.get_logger().info(
            "Warp calibration: click 4 floor corners in order "
            "top-left, top-right, bottom-right, bottom-left."
        )

    def mouse_callback(self, event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN or not self.warp_calibrating:
            return

        self.warp_calibration_points.append([float(x), float(y)])
        self.get_logger().info(
            f"Warp point {len(self.warp_calibration_points)}/4: ({x}, {y})"
        )

        if len(self.warp_calibration_points) == 4:
            self.set_warp_points(np.array(self.warp_calibration_points, dtype=np.float32))
            self.warp_calibrating = False
            self.warp_enabled = True
            self.get_logger().info("Warp calibration done. Bird-eye view enabled.")

    def draw_warp_calibration(self, frame):
        if not self.warp_calibrating:
            return

        for idx, point in enumerate(self.warp_calibration_points):
            px, py = int(point[0]), int(point[1])
            cv2.circle(frame, (px, py), 6, (0, 255, 255), -1, cv2.LINE_AA)
            cv2.putText(
                frame,
                str(idx + 1),
                (px + 8, py - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2,
            )

        cv2.putText(
            frame,
            "Click 4 floor corners: TL, TR, BR, BL",
            (20, frame.shape[0] - 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
        )

    def apply_warp(self, frame):
        if not self.warp_enabled or self.warp_matrix is None or self.warp_calibrating:
            return frame

        return cv2.warpPerspective(
            frame,
            self.warp_matrix,
            (self.warp_width, self.warp_height),
            flags=cv2.INTER_LINEAR,
        )

    def trajectory_callback(self, msg):
        self.latest_desired = (float(msg.x), float(msg.y), float(msg.z))

        if self.trajectory_anchor_pixel is None:
            return

        if self.trajectory_anchor_desired is None:
            self.trajectory_anchor_desired = (
                self.latest_desired[0],
                self.latest_desired[1],
            )
            self.trajectory_anchor_desired_theta = self.latest_desired[2]
            self.trajectory_points.clear()

        self.trajectory_points.append((self.latest_desired[0], self.latest_desired[1]))

    def trajectory_mode_callback(self, msg):
        mode = msg.data.strip().lower()
        if mode and mode in ('actual', 'history') and mode != self.trajectory_mode:
            self.trajectory_mode = mode
            self.clear_trajectory_overlay()
            self.get_logger().info(f"trajectory_mode={self.trajectory_mode}")

    def clear_trajectory_overlay(self):
        self.trajectory_points.clear()
        self.actual_path_pixels.clear()
        self.trajectory_anchor_desired = None
        self.trajectory_anchor_desired_theta = None
        self.trajectory_anchor_pixel = None
        self.trajectory_anchor_camera_yaw = None
        self.trajectory_px_per_meter = None
        self.get_logger().info("Camera trajectory overlay cleared.")

    def add_actual_path_point(self, pixel):
        if pixel is None:
            return

        if self.actual_path_pixels:
            last = self.actual_path_pixels[-1]
            dist = math.hypot(pixel[0] - last[0], pixel[1] - last[1])
            if dist < self.actual_min_pixel_step:
                return

        self.actual_path_pixels.append((float(pixel[0]), float(pixel[1])))

    def draw_actual_path_overlay(self, frame):
        if len(self.actual_path_pixels) < 2:
            return False

        points = [
            (int(round(px)), int(round(py)))
            for px, py in self.actual_path_pixels
        ]

        for p0, p1 in zip(points, points[1:]):
            cv2.line(frame, p0, p1, (0, 0, 255), 1, cv2.LINE_AA)

        cv2.circle(frame, points[-1], 3, (0, 0, 255), -1, cv2.LINE_AA)
        return True

    def map_offset_to_pixel(self, anchor_pixel, px_per_meter, dx, dy):
        dy *= self.trajectory_y_sign

        rotation = self.trajectory_rotation
        if (
            self.trajectory_auto_align_yaw
            and self.trajectory_anchor_camera_yaw is not None
            and self.trajectory_anchor_desired_theta is not None
        ):
            rotation += self.trajectory_anchor_camera_yaw - self.trajectory_anchor_desired_theta

        cos_r = math.cos(rotation)
        sin_r = math.sin(rotation)
        x_rot = cos_r * dx - sin_r * dy
        y_rot = sin_r * dx + cos_r * dy

        scale = px_per_meter * self.trajectory_scale
        px = anchor_pixel[0] + x_rot * scale
        py = anchor_pixel[1] - y_rot * scale

        return int(round(px)), int(round(py))

    def marker_pixels_per_meter(self, img_pts):
        side_lengths = []
        for i in range(4):
            p0 = img_pts[i]
            p1 = img_pts[(i + 1) % 4]
            side_lengths.append(float(np.linalg.norm(p1 - p0)))

        avg_side_px = sum(side_lengths) / len(side_lengths)
        return avg_side_px / max(self.marker_size, 1e-6)

    def desired_to_pixel(self, x_des, y_des):
        if (
            self.trajectory_anchor_desired is None
            or self.trajectory_anchor_pixel is None
            or self.trajectory_px_per_meter is None
        ):
            return None

        dx = x_des - self.trajectory_anchor_desired[0]
        dy = y_des - self.trajectory_anchor_desired[1]

        return self.map_offset_to_pixel(
            self.trajectory_anchor_pixel,
            self.trajectory_px_per_meter,
            dx,
            dy,
        )

    def draw_circle_reference_overlay(self, frame):
        """Vẽ quỹ đạo tròn 2D trên màn hình — luôn tròn đều (giống hình mẫu),
        không bị méo theo phối cảnh camera."""
        if (
            self.latest_desired is None
            or self.current_tag_pixel is None
            or self.current_px_per_meter is None
        ):
            return False

        theta_d = self.latest_desired[2]
        radius = self.circle_radius

        # Tâm tròn lệch so với tag để tag nằm trên chu vi vòng tròn
        center_dx = -radius * math.sin(theta_d)
        center_dy = radius * math.cos(theta_d)

        pixels = []
        for i in range(181):
            a = 2.0 * math.pi * i / 180.0
            dx = center_dx + radius * math.cos(a)
            dy = center_dy + radius * math.sin(a)
            pixels.append(
                self.map_offset_to_pixel(
                    self.current_tag_pixel,
                    self.current_px_per_meter,
                    dx,
                    dy,
                )
            )

        h, w = frame.shape[:2]
        margin = 200
        for p0, p1 in zip(pixels, pixels[1:]):
            if not (
                -margin <= p0[0] <= w + margin
                and -margin <= p0[1] <= h + margin
                and -margin <= p1[0] <= w + margin
                and -margin <= p1[1] <= h + margin
            ):
                continue
            cv2.line(frame, p0, p1, (0, 0, 255), 1, cv2.LINE_AA)

        cv2.circle(
            frame,
            (int(round(self.current_tag_pixel[0])), int(round(self.current_tag_pixel[1]))),
            3,
            (0, 0, 255),
            -1,
            cv2.LINE_AA,
        )
        return True

    def draw_trajectory_overlay(self, frame, camera_matrix):
        if not self.draw_trajectory_enabled:
            return

        if self.trajectory_mode == 'actual':
            self.draw_actual_path_overlay(frame)
            return

        if len(self.trajectory_points) < 2:
            return

        if self.trajectory_anchor_pixel is None:
            return

        pixels = []
        h, w = frame.shape[:2]
        margin = 200

        for x_des, y_des in self.trajectory_points:
            pixel = self.desired_to_pixel(x_des, y_des)
            if pixel is None:
                pixels.append(None)
                continue

            px, py = pixel
            if -margin <= px <= w + margin and -margin <= py <= h + margin:
                pixels.append((px, py))
            else:
                pixels.append(None)

        for p0, p1 in zip(pixels, pixels[1:]):
            if p0 is None or p1 is None:
                continue
            cv2.line(frame, p0, p1, (0, 0, 255), 1, cv2.LINE_AA)

        if pixels[-1] is not None:
            cv2.circle(frame, pixels[-1], 3, (0, 0, 255), -1, cv2.LINE_AA)

    def timer_callback(self):
        try:
            # Lấy frame mới nhất, nếu không có thì bỏ qua lượt này
            frame = self.q.get(timeout=0.05)
        except queue.Empty:
            return

        h_in, w_in = frame.shape[:2]

        # SỬA LỖI ODOMETRY: NGUYÊN NHÂN LÀ DO CAMERA BỊ ÉP HÌNH (SQUISHED)
        # Luồng 640x480 không phải bị crop, mà bị ÉP từ 1280x720 xuống.
        # Do đó trục X và trục Y bị thu nhỏ với 2 tỷ lệ khác nhau.
        scale_x = w_in / 1280.0
        scale_y = h_in / 720.0

        camera_matrix = self.base_camera_matrix.copy()
        camera_matrix[0, 0] *= scale_x     # fx scale theo chiều ngang
        camera_matrix[1, 1] *= scale_y     # fy scale theo chiều dọc
        camera_matrix[0, 2] *= scale_x     # cx scale theo chiều ngang
        camera_matrix[1, 2] *= scale_y     # cy scale theo chiều dọc
        
        scaled_camera_matrix = camera_matrix.copy()
        dist_coeffs = self.dist_coeffs

        # ===== DETECT TRÊN ẢNH GỐC (FISHEYE) =====
        # Vì Homography được calib trên ảnh fisheye, nên phải detect trên cùng loại ảnh.
        # solvePnP cũng xử lý đúng với dist_coeffs gốc.
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        detections = self.detector.detect(gray)

        for det in detections:
            if det.tag_id != self.marker_id:
                continue

            img_pts = det.corners.astype(np.float32)
            success, rvec, tvec = cv2.solvePnP(self.marker_3D, img_pts, camera_matrix, dist_coeffs,
                                               flags=cv2.SOLVEPNP_IPPE_SQUARE)
            if not success:
                continue

            R_orig, _ = cv2.Rodrigues(rvec)
            
            # ===== TÍNH TỌA ĐỘ MẶT PHẲNG SÀN BÙ NGHIÊNG CAMERA =====
            # Lấy góc yaw từ rotation matrix
            yaw_cam = yaw_from_rotation_matrix(R_orig)
            
            # Khử góc quay quanh trục Z của Tag để lấy ma trận nghiêng thuần túy của Camera so với Sàn
            Rz_yaw = np.array([
                [math.cos(yaw_cam), -math.sin(yaw_cam), 0],
                [math.sin(yaw_cam),  math.cos(yaw_cam), 0],
                [0,                  0,                 1]
            ])
            
            # R_floor2cam là ma trận cố định biểu diễn độ nghiêng của camera so với mặt sàn.
            # Nó không bị thay đổi dù robot (tag) có xoay tròn tại chỗ.
            R_floor2cam = R_orig @ Rz_yaw.T  
            
            tag_center = np.mean(img_pts, axis=0)

            if self.homography_matrix is not None:
                # ---------------------------------------------------------
                # SỬ DỤNG HOMOGRAPHY ĐỂ TÍNH X, Y (CỰC KỲ CHÍNH XÁC THEO SÀN)
                # ---------------------------------------------------------
                pixel_h = np.array([[tag_center[0]], [tag_center[1]], [1.0]], dtype=np.float32)
                computed = self.homography_matrix @ pixel_h
                computed_x = float(computed[0, 0] / computed[2, 0])
                computed_y = float(computed[1, 0] / computed[2, 0])

                # Chuyển đổi sang hệ trục Toán học (X = Sang Phải, Y = Lên Trên)
                # Áp dụng hệ số bù méo ống kính (barrel distortion compensation)
                robot_x = computed_x * self.homography_scale_x
                robot_y = computed_y * self.homography_scale_y
            else:
                # ---------------------------------------------------------
                # DÙNG SOLVEPNP DỰ PHÒNG NẾU KHÔNG CÓ FILE HOMOGRAPHY
                # ---------------------------------------------------------
                # Chiếu tọa độ tvec (trong hệ camera) xuống mặt phẳng sàn (đã khử nghiêng)
                pos_floor = R_floor2cam.T @ tvec
                # Chuyển sang hệ trục Toán học (X = Sang Phải, Y = Lên Trên)
                # pos_floor[0] là Right, pos_floor[1] là Down
                robot_x = pos_floor[0, 0]
                robot_y = -pos_floor[1, 0]
            
            # Hệ tọa độ Yaw mới: X hướng Phải (0 độ), Y hướng Lên Trên (+90 độ)
            yaw_deg = normalize_angle_deg(-math.degrees(yaw_cam))

            if self.last_raw_yaw_deg is None:
                self.raw_yaw_unwrapped = yaw_deg
            else:
                self.raw_yaw_unwrapped += angle_diff_deg(yaw_deg, self.last_raw_yaw_deg)
            self.last_raw_yaw_deg = yaw_deg

            if self.prefilter_pose:
                self.pose['x'] = self.kalman_x.update(robot_x)
                self.pose['y'] = self.kalman_y.update(robot_y)
                filtered_unwrapped = self.kalman_yaw.update(self.raw_yaw_unwrapped)
                self.pose['yaw'] = normalize_angle_deg(filtered_unwrapped)
            else:
                self.pose['x'] = float(robot_x)
                self.pose['y'] = float(robot_y)
                self.pose['yaw'] = float(yaw_deg)

            self.pose['yaw_unwrapped'] = float(self.raw_yaw_unwrapped)
            self.latest_tag_depth = float(tvec[2][0])
            tag_center = np.mean(img_pts, axis=0)
            self.current_tag_pixel = (float(tag_center[0]), float(tag_center[1]))
            self.current_px_per_meter = self.marker_pixels_per_meter(img_pts)
            # Lưu thông số camera + pose để dùng projectPoints vẽ quỹ đạo
            self._latest_rvec = rvec.copy()
            self._latest_tvec = tvec.copy()
            self._latest_cam_matrix = camera_matrix.copy()
            self._latest_dist_coeffs = dist_coeffs.copy()
            self.add_actual_path_point(self.current_tag_pixel)

            if (
                self.trajectory_anchor_pixel is None
                and self.latest_desired is not None
            ):
                self.trajectory_anchor_pixel = (
                    self.current_tag_pixel[0],
                    self.current_tag_pixel[1],
                )
                self.trajectory_anchor_camera_yaw = math.radians(float(self.pose['yaw']))
                if self.trajectory_pixels_per_meter_param > 0.0:
                    self.trajectory_px_per_meter = self.trajectory_pixels_per_meter_param
                else:
                    self.trajectory_px_per_meter = self.current_px_per_meter
                self.trajectory_anchor_desired = (
                    self.latest_desired[0],
                    self.latest_desired[1],
                )
                self.trajectory_anchor_desired_theta = self.latest_desired[2]
                self.trajectory_points.clear()
                self.trajectory_points.append(
                    (self.latest_desired[0], self.latest_desired[1])
                )
                self.get_logger().info(
                    "Camera trajectory anchor set: "
                    f"pixel=({self.trajectory_anchor_pixel[0]:.1f}, "
                    f"{self.trajectory_anchor_pixel[1]:.1f}), "
                    f"px_per_m={self.trajectory_px_per_meter:.1f}, "
                    f"yaw={self.pose['yaw']:.1f} deg"
                )

            # Lưu vị trí ban đầu
            if not self.pose_saved:
                self.write_pose_to_yaml()
                self.pose_saved = True

            # Publish dữ liệu chuẩn Odometry cho EKF
            odom_msg = Odometry()
            odom_msg.header.stamp = self.get_clock().now().to_msg()
            odom_msg.header.frame_id = 'odom'
            odom_msg.child_frame_id = 'base_link'

            # Tọa độ X, Y
            odom_msg.pose.pose.position.x = float(self.pose['x'])
            odom_msg.pose.pose.position.y = float(self.pose['y'])
            odom_msg.pose.pose.position.z = 0.0

            # Chuyển đổi Yaw sang Quaternion cho Orientation
            quat = quaternion_from_yaw(math.radians(self.pose['yaw']))
            odom_msg.pose.pose.orientation.x = quat[0]
            odom_msg.pose.pose.orientation.y = quat[1]
            odom_msg.pose.pose.orientation.z = quat[2]
            odom_msg.pose.pose.orientation.w = quat[3]

            # Ma trận hiệp phương sai (Covariance): đúng đơn vị variance.
            P = odom_msg.pose.covariance
            P[0] = self.camera_x_variance
            P[7] = self.camera_y_variance
            P[14] = 1e6
            P[21] = 1e6
            P[28] = 1e6
            P[35] = self.camera_yaw_variance

            T = odom_msg.twist.covariance
            T[0] = 1e6
            T[7] = 1e6
            T[14] = 1e6
            T[21] = 1e6
            T[28] = 1e6
            T[35] = 1e6

            self.pose_pub.publish(odom_msg)

            # --- THÊM PHẦN VISUALIZATION ---
            # 1. Vẽ khung vuông bao quanh AprilTag (màu xanh lá)
            pts = img_pts.astype(int)
            for i in range(4):
                cv2.line(frame, tuple(pts[i]), tuple(pts[(i+1)%4]), (0, 255, 0), 2)
            
            cv2.putText(frame, f"AprilTag ID: {det.tag_id}", tuple(pts[3] + [0, 20]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec, tvec, 0.1, 2)

            break

        self.draw_trajectory_overlay(frame, camera_matrix)
        self.draw_warp_calibration(frame)

        display_frame = self.apply_warp(frame)

        if self.display_resize:
            display_frame = cv2.resize(
                display_frame,
                (self.display_width, self.display_height),
                interpolation=cv2.INTER_LINEAR,
            )

        display_h, display_w = display_frame.shape[:2]
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        timestamp_box_x = max(0, display_w - 465)
        timestamp_text_x = max(10, display_w - 360)
        cv2.rectangle(display_frame, (timestamp_box_x, 0), (display_w, 70), (0, 0, 0), -1)
        cv2.putText(display_frame, timestamp, (timestamp_text_x, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(display_frame, timestamp, (timestamp_text_x, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

        if self.pose['x'] is not None and self.pose['y'] is not None and self.pose['yaw'] is not None:
            overlay = display_frame.copy()
            cv2.rectangle(overlay, (10, 10), (235, 95), (0, 0, 0), -1)
            display_frame = cv2.addWeighted(overlay, 0.35, display_frame, 0.65, 0)
            cv2.putText(display_frame, f"X: {self.pose['x']:.3f} m", (20, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(display_frame, f"Y: {self.pose['y']:.3f} m", (20, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(display_frame, f"Yaw: {self.pose['yaw']:.2f} deg", (20, 85),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

        # Show cửa sổ hình ảnh (Hiển thị liên tục kể cả khi không thấy tag)
        cv2.imshow(self.window_name, display_frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('r'):
            self.clear_trajectory_overlay()
        elif key == ord('c'):
            self.start_warp_calibration()
        elif key == ord('u'):
            self.warp_enabled = not self.warp_enabled
            self.get_logger().info(f"warp_enabled={self.warp_enabled}")
        elif key == ord('y'):
            self.trajectory_y_sign *= -1.0
            self.clear_trajectory_overlay()
            self.get_logger().info(f"trajectory_y_sign={self.trajectory_y_sign:+.1f}")
        elif key == ord('a'):
            self.trajectory_rotation -= math.radians(5.0)
            self.clear_trajectory_overlay()
            self.get_logger().info(
                f"trajectory_rotation_deg={math.degrees(self.trajectory_rotation):+.1f}"
            )
        elif key == ord('d'):
            self.trajectory_rotation += math.radians(5.0)
            self.clear_trajectory_overlay()
            self.get_logger().info(
                f"trajectory_rotation_deg={math.degrees(self.trajectory_rotation):+.1f}"
            )
        elif key in (ord('+'), ord('=')):
            self.trajectory_scale *= 1.1
            self.clear_trajectory_overlay()
            self.get_logger().info(f"trajectory_scale={self.trajectory_scale:.3f}")
        elif key in (ord('-'), ord('_')):
            self.trajectory_scale /= 1.1
            self.clear_trajectory_overlay()
            self.get_logger().info(f"trajectory_scale={self.trajectory_scale:.3f}")

def main(args=None):
    rclpy.init(args=args)
    node = CameraPoseEstimator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()
