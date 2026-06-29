import math

import numpy as np
import rclpy
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import Imu
from std_msgs.msg import Empty
from tf2_ros import TransformBroadcaster


def normalize_angle(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


def yaw_from_quaternion(q):
    t3 = 2.0 * (q.w * q.z + q.x * q.y)
    t4 = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(t3, t4)


def quaternion_from_yaw(yaw):
    half = yaw * 0.5
    return 0.0, 0.0, math.sin(half), math.cos(half)


def covariance_is_valid(covariance, index):
    return 0.0 < covariance[index] < 1e5


class CustomEkfNode(Node):
    """
    Lightweight 2D EKF for differential-drive AMR.

    State: [x, y, theta, v, w]
      - /odom_raw initializes x/y/yaw and updates v/yaw/w in the controller odom frame
      - /odom_camera is aligned into the odom frame before optional x/y correction
      - /imu/data can be enabled as an alternate yaw-rate source only
      - prediction integrates v and w into x, y, theta
    """

    def __init__(self):
        super().__init__('custom_ekf_node')

        self.declare_parameter('frequency', 20.0)
        self.declare_parameter('sensor_timeout', 0.5)
        self.declare_parameter('odom_topic', '/odom_raw')
        self.declare_parameter('camera_topic', '/odom_camera')
        self.declare_parameter('imu_topic', '/imu/data')
        self.declare_parameter('output_topic', '/odometry/filtered')
        self.declare_parameter('reset_topic', '/reset_odom')
        self.declare_parameter('odom_frame', 'odom')
        self.declare_parameter('base_link_frame', 'base_link')
        self.declare_parameter('publish_tf', True)
        self.declare_parameter('use_odom_pose', True)
        self.declare_parameter('use_odom_yaw', True)
        self.declare_parameter('use_odom_w', True)
        self.declare_parameter('use_camera_pose', False)
        self.declare_parameter('use_camera_yaw', False)
        self.declare_parameter('use_camera_yaw_for_alignment', True)
        self.declare_parameter('use_imu_w', False)
        self.declare_parameter('imu_w_sign', 1.0)
        self.declare_parameter('odom_w_sign', 1.0)
        self.declare_parameter('align_camera_to_odom', True)

        self.declare_parameter('initial_covariance', 0.05)
        self.declare_parameter('process_noise_x', 0.002)
        self.declare_parameter('process_noise_y', 0.002)
        self.declare_parameter('process_noise_theta', 0.004)
        self.declare_parameter('process_noise_v', 0.08)
        self.declare_parameter('process_noise_w', 0.12)

        self.declare_parameter('odom_x_variance', 0.01)
        self.declare_parameter('odom_y_variance', 0.01)
        self.declare_parameter('odom_yaw_variance', 0.08)
        self.declare_parameter('odom_v_variance', 0.04)
        self.declare_parameter('odom_w_variance', 0.60)
        self.declare_parameter('camera_x_variance', 0.01)
        self.declare_parameter('camera_y_variance', 0.01)
        self.declare_parameter('camera_yaw_variance', 0.05)
        self.declare_parameter('use_camera_msg_covariance', False)
        self.declare_parameter('imu_w_variance', 0.01)

        # Outlier rejection thresholds
        self.declare_parameter('max_jump_xy', 0.30)      # m
        self.declare_parameter('max_jump_yaw', 0.52)     # rad ≈ 30 deg

        self.frequency = float(self.get_parameter('frequency').value)
        self.sensor_timeout = float(self.get_parameter('sensor_timeout').value)
        self.odom_topic = self.get_parameter('odom_topic').value
        self.camera_topic = self.get_parameter('camera_topic').value
        self.imu_topic = self.get_parameter('imu_topic').value
        self.output_topic = self.get_parameter('output_topic').value
        self.reset_topic = self.get_parameter('reset_topic').value
        self.odom_frame = self.get_parameter('odom_frame').value
        self.base_link_frame = self.get_parameter('base_link_frame').value
        self.publish_tf = bool(self.get_parameter('publish_tf').value)
        self.use_odom_pose = bool(self.get_parameter('use_odom_pose').value)
        self.use_odom_yaw = bool(self.get_parameter('use_odom_yaw').value)
        self.use_odom_w = bool(self.get_parameter('use_odom_w').value)
        self.use_camera_pose = bool(self.get_parameter('use_camera_pose').value)
        self.use_camera_yaw = bool(self.get_parameter('use_camera_yaw').value)
        self.use_camera_yaw_for_alignment = bool(
            self.get_parameter('use_camera_yaw_for_alignment').value
        )
        self.use_imu_w = bool(self.get_parameter('use_imu_w').value)
        self.imu_w_sign = float(self.get_parameter('imu_w_sign').value)
        self.odom_w_sign = float(self.get_parameter('odom_w_sign').value)
        self.align_camera_to_odom = bool(self.get_parameter('align_camera_to_odom').value)

        initial_covariance = float(self.get_parameter('initial_covariance').value)
        self.x = np.zeros((5, 1), dtype=float)
        self.p = np.eye(5, dtype=float) * initial_covariance

        self.q_base = np.diag([
            float(self.get_parameter('process_noise_x').value),
            float(self.get_parameter('process_noise_y').value),
            float(self.get_parameter('process_noise_theta').value),
            float(self.get_parameter('process_noise_v').value),
            float(self.get_parameter('process_noise_w').value),
        ])

        self.r_odom_x = float(self.get_parameter('odom_x_variance').value)
        self.r_odom_y = float(self.get_parameter('odom_y_variance').value)
        self.r_odom_yaw = float(self.get_parameter('odom_yaw_variance').value)
        self.r_odom_v = float(self.get_parameter('odom_v_variance').value)
        self.r_odom_w = float(self.get_parameter('odom_w_variance').value)
        self.r_camera_x = float(self.get_parameter('camera_x_variance').value)
        self.r_camera_y = float(self.get_parameter('camera_y_variance').value)
        self.r_camera_yaw = float(self.get_parameter('camera_yaw_variance').value)
        self.use_camera_msg_covariance = bool(
            self.get_parameter('use_camera_msg_covariance').value
        )
        self.r_imu_w = float(self.get_parameter('imu_w_variance').value)

        self.max_jump_xy = float(self.get_parameter('max_jump_xy').value)
        self.max_jump_yaw = float(self.get_parameter('max_jump_yaw').value)
        self.camera_reject_count = 0

        self.initialized = False
        self.last_predict_time = None
        self.last_odom_time = None
        self.last_camera_time = None
        self.last_imu_time = None
        self.camera_alignment = None

        self.odom_sub = self.create_subscription(
            Odometry, self.odom_topic, self.odom_callback, 10
        )
        self.camera_sub = self.create_subscription(
            Odometry, self.camera_topic, self.camera_callback, 10
        )
        self.imu_sub = self.create_subscription(
            Imu, self.imu_topic, self.imu_callback, 10
        )
        self.reset_sub = self.create_subscription(
            Empty, self.reset_topic, self.reset_callback, 10
        )
        self.odom_pub = self.create_publisher(Odometry, self.output_topic, 10)
        self.camera_aligned_pub = self.create_publisher(Odometry, '/odom_camera_aligned', 10)

        self.tf_broadcaster = TransformBroadcaster(self) if self.publish_tf else None

        period = 1.0 / max(self.frequency, 1.0)
        self.timer = self.create_timer(period, self.timer_callback)

        self.get_logger().info(
            "Custom EKF started: "
            f"{self.odom_topic} + {self.camera_topic} + {self.imu_topic} -> {self.output_topic}, "
            f"publish_tf={self.publish_tf}, "
            f"use_camera_pose={self.use_camera_pose}, use_camera_yaw={self.use_camera_yaw}, "
            f"align_camera_to_odom={self.align_camera_to_odom}"
        )

    def now_sec(self):
        return self.get_clock().now().nanoseconds / 1e9

    def reset_callback(self, _msg):
        self.x[:, 0] = 0.0
        self.p = np.eye(5, dtype=float) * float(
            self.get_parameter('initial_covariance').value
        )
        self.initialized = False
        self.last_predict_time = None
        self.last_odom_time = None
        self.last_camera_time = None
        self.last_imu_time = None
        self.camera_alignment = None
        self.get_logger().info("Custom EKF reset. Waiting for /odom_raw.")

    def initialize_from_odom(self, msg, now_s):
        yaw = yaw_from_quaternion(msg.pose.pose.orientation)
        self.x[0, 0] = msg.pose.pose.position.x
        self.x[1, 0] = msg.pose.pose.position.y
        self.x[2, 0] = yaw
        self.x[3, 0] = msg.twist.twist.linear.x
        self.x[4, 0] = self.odom_w_sign * msg.twist.twist.angular.z
        self.last_predict_time = now_s
        self.last_odom_time = now_s
        self.initialized = True
        self.get_logger().info(
            f"Custom EKF initialized: x={self.x[0,0]:.3f}, "
            f"y={self.x[1,0]:.3f}, yaw={math.degrees(yaw):.1f} deg"
        )

    def predict_to(self, now_s):
        if not self.initialized:
            return

        if self.last_predict_time is None:
            self.last_predict_time = now_s
            return

        dt = now_s - self.last_predict_time
        if dt <= 0.0:
            return

        dt = min(dt, 0.20)
        theta = self.x[2, 0]
        v = self.x[3, 0]
        w = self.x[4, 0]

        self.x[0, 0] += v * math.cos(theta) * dt
        self.x[1, 0] += v * math.sin(theta) * dt
        self.x[2, 0] = normalize_angle(theta + w * dt)

        f = np.eye(5, dtype=float)
        f[0, 2] = -v * math.sin(theta) * dt
        f[0, 3] = math.cos(theta) * dt
        f[1, 2] = v * math.cos(theta) * dt
        f[1, 3] = math.sin(theta) * dt
        f[2, 4] = dt

        self.p = f @ self.p @ f.T + self.q_base * dt
        self.p = 0.5 * (self.p + self.p.T)
        self.last_predict_time = now_s

    def ekf_update(self, z, h, r):
        innovation = z - h @ self.x
        for idx in range(innovation.shape[0]):
            # Measurements for theta need angle wrapping.
            state_indices = np.flatnonzero(h[idx])
            if len(state_indices) == 1 and state_indices[0] == 2:
                innovation[idx, 0] = normalize_angle(innovation[idx, 0])

        s = h @ self.p @ h.T + r
        k = self.p @ h.T @ np.linalg.inv(s)
        self.x = self.x + k @ innovation
        self.x[2, 0] = normalize_angle(self.x[2, 0])

        i = np.eye(5, dtype=float)
        # Joseph form keeps covariance symmetric and positive semi-definite.
        self.p = (i - k @ h) @ self.p @ (i - k @ h).T + k @ r @ k.T
        self.p = 0.5 * (self.p + self.p.T)

    def camera_variance(self, covariance, index, fallback):
        if self.use_camera_msg_covariance and covariance_is_valid(covariance, index):
            return covariance[index]
        return fallback

    def odom_callback(self, msg):
        now_s = self.now_sec()
        if not self.initialized:
            if not self.align_camera_to_odom:
                # Phải đợi camera để lấy tọa độ tuyệt đối ban đầu
                return
            self.initialize_from_odom(msg, now_s)
            return

        self.predict_to(now_s)
        self.last_odom_time = now_s

        z_values = []
        h_rows = []
        r_values = []

        if self.use_odom_pose:
            z_values.extend([
                msg.pose.pose.position.x,
                msg.pose.pose.position.y,
            ])
            h_rows.extend([
                [1.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 0.0],
            ])
            r_values.extend([
                msg.pose.covariance[0] if covariance_is_valid(msg.pose.covariance, 0) else self.r_odom_x,
                msg.pose.covariance[7] if covariance_is_valid(msg.pose.covariance, 7) else self.r_odom_y,
            ])

        z_values.append(msg.twist.twist.linear.x)
        h_rows.append([0.0, 0.0, 0.0, 1.0, 0.0])
        r_values.append(
            msg.twist.covariance[0] if covariance_is_valid(msg.twist.covariance, 0) else self.r_odom_v
        )

        if self.use_odom_yaw:
            z_values.append(yaw_from_quaternion(msg.pose.pose.orientation))
            h_rows.append([0.0, 0.0, 1.0, 0.0, 0.0])
            r_values.append(
                msg.pose.covariance[35]
                if covariance_is_valid(msg.pose.covariance, 35)
                else self.r_odom_yaw
            )

        if self.use_odom_w:
            z_values.append(self.odom_w_sign * msg.twist.twist.angular.z)
            h_rows.append([0.0, 0.0, 0.0, 0.0, 1.0])
            r_values.append(
                msg.twist.covariance[35]
                if covariance_is_valid(msg.twist.covariance, 35)
                else self.r_odom_w
            )

        if z_values:
            z = np.array(z_values, dtype=float).reshape((-1, 1))
            h = np.array(h_rows, dtype=float)
            r = np.diag(r_values)
            self.ekf_update(z, h, r)

    def camera_callback(self, msg):
        if not self.use_camera_pose and not self.use_camera_yaw:
            return

        now_s = self.now_sec()

        if not self.initialized:
            if not self.align_camera_to_odom:
                # Initialize EKF directly in Camera Frame
                self.x[0, 0] = msg.pose.pose.position.x
                self.x[1, 0] = msg.pose.pose.position.y
                self.x[2, 0] = yaw_from_quaternion(msg.pose.pose.orientation)
                self.x[3, 0] = 0.0
                self.x[4, 0] = 0.0
                self.last_predict_time = now_s
                self.last_camera_time = now_s
                self.initialized = True
                self.get_logger().info(
                    f"Custom EKF initialized from Camera Frame: x={self.x[0,0]:.3f}, "
                    f"y={self.x[1,0]:.3f}, yaw={math.degrees(self.x[2,0]):.1f} deg"
                )
            return

        self.predict_to(now_s)
        self.last_camera_time = now_s
        aligned_msg = self.align_camera_msg(msg)
        self.camera_aligned_pub.publish(aligned_msg)

        z_values = []
        h_rows = []
        r_values = []

        if self.use_camera_pose:
            z_values.extend([
                aligned_msg.pose.pose.position.x,
                aligned_msg.pose.pose.position.y,
            ])
            h_rows.extend([
                [1.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 0.0],
            ])
            r_values.extend([
                self.camera_variance(msg.pose.covariance, 0, self.r_camera_x),
                self.camera_variance(msg.pose.covariance, 7, self.r_camera_y),
            ])

        if self.use_camera_yaw:
            z_values.append(yaw_from_quaternion(aligned_msg.pose.pose.orientation))
            h_rows.append([0.0, 0.0, 1.0, 0.0, 0.0])
            r_values.append(
                self.camera_variance(msg.pose.covariance, 35, self.r_camera_yaw)
            )

        if z_values:
            z = np.array(z_values, dtype=float).reshape((-1, 1))
            h = np.array(h_rows, dtype=float)
            r = np.diag(r_values)

            # --- OUTLIER REJECTION ---
            innovation = z - h @ self.x
            # Normalize angle innovation for yaw
            for idx in range(innovation.shape[0]):
                state_indices = np.flatnonzero(h[idx])
                if len(state_indices) == 1 and state_indices[0] == 2:
                    innovation[idx, 0] = normalize_angle(innovation[idx, 0])

            # Check jump thresholds
            reject = False
            for idx in range(innovation.shape[0]):
                state_indices = np.flatnonzero(h[idx])
                if len(state_indices) == 1:
                    si = state_indices[0]
                    jump = abs(innovation[idx, 0])
                    if si in (0, 1) and jump > self.max_jump_xy:
                        reject = True
                        break
                    if si == 2 and jump > self.max_jump_yaw:
                        reject = True
                        break

            if reject:
                self.camera_reject_count += 1
                self.get_logger().warn(
                    f"Camera outlier rejected (#{self.camera_reject_count}): "
                    f"innovation = {innovation.flatten()}",
                    throttle_duration_sec=1.0,
                )
                return

            self.ekf_update(z, h, r)

    def imu_callback(self, msg):
        if not self.use_imu_w:
            return

        if not self.initialized:
            return

        now_s = self.now_sec()
        self.predict_to(now_s)
        self.last_imu_time = now_s

        variance = (
            msg.angular_velocity_covariance[8]
            if covariance_is_valid(msg.angular_velocity_covariance, 8)
            else self.r_imu_w
        )
        z = np.array([[self.imu_w_sign * msg.angular_velocity.z]], dtype=float)
        h = np.array([[0.0, 0.0, 0.0, 0.0, 1.0]], dtype=float)
        r = np.array([[variance]], dtype=float)
        self.ekf_update(z, h, r)

    def timer_callback(self):
        if not self.initialized:
            return

        now_s = self.now_sec()
        self.predict_to(now_s)
        self.apply_sensor_timeout(now_s)
        stamp = self.get_clock().now().to_msg()
        self.publish_odometry(stamp)
        if self.publish_tf:
            self.publish_transform(stamp)

    def apply_sensor_timeout(self, now_s):
        if (
            self.last_odom_time is not None
            and now_s - self.last_odom_time > self.sensor_timeout
        ):
            self.x[3, 0] = 0.0
            if self.use_odom_w:
                self.x[4, 0] = 0.0
            self.get_logger().warn(
                "Odometry timeout in custom EKF. Holding odom velocity at 0.",
                throttle_duration_sec=2.0,
            )

        if self.use_imu_w and (
            self.last_imu_time is None
            or now_s - self.last_imu_time > self.sensor_timeout
        ):
            self.x[4, 0] = 0.0
            self.get_logger().warn(
                "IMU timeout in custom EKF. Holding yaw rate at 0.",
                throttle_duration_sec=2.0,
            )

    def align_camera_msg(self, msg):
        cam_x = msg.pose.pose.position.x
        cam_y = msg.pose.pose.position.y
        cam_yaw = yaw_from_quaternion(msg.pose.pose.orientation)

        if not self.align_camera_to_odom:
            # Không align: Sử dụng trực tiếp hệ tọa độ Camera làm hệ tọa độ EKF World
            aligned = Odometry()
            aligned.header.stamp = msg.header.stamp
            aligned.header.frame_id = self.odom_frame
            aligned.child_frame_id = self.base_link_frame
            aligned.pose.pose.position.x = float(cam_x)
            aligned.pose.pose.position.y = float(cam_y)
            aligned.pose.pose.position.z = 0.0
            aligned.pose.pose.orientation = msg.pose.pose.orientation
            aligned.pose.covariance = list(msg.pose.covariance)
            aligned.twist.twist = msg.twist.twist
            aligned.twist.covariance = list(msg.twist.covariance)
            return aligned

        if self.camera_alignment is None:
            self.camera_alignment = {
                'cam_x0': cam_x,
                'cam_y0': cam_y,
                'cam_yaw0': cam_yaw,
                'odom_x0': float(self.x[0, 0]),
                'odom_y0': float(self.x[1, 0]),
                'odom_yaw0': float(self.x[2, 0]),
            }
            self.get_logger().info(
                "Camera aligned to odom frame: "
                f"cam=({cam_x:.3f}, {cam_y:.3f}, {math.degrees(cam_yaw):.1f} deg), "
                f"odom=({self.x[0,0]:.3f}, {self.x[1,0]:.3f}, "
                f"{math.degrees(self.x[2,0]):.1f} deg)"
            )

        a = self.camera_alignment
        dx = cam_x - a['cam_x0']
        dy = cam_y - a['cam_y0']

        if self.use_camera_yaw_for_alignment:
            rot = a['odom_yaw0'] - a['cam_yaw0']
        else:
            rot = 0.0

        cos_r = math.cos(rot)
        sin_r = math.sin(rot)

        aligned_x = a['odom_x0'] + cos_r * dx - sin_r * dy
        aligned_y = a['odom_y0'] + sin_r * dx + cos_r * dy
        aligned_yaw = normalize_angle(
            a['odom_yaw0'] + normalize_angle(cam_yaw - a['cam_yaw0'])
        )

        aligned = Odometry()
        aligned.header.stamp = msg.header.stamp
        aligned.header.frame_id = self.odom_frame
        aligned.child_frame_id = self.base_link_frame
        aligned.pose.pose.position.x = float(aligned_x)
        aligned.pose.pose.position.y = float(aligned_y)
        aligned.pose.pose.position.z = 0.0

        qx, qy, qz, qw = quaternion_from_yaw(aligned_yaw)
        aligned.pose.pose.orientation.x = qx
        aligned.pose.pose.orientation.y = qy
        aligned.pose.pose.orientation.z = qz
        aligned.pose.pose.orientation.w = qw
        aligned.pose.covariance = list(msg.pose.covariance)
        aligned.twist.twist = msg.twist.twist
        aligned.twist.covariance = list(msg.twist.covariance)
        return aligned

    def publish_odometry(self, stamp):
        msg = Odometry()
        msg.header.stamp = stamp
        msg.header.frame_id = self.odom_frame
        msg.child_frame_id = self.base_link_frame

        msg.pose.pose.position.x = float(self.x[0, 0])
        msg.pose.pose.position.y = float(self.x[1, 0])
        msg.pose.pose.position.z = 0.0

        qx, qy, qz, qw = quaternion_from_yaw(self.x[2, 0])
        msg.pose.pose.orientation.x = qx
        msg.pose.pose.orientation.y = qy
        msg.pose.pose.orientation.z = qz
        msg.pose.pose.orientation.w = qw

        msg.twist.twist.linear.x = float(self.x[3, 0])
        msg.twist.twist.linear.y = 0.0
        msg.twist.twist.angular.z = float(self.x[4, 0])

        msg.pose.covariance[0] = float(self.p[0, 0])
        msg.pose.covariance[7] = float(self.p[1, 1])
        msg.pose.covariance[14] = 1e6
        msg.pose.covariance[21] = 1e6
        msg.pose.covariance[28] = 1e6
        msg.pose.covariance[35] = float(self.p[2, 2])

        msg.twist.covariance[0] = float(self.p[3, 3])
        msg.twist.covariance[7] = 1e6
        msg.twist.covariance[14] = 1e6
        msg.twist.covariance[21] = 1e6
        msg.twist.covariance[28] = 1e6
        msg.twist.covariance[35] = float(self.p[4, 4])

        self.odom_pub.publish(msg)

    def publish_transform(self, stamp):
        msg = TransformStamped()
        msg.header.stamp = stamp
        msg.header.frame_id = self.odom_frame
        msg.child_frame_id = self.base_link_frame
        msg.transform.translation.x = float(self.x[0, 0])
        msg.transform.translation.y = float(self.x[1, 0])
        msg.transform.translation.z = 0.0

        qx, qy, qz, qw = quaternion_from_yaw(self.x[2, 0])
        msg.transform.rotation.x = qx
        msg.transform.rotation.y = qy
        msg.transform.rotation.z = qz
        msg.transform.rotation.w = qw
        self.tf_broadcaster.sendTransform(msg)


def main(args=None):
    rclpy.init(args=args)
    node = CustomEkfNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
