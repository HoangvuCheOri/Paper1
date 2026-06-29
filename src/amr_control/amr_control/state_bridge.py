import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, Empty
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu
import math
import time


def quaternion_from_euler(roll, pitch, yaw):
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    q = [0.0] * 4
    q[0] = sr * cp * cy - cr * sp * sy
    q[1] = cr * sp * cy + sr * cp * sy
    q[2] = cr * cp * sy - sr * sp * cy
    q[3] = cr * cp * cy + sr * sp * sy
    return q


class StateBridgeNode(Node):
    """
    Nhận raw sensor data từ robot_serial_bridge (/robot_state: 3 trường):
      [0] rpmL_signed * 10
      [1] rpmR_signed * 10
      [2] gyro_z * 1000  (rad/s)

    Publish:
      /odom_raw  (Odometry) → EKF
      /imu/data  (Imu)      → EKF

    FIX ESP-NOW:
      - dt clamp chặt hơn (max 0.12s thay vì 0.2s) để 1 gói drop không
        gây odometry jump lớn.
      - Nếu dt > PACKET_LOSS_THRESHOLD: coi như v=0, w=0 trong chu kỳ đó
        (robot không dịch chuyển được nhiều trong thời gian mất gói ngắn)
        → tránh odometry vọt xa so với thực tế.
      - Low-pass filter nhẹ trên v và w để giảm chattering từ jitter packet.
    """

    WHEEL_DIAMETER = 0.063
    WHEEL_BASE     = 0.17
    PPR            = 937.0
    DT             = 0.05          # Nominal period (s)
    WHEEL_CALIB_L  = 1.0

    # Nếu dt > ngưỡng này → coi như mất gói, KHÔNG tích phân odometry
    # (0.12s = chịu được 1 gói drop, ≈2.4×DT nominal)
    PACKET_LOSS_THRESHOLD = 0.12

    ODOM_SCALE_FACTOR = 1.0

    # Low-pass filter coefficient cho v và w  (0=giữ nguyên cũ, 1=không lọc)
    # 0.6 → giảm jitter mà không lag quá nhiều
    LPF_ALPHA = 0.6

    def __init__(self):
        super().__init__('state_bridge_node')

        self.state_sub = self.create_subscription(
            Float32MultiArray, '/robot_state', self.state_callback, 10
        )
        self.reset_sub = self.create_subscription(
            Empty, '/reset_odom', self.reset_callback, 10
        )

        self.odom_pub = self.create_publisher(Odometry, '/odom_raw', 10)
        self.imu_pub  = self.create_publisher(Imu,      '/imu/data', 10)

        self.x     = 0.0
        self.y     = 0.0
        self.theta = 0.0

        self._last_time = None

        # LPF state
        self._v_filt = 0.0
        self._w_filt = 0.0

        # Diagnostic: đếm gói bị drop
        self._pkt_total  = 0
        self._pkt_drop   = 0

        self.get_logger().info(
            "StateBridge (ESP-NOW hardened): "
            "PACKET_LOSS_THRESHOLD=%.3fs  LPF_ALPHA=%.2f" %
            (self.PACKET_LOSS_THRESHOLD, self.LPF_ALPHA)
        )

    def reset_callback(self, msg):
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0
        self._last_time = None
        self._v_filt = 0.0
        self._w_filt = 0.0
        self.get_logger().info("Odometry reset to (0, 0, 0)")

    def state_callback(self, msg):
        if len(msg.data) < 3:
            self.get_logger().warn(f"Expected 3 fields, got {len(msg.data)}")
            return

        # ── Đo dt thực tế ───────────────────────────────────────────────
        now = time.monotonic()
        if self._last_time is None:
            dt = self.DT
        else:
            dt = now - self._last_time
        self._last_time = now

        self._pkt_total += 1

        # ── Phát hiện mất gói ESP-NOW ───────────────────────────────────
        # Nếu dt quá lớn: nhiều khả năng có 1+ gói bị drop trước gói này.
        # Tích phân với dt lớn sẽ làm odometry nhảy. Dùng dt nominal thay thế
        # nhưng log cảnh báo để biết tỉ lệ drop.
        if dt > self.PACKET_LOSS_THRESHOLD:
            self._pkt_drop += 1
            drop_rate = 100.0 * self._pkt_drop / self._pkt_total
            self.get_logger().warn(
                f"[ESP-NOW] Packet delay {dt*1000:.0f}ms — "
                f"drop rate {drop_rate:.1f}% ({self._pkt_drop}/{self._pkt_total})",
                throttle_duration_sec=5.0
            )
            # Clamp dt về nominal: tránh odometry jump
            # (Chấp nhận tích phân nhỏ hơn thực tế trong 1 chu kỳ mất gói)
            dt = self.DT
        else:
            # Clamp dt bình thường trong khoảng hợp lệ
            dt = max(0.02, min(dt, self.PACKET_LOSS_THRESHOLD))

        # ── Giải mã ─────────────────────────────────────────────────────
        rpm_L_raw = float(msg.data[0]) / 10.0
        rpm_R     = float(msg.data[1]) / 10.0
        # Đảo dấu gyro_z để khớp chuẩn ROS2 (quay CCW là dương)
        gyro_z    = float(msg.data[2]) / 1000.0

        # Hệ số hiệu chỉnh bánh trái (khớp STM32 1.22f)
        rpm_L = math.copysign(abs(rpm_L_raw) * self.WHEEL_CALIB_L, rpm_L_raw)

        # ── Forward kinematics ──────────────────────────────────────────
        v_L = rpm_L * (math.pi * self.WHEEL_DIAMETER / 60.0)
        v_R = rpm_R * (math.pi * self.WHEEL_DIAMETER / 60.0)

        v_raw     = (v_R + v_L) / 2.0
        w_enc_raw = (v_R - v_L) / self.WHEEL_BASE

        # ── Low-pass filter (giảm chattering từ jitter ESP-NOW) ─────────
        v     = self.LPF_ALPHA * v_raw     + (1.0 - self.LPF_ALPHA) * self._v_filt
        w_enc = self.LPF_ALPHA * w_enc_raw + (1.0 - self.LPF_ALPHA) * self._w_filt
        self._v_filt = v
        self._w_filt = w_enc

        # KHÔNG fusion gyro vào w ở đây — EKF tự fusion /imu/data
        w = w_enc

        # ── Tích phân odometry (midpoint method) ────────────────────────
        theta_old  = self.theta
        self.theta += w * dt
        self.theta  = math.atan2(math.sin(self.theta), math.cos(self.theta))

        theta_mid = (theta_old + self.theta) / 2.0
        self.x += v * math.cos(theta_mid) * dt * self.ODOM_SCALE_FACTOR
        self.y += v * math.sin(theta_mid) * dt * self.ODOM_SCALE_FACTOR

        # ── Publish ──────────────────────────────────────────────────────
        stamp = self.get_clock().now().to_msg()
        self._publish_odometry(stamp, v, w)
        self._publish_imu(stamp, gyro_z)

    def _publish_odometry(self, stamp, v, w):
        odom = Odometry()
        odom.header.stamp    = stamp
        odom.header.frame_id = 'odom'
        odom.child_frame_id  = 'base_link'

        odom.pose.pose.position.x = self.x
        odom.pose.pose.position.y = self.y
        odom.pose.pose.position.z = 0.0

        q = quaternion_from_euler(0.0, 0.0, self.theta)
        odom.pose.pose.orientation.x = q[0]
        odom.pose.pose.orientation.y = q[1]
        odom.pose.pose.orientation.z = q[2]
        odom.pose.pose.orientation.w = q[3]

        # Tăng độ tin cậy của wheel odom khi chạy không camera
        odom.pose.covariance[0]  = 0.01   # x
        odom.pose.covariance[7]  = 0.01   # y
        odom.pose.covariance[14] = 1e6
        odom.pose.covariance[21] = 1e6
        odom.pose.covariance[28] = 1e6
        odom.pose.covariance[35] = 0.03   # yaw

        odom.twist.twist.linear.x  = v
        odom.twist.twist.linear.y  = 0.0
        odom.twist.twist.angular.z = w

        odom.twist.covariance[0]  = 0.03
        odom.twist.covariance[7]  = 1e6
        odom.twist.covariance[14] = 1e6
        odom.twist.covariance[21] = 1e6
        odom.twist.covariance[28] = 1e6
        odom.twist.covariance[35] = 1.2   # tăng từ 1.0

        self.odom_pub.publish(odom)

    def _publish_imu(self, stamp, gyro_z):
        imu = Imu()
        imu.header.stamp    = stamp
        imu.header.frame_id = 'base_link'

        imu.orientation_covariance[0] = -1.0

        imu.angular_velocity.x = 0.0
        imu.angular_velocity.y = 0.0
        imu.angular_velocity.z = gyro_z

        imu.angular_velocity_covariance[0] = 1e6
        imu.angular_velocity_covariance[4] = 1e6
        imu.angular_velocity_covariance[8] = 0.005

        imu.linear_acceleration_covariance[0] = -1.0

        self.imu_pub.publish(imu)


def main(args=None):
    rclpy.init(args=args)
    node = StateBridgeNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()