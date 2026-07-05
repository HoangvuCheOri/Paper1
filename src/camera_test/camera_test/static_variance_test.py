#!/usr/bin/env python3
"""
Measure camera pose variance while the robot is stationary.

Test 1: Robot đứng yên 60s → Log /odom_camera → Tính std_x, std_y, std_yaw
→ Đây chính là R thực tế cho EKF.

Cách chạy:
  Terminal 1: ros2 run amr_control camera_node
  Terminal 2:
    cd ~/ros2_ws && source install/setup.bash
    python3 src/camera_test/camera_test/static_variance_test.py

Robot phải ĐỨNG YÊN trong khi chạy script này!
"""

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Pose2D
import math
import numpy as np
import time
import sys


def yaw_from_quaternion(q):
    t3 = 2.0 * (q.w * q.z + q.x * q.y)
    t4 = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(t3, t4)


class StaticVarianceTest(Node):
    def __init__(self, duration=60.0):
        super().__init__('static_variance_test')

        self.duration = duration
        self.data_x = []
        self.data_y = []
        self.data_yaw = []
        self.start_time = None

        # Subscribe cả 2 topic camera
        self.cam_odom_sub = self.create_subscription(
            Odometry, '/odom_camera', self.odom_camera_cb, 10
        )
        self.cam_pose_sub = self.create_subscription(
            Pose2D, '/apriltag_pose', self.apriltag_pose_cb, 10
        )
        self.source = None

        self.timer = self.create_timer(1.0, self.progress_report)

        self.get_logger().info(
            f"=== STATIC VARIANCE TEST ===\n"
            f"  Robot phải ĐỨNG YÊN!\n"
            f"  Thu thập dữ liệu trong {self.duration:.0f} giây...\n"
            f"  Đang chờ dữ liệu từ /odom_camera hoặc /apriltag_pose..."
        )

    def odom_camera_cb(self, msg):
        if self.source is None:
            self.source = '/odom_camera'
            self.get_logger().info(f"Nhận dữ liệu từ {self.source}")
        if self.source != '/odom_camera':
            return

        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        yaw = yaw_from_quaternion(msg.pose.pose.orientation)
        self._record(x, y, yaw)

    def apriltag_pose_cb(self, msg):
        if self.source is None:
            self.source = '/apriltag_pose'
            self.get_logger().info(f"Nhận dữ liệu từ {self.source}")
        if self.source != '/apriltag_pose':
            return

        self._record(msg.x, msg.y, math.radians(msg.theta))

    def _record(self, x, y, yaw_rad):
        if self.start_time is None:
            self.start_time = time.time()

        elapsed = time.time() - self.start_time
        if elapsed > self.duration:
            self.compute_and_exit()
            return

        self.data_x.append(x)
        self.data_y.append(y)
        self.data_yaw.append(yaw_rad)

    def progress_report(self):
        if self.start_time is None:
            return
        elapsed = time.time() - self.start_time
        n = len(self.data_x)
        remaining = max(0, self.duration - elapsed)
        self.get_logger().info(
            f"  [{elapsed:.0f}/{self.duration:.0f}s] Đã thu {n} mẫu | Còn {remaining:.0f}s"
        )

    def compute_and_exit(self):
        n = len(self.data_x)
        if n < 10:
            self.get_logger().error(f"Chỉ thu được {n} mẫu, quá ít! Kiểm tra camera.")
            rclpy.shutdown()
            return

        x = np.array(self.data_x)
        y = np.array(self.data_y)
        yaw = np.array(self.data_yaw)

        # Tính mean và std
        mean_x, std_x = np.mean(x), np.std(x)
        mean_y, std_y = np.mean(y), np.std(y)
        mean_yaw_deg = np.degrees(np.mean(yaw))
        std_yaw = np.std(yaw)
        std_yaw_deg = np.degrees(std_yaw)

        # Variance = std^2 → đây chính là R cho EKF
        var_x = std_x ** 2
        var_y = std_y ** 2
        var_yaw = std_yaw ** 2

        report = f"""
╔══════════════════════════════════════════════════════════════╗
║             KẾT QUẢ TEST TĨNH (STATIC VARIANCE)            ║
╠══════════════════════════════════════════════════════════════╣
║  Nguồn dữ liệu: {self.source:<40s}  ║
║  Số mẫu:        {n:<40d}  ║
║  Thời gian:     {self.duration:.0f}s{'':<38s}  ║
╠══════════════════════════════════════════════════════════════╣
║                    MEAN (Trung bình)                        ║
║  X:    {mean_x:+.4f} m                                        ║
║  Y:    {mean_y:+.4f} m                                        ║
║  Yaw:  {mean_yaw_deg:+.2f} deg                                     ║
╠══════════════════════════════════════════════════════════════╣
║                  STD (Độ lệch chuẩn)                        ║
║  std_x:    {std_x:.6f} m   ({std_x*100:.2f} cm)                    ║
║  std_y:    {std_y:.6f} m   ({std_y*100:.2f} cm)                    ║
║  std_yaw:  {std_yaw_deg:.4f} deg  ({std_yaw:.6f} rad)              ║
╠══════════════════════════════════════════════════════════════╣
║           HỆ SỐ R CHO EKF (variance = std²)                ║
║                                                              ║
║  camera_x_variance:   {var_x:.6f}                              ║
║  camera_y_variance:   {var_y:.6f}                              ║
║  camera_yaw_variance: {var_yaw:.6f}                            ║
║                                                              ║
║  Copy vào custom_ekf.yaml:                                   ║
║    camera_x_variance: {var_x:.6f}                              ║
║    camera_y_variance: {var_y:.6f}                              ║
║    camera_yaw_variance: {var_yaw:.6f}                          ║
╚══════════════════════════════════════════════════════════════╝
"""
        print(report)
        self.get_logger().info("Test hoàn tất! Bạn có thể dừng chương trình (Ctrl+C).")
        rclpy.shutdown()


def main(args=None):
    rclpy.init(args=args)

    duration = 30.0  # Mặc định 30 giây, đủ để có khoảng 900 mẫu ở 30Hz
    if len(sys.argv) > 1:
        try:
            duration = float(sys.argv[1])
        except ValueError:
            pass

    node = StaticVarianceTest(duration=duration)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()


if __name__ == '__main__':
    main()
